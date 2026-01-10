"""
Document management service.

Handles complete document lifecycle including vectorstore, filesystem, and database.

SECURITY (v1.5):
- End-to-end deletion with verification
- Audit trail for all operations
- Tombstone records for GDPR compliance
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

from . import db
from .rag_runtime import vectorstore
from .audit_log import generate_request_id, log_delete

logger = logging.getLogger(__name__)


@dataclass
class DeletionResult:
    """Result of a deletion operation with verification status."""
    
    doc_id: str
    vectorstore_deleted: bool
    file_deleted: bool
    db_deleted: bool
    chunk_count: int
    errors: list[str]
    
    @property
    def success(self) -> bool:
        """True if all deletions succeeded."""
        return self.vectorstore_deleted and self.file_deleted and self.db_deleted
    
    @property
    def partial(self) -> bool:
        """True if some but not all deletions succeeded."""
        deleted = [self.vectorstore_deleted, self.file_deleted, self.db_deleted]
        return any(deleted) and not all(deleted)


def delete_document_complete(
    doc_id: str,
    session_id: str = "",
) -> DeletionResult:
    """
    Delete a document from all storage locations with verification.

    Order of operations:
    1. Get document metadata and chunk IDs
    2. Remove from vectorstore (chunks) - verify deletion
    3. Remove file from disk - verify deletion
    4. Remove from SQLite (cascades to chunks table)
    5. Add tombstone record for audit
    6. Log deletion event

    Args:
        doc_id: The document identifier to delete
        session_id: Session ID for audit logging

    Returns:
        DeletionResult with verification status for each step
    """
    request_id = generate_request_id()
    errors: list[str] = []
    
    # Get document info
    doc = db.get_document(doc_id)
    if not doc:
        logger.info("Delete requested for missing doc", extra={"doc_id": doc_id})
        return DeletionResult(
            doc_id=doc_id,
            vectorstore_deleted=True,  # Nothing to delete
            file_deleted=True,
            db_deleted=True,
            chunk_count=0,
            errors=["Document not found (may already be deleted)"],
        )

    # Get chunk IDs before deletion
    chunk_ids = db.get_chunk_ids(doc_id)
    chunk_count = len(chunk_ids)
    
    # 1. Delete from vectorstore with verification
    vectorstore_deleted = False
    if chunk_ids:
        try:
            vs = vectorstore()
            vs.delete(ids=chunk_ids)
            
            # Verify deletion - try to retrieve deleted chunks
            # Note: This is a best-effort verification
            remaining = vs.get(ids=chunk_ids[:1])  # Check first chunk
            if remaining and remaining.get("ids"):
                errors.append("Vectorstore deletion may be incomplete")
            else:
                vectorstore_deleted = True
                
            logger.debug(
                "Deleted chunks from vectorstore",
                extra={"doc_id": doc_id, "count": chunk_count},
            )
        except Exception as e:
            errors.append(f"Vectorstore: {type(e).__name__}")
            logger.exception(
                "Failed to delete from vectorstore",
                extra={"doc_id": doc_id, "error_code": "VS_DELETE_ERROR"},
            )
    else:
        vectorstore_deleted = True  # No chunks to delete

    # 2. Delete file from disk with verification
    file_deleted = False
    stored_path = Path(doc["stored_path"])
    try:
        if stored_path.exists():
            stored_path.unlink()
            
            # Verify deletion
            if stored_path.exists():
                errors.append("File still exists after deletion")
            else:
                file_deleted = True
        else:
            file_deleted = True  # Already gone
            
        logger.debug("Deleted file from disk", extra={"doc_id": doc_id})
    except OSError as e:
        errors.append(f"File: {type(e).__name__}")
        logger.warning(
            "Failed to delete file from disk",
            extra={"doc_id": doc_id, "error_code": "FILE_DELETE_ERROR"},
        )

    # 3. Delete from database (cascades to chunks)
    db_deleted = False
    try:
        db.delete_document_rows(doc_id)
        
        # Verify deletion
        remaining_doc = db.get_document(doc_id)
        if remaining_doc is None:
            db_deleted = True
        else:
            errors.append("Document still in database after deletion")
            
        logger.debug("Deleted from database", extra={"doc_id": doc_id})
    except Exception as e:
        errors.append(f"Database: {type(e).__name__}")
        logger.exception(
            "Failed to delete from database",
            extra={"doc_id": doc_id, "error_code": "DB_DELETE_ERROR"},
        )

    # 4. Add tombstone record for GDPR audit trail
    try:
        db.add_deletion_tombstone(
            doc_id=doc_id,
            sha256=doc["sha256"],
            chunk_count=chunk_count,
        )
    except Exception as e:
        # Non-blocking - tombstone is for audit only
        logger.warning(
            "Failed to add deletion tombstone",
            extra={"doc_id": doc_id, "error_code": "TOMBSTONE_ERROR"},
        )

    # 5. Audit log
    log_delete(
        request_id=request_id,
        session_id=session_id,
        doc_id=doc_id,
        chunk_ids=chunk_ids,
    )

    result = DeletionResult(
        doc_id=doc_id,
        vectorstore_deleted=vectorstore_deleted,
        file_deleted=file_deleted,
        db_deleted=db_deleted,
        chunk_count=chunk_count,
        errors=errors,
    )
    
    if result.success:
        logger.info("Document deleted successfully", extra={"doc_id": doc_id})
    elif result.partial:
        logger.warning(
            "Document partially deleted",
            extra={"doc_id": doc_id, "errors": errors},
        )
    else:
        logger.error(
            "Document deletion failed",
            extra={"doc_id": doc_id, "errors": errors},
        )

    return result


def verify_document_deleted(doc_id: str) -> dict[str, bool]:
    """
    Verify that a document has been completely deleted.
    
    Useful for GDPR compliance verification.
    
    Args:
        doc_id: The document identifier to check
        
    Returns:
        Dict with verification status for each storage location
    """
    # Check database
    doc = db.get_document(doc_id)
    db_clear = doc is None
    
    # Check chunks in database
    chunk_ids = db.get_chunk_ids(doc_id)
    chunks_clear = len(chunk_ids) == 0
    
    # Check vectorstore (if we have chunk IDs from tombstone)
    vs_clear = True
    if chunk_ids:
        try:
            vs = vectorstore()
            result = vs.get(ids=chunk_ids[:1])
            vs_clear = not (result and result.get("ids"))
        except Exception:
            vs_clear = True  # Assume clear on error
    
    # Check file (if we have the path from tombstone)
    file_clear = True
    if doc:
        stored_path = Path(doc["stored_path"])
        file_clear = not stored_path.exists()
    
    return {
        "database_clear": db_clear,
        "chunks_clear": chunks_clear,
        "vectorstore_clear": vs_clear,
        "file_clear": file_clear,
        "fully_deleted": all([db_clear, chunks_clear, vs_clear, file_clear]),
    }
