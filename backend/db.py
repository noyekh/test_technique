"""
Database layer for document and conversation persistence.

Design decisions:
- SQLite chosen for PoC simplicity (no external DB server required)
- Row factory for dict-like access to results
- PRAGMA foreign_keys=ON for cascade deletes (chunks when doc deleted)
- UUID hex for IDs (no collision risk, URL-safe)
- ISO 8601 timestamps in UTC for consistency

Tables:
- documents: uploaded files metadata + SHA256 for dedup
- chunks: mapping doc_id -> chunk_ids for vectorstore sync
- conversations: chat sessions
- messages: chat history per conversation
- deletions: tombstone records for GDPR audit trail (v1.5)

For production, consider:
- Connection pooling (sqlite3 doesn't need it, but PostgreSQL would)
- Async operations (aiosqlite)
- Migrations (alembic)
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Iterable, Optional

from .settings import SQLITE_PATH


def _utcnow() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _id() -> str:
    """Generate a new UUID hex string."""
    return uuid.uuid4().hex


def connect() -> sqlite3.Connection:
    """
    Create a database connection with proper configuration.
    
    Configuration:
    - check_same_thread=False: Allows use from Streamlit's threading model
    - row_factory=sqlite3.Row: Dict-like access to query results
    - PRAGMA foreign_keys=ON: Enables cascade deletes
    """
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SQLITE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """
    Initialize database schema.
    
    Idempotent: safe to call multiple times.
    Creates tables and indexes if they don't exist.
    """
    conn = connect()
    cur = conn.cursor()

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        original_name TEXT NOT NULL,
        stored_path TEXT NOT NULL,
        ext TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        created_at TEXT NOT NULL
    );
    """
    )

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
    );
    """
    )

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS conversations (
        conv_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """
    )

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS messages (
        msg_id TEXT PRIMARY KEY,
        conv_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(conv_id) REFERENCES conversations(conv_id) ON DELETE CASCADE
    );
    """
    )
    
    # v1.5: Deletion tombstones for GDPR audit trail
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS deletions (
        deletion_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        chunk_count INTEGER NOT NULL,
        deleted_at TEXT NOT NULL
    );
    """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conv_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_deletions_doc_id ON deletions(doc_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_deletions_sha256 ON deletions(sha256);")

    conn.commit()
    conn.close()


# ---------------- documents ----------------

def add_document(
    original_name: str, stored_path: str, ext: str, sha256: str, size_bytes: int
) -> str:
    """
    Add a new document to the database.
    
    Args:
        original_name: Original filename as uploaded
        stored_path: Path where file is stored on disk
        ext: File extension (without dot)
        sha256: Content hash for deduplication
        size_bytes: File size in bytes
        
    Returns:
        Generated document ID
    """
    doc_id = _id()
    conn = connect()
    conn.execute(
        """
        INSERT INTO documents(doc_id, original_name, stored_path, ext, sha256, size_bytes, created_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (doc_id, original_name, stored_path, ext, sha256, size_bytes, _utcnow()),
    )
    conn.commit()
    conn.close()
    return doc_id


def list_documents():
    """
    List all documents, most recent first.
    
    Returns:
        List of document Row objects
    """
    conn = connect()
    rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    conn.close()
    return rows


def get_document(doc_id: str):
    """
    Get a single document by ID.
    
    Args:
        doc_id: Document identifier
        
    Returns:
        Document Row or None if not found
    """
    conn = connect()
    row = conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    conn.close()
    return row


def get_document_by_sha256(sha256: str):
    """
    Find a document by content hash.
    
    Used for deduplication on upload.
    
    Args:
        sha256: Content hash to search for
        
    Returns:
        Document Row or None if not found
    """
    conn = connect()
    row = conn.execute("SELECT * FROM documents WHERE sha256=?", (sha256,)).fetchone()
    conn.close()
    return row


def add_chunks(doc_id: str, chunk_ids: Iterable[str]) -> None:
    """
    Record chunk IDs for a document.
    
    This creates the mapping between doc_id and the chunk IDs
    stored in the vectorstore, enabling proper cleanup on deletion.
    
    Args:
        doc_id: Parent document ID
        chunk_ids: Iterable of chunk IDs as stored in vectorstore
    """
    conn = connect()
    conn.executemany(
        "INSERT INTO chunks(chunk_id, doc_id, chunk_index) VALUES (?,?,?)",
        [(cid, doc_id, i) for i, cid in enumerate(chunk_ids)],
    )
    conn.commit()
    conn.close()


def get_chunk_ids(doc_id: str) -> list[str]:
    """
    Get all chunk IDs for a document.
    
    Used during deletion to remove chunks from vectorstore.
    
    Args:
        doc_id: Document identifier
        
    Returns:
        List of chunk IDs in order
    """
    conn = connect()
    rows = conn.execute(
        "SELECT chunk_id FROM chunks WHERE doc_id=? ORDER BY chunk_index ASC", (doc_id,)
    ).fetchall()
    conn.close()
    return [r["chunk_id"] for r in rows]


def delete_document_rows(doc_id: str) -> None:
    """
    Delete document from database.
    
    Note: This only removes the database record. For complete deletion
    including vectorstore and filesystem, use documents.delete_document_complete().
    
    CASCADE delete will automatically remove associated chunks.
    
    Args:
        doc_id: Document identifier to delete
    """
    conn = connect()
    conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
    conn.commit()
    conn.close()


# ---------------- deletions (v1.5 tombstones) ----------------

def add_deletion_tombstone(doc_id: str, sha256: str, chunk_count: int) -> str:
    """
    Add a tombstone record for a deleted document.
    
    This creates an audit trail for GDPR compliance, recording that
    a document was deleted without storing its content.
    
    Args:
        doc_id: The deleted document's ID
        sha256: Content hash of the deleted document
        chunk_count: Number of chunks that were deleted
        
    Returns:
        Generated deletion ID
    """
    deletion_id = _id()
    conn = connect()
    conn.execute(
        """
        INSERT INTO deletions(deletion_id, doc_id, sha256, chunk_count, deleted_at)
        VALUES (?,?,?,?,?)
        """,
        (deletion_id, doc_id, sha256, chunk_count, _utcnow()),
    )
    conn.commit()
    conn.close()
    return deletion_id


def get_deletion_tombstone(doc_id: str):
    """
    Get deletion tombstone for a document.
    
    Args:
        doc_id: Document identifier
        
    Returns:
        Deletion Row or None if no tombstone exists
    """
    conn = connect()
    row = conn.execute(
        "SELECT * FROM deletions WHERE doc_id=? ORDER BY deleted_at DESC LIMIT 1",
        (doc_id,)
    ).fetchone()
    conn.close()
    return row


def list_deletions(limit: int = 100):
    """
    List recent deletion tombstones.
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        List of deletion Row objects
    """
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM deletions ORDER BY deleted_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return rows


# ---------------- conversations ----------------

def ensure_default_conversation() -> str:
    """
    Ensure at least one conversation exists.
    
    Creates a default conversation if none exists.
    
    Returns:
        ID of the first/default conversation
    """
    conn = connect()
    row = conn.execute(
        "SELECT conv_id FROM conversations ORDER BY created_at ASC LIMIT 1"
    ).fetchone()
    if row:
        conn.close()
        return row["conv_id"]

    conv_id = _id()
    conn.execute(
        "INSERT INTO conversations(conv_id, title, created_at) VALUES (?,?,?)",
        (conv_id, "Conversation 1", _utcnow()),
    )
    conn.commit()
    conn.close()
    return conv_id


def new_conversation(title: Optional[str] = None) -> str:
    """
    Create a new conversation.
    
    Args:
        title: Optional title, defaults to "Conversation {id[:6]}"
        
    Returns:
        Generated conversation ID
    """
    conv_id = _id()
    title = title or f"Conversation {conv_id[:6]}"
    conn = connect()
    conn.execute(
        "INSERT INTO conversations(conv_id, title, created_at) VALUES (?,?,?)",
        (conv_id, title, _utcnow()),
    )
    conn.commit()
    conn.close()
    return conv_id


def list_conversations():
    """
    List all conversations, most recent first.
    
    Returns:
        List of conversation Row objects
    """
    conn = connect()
    rows = conn.execute("SELECT * FROM conversations ORDER BY created_at DESC").fetchall()
    conn.close()
    return rows


def add_message(conv_id: str, role: str, content: str) -> None:
    """
    Add a message to a conversation.
    
    Args:
        conv_id: Conversation ID
        role: Message role ("user" or "assistant")
        content: Message content
    """
    conn = connect()
    conn.execute(
        "INSERT INTO messages(msg_id, conv_id, role, content, created_at) VALUES (?,?,?,?,?)",
        (_id(), conv_id, role, content, _utcnow()),
    )
    conn.commit()
    conn.close()


def get_messages(conv_id: str):
    """
    Get all messages in a conversation.
    
    Args:
        conv_id: Conversation ID
        
    Returns:
        List of message Row objects in chronological order
    """
    conn = connect()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE conv_id=? ORDER BY created_at ASC",
        (conv_id,),
    ).fetchall()
    conn.close()
    return rows
