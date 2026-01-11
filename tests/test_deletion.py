"""
Tests for end-to-end document deletion.
"""

from unittest.mock import patch

from backend.documents import DeletionResult, delete_document_complete


def test_deletion_result_success():
    """Test DeletionResult success property."""
    result = DeletionResult(
        doc_id="test",
        vectorstore_deleted=True,
        file_deleted=True,
        db_deleted=True,
        chunk_count=5,
        errors=[],
    )
    assert result.success
    assert not result.partial


def test_deletion_result_partial():
    """Test DeletionResult partial property."""
    result = DeletionResult(
        doc_id="test",
        vectorstore_deleted=True,
        file_deleted=False,
        db_deleted=True,
        chunk_count=5,
        errors=["File: PermissionError"],
    )
    assert not result.success
    assert result.partial


def test_deletion_result_failure():
    """Test DeletionResult when nothing deleted."""
    result = DeletionResult(
        doc_id="test",
        vectorstore_deleted=False,
        file_deleted=False,
        db_deleted=False,
        chunk_count=5,
        errors=["Error 1", "Error 2"],
    )
    assert not result.success
    assert not result.partial


def test_delete_missing_document():
    """Test deleting a document that doesn't exist."""
    with patch("backend.documents.db") as mock_db:
        mock_db.get_document.return_value = None

        result = delete_document_complete("nonexistent")

        assert result.success  # Nothing to delete = success
        assert "not found" in result.errors[0].lower()
