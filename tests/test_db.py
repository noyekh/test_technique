"""
Tests for database layer (SQLite persistence).

Tests cover:
- Document CRUD operations
- Chunk tracking for vectorstore sync
- Deletion tombstones for GDPR compliance
- Conversation and message persistence
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import db module once at module level
from backend import db as db_module


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        # Patch where it's used (db module), not where defined (settings)
        with patch.object(db_module, "SQLITE_PATH", db_path):
            db_module.init_db()
            yield db_module


def test_init_db_creates_tables(temp_db):
    """Test that init_db creates all required tables."""
    conn = temp_db.connect()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {row["name"] for row in cursor.fetchall()}
    conn.close()

    expected = {"documents", "chunks", "conversations", "messages", "deletions"}
    assert expected.issubset(tables)


def test_add_document_returns_id(temp_db):
    """Test that add_document returns a valid document ID."""
    doc_id = temp_db.add_document(
        original_name="test.txt",
        stored_path="/data/test.txt",
        ext="txt",
        sha256="abc123",
        size_bytes=100,
    )

    assert doc_id is not None
    assert len(doc_id) == 32  # UUID hex


def test_get_document_returns_correct_data(temp_db):
    """Test that get_document returns the saved data."""
    doc_id = temp_db.add_document(
        original_name="contract.pdf",
        stored_path="/data/contract.pdf",
        ext="pdf",
        sha256="sha256hash",
        size_bytes=2048,
    )

    doc = temp_db.get_document(doc_id)

    assert doc is not None
    assert doc["original_name"] == "contract.pdf"
    assert doc["stored_path"] == "/data/contract.pdf"
    assert doc["sha256"] == "sha256hash"
    assert doc["size_bytes"] == 2048


def test_get_document_returns_none_for_missing(temp_db):
    """Test that get_document returns None for non-existent document."""
    doc = temp_db.get_document("nonexistent_id")
    assert doc is None


def test_get_document_by_sha256(temp_db):
    """Test finding document by content hash."""
    temp_db.add_document(
        original_name="unique.txt",
        stored_path="/data/unique.txt",
        ext="txt",
        sha256="unique_hash_123",
        size_bytes=50,
    )

    doc = temp_db.get_document_by_sha256("unique_hash_123")

    assert doc is not None
    assert doc["original_name"] == "unique.txt"


def test_get_document_by_sha256_returns_none_for_missing(temp_db):
    """Test that sha256 lookup returns None when not found."""
    doc = temp_db.get_document_by_sha256("nonexistent_hash")
    assert doc is None


def test_list_documents_returns_all(temp_db):
    """Test that list_documents returns all documents."""
    temp_db.add_document("doc1.txt", "/data/1", "txt", "h1", 10)
    temp_db.add_document("doc2.txt", "/data/2", "txt", "h2", 20)
    temp_db.add_document("doc3.txt", "/data/3", "txt", "h3", 30)

    docs = temp_db.list_documents()

    assert len(docs) == 3


def test_list_documents_ordered_by_date_desc(temp_db):
    """Test that documents are returned most recent first."""
    temp_db.add_document("first.txt", "/data/1", "txt", "h1", 10)
    temp_db.add_document("second.txt", "/data/2", "txt", "h2", 20)
    temp_db.add_document("third.txt", "/data/3", "txt", "h3", 30)

    docs = temp_db.list_documents()

    # Most recent (third) should be first
    assert docs[0]["original_name"] == "third.txt"


# ============================================================================
# CHUNKS
# ============================================================================


def test_add_chunks_stores_mapping(temp_db):
    """Test that chunks are correctly associated with document."""
    doc_id = temp_db.add_document("doc.txt", "/data/doc", "txt", "hash", 100)
    chunk_ids = ["chunk_0", "chunk_1", "chunk_2"]

    temp_db.add_chunks(doc_id, chunk_ids)

    retrieved = temp_db.get_chunk_ids(doc_id)
    assert retrieved == chunk_ids


def test_get_chunk_ids_preserves_order(temp_db):
    """Test that chunk IDs are returned in correct order."""
    doc_id = temp_db.add_document("doc.txt", "/data/doc", "txt", "hash", 100)
    chunk_ids = ["z_last", "a_first", "m_middle"]

    temp_db.add_chunks(doc_id, chunk_ids)

    retrieved = temp_db.get_chunk_ids(doc_id)
    # Should preserve insertion order, not alphabetical
    assert retrieved == ["z_last", "a_first", "m_middle"]


def test_get_chunk_ids_returns_empty_for_missing(temp_db):
    """Test that get_chunk_ids returns empty list for non-existent document."""
    chunk_ids = temp_db.get_chunk_ids("nonexistent_doc")
    assert chunk_ids == []


def test_delete_document_cascades_to_chunks(temp_db):
    """Test that deleting a document also deletes its chunks."""
    doc_id = temp_db.add_document("doc.txt", "/data/doc", "txt", "hash", 100)
    temp_db.add_chunks(doc_id, ["c1", "c2", "c3"])

    # Verify chunks exist
    assert len(temp_db.get_chunk_ids(doc_id)) == 3

    # Delete document
    temp_db.delete_document_rows(doc_id)

    # Verify document and chunks are gone
    assert temp_db.get_document(doc_id) is None
    assert temp_db.get_chunk_ids(doc_id) == []


# ============================================================================
# DELETIONS (TOMBSTONES)
# ============================================================================


def test_add_deletion_tombstone(temp_db):
    """Test creating a deletion tombstone."""
    deletion_id = temp_db.add_deletion_tombstone(
        doc_id="deleted_doc_123",
        sha256="deleted_hash",
        chunk_count=5,
    )

    assert deletion_id is not None
    assert len(deletion_id) == 32


def test_get_deletion_tombstone(temp_db):
    """Test retrieving a deletion tombstone."""
    temp_db.add_deletion_tombstone(
        doc_id="doc_to_verify",
        sha256="verify_hash",
        chunk_count=10,
    )

    tombstone = temp_db.get_deletion_tombstone("doc_to_verify")

    assert tombstone is not None
    assert tombstone["sha256"] == "verify_hash"
    assert tombstone["chunk_count"] == 10


def test_get_deletion_tombstone_returns_none_for_missing(temp_db):
    """Test that tombstone lookup returns None when not found."""
    tombstone = temp_db.get_deletion_tombstone("never_deleted")
    assert tombstone is None


def test_list_deletions(temp_db):
    """Test listing deletion tombstones."""
    temp_db.add_deletion_tombstone("d1", "h1", 1)
    temp_db.add_deletion_tombstone("d2", "h2", 2)
    temp_db.add_deletion_tombstone("d3", "h3", 3)

    deletions = temp_db.list_deletions(limit=10)

    assert len(deletions) == 3


def test_list_deletions_respects_limit(temp_db):
    """Test that list_deletions respects the limit parameter."""
    for i in range(10):
        temp_db.add_deletion_tombstone(f"d{i}", f"h{i}", i)

    deletions = temp_db.list_deletions(limit=5)

    assert len(deletions) == 5


# ============================================================================
# CONVERSATIONS
# ============================================================================


def test_ensure_default_conversation_creates_one(temp_db):
    """Test that ensure_default_conversation creates a conversation if none exists."""
    conv_id = temp_db.ensure_default_conversation()

    assert conv_id is not None
    assert len(conv_id) == 32


def test_ensure_default_conversation_returns_same(temp_db):
    """Test that ensure_default_conversation returns the same ID on repeated calls."""
    conv_id_1 = temp_db.ensure_default_conversation()
    conv_id_2 = temp_db.ensure_default_conversation()

    assert conv_id_1 == conv_id_2


def test_new_conversation(temp_db):
    """Test creating a new conversation."""
    conv_id = temp_db.new_conversation(title="Test Chat")

    assert conv_id is not None
    convs = temp_db.list_conversations()
    titles = [c["title"] for c in convs]
    assert "Test Chat" in titles


def test_new_conversation_default_title(temp_db):
    """Test that new_conversation uses default title if none provided."""
    temp_db.new_conversation()

    convs = temp_db.list_conversations()
    assert len(convs) == 1
    assert convs[0]["title"] == "Nouvelle conversation"


def test_add_message_and_retrieve(temp_db):
    """Test adding and retrieving messages."""
    conv_id = temp_db.new_conversation("Test")

    temp_db.add_message(conv_id, "user", "Hello")
    temp_db.add_message(conv_id, "assistant", "Hi there!")

    messages = temp_db.get_messages(conv_id)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hi there!"


def test_messages_ordered_chronologically(temp_db):
    """Test that messages are returned in chronological order."""
    conv_id = temp_db.new_conversation("Test")

    temp_db.add_message(conv_id, "user", "First")
    temp_db.add_message(conv_id, "assistant", "Second")
    temp_db.add_message(conv_id, "user", "Third")

    messages = temp_db.get_messages(conv_id)

    assert messages[0]["content"] == "First"
    assert messages[1]["content"] == "Second"
    assert messages[2]["content"] == "Third"


def test_get_messages_returns_empty_for_new_conversation(temp_db):
    """Test that new conversation has no messages."""
    conv_id = temp_db.new_conversation("Empty")

    messages = temp_db.get_messages(conv_id)

    assert messages == []


# ============================================================================
# v1.10: SOURCES PERSISTENCE
# ============================================================================


def test_add_message_with_sources(temp_db):
    """Test adding a message with source metadata."""
    conv_id = temp_db.new_conversation("Test Sources")
    sources = [
        {"i": 1, "source": "doc1.txt", "score": 0.85, "chunk": 0},
        {"i": 2, "source": "doc2.txt", "score": 0.72, "chunk": 1},
    ]

    temp_db.add_message(conv_id, "assistant", "Answer with sources", sources=sources)

    messages = temp_db.get_messages(conv_id)
    assert len(messages) == 1
    assert messages[0]["sources"] == sources


def test_add_message_without_sources(temp_db):
    """Test that messages without sources have None for sources field."""
    conv_id = temp_db.new_conversation("Test No Sources")

    temp_db.add_message(conv_id, "user", "Question without sources")

    messages = temp_db.get_messages(conv_id)
    assert len(messages) == 1
    assert messages[0]["sources"] is None


def test_get_messages_returns_sources_field(temp_db):
    """Test that get_messages always returns a sources field."""
    conv_id = temp_db.new_conversation("Test")

    temp_db.add_message(conv_id, "user", "Hello")
    temp_db.add_message(
        conv_id,
        "assistant",
        "Hi!",
        sources=[{"i": 1, "source": "test.txt", "score": 0.9, "chunk": 0}],
    )

    messages = temp_db.get_messages(conv_id)

    assert len(messages) == 2
    # User message has no sources
    assert messages[0]["sources"] is None
    # Assistant message has sources
    assert messages[1]["sources"] is not None
    assert len(messages[1]["sources"]) == 1


# ============================================================================
# v1.10: AUTO-TITLE CONVERSATIONS
# ============================================================================


def test_auto_title_from_first_user_message(temp_db):
    """Test that conversation title is auto-generated from first user message."""
    conv_id = temp_db.new_conversation()  # Default title "Nouvelle conversation"

    temp_db.add_message(conv_id, "user", "Quelle est la fiscalité des dividendes?")

    convs = temp_db.list_conversations()
    conv = next(c for c in convs if c["conv_id"] == conv_id)

    # Title should be set from user message
    assert conv["title"] == "Quelle est la fiscalité des dividendes?"


def test_auto_title_truncates_long_messages(temp_db):
    """Test that auto-title truncates messages longer than 50 chars."""
    conv_id = temp_db.new_conversation()

    long_message = "Cette question est extrêmement longue et dépasse les cinquante caractères autorisés pour le titre"
    temp_db.add_message(conv_id, "user", long_message)

    convs = temp_db.list_conversations()
    conv = next(c for c in convs if c["conv_id"] == conv_id)

    # Title should be truncated and end with "..."
    assert len(conv["title"]) <= 53  # 50 + "..."
    assert conv["title"].endswith("...")


def test_auto_title_only_on_generic_title(temp_db):
    """Test that auto-title only applies when title is generic."""
    conv_id = temp_db.new_conversation(title="Custom Title")

    temp_db.add_message(conv_id, "user", "This should not change the title")

    convs = temp_db.list_conversations()
    conv = next(c for c in convs if c["conv_id"] == conv_id)

    # Title should remain unchanged
    assert conv["title"] == "Custom Title"


def test_auto_title_only_from_user_message(temp_db):
    """Test that auto-title only triggers on user messages, not assistant."""
    conv_id = temp_db.new_conversation()

    # Assistant message first (shouldn't trigger auto-title)
    temp_db.add_message(conv_id, "assistant", "Welcome! How can I help?")

    convs = temp_db.list_conversations()
    conv = next(c for c in convs if c["conv_id"] == conv_id)

    # Title should still be generic
    assert conv["title"] == "Nouvelle conversation"


# ============================================================================
# v1.10: UPDATED_AT ORDERING
# ============================================================================


def test_conversations_ordered_by_updated_at(temp_db):
    """Test that conversations are ordered by most recently used."""
    import time

    # Create three conversations
    conv1 = temp_db.new_conversation("First")
    time.sleep(0.01)  # Small delay to ensure different timestamps
    conv2 = temp_db.new_conversation("Second")
    time.sleep(0.01)
    conv3 = temp_db.new_conversation("Third")

    # Add message to first conversation (makes it most recent)
    time.sleep(0.01)
    temp_db.add_message(conv1, "user", "Update first conversation")

    convs = temp_db.list_conversations()

    # First conversation should now be first (most recently updated)
    assert convs[0]["conv_id"] == conv1
    # Third was created last but not updated, so second
    assert convs[1]["conv_id"] == conv3
    # Second is oldest by activity
    assert convs[2]["conv_id"] == conv2


def test_add_message_updates_conversation_timestamp(temp_db):
    """Test that adding a message updates the conversation's updated_at."""
    conv_id = temp_db.new_conversation("Test")

    convs_before = temp_db.list_conversations()
    updated_before = convs_before[0]["updated_at"]

    import time

    time.sleep(0.01)

    temp_db.add_message(conv_id, "user", "New message")

    convs_after = temp_db.list_conversations()
    updated_after = convs_after[0]["updated_at"]

    assert updated_after > updated_before


def test_update_conversation_title(temp_db):
    """Test manually updating a conversation title."""
    conv_id = temp_db.new_conversation("Original Title")

    temp_db.update_conversation_title(conv_id, "New Title")

    convs = temp_db.list_conversations()
    conv = next(c for c in convs if c["conv_id"] == conv_id)

    assert conv["title"] == "New Title"


def test_delete_conversation(temp_db):
    """Test deleting a conversation."""
    conv_id = temp_db.new_conversation("To Delete")
    temp_db.add_message(conv_id, "user", "Message to delete")

    # Verify conversation exists
    convs = temp_db.list_conversations()
    assert any(c["conv_id"] == conv_id for c in convs)

    # Delete
    result = temp_db.delete_conversation(conv_id)

    assert result is True
    # Verify conversation is gone
    convs = temp_db.list_conversations()
    assert not any(c["conv_id"] == conv_id for c in convs)
    # Verify messages are also gone (CASCADE)
    msgs = temp_db.get_messages(conv_id)
    assert msgs == []


def test_delete_conversation_nonexistent(temp_db):
    """Test deleting a non-existent conversation returns False."""
    result = temp_db.delete_conversation("nonexistent_id")
    assert result is False
