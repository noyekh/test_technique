"""
Tests for audit logging with allowlist policy.
"""

import json
from unittest.mock import patch

from backend.audit_log import (
    AuditEvent,
    RequestTimer,
    generate_request_id,
    log_delete,
    log_error,
    log_query,
    log_upload,
    utcnow_iso,
)


def test_audit_event_serialization():
    """Test that audit events serialize to valid JSON."""
    event = AuditEvent(
        event_type="query",
        request_id="abc123",
        timestamp="2026-01-10T12:00:00Z",
        doc_ids=["doc1", "doc2"],
        scores=[0.9, 0.8],
        verdict="answer",
    )

    json_str = event.to_json()
    parsed = json.loads(json_str)

    assert parsed["event_type"] == "query"
    assert parsed["request_id"] == "abc123"
    assert parsed["doc_ids"] == ["doc1", "doc2"]
    assert parsed["scores"] == [0.9, 0.8]


def test_audit_event_no_content_fields():
    """Verify that AuditEvent has no fields for sensitive content."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(AuditEvent)}

    # These should NOT be in the audit event
    forbidden_fields = {
        "question",
        "prompt",
        "answer",
        "response",
        "content",
        "text",
        "chunk_content",
        "quote",
        "filename",
    }

    assert field_names.isdisjoint(forbidden_fields), (
        f"Audit event contains forbidden fields: {field_names & forbidden_fields}"
    )


def test_generate_request_id():
    """Test request ID generation."""
    id1 = generate_request_id()
    id2 = generate_request_id()

    assert len(id1) == 16
    assert id1 != id2  # Should be unique


def test_utcnow_iso_format():
    """Test timestamp format."""
    ts = utcnow_iso()

    # Should be ISO format
    assert "T" in ts
    assert ts.endswith("+00:00") or ts.endswith("Z")


def test_request_timer():
    """Test request timing context manager."""
    import time

    with RequestTimer() as timer:
        time.sleep(0.01)  # 10ms

    assert timer.elapsed_ms >= 10
    assert timer.elapsed_ms < 1000  # Sanity check


def test_log_query_structure():
    """Test that log_query produces correct structure."""
    # Mock the logger to capture the output
    with patch("backend.audit_log._get_audit_logger") as mock_logger:
        mock_logger.return_value.info = lambda x: captured.append(x)
        captured = []

        log_query(
            request_id="req123",
            session_id="sess456",
            doc_ids=["doc1"],
            chunk_ids=["doc1:0", "doc1:1"],
            scores=[0.9, 0.85],
            latency_ms=150,
            model="gpt-4o-mini",
            verdict="answer",
        )

        assert len(captured) == 1
        parsed = json.loads(captured[0])

        assert parsed["event_type"] == "query"
        assert parsed["request_id"] == "req123"
        assert parsed["doc_ids"] == ["doc1"]
        assert parsed["verdict"] == "answer"
        # Verify no content fields
        assert "question" not in parsed
        assert "answer" not in parsed


def test_log_upload_structure():
    """Test that log_upload produces correct structure."""
    with patch("backend.audit_log._get_audit_logger") as mock_logger:
        mock_logger.return_value.info = lambda x: captured.append(x)
        captured = []

        log_upload(
            request_id="req123",
            session_id="sess456",
            doc_id="newdoc",
            chunk_ids=["newdoc:0", "newdoc:1", "newdoc:2"],
        )

        assert len(captured) == 1
        parsed = json.loads(captured[0])

        assert parsed["event_type"] == "upload"
        assert parsed["doc_ids"] == ["newdoc"]
        assert len(parsed["chunk_ids"]) == 3
        # Verify no content fields
        assert "filename" not in parsed
        assert "content" not in parsed


def test_log_delete_structure():
    """Test that log_delete produces correct structure."""
    with patch("backend.audit_log._get_audit_logger") as mock_logger:
        mock_logger.return_value.info = lambda x: captured.append(x)
        captured = []

        log_delete(
            request_id="req123",
            session_id="sess456",
            doc_id="deleted_doc",
            chunk_ids=["deleted_doc:0"],
        )

        assert len(captured) == 1
        parsed = json.loads(captured[0])

        assert parsed["event_type"] == "delete"
        assert parsed["doc_ids"] == ["deleted_doc"]


def test_log_error_no_message():
    """Test that log_error uses error_code, not error message."""
    with patch("backend.audit_log._get_audit_logger") as mock_logger:
        mock_logger.return_value.info = lambda x: captured.append(x)
        captured = []

        log_error(
            request_id="req123",
            session_id="sess456",
            error_code="RATE_LIMITED",
        )

        assert len(captured) == 1
        parsed = json.loads(captured[0])

        assert parsed["error_code"] == "RATE_LIMITED"
        # Should not contain error message (could have user content)
        assert "message" not in parsed
        assert "error_message" not in parsed
