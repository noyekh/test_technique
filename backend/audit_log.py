"""
Audit logging with strict allowlist policy.

CRITICAL: This module enforces a strict allowlist policy for logging.
We NEVER log any content that could contain sensitive information.

Allowlist (what we log):
- request_id: Unique identifier for request tracing
- timestamp: When the event occurred
- user_session_id: Anonymized session identifier
- doc_ids / chunk_ids: Document identifiers (not content)
- scores: Relevance scores (numeric only)
- latency_ms: Response time
- model: LLM model used
- verdict: "answer" or "refusal"
- token_counts: Input/output token counts
- event_type: Type of audit event
- error_code: Error codes (not messages with user content)

Blocklist (NEVER log):
- prompts / questions
- chunks / document content
- LLM responses
- quotes / citations text
- filenames (use doc_id instead)
- user inputs of any kind

Design decisions:
- Separate audit logger from application logger
- Structured JSON format for machine parsing
- File rotation to prevent unbounded growth
- Explicit function interface to prevent accidental content logging
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from typing import Literal

from .settings import DATA_DIR

AUDIT_LOG_PATH = DATA_DIR / "audit.jsonl"


@dataclass(frozen=True)
class AuditEvent:
    """
    Structured audit event with allowlist-only fields.

    All fields are either identifiers, numeric values, or controlled enums.
    No free-text content is allowed.
    """
    event_type: Literal["query", "upload", "delete", "refusal", "error", "auth"]
    request_id: str
    timestamp: str
    session_id: str = ""
    doc_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    latency_ms: int = 0
    model: str = ""
    verdict: Literal["answer", "refusal", "error", ""] = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error_code: str = ""

    def to_json(self) -> str:
        """Serialize to JSON line."""
        return json.dumps(asdict(self), ensure_ascii=False)


def _get_audit_logger() -> logging.Logger:
    """
    Get or create the audit logger with file rotation.

    Separate from application logging to ensure audit events
    are captured even if app logging fails.
    """
    logger = logging.getLogger("audit")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False  # Don't send to root logger

    # Ensure directory exists
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Rotating file handler: 10MB max, keep 5 backups
    handler = RotatingFileHandler(
        AUDIT_LOG_PATH,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    return logger


def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return uuid.uuid4().hex[:16]


def utcnow_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


def log_query(
    request_id: str,
    session_id: str,
    doc_ids: list[str],
    chunk_ids: list[str],
    scores: list[float],
    latency_ms: int,
    model: str,
    verdict: Literal["answer", "refusal"],
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """
    Log a RAG query event.

    Note: We deliberately do NOT accept question or answer text.
    """
    event = AuditEvent(
        event_type="query",
        request_id=request_id,
        timestamp=utcnow_iso(),
        session_id=session_id,
        doc_ids=doc_ids,
        chunk_ids=chunk_ids,
        scores=scores,
        latency_ms=latency_ms,
        model=model,
        verdict=verdict,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    _get_audit_logger().info(event.to_json())


def log_upload(
    request_id: str,
    session_id: str,
    doc_id: str,
    chunk_ids: list[str],
) -> None:
    """
    Log a document upload event.

    Note: We log doc_id, not filename (which could be sensitive).
    """
    event = AuditEvent(
        event_type="upload",
        request_id=request_id,
        timestamp=utcnow_iso(),
        session_id=session_id,
        doc_ids=[doc_id],
        chunk_ids=chunk_ids,
    )
    _get_audit_logger().info(event.to_json())


def log_delete(
    request_id: str,
    session_id: str,
    doc_id: str,
    chunk_ids: list[str],
) -> None:
    """
    Log a document deletion event.

    This creates an audit trail for GDPR compliance.
    """
    event = AuditEvent(
        event_type="delete",
        request_id=request_id,
        timestamp=utcnow_iso(),
        session_id=session_id,
        doc_ids=[doc_id],
        chunk_ids=chunk_ids,
    )
    _get_audit_logger().info(event.to_json())


def log_error(
    request_id: str,
    session_id: str,
    error_code: str,
) -> None:
    """
    Log an error event.

    Note: We log error_code, not error message (which could contain user input).
    """
    event = AuditEvent(
        event_type="error",
        request_id=request_id,
        timestamp=utcnow_iso(),
        session_id=session_id,
        error_code=error_code,
        verdict="error",
    )
    _get_audit_logger().info(event.to_json())


def log_auth(
    request_id: str,
    session_id: str,
    action: Literal["login_success", "login_failed", "logout"],
    username: str = "",
) -> None:
    """
    Log authentication event.

    Note: We log action type, not user details (privacy).
    """
    event = AuditEvent(
        event_type="auth",
        request_id=request_id,
        timestamp=utcnow_iso(),
        session_id=session_id,
        error_code=action,  # Reuse field for action type
    )
    _get_audit_logger().info(event.to_json())


class RequestTimer:
    """Context manager for timing requests."""

    def __init__(self):
        self.start_time: float = 0
        self.elapsed_ms: int = 0

    def __enter__(self) -> RequestTimer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = int((time.perf_counter() - self.start_time) * 1000)
