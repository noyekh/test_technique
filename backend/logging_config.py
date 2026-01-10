"""
Logging configuration for the application.

Design decisions:
- Basic format: Timestamp | Level | Logger | Message
- stdout output: Compatible with container logging (Docker, K8s)
- INFO level default: Verbose enough for debugging, not too noisy
- Idempotent setup: Safe to call multiple times

SECURITY (v1.5):
- Application logs should NOT contain sensitive content
- Use audit_log.py for structured audit events
- Never log prompts, questions, or responses here

For production, consider:
- structlog for JSON logging (machine-parseable)
- Log aggregation (ELK, Datadog, CloudWatch)
- Correlation IDs for request tracing
- Separate error log file
- Log rotation

Usage:
    from backend.logging_config import setup_logging
    setup_logging()  # Call once at startup
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure minimal structured-ish logging format.
    Idempotent: won't add duplicate handlers if already configured.
    """
    root = logging.getLogger()
    if root.handlers:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
