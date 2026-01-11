"""
File handling utilities for secure upload management.

Design decisions:
- SHA256 hashing: Content-based deduplication, prevents re-indexing identical files
- Safe filename sanitization: Prevents path traversal and special char issues
- Content-addressed storage: {hash}_{filename} pattern ensures uniqueness

Security considerations:
- Only alphanumeric, dots, dashes, underscores allowed in filenames
- 200 char limit prevents filesystem issues
- Hash prefix makes filenames unpredictable
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .settings import UPLOAD_DIR

SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")


def sha256_bytes(b: bytes) -> str:
    """
    Compute SHA256 hash of bytes.

    Args:
        b: Raw bytes to hash

    Returns:
        Hex-encoded hash string
    """
    return hashlib.sha256(b).hexdigest()


def safe_filename(name: str) -> str:
    """
    Sanitize filename to prevent path traversal and special chars.

    Args:
        name: Original filename

    Returns:
        Sanitized filename (max 200 chars)
    """
    name = (name or "").strip()
    name = SAFE_NAME.sub("_", name)
    return name[:200] if len(name) > 200 else name


def infer_ext(filename: str) -> str:
    """
    Extract file extension from filename.

    Args:
        filename: Original filename

    Returns:
        Lowercase extension without dot, or "txt" as default
    """
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext if ext else "txt"


def build_stored_path(digest: str, original_name: str) -> Path:
    """
    Build content-addressed storage path.

    Args:
        digest: SHA256 hash of content
        original_name: Original filename

    Returns:
        Path in format: uploads/{hash}_{sanitized_name}
    """
    original_name = safe_filename(original_name)
    return UPLOAD_DIR / f"{digest}_{original_name}"


def save_upload(original_name: str, raw: bytes) -> tuple[str, Path, str, int]:
    """
    Save an uploaded file to disk with content-addressed naming.

    Args:
        original_name: Original filename as uploaded
        raw: File content as bytes

    Returns:
        Tuple of (sha256, stored_path, extension, size_bytes)
    """
    digest = sha256_bytes(raw)
    ext = infer_ext(original_name)
    stored_path = build_stored_path(digest, original_name)
    stored_path.write_bytes(raw)
    return digest, stored_path, ext, len(raw)
