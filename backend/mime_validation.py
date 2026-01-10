"""
MIME type validation for uploaded files.

Design decisions:
- python-magic (libmagic): Industry standard for MIME detection
- Graceful fallback: Works without libmagic (extension-only mode)
- Whitelist approach: Only allow known-safe types

Why MIME validation matters:
- Extension can be spoofed (malicious.exe → malicious.txt)
- Magic bytes reveal true file type
- Prevents processing of unexpected formats

Supported types:
- text/plain: .txt files
- text/csv: .csv files (also accepts text/plain, application/vnd.ms-excel)
- text/html: .html/.htm files (also accepts application/xhtml+xml)

Fallback behavior:
- If libmagic unavailable: Trust extension (acceptable for internal PoC)
- If MIME detection fails: Trust extension with warning log
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ALLOWED_EXTS = {"txt", "csv", "html", "htm"}

# Common MIME strings returned by libmagic for these types
ALLOWED_MIME_BY_EXT = {
    "txt": {"text/plain"},
    "csv": {"text/csv", "text/plain", "application/vnd.ms-excel"},
    "html": {"text/html", "application/xhtml+xml"},
    "htm": {"text/html", "application/xhtml+xml"},
}


def detect_mime(raw: bytes) -> str | None:
    """
    Detect MIME type using python-magic (libmagic).
    
    Args:
        raw: File content as bytes
        
    Returns:
        Detected MIME type or None if unavailable
    """
    try:
        import magic  # type: ignore
    except Exception as e:
        logger.warning(
            "python-magic unavailable, falling back to extension",
            extra={"error_code": "MAGIC_UNAVAILABLE"},
        )
        return None

    try:
        m = magic.Magic(mime=True)
        out = m.from_buffer(raw)
        if not out:
            return None
        # strip charset if present: "text/plain; charset=us-ascii"
        return out.split(";")[0].strip().lower()
    except Exception as e:
        logger.warning(
            "MIME detection failed, falling back to extension",
            extra={"error_code": "MIME_DETECTION_ERROR"},
        )
        return None


def validate_mime(ext: str, raw: bytes) -> tuple[bool, str | None]:
    """
    Validate file MIME type against extension.
    
    Args:
        ext: File extension (without dot)
        raw: File content as bytes
        
    Returns:
        Tuple of (is_valid, detected_mime)
        If detection unavailable, returns (True, None) for PoC fallback
    """
    ext = (ext or "").lower()
    if ext not in ALLOWED_EXTS:
        return False, None

    mime = detect_mime(raw)
    if mime is None:
        # fallback mode: allow
        return True, None

    allowed = ALLOWED_MIME_BY_EXT.get(ext, set())
    ok = mime in allowed
    return ok, mime
