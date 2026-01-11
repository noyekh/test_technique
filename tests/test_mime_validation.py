"""
Tests for MIME type validation.

Tests cover:
- Extension whitelist validation
- MIME type detection and matching
- Fallback behavior when libmagic unavailable
"""

from unittest.mock import patch

from backend.mime_validation import (
    ALLOWED_EXTS,
    ALLOWED_MIME_BY_EXT,
    validate_mime,
)

# ============================================================================
# EXTENSION WHITELIST TESTS
# ============================================================================


def test_allowed_extensions():
    """Test that expected extensions are in the whitelist."""
    assert "txt" in ALLOWED_EXTS
    assert "csv" in ALLOWED_EXTS
    assert "html" in ALLOWED_EXTS
    assert "htm" in ALLOWED_EXTS


def test_disallowed_extensions():
    """Test that dangerous extensions are not in the whitelist."""
    dangerous = ["exe", "dll", "sh", "bat", "py", "js", "php"]
    for ext in dangerous:
        assert ext not in ALLOWED_EXTS


def test_validate_mime_rejects_unknown_extension():
    """Test that unknown extensions are rejected."""
    ok, detected = validate_mime("exe", b"MZ\x90\x00")  # PE header
    assert ok is False


def test_validate_mime_rejects_empty_extension():
    """Test that empty extension is rejected."""
    ok, detected = validate_mime("", b"some content")
    assert ok is False


# ============================================================================
# MIME DETECTION TESTS (with mocked magic)
# ============================================================================


def test_detect_mime_returns_none_when_import_fails():
    """Test that detect_mime returns None when magic import fails."""
    # Mock the import to raise ImportError
    import importlib
    import sys

    # Save original magic module if it exists
    original_magic = sys.modules.get("magic")

    try:
        # Force magic to be "unavailable" by making import raise
        sys.modules["magic"] = None  # type: ignore

        # Reimport to clear any cached reference
        from backend import mime_validation

        importlib.reload(mime_validation)

        # Now detect_mime should return None (import will raise)
        result = mime_validation.detect_mime(b"test content")
        assert result is None
    finally:
        # Restore original state
        if original_magic is not None:
            sys.modules["magic"] = original_magic
        elif "magic" in sys.modules:
            del sys.modules["magic"]

        # Reload to restore normal behavior
        from backend import mime_validation

        importlib.reload(mime_validation)


def test_validate_mime_txt_valid():
    """Test valid text/plain detection for .txt files."""
    with patch("backend.mime_validation.detect_mime") as mock:
        mock.return_value = "text/plain"

        ok, detected = validate_mime("txt", b"Hello world")

        assert ok is True
        assert detected == "text/plain"


def test_validate_mime_txt_invalid():
    """Test rejection of non-text MIME for .txt files."""
    with patch("backend.mime_validation.detect_mime") as mock:
        mock.return_value = "application/pdf"

        ok, detected = validate_mime("txt", b"fake content")

        assert ok is False
        assert detected == "application/pdf"


def test_validate_mime_csv_accepts_text_plain():
    """Test that CSV accepts text/plain MIME (common for CSV files)."""
    with patch("backend.mime_validation.detect_mime") as mock:
        mock.return_value = "text/plain"

        ok, detected = validate_mime("csv", b"a,b,c\n1,2,3")

        assert ok is True


def test_validate_mime_csv_accepts_text_csv():
    """Test that CSV accepts text/csv MIME."""
    with patch("backend.mime_validation.detect_mime") as mock:
        mock.return_value = "text/csv"

        ok, detected = validate_mime("csv", b"a,b,c\n1,2,3")

        assert ok is True


def test_validate_mime_csv_accepts_excel_mime():
    """Test that CSV accepts application/vnd.ms-excel (legacy Excel association)."""
    with patch("backend.mime_validation.detect_mime") as mock:
        mock.return_value = "application/vnd.ms-excel"

        ok, detected = validate_mime("csv", b"a,b,c\n1,2,3")

        assert ok is True


def test_validate_mime_html_accepts_text_html():
    """Test that HTML accepts text/html MIME."""
    with patch("backend.mime_validation.detect_mime") as mock:
        mock.return_value = "text/html"

        ok, detected = validate_mime("html", b"<html></html>")

        assert ok is True


def test_validate_mime_html_accepts_xhtml():
    """Test that HTML accepts application/xhtml+xml MIME."""
    with patch("backend.mime_validation.detect_mime") as mock:
        mock.return_value = "application/xhtml+xml"

        ok, detected = validate_mime("html", b"<html></html>")

        assert ok is True


def test_validate_mime_htm_same_as_html():
    """Test that .htm extension has same rules as .html."""
    with patch("backend.mime_validation.detect_mime") as mock:
        mock.return_value = "text/html"

        ok, detected = validate_mime("htm", b"<html></html>")

        assert ok is True


def test_validate_mime_html_rejects_javascript():
    """Test that HTML rejects JavaScript MIME type."""
    with patch("backend.mime_validation.detect_mime") as mock:
        mock.return_value = "application/javascript"

        ok, detected = validate_mime("html", b"<script>bad()</script>")

        assert ok is False


# ============================================================================
# FALLBACK BEHAVIOR TESTS
# ============================================================================


def test_validate_mime_fallback_allows_valid_extension():
    """Test that fallback mode allows files with valid extension."""
    with patch("backend.mime_validation.detect_mime") as mock:
        mock.return_value = None  # Simulate magic unavailable

        ok, detected = validate_mime("txt", b"some content")

        assert ok is True
        assert detected is None


def test_validate_mime_fallback_still_checks_extension():
    """Test that fallback mode still validates extension whitelist."""
    with patch("backend.mime_validation.detect_mime") as mock:
        mock.return_value = None  # Simulate magic unavailable

        ok, detected = validate_mime("exe", b"MZ\x90\x00")

        assert ok is False


def test_allowed_mime_by_ext_completeness():
    """Test that all allowed extensions have MIME mappings."""
    for ext in ALLOWED_EXTS:
        assert ext in ALLOWED_MIME_BY_EXT, f"Missing MIME mapping for .{ext}"
        assert len(ALLOWED_MIME_BY_EXT[ext]) > 0, f"Empty MIME set for .{ext}"


# ============================================================================
# MIME STRING NORMALIZATION
# ============================================================================


def test_validate_mime_case_insensitive_extension():
    """Test that extension comparison is case-insensitive."""
    with patch("backend.mime_validation.detect_mime") as mock:
        mock.return_value = "text/plain"

        ok1, _ = validate_mime("TXT", b"content")
        ok2, _ = validate_mime("Txt", b"content")
        ok3, _ = validate_mime("txt", b"content")

        assert ok1 is True
        assert ok2 is True
        assert ok3 is True
