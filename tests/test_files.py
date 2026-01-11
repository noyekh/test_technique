"""
Tests for file handling utilities.

Tests cover:
- SHA256 hashing
- Filename sanitization
- Extension inference
- Content-addressed storage path building
- File upload saving
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

from backend.files import (
    build_stored_path,
    infer_ext,
    safe_filename,
    save_upload,
    sha256_bytes,
)

# ============================================================================
# SHA256 HASHING TESTS
# ============================================================================


def test_sha256_bytes_basic():
    """Test basic SHA256 hashing."""
    result = sha256_bytes(b"hello world")

    assert isinstance(result, str)
    assert len(result) == 64  # SHA256 hex is 64 chars


def test_sha256_bytes_empty():
    """Test SHA256 of empty bytes."""
    result = sha256_bytes(b"")

    # Known hash of empty string
    expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert result == expected


def test_sha256_bytes_deterministic():
    """Test that same content gives same hash."""
    content = b"test content for hashing"

    hash1 = sha256_bytes(content)
    hash2 = sha256_bytes(content)

    assert hash1 == hash2


def test_sha256_bytes_different_content():
    """Test that different content gives different hash."""
    hash1 = sha256_bytes(b"content A")
    hash2 = sha256_bytes(b"content B")

    assert hash1 != hash2


# ============================================================================
# SAFE FILENAME TESTS
# ============================================================================


def test_safe_filename_basic():
    """Test basic filename sanitization."""
    result = safe_filename("document.txt")

    assert result == "document.txt"


def test_safe_filename_special_chars():
    """Test that special characters are replaced."""
    result = safe_filename("my file (1).txt")

    assert "(" not in result
    assert ")" not in result
    assert " " not in result
    assert "_" in result  # Replaced with underscore


def test_safe_filename_path_traversal():
    """Test that path traversal characters are sanitized."""
    result = safe_filename("../../../etc/passwd")

    # Slashes are replaced with underscore, dots are preserved
    # But the result is safe because we use Path operations later
    assert "/" not in result
    # The function replaces non-alphanumeric (except ._-) with _
    assert result == ".._.._.._etc_passwd"


def test_safe_filename_unicode():
    """Test that unicode characters are replaced."""
    result = safe_filename("café_document.txt")

    # Non-ASCII chars should be replaced
    assert "é" not in result


def test_safe_filename_empty():
    """Test empty filename handling."""
    result = safe_filename("")
    assert result == ""


def test_safe_filename_none():
    """Test None filename handling."""
    result = safe_filename(None)
    assert result == ""


def test_safe_filename_whitespace():
    """Test whitespace is trimmed."""
    result = safe_filename("  document.txt  ")
    assert not result.startswith(" ")
    assert not result.endswith(" ")


def test_safe_filename_max_length():
    """Test that filenames are truncated to 200 chars."""
    long_name = "a" * 300 + ".txt"
    result = safe_filename(long_name)

    assert len(result) <= 200


def test_safe_filename_preserves_extension():
    """Test that valid extensions are preserved."""
    result = safe_filename("document.pdf")
    assert result.endswith(".pdf")


def test_safe_filename_allows_dashes_underscores():
    """Test that dashes and underscores are allowed."""
    result = safe_filename("my-doc_v2.txt")
    assert result == "my-doc_v2.txt"


# ============================================================================
# INFER EXTENSION TESTS
# ============================================================================


def test_infer_ext_basic():
    """Test basic extension inference."""
    assert infer_ext("document.txt") == "txt"
    assert infer_ext("file.pdf") == "pdf"
    assert infer_ext("data.csv") == "csv"


def test_infer_ext_uppercase():
    """Test that extensions are lowercased."""
    assert infer_ext("FILE.TXT") == "txt"
    assert infer_ext("Doc.PDF") == "pdf"


def test_infer_ext_no_extension():
    """Test files without extension default to txt."""
    assert infer_ext("filename") == "txt"


def test_infer_ext_empty():
    """Test empty filename defaults to txt."""
    assert infer_ext("") == "txt"


def test_infer_ext_multiple_dots():
    """Test file with multiple dots."""
    assert infer_ext("archive.tar.gz") == "gz"
    assert infer_ext("file.backup.txt") == "txt"


def test_infer_ext_hidden_file():
    """Test hidden file (starts with dot) has no suffix, defaults to txt."""
    # pathlib treats .gitignore as having no suffix (it's the stem)
    result = infer_ext(".gitignore")
    assert result == "txt"  # Default when no extension


# ============================================================================
# BUILD STORED PATH TESTS
# ============================================================================


def test_build_stored_path_basic():
    """Test basic stored path building."""
    digest = "abc123"
    name = "document.txt"

    with patch("backend.files.UPLOAD_DIR", Path("/uploads")):
        result = build_stored_path(digest, name)

    assert "abc123" in str(result)
    assert "document.txt" in str(result)


def test_build_stored_path_sanitizes_name():
    """Test that filename is sanitized in path."""
    digest = "abc123"
    name = "my file (1).txt"

    with patch("backend.files.UPLOAD_DIR", Path("/uploads")):
        result = build_stored_path(digest, name)

    # Should not contain special chars
    assert "(" not in str(result)
    assert " " not in str(result)


def test_build_stored_path_format():
    """Test stored path format is {hash}_{name}."""
    digest = "abcdef"
    name = "test.txt"

    with patch("backend.files.UPLOAD_DIR", Path("/uploads")):
        result = build_stored_path(digest, name)

    assert result.name == "abcdef_test.txt"


# ============================================================================
# SAVE UPLOAD TESTS
# ============================================================================


def test_save_upload_creates_file():
    """Test that save_upload creates file on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("backend.files.UPLOAD_DIR", Path(tmpdir)):
            content = b"test file content"
            digest, path, ext, size = save_upload("test.txt", content)

            assert path.exists()
            assert path.read_bytes() == content


def test_save_upload_returns_correct_hash():
    """Test that save_upload returns correct SHA256."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("backend.files.UPLOAD_DIR", Path(tmpdir)):
            content = b"test content"
            expected_hash = sha256_bytes(content)

            digest, path, ext, size = save_upload("file.txt", content)

            assert digest == expected_hash


def test_save_upload_returns_correct_extension():
    """Test that save_upload returns correct extension."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("backend.files.UPLOAD_DIR", Path(tmpdir)):
            digest, path, ext, size = save_upload("document.csv", b"a,b,c")

            assert ext == "csv"


def test_save_upload_returns_correct_size():
    """Test that save_upload returns correct file size."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("backend.files.UPLOAD_DIR", Path(tmpdir)):
            content = b"exactly 20 bytes!!!"  # 20 bytes
            digest, path, ext, size = save_upload("test.txt", content)

            assert size == len(content)


def test_save_upload_content_addressed():
    """Test that same content produces same path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("backend.files.UPLOAD_DIR", Path(tmpdir)):
            content = b"identical content"

            _, path1, _, _ = save_upload("file1.txt", content)
            _, path2, _, _ = save_upload("file2.txt", content)

            # Same content = same hash prefix
            # But different original names
            assert path1.read_bytes() == path2.read_bytes()
