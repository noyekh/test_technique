"""
Tests for document ingestion and parsing.
"""

from backend.ingest import normalize_text, read_to_text


def test_normalize_text_basic():
    """Test basic text normalization."""
    assert normalize_text("  foo\r\nbar  ") == "foo\nbar"


def test_normalize_text_preserves_content():
    """Test that normalization preserves actual content."""
    text = "Line 1\n\nLine 2\nLine 3"
    assert normalize_text(text) == text


def test_normalize_text_handles_empty():
    """Test handling of empty input."""
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_html_extraction_basic():
    """Test basic HTML text extraction."""
    html = b"<html><body><p>Test content</p></body></html>"
    out = read_to_text("html", html)
    assert "Test content" in out


def test_html_extraction_strips_scripts():
    """Test that script tags are removed from HTML."""
    html = b"<html><body><p>Test</p><script>bad()</script></body></html>"
    out = read_to_text("html", html)
    assert "Test" in out
    assert "bad()" not in out


def test_html_extraction_strips_styles():
    """Test that style tags are removed from HTML."""
    html = b"<html><head><style>body{color:red}</style></head><body><p>Test</p></body></html>"
    out = read_to_text("html", html)
    assert "Test" in out
    assert "color:red" not in out


def test_html_extraction_strips_nav():
    """Test that navigation elements are removed."""
    html = b"<html><body><nav>Menu</nav><main>Content</main></body></html>"
    out = read_to_text("html", html)
    assert "Content" in out
    assert "Menu" not in out


def test_csv_extraction_preserves_structure():
    """Test that CSV parsing preserves column names."""
    csv = b"name,value\nAlice,100\nBob,200"
    out = read_to_text("csv", csv)
    assert "name: Alice" in out
    assert "value: 100" in out


def test_txt_extraction():
    """Test plain text extraction."""
    txt = b"Simple text content"
    out = read_to_text("txt", txt)
    assert out == "Simple text content"
