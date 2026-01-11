"""
Document ingestion and text extraction.

Supported formats:
- .txt: Plain text (UTF-8 with fallback)
- .csv: Tabular data (preserves column semantics)
- .html/.htm: Web pages (strips scripts, styles, nav)

Design decisions:
- BeautifulSoup + lxml: Fast, robust HTML parsing
- pandas for CSV: Handles edge cases (quoting, encoding)
- UTF-8 with errors="ignore": Graceful handling of bad encoding
- Semantic CSV format "col: val": Preserves meaning for RAG retrieval

For production, consider:
- PDF support (pypdf, pdfplumber)
- DOCX support (python-docx)
- OCR for scanned documents (tesseract)
- Language detection
"""

from __future__ import annotations

import html
import pandas as pd
from bs4 import BeautifulSoup


def normalize_text(s: str) -> str:
    """
    Normalize text: CRLF -> LF, strip trailing whitespace, trim.
    
    Args:
        s: Raw text string
        
    Returns:
        Normalized text
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = s.strip()
    return s


def read_to_text(ext: str, raw: bytes) -> str:
    """
    Convert raw file bytes to normalized text based on extension.
    
    Args:
        ext: File extension (without dot)
        raw: File content as bytes
        
    Returns:
        Extracted and normalized text
        
    Raises:
        ValueError: If extension is not supported
    """
    ext = (ext or "").lower()

    if ext == "txt":
        return normalize_text(raw.decode("utf-8", errors="ignore"))

    if ext == "csv":
        # Preserve semantic structure: "col: val | col: val"
        df = pd.read_csv(
            pd.io.common.BytesIO(raw),
            dtype=str,
            keep_default_na=False,
            encoding_errors="ignore",
        )
        lines: list[str] = []
        for _, row in df.iterrows():
            parts = []
            for col, val in row.items():
                val = (val or "").strip()
                if val:
                    parts.append(f"{col}: {val}")
            if parts:
                lines.append(" | ".join(parts))
        return normalize_text("\n".join(lines))

    if ext in {"html", "htm"}:
        soup = BeautifulSoup(raw.decode("utf-8", errors="ignore"), "lxml")
        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # Decode HTML entities (&amp; -> &, &lt; -> <, etc.)
        text = html.unescape(text)
        return normalize_text(text)

    raise ValueError(f"Extension non supportée: .{ext}")
