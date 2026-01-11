"""
Input sanitization and prompt injection protection.

Design decisions:
- Blocklist approach: Simple, transparent, easy to extend
- Control character removal: Prevents hidden instructions
- Length limiting: Prevents token stuffing attacks
- Bilingual patterns: French + English coverage

Known limitations (acceptable for PoC):
- Blocklist can be bypassed with synonyms, typos, encoding tricks
- No ML-based detection (would require additional dependencies)
- No rate limiting on failed attempts

For production, consider:
- ML-based injection detection (e.g., rebuff, guardrails)
- Input/output firewalls
- Semantic similarity to known attacks
- Logging of blocked attempts for analysis

SECURITY:
- This is defense-in-depth layer 1 (input sanitization)
- Layer 2 is the hardened system prompt in rag_core.py
- Layer 3 is structured output validation
- Layer 4 is citation verification
"""

from __future__ import annotations

import re

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

_FORBIDDEN = [
    "ignore previous",
    "disregard previous",
    "forget the instructions",
    "oublie les instructions",
    "ignore les instructions",
    "system prompt",
    "système:",
    "system:",
    "ignore all",
    "ignore tout",
    "new instructions",
    "nouvelles instructions",
    "ignore above",
    "disregard above",
    "override",
]


def sanitize_question(q: str, max_len: int = 2000) -> str:
    """
    Sanitize user input for basic prompt injection protection.

    This is NOT a complete defense - see rag_core.py for the
    hardened system prompt that treats all sources as untrusted.

    Args:
        q: User's question
        max_len: Maximum allowed length

    Returns:
        Sanitized question, or empty string if injection detected
    """
    q = (q or "").strip()
    q = _CONTROL_CHARS.sub("", q)
    q = q[:max_len]

    low = q.lower()
    if any(p in low for p in _FORBIDDEN):
        return ""

    return q
