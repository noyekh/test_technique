"""
Tests for citation verification module.

Tests cover:
- Citation extraction from LLM responses
- Basic citation validation (range check)
- Presence verification (word overlap)
- Semantic verification (embedding similarity)
- Full answer verification pipeline
- Ungrounded claim filtering
"""

from dataclasses import dataclass
from typing import Any

from backend.citation_verifier import (
    CitationCheck,
    VerificationResult,
    compute_similarity,
    extract_citations,
    filter_ungrounded_claims,
    verify_answer,
    verify_citation_basic,
    verify_citation_presence,
    verify_citation_semantic,
)

# ============================================================================
# TEST FIXTURES
# ============================================================================


@dataclass
class FakeDoc:
    """Fake document for testing."""

    page_content: str
    metadata: dict[str, Any]


def fake_embed_fn(text: str) -> list[float]:
    """Simple fake embedding function for testing."""
    # Return a simple embedding based on text length and first char
    # This creates somewhat different vectors for different texts
    base = [ord(c) / 255 for c in text[:10].ljust(10)]
    return base + [len(text) / 1000] * 6  # 16-dim vector


def constant_embed_fn(text: str) -> list[float]:
    """Returns constant embedding for testing zero similarity cases."""
    return [0.5] * 16


# ============================================================================
# EXTRACT CITATIONS TESTS
# ============================================================================


def test_extract_citations_single():
    """Test extracting a single citation."""
    text = "Le contrat prévoit une garantie de 2 ans [Source 1]."
    citations = extract_citations(text)

    assert len(citations) == 1
    assert citations[0][1] == 1
    assert "[Source 1]" in citations[0][0]


def test_extract_citations_multiple():
    """Test extracting multiple citations."""
    text = "Article 1 définit la garantie [Source 1]. Article 2 précise les délais [Source 2]."
    citations = extract_citations(text)

    assert len(citations) == 2
    assert citations[0][1] == 1
    assert citations[1][1] == 2


def test_extract_citations_none():
    """Test text without citations."""
    text = "Le contrat prévoit une garantie de 2 ans."
    citations = extract_citations(text)

    assert len(citations) == 0


def test_extract_citations_case_insensitive():
    """Test that citation extraction is case-insensitive."""
    text = "Voir [source 1] et [SOURCE 2]."
    citations = extract_citations(text)

    # Both citations should be extracted from the same sentence
    assert len(citations) == 2
    assert citations[0][1] == 1
    assert citations[1][1] == 2


def test_extract_citations_with_surrounding_text():
    """Test that extracted citation includes surrounding sentence."""
    text = "La responsabilité est limitée à 1000 euros [Source 3]. Fin."
    citations = extract_citations(text)

    assert len(citations) == 1
    assert "1000 euros" in citations[0][0]


# ============================================================================
# COMPUTE SIMILARITY TESTS
# ============================================================================


def test_compute_similarity_identical():
    """Test similarity of identical texts."""
    similarity = compute_similarity("test text", "test text", fake_embed_fn)
    assert similarity > 0.99  # Should be ~1.0


def test_compute_similarity_different():
    """Test similarity of different texts."""
    similarity = compute_similarity("short", "completely different longer text here", fake_embed_fn)
    # Different texts should have lower similarity
    assert similarity < 0.99


def test_compute_similarity_empty():
    """Test similarity with empty text returns 0."""

    def zero_embed(text: str) -> list[float]:
        if not text:
            return [0.0] * 16
        return [1.0] * 16

    similarity = compute_similarity("", "text", zero_embed)
    assert similarity == 0.0


def test_compute_similarity_exception_returns_zero():
    """Test that exceptions return 0.0."""

    def failing_embed(text: str) -> list[float]:
        raise ValueError("Embedding failed")

    similarity = compute_similarity("text1", "text2", failing_embed)
    assert similarity == 0.0


# ============================================================================
# BASIC VERIFICATION TESTS
# ============================================================================


def test_verify_citation_basic_valid():
    """Test valid citation numbers."""
    assert verify_citation_basic(1, 5) is True
    assert verify_citation_basic(3, 5) is True
    assert verify_citation_basic(5, 5) is True


def test_verify_citation_basic_invalid_zero():
    """Test that citation 0 is invalid."""
    assert verify_citation_basic(0, 5) is False


def test_verify_citation_basic_invalid_over():
    """Test that citation > max is invalid."""
    assert verify_citation_basic(6, 5) is False
    assert verify_citation_basic(99, 5) is False


def test_verify_citation_basic_negative():
    """Test that negative citations are invalid."""
    assert verify_citation_basic(-1, 5) is False


# ============================================================================
# PRESENCE VERIFICATION TESTS
# ============================================================================


def test_verify_citation_presence_valid():
    """Test presence verification with overlapping words."""
    # Use words that are NOT stopwords (le, la, de, du, etc. are removed)
    claim = "contrat prévoit garantie conformité produit"
    source = "Article vendeur garantit conformité produit contrat"

    is_valid, reason = verify_citation_presence(claim, source, min_overlap_words=3)

    assert is_valid is True
    assert "overlapping words" in reason


def test_verify_citation_presence_invalid():
    """Test presence verification with no overlap."""
    claim = "Le contrat prévoit une garantie de conformité."
    source = "Météo: Il fait beau aujourd'hui en France."

    is_valid, reason = verify_citation_presence(claim, source, min_overlap_words=3)

    assert is_valid is False


def test_verify_citation_presence_stopwords_ignored():
    """Test that stopwords are not counted."""
    claim = "Le la les de du"  # Only stopwords
    source = "Le la les de du article"

    is_valid, reason = verify_citation_presence(claim, source, min_overlap_words=3)

    # Should fail because stopwords are ignored
    assert is_valid is False


def test_verify_citation_presence_custom_threshold():
    """Test presence verification with custom threshold."""
    claim = "garantie conformité"
    source = "garantie conformité produit"

    # With threshold 2, should pass
    is_valid_2, _ = verify_citation_presence(claim, source, min_overlap_words=2)
    assert is_valid_2 is True

    # With threshold 5, should fail
    is_valid_5, _ = verify_citation_presence(claim, source, min_overlap_words=5)
    assert is_valid_5 is False


# ============================================================================
# SEMANTIC VERIFICATION TESTS
# ============================================================================


def test_verify_citation_semantic_valid():
    """Test semantic verification with similar texts."""
    claim = "test content"
    source = "test content here"

    is_valid, confidence, reason = verify_citation_semantic(
        claim, source, fake_embed_fn, threshold=0.5
    )

    assert is_valid is True
    assert confidence > 0.5
    assert "similarity" in reason.lower()


def test_verify_citation_semantic_invalid():
    """Test semantic verification with different texts."""
    claim = "aaa"
    source = "zzz completely different"

    is_valid, confidence, reason = verify_citation_semantic(
        claim, source, fake_embed_fn, threshold=0.99
    )

    # With very high threshold, should fail
    assert is_valid is False
    assert "low" in reason.lower()


def test_verify_citation_semantic_threshold():
    """Test semantic verification respects threshold."""
    claim = "text"
    source = "text"

    # Low threshold should pass
    is_valid_low, _, _ = verify_citation_semantic(claim, source, fake_embed_fn, threshold=0.1)
    assert is_valid_low is True


# ============================================================================
# FULL VERIFICATION PIPELINE TESTS
# ============================================================================


def test_verify_answer_empty():
    """Test verification of empty answer."""
    result = verify_answer("", [], [], None)

    assert result.original_answer == ""
    assert result.verified_answer == ""
    assert result.total_citations == 0
    assert result.valid_citations == 0


def test_verify_answer_basic_level():
    """Test verification at basic level."""
    answer = "Le contrat dit X [Source 1]. Le contrat dit Y [Source 99]."
    sources_meta = [{"source": "doc.txt"}]
    source_texts = ["Le contrat stipule X et autres clauses."]

    result = verify_answer(
        answer,
        sources_meta,
        source_texts,
        verification_level="basic",
    )

    assert result.total_citations == 2
    assert result.valid_citations == 1  # Source 99 is invalid
    assert 99 in result.removed_citations


def test_verify_answer_presence_level():
    """Test verification at presence level."""
    answer = "La garantie est de 2 ans [Source 1]. Le ciel est bleu [Source 2]."
    sources_meta = [{"source": "contrat.txt"}, {"source": "autre.txt"}]
    source_texts = [
        "Article 1: La garantie de conformité est de 2 ans.",
        "Article 2: Les conditions de vente sont définies.",
    ]

    result = verify_answer(
        answer,
        sources_meta,
        source_texts,
        verification_level="presence",
        min_overlap_words=2,
    )

    # First citation should pass (word overlap with garantie, ans)
    # Second citation should fail (no overlap with "ciel bleu")
    assert result.total_citations == 2
    assert 2 in result.removed_citations


def test_verify_answer_semantic_level():
    """Test verification at semantic level."""
    answer = "Information importante [Source 1]."
    sources_meta = [{"source": "doc.txt"}]
    source_texts = ["Information importante dans ce document."]

    result = verify_answer(
        answer,
        sources_meta,
        source_texts,
        embed_fn=fake_embed_fn,
        verification_level="semantic",
        semantic_threshold=0.5,
        min_overlap_words=1,
    )

    assert result.total_citations == 1


def test_verify_answer_removes_invalid_sentences():
    """Test that invalid citations are removed from verified answer."""
    answer = "Valid claim [Source 1]. Invalid claim [Source 99]. Another valid [Source 1]."
    sources_meta = [{"source": "doc.txt"}]
    source_texts = ["Valid claim content here."]

    result = verify_answer(
        answer,
        sources_meta,
        source_texts,
        verification_level="basic",
    )

    assert "Source 99" not in result.verified_answer
    assert "Invalid claim" not in result.verified_answer


def test_verify_answer_returns_checks():
    """Test that verification returns detailed checks."""
    answer = "Claim A [Source 1]. Claim B [Source 2]."
    sources_meta = [{"source": "a.txt"}, {"source": "b.txt"}]
    source_texts = ["Content A", "Content B"]

    result = verify_answer(
        answer,
        sources_meta,
        source_texts,
        verification_level="basic",
    )

    assert len(result.checks) == 2
    for check in result.checks:
        assert isinstance(check, CitationCheck)
        assert check.citation_num in [1, 2]


# ============================================================================
# FILTER UNGROUNDED CLAIMS TESTS
# ============================================================================


def test_filter_ungrounded_claims_presence_mode():
    """Test filtering with presence mode (no embed_fn)."""
    answer = "Garantie deux ans [Source 1]. Unrelated text [Source 2]."
    sources_meta = [{"source": "contrat.txt"}, {"source": "autre.txt"}]
    source_texts = [
        "La garantie est de deux ans selon l'article 1.",
        "Conditions générales de vente.",
    ]

    verified = filter_ungrounded_claims(answer, sources_meta, source_texts, embed_fn=None)

    # Should use presence level without embed_fn
    assert isinstance(verified, str)


def test_filter_ungrounded_claims_semantic_mode():
    """Test filtering with semantic mode (with embed_fn)."""
    answer = "Test content [Source 1]."
    sources_meta = [{"source": "doc.txt"}]
    source_texts = ["Test content in document."]

    verified = filter_ungrounded_claims(answer, sources_meta, source_texts, embed_fn=fake_embed_fn)

    # Should use semantic level with embed_fn
    assert isinstance(verified, str)


def test_filter_ungrounded_claims_empty():
    """Test filtering empty answer."""
    verified = filter_ungrounded_claims("", [], [], None)
    assert verified == ""


# ============================================================================
# VERIFICATION RESULT DATACLASS TESTS
# ============================================================================


def test_verification_result_dataclass():
    """Test VerificationResult dataclass."""
    result = VerificationResult(
        original_answer="Original [Source 1].",
        verified_answer="Original [Source 1].",
        total_citations=1,
        valid_citations=1,
        removed_citations=[],
        checks=[],
    )

    assert result.original_answer == "Original [Source 1]."
    assert result.valid_citations == 1
    assert len(result.removed_citations) == 0


def test_citation_check_dataclass():
    """Test CitationCheck dataclass."""
    check = CitationCheck(
        citation_num=1,
        claim_text="Test claim [Source 1].",
        source_text="Source content",
        is_valid=True,
        confidence=0.95,
        reason="Citation verified",
    )

    assert check.citation_num == 1
    assert check.is_valid is True
    assert check.confidence == 0.95
