"""
Tests for multi-query expansion module.

Tests cover:
- Query variant generation
- Document deduplication
- Full expand and retrieve pipeline
- Error handling and fallbacks
"""

from dataclasses import dataclass
from typing import Any

from backend.multi_query import (
    deduplicate_documents,
    expand_and_retrieve,
    generate_query_variants,
)

# ============================================================================
# TEST FIXTURES
# ============================================================================


@dataclass
class FakeDoc:
    """Fake document for testing."""

    page_content: str
    metadata: dict[str, Any]


def fake_llm_expand(prompt: str) -> list[str]:
    """Fake LLM that returns predefined variants."""
    return [
        "Reformulation 1 de la question",
        "Reformulation 2 de la question",
        "Reformulation 3 de la question",
    ]


def fake_llm_expand_empty(prompt: str) -> list[str]:
    """Fake LLM that returns empty list."""
    return []


def fake_llm_expand_with_duplicates(prompt: str) -> list[str]:
    """Fake LLM that returns duplicates."""
    return [
        "Reformulation 1",
        "Reformulation 1",  # Duplicate
        "Reformulation 2",
    ]


def fake_llm_expand_failing(prompt: str) -> list[str]:
    """Fake LLM that raises exception."""
    raise ValueError("LLM expansion failed")


def fake_retriever(query: str, k: int) -> list[tuple[FakeDoc, float]]:
    """Fake retriever that returns docs based on query."""
    return [
        (FakeDoc(page_content=f"Doc for: {query}", metadata={"source": "test.txt"}), 0.9),
        (FakeDoc(page_content=f"Another doc for: {query}", metadata={"source": "test2.txt"}), 0.8),
    ]


def fake_retriever_empty(query: str, k: int) -> list[tuple[FakeDoc, float]]:
    """Fake retriever that returns no documents."""
    return []


def fake_retriever_failing(query: str, k: int) -> list[tuple[FakeDoc, float]]:
    """Fake retriever that raises exception."""
    raise ValueError("Retrieval failed")


# ============================================================================
# GENERATE QUERY VARIANTS TESTS
# ============================================================================


def test_generate_query_variants_includes_original():
    """Test that original question is always included."""
    original = "Question originale"
    variants = generate_query_variants(original, fake_llm_expand, num_variants=3)

    assert original in variants
    assert variants[0] == original  # First should be original


def test_generate_query_variants_adds_generated():
    """Test that generated variants are added."""
    variants = generate_query_variants("Question", fake_llm_expand, num_variants=3)

    # Should have original + up to 3 variants
    assert len(variants) <= 4
    assert len(variants) >= 1


def test_generate_query_variants_empty_question():
    """Test that empty question returns empty list."""
    variants = generate_query_variants("", fake_llm_expand, num_variants=3)
    assert variants == []


def test_generate_query_variants_whitespace_question():
    """Test that whitespace-only question returns empty list."""
    variants = generate_query_variants("   ", fake_llm_expand, num_variants=3)
    assert variants == []


def test_generate_query_variants_no_llm_output():
    """Test handling of empty LLM output."""
    variants = generate_query_variants("Question", fake_llm_expand_empty, num_variants=3)

    # Should still have original
    assert len(variants) >= 1
    assert "Question" in variants


def test_generate_query_variants_deduplicates():
    """Test that duplicate variants are removed."""
    variants = generate_query_variants("Question", fake_llm_expand_with_duplicates, num_variants=3)

    # Should not have duplicates
    assert len(variants) == len(set(variants))


def test_generate_query_variants_respects_limit():
    """Test that variant count respects num_variants."""
    variants = generate_query_variants("Question", fake_llm_expand, num_variants=2)

    # Original + 2 variants max
    assert len(variants) <= 3


def test_generate_query_variants_llm_failure():
    """Test graceful handling of LLM failure."""
    variants = generate_query_variants("Question", fake_llm_expand_failing, num_variants=3)

    # Should fall back to just original
    assert len(variants) == 1
    assert "Question" in variants


def test_generate_query_variants_strips_whitespace():
    """Test that variants are stripped of whitespace."""

    def llm_with_whitespace(prompt: str) -> list[str]:
        return ["  variant with spaces  ", "\nvariant with newlines\n"]

    variants = generate_query_variants("Question", llm_with_whitespace, num_variants=2)

    for v in variants:
        assert v == v.strip()


# ============================================================================
# DEDUPLICATE DOCUMENTS TESTS
# ============================================================================


def test_deduplicate_documents_basic():
    """Test basic deduplication."""
    docs = [
        (FakeDoc(page_content="Same content", metadata={}), 0.9),
        (FakeDoc(page_content="Same content", metadata={}), 0.8),
        (FakeDoc(page_content="Different content", metadata={}), 0.7),
    ]

    result = deduplicate_documents(docs)

    # Should have 2 unique docs
    assert len(result) == 2


def test_deduplicate_documents_keeps_highest_score():
    """Test that highest score is kept for duplicates."""
    docs = [
        (FakeDoc(page_content="Duplicate doc", metadata={}), 0.5),
        (FakeDoc(page_content="Duplicate doc", metadata={}), 0.9),  # Higher score
        (FakeDoc(page_content="Duplicate doc", metadata={}), 0.7),
    ]

    result = deduplicate_documents(docs)

    assert len(result) == 1
    assert result[0][1] == 0.9  # Highest score kept


def test_deduplicate_documents_empty():
    """Test deduplication of empty list."""
    result = deduplicate_documents([])
    assert result == []


def test_deduplicate_documents_sorted_by_score():
    """Test that result is sorted by score descending."""
    docs = [
        (FakeDoc(page_content="Low score", metadata={}), 0.3),
        (FakeDoc(page_content="High score", metadata={}), 0.9),
        (FakeDoc(page_content="Medium score", metadata={}), 0.6),
    ]

    result = deduplicate_documents(docs)

    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)


def test_deduplicate_documents_uses_content_prefix():
    """Test that deduplication uses first 100 chars of content."""
    # Same first 100 chars but different after
    prefix = "A" * 100
    docs = [
        (FakeDoc(page_content=prefix + "suffix1", metadata={}), 0.9),
        (FakeDoc(page_content=prefix + "suffix2", metadata={}), 0.8),
    ]

    result = deduplicate_documents(docs)

    # Should be considered duplicates (same first 100 chars)
    assert len(result) == 1


# ============================================================================
# EXPAND AND RETRIEVE TESTS
# ============================================================================


def test_expand_and_retrieve_basic():
    """Test basic expand and retrieve."""
    result = expand_and_retrieve(
        question="Test question",
        retriever_fn=fake_retriever,
        llm_expand_fn=fake_llm_expand,
        k_per_query=2,
        num_variants=2,
        enabled=True,
    )

    assert len(result) > 0
    for doc, score in result:
        assert hasattr(doc, "page_content")
        assert isinstance(score, float)


def test_expand_and_retrieve_disabled():
    """Test that disabled mode uses single query."""
    call_count = 0

    def counting_retriever(query: str, k: int) -> list[tuple[FakeDoc, float]]:
        nonlocal call_count
        call_count += 1
        return fake_retriever(query, k)

    expand_and_retrieve(
        question="Test",
        retriever_fn=counting_retriever,
        llm_expand_fn=fake_llm_expand,
        k_per_query=2,
        num_variants=3,
        enabled=False,  # Disabled
    )

    # Should only call retriever once (no expansion)
    assert call_count == 1


def test_expand_and_retrieve_empty_question():
    """Test expand and retrieve with empty question."""
    result = expand_and_retrieve(
        question="",
        retriever_fn=fake_retriever,
        llm_expand_fn=fake_llm_expand,
        k_per_query=2,
        num_variants=2,
        enabled=True,
    )

    assert result == []


def test_expand_and_retrieve_deduplicates():
    """Test that results are deduplicated."""

    def retriever_with_duplicates(query: str, k: int) -> list[tuple[FakeDoc, float]]:
        # Always return same document
        return [(FakeDoc(page_content="Same doc", metadata={}), 0.9)]

    result = expand_and_retrieve(
        question="Test",
        retriever_fn=retriever_with_duplicates,
        llm_expand_fn=fake_llm_expand,
        k_per_query=2,
        num_variants=3,
        enabled=True,
    )

    # Should have only 1 unique doc despite multiple queries
    assert len(result) == 1


def test_expand_and_retrieve_fallback_on_error():
    """Test fallback to single query on error."""

    def failing_llm(prompt: str) -> list[str]:
        raise ValueError("LLM failed")

    # Should not raise, should fall back
    result = expand_and_retrieve(
        question="Test",
        retriever_fn=fake_retriever,
        llm_expand_fn=failing_llm,
        k_per_query=2,
        num_variants=3,
        enabled=True,
    )

    # Should still get results (from fallback single query)
    assert len(result) >= 0


def test_expand_and_retrieve_combines_results():
    """Test that results from all variants are combined."""
    query_count = 0

    def tracking_retriever(query: str, k: int) -> list[tuple[FakeDoc, float]]:
        nonlocal query_count
        query_count += 1
        return [(FakeDoc(page_content=f"Doc {query_count}", metadata={}), 0.9 - query_count * 0.1)]

    expand_and_retrieve(
        question="Test",
        retriever_fn=tracking_retriever,
        llm_expand_fn=fake_llm_expand,  # Returns 3 variants
        k_per_query=1,
        num_variants=3,
        enabled=True,
    )

    # Should have called retriever for original + variants
    assert query_count >= 2
