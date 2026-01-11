"""
RAG Facade - Backward-compatible API for pages.

This module provides the same API as the previous monolithic rag.py,
but delegates to rag_core (logic) and rag_runtime (adapters).

Pages can continue using:
    from backend.rag import answer_question, stream_answer_tokens, ...

without knowing about the internal refactoring.

Design:
- answer_question_buffered() is the recommended entry point
- stream_answer_tokens() is deprecated for legal contexts

v1.9 Changes:
- Added citation verification (verify_citation_fn)
- Pipeline: Query → Multi-query → Hybrid → Rerank → LLM → Citation verification
"""

from __future__ import annotations

from typing import Any, Iterator

from .rag_core import (
    answer_question as core_answer,
    answer_question_buffered as core_answer_buffered,
    stream_answer as core_stream,
    validate_inline_citations,
    postprocess_micro_quotes,
    refusal,
)
from .rag_runtime import (
    make_config,
    retriever_fn,
    llm_invoke_structured,
    llm_stream_tokens,
    vectorstore,
    chunk_text,
    add_doc_chunks,
)
from .citation_verifier import verify_answer
from .security import sanitize_question
from .settings import settings

# Re-export for backward compatibility
__all__ = [
    "answer_question",
    "answer_question_buffered",
    "stream_answer_tokens",
    "validate_streamed_answer",
    "chunk_text",
    "add_doc_chunks",
    "vectorstore",
    "refusal",
]


def _citation_verify_fn(
    answer: str,
    sources_meta: list[dict[str, Any]],
    source_texts: list[str],
    level: str,
    threshold: float,
    min_words: int,
) -> str:
    """
    Citation verification wrapper for rag_core.

    v1.9: Verifies each citation against source documents.
    """
    result = verify_answer(
        answer_text=answer,
        sources_meta=sources_meta,
        source_texts=source_texts,
        embed_fn=None,  # Semantic check requires embeddings, not yet implemented
        verification_level=level,
        semantic_threshold=threshold,
        min_overlap_words=min_words,
    )
    return result.verified_answer


def answer_question_buffered(
    question: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, Any]], list[str], list[str]]:
    """
    Answer a question using RAG with structured output and full validation.

    This is the RECOMMENDED entry point (non-streaming, fully validated).
    Returns complete audit information without exposing any response until validated.

    v1.11 Pipeline:
    Query (pre-contextualized) → Multi-query (3 variants) →
    Hybrid BM25+Dense (top_k=100) → Rerank (top_n=15) → LLM → Citation verification

    Note: Query contextualization should be done by the caller (UI layer)
    before calling this function. The history is still passed for LLM context.

    Args:
        question: User's question (should be pre-contextualized if needed)
        history: Optional conversation history for LLM context

    Returns:
        Tuple of (answer_text, sources_metadata, doc_ids, chunk_ids)
    """
    # Sanitize before passing to core
    question = sanitize_question(question, settings.max_question_len)
    if not question:
        return refusal(), [], [], []

    cfg = make_config()
    return core_answer_buffered(
        question=question,
        retriever=retriever_fn,
        llm_invoke_structured=llm_invoke_structured,
        cfg=cfg,
        citation_verify_fn=_citation_verify_fn if settings.citation_verification_enabled else None,
        history=history,
    )


def answer_question(
    question: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Answer a question using RAG with structured output.

    Args:
        question: User's question (should be pre-contextualized if needed)
        history: Optional conversation history for LLM context

    Returns:
        Tuple of (answer_text, sources_metadata)
    """
    # Sanitize before passing to core
    question = sanitize_question(question, settings.max_question_len)
    if not question:
        return refusal(), []

    cfg = make_config()
    return core_answer(
        question=question,
        retriever=retriever_fn,
        llm_invoke_structured=llm_invoke_structured,
        cfg=cfg,
        history=history,
    )


def stream_answer_tokens(
    question: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[str | None, list[dict[str, Any]], int, Iterator[str]]:
    """
    Stream an answer token by token.

    WARNING: This function is DEPRECATED for legal contexts.
    Use answer_question_buffered() instead to ensure validation before display.

    Args:
        question: User's question (should be pre-contextualized if needed)
        history: Optional conversation history for LLM context

    Returns:
        Tuple of (refusal_or_none, sources_meta, max_src, token_iterator)

        - If refusal_or_none is not None: display it, ignore iterator
        - Otherwise: consume iterator, then validate with validate_streamed_answer()
    """
    # Sanitize before passing to core
    question = sanitize_question(question, settings.max_question_len)
    if not question:
        return refusal(), [], 0, iter(())

    cfg = make_config()
    return core_stream(
        question=question,
        retriever=retriever_fn,
        llm_stream=llm_stream_tokens,
        cfg=cfg,
        history=history,
    )


def validate_streamed_answer(answer_text: str, max_src: int) -> bool:
    """
    Validate citations in a streamed answer.

    Should be called after consuming the token iterator from stream_answer_tokens().

    Args:
        answer_text: The accumulated answer text
        max_src: Maximum valid source number

    Returns:
        True if citations are valid
    """
    return validate_inline_citations(answer_text, max_src)
