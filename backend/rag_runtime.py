"""
RAG Runtime - Streamlit and LangChain adapters.

This module provides concrete implementations of the interfaces defined
in rag_core.py. It handles:
- Streamlit caching (@st.cache_resource)
- LangChain components (embeddings, vectorstore, LLM)
- Conversion between generic message format and LangChain prompts

The separation allows rag_core to be tested without these dependencies.

Design decisions:

Why LangChain?
- Unified interface for multiple LLM providers
- Built-in retry, streaming, structured output
- Active community, good documentation
- Easy to swap providers (OpenAI → Anthropic → local)

Why Chroma over FAISS?
- Persistent storage out of the box
- Better CRUD operations (delete by ID)
- Metadata filtering support
- Good enough performance for PoC scale

Why @st.cache_resource?
- Singleton pattern for expensive resources
- Survives reruns (essential for Streamlit)
- Automatic invalidation on code changes

Why temperature=0?
- Deterministic outputs for legal context
- Reproducible answers
- Reduced hallucination risk

v1.7 Changes:

LLM: GPT-4.1-mini
- Released April 2025 with 1M token context window
- +17% improvement on legal cross-referencing (Thomson Reuters)
- $0.40/$1.60 per 1M tokens (input/output)

Embeddings: Voyage AI voyage-3-large
- State-of-the-art embeddings (Voyage AI, January 2025)
- Surpasses voyage-law-2 on legal benchmarks
- 200M tokens FREE, then $0.05/M tokens
- 32K context window (vs 8K for OpenAI)
- Recommended by Anthropic for RAG applications

v1.6 Changes (retained):

Token-aware chunking:
- Uses tiktoken (cl100k_base) for precise token measurement
- 768 tokens optimal for French legal text (Chroma Research 2024)
- Legal-specific separators preserve Article/Alinéa boundaries

Hybrid BM25+Dense retrieval:
- EnsembleRetriever combines lexical (BM25) and semantic (dense) search
- BM25 weighted 0.6 for exact legal citations ("Article L.121-1")
- Based on Anthropic Contextual Retrieval (-49% retrieval failures)
- MDPI 2025: BM25 achieves ROUGE-L 0.8894 on legal text
"""

from __future__ import annotations

import logging
from typing import Iterator

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from .rag_core import RagConfig
from .settings import CHROMA_DIR, settings

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================


class RagAnswer(BaseModel):
    """Structured output model for RAG answers."""

    answer: str = Field(
        ..., description="Réponse en français, basée uniquement sur le contexte."
    )
    citations: list[int] = Field(
        default_factory=list, description="Numéros de SOURCES utilisées (ex: [1, 3])."
    )


# ============================================================================
# CACHED RESOURCES (Streamlit)
# ============================================================================


@st.cache_resource
def embeddings():
    """
    Get cached Voyage AI embeddings model.
    
    v1.7: Uses voyage-3-large which surpasses voyage-law-2 on legal benchmarks.
    - 200M tokens FREE tier, then $0.05/M tokens
    - 32K context window (vs 8K for OpenAI)
    - Recommended by Anthropic for RAG applications
    
    Requires VOYAGE_API_KEY environment variable.
    """
    return VoyageAIEmbeddings(
        model=settings.voyage_embed_model,
        voyage_api_key=None,  # Uses VOYAGE_API_KEY env var
    )


@st.cache_resource
def vectorstore():
    """Get cached Chroma vectorstore."""
    return Chroma(
        collection_name=settings.rag_collection,
        embedding_function=embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


@st.cache_resource
def llm():
    """Get cached LLM for structured output (non-streaming)."""
    return ChatOpenAI(model=settings.openai_chat_model, temperature=0)


@st.cache_resource
def llm_streaming():
    """Get cached LLM for streaming output."""
    return ChatOpenAI(model=settings.openai_chat_model, temperature=0, streaming=True)


# ============================================================================
# CONFIGURATION FACTORY
# ============================================================================


def make_config() -> RagConfig:
    """Create RagConfig from application settings."""
    return RagConfig(
        min_relevance=settings.rag_min_relevance,
        keep_ratio=settings.rag_keep_ratio,
        keep_floor=settings.rag_keep_floor,
        top_k=settings.rag_top_k,
        retrieve_k=settings.rag_top_k,
        max_question_len=settings.max_question_len,
        max_answer_chars=settings.max_answer_chars,
    )


# ============================================================================
# RETRIEVER IMPLEMENTATION (v1.6 - Hybrid BM25+Dense)
# ============================================================================

# In-memory document store for BM25 (rebuilt on each session)
_bm25_docs: list[Document] = []


def _rebuild_bm25_index() -> None:
    """
    Rebuild the BM25 index from vectorstore documents.
    
    Called when documents are added or removed.
    BM25 requires all documents in memory for TF-IDF calculation.
    """
    global _bm25_docs
    try:
        vs = vectorstore()
        # Get all documents from Chroma
        # Note: This loads all docs - acceptable for PoC scale (<10k docs)
        result = vs.get(include=["documents", "metadatas"])
        
        if result and result.get("documents"):
            _bm25_docs = [
                Document(
                    page_content=doc,
                    metadata=meta if meta else {}
                )
                for doc, meta in zip(result["documents"], result.get("metadatas", [{}] * len(result["documents"])))
            ]
            logger.debug(f"BM25 index rebuilt with {len(_bm25_docs)} documents")
        else:
            _bm25_docs = []
    except Exception as e:
        logger.warning(f"Failed to rebuild BM25 index: {e}")
        _bm25_docs = []


def retriever_fn(question: str, k: int) -> list[tuple[Document, float]]:
    """
    Retrieve documents using hybrid BM25+Dense search.

    v1.6: Combines lexical (BM25) and semantic (dense) retrieval.
    - BM25 excels at exact matches ("Article L.121-1", "42 U.S.C. § 1983")
    - Dense excels at semantic similarity ("responsabilité civile" ≈ "tort liability")
    - Combined approach reduces retrieval failures by 49% (Anthropic, Sept 2024)

    Args:
        question: User's search query
        k: Number of documents to retrieve

    Returns:
        List of (Document, relevance_score) tuples
    """
    if not settings.hybrid_search:
        # Fallback to dense-only (v1.5 behavior)
        try:
            return vectorstore().similarity_search_with_relevance_scores(question, k=k)
        except Exception as e:
            logger.exception("Retrieval failed", extra={"error_code": "RETRIEVAL_ERROR"})
            return []

    # Hybrid search
    try:
        vs = vectorstore()
        
        # Dense retriever
        dense_retriever = vs.as_retriever(search_kwargs={"k": k * 2})
        
        # BM25 retriever (requires documents in memory)
        if not _bm25_docs:
            _rebuild_bm25_index()
        
        if not _bm25_docs:
            # No documents indexed yet, fall back to dense-only
            logger.info("No documents for BM25, using dense-only retrieval")
            return vs.similarity_search_with_relevance_scores(question, k=k)
        
        bm25_retriever = BM25Retriever.from_documents(
            _bm25_docs,
            k=k * 2  # Retrieve more, ensemble will dedupe
        )
        
        # Ensemble with Reciprocal Rank Fusion
        ensemble = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[1 - settings.bm25_weight, settings.bm25_weight]
        )
        
        # Get documents (EnsembleRetriever returns Documents, not scores)
        docs = ensemble.invoke(question)[:k]
        
        # Get scores from dense retriever for the retrieved docs
        # (BM25 scores aren't directly comparable to cosine similarity)
        if docs:
            scored_results = vs.similarity_search_with_relevance_scores(question, k=k * 2)
            score_map = {doc.page_content[:100]: score for doc, score in scored_results}
            
            results = []
            for doc in docs:
                # Look up score, default to 0.5 if from BM25-only
                score = score_map.get(doc.page_content[:100], 0.5)
                results.append((doc, score))
            
            return results
        
        return []
        
    except Exception as e:
        logger.exception("Hybrid retrieval failed", extra={"error_code": "HYBRID_RETRIEVAL_ERROR"})
        # Fallback to dense-only on error
        try:
            return vectorstore().similarity_search_with_relevance_scores(question, k=k)
        except Exception:
            return []


# ============================================================================
# CHUNKING (v1.6 - Token-aware with legal separators)
# ============================================================================

# French legal document separators (priority order)
# Preserves Article/Alinéa boundaries critical for legal citation accuracy
LEGAL_SEPARATORS_FR = [
    # Major structural boundaries (highest priority)
    "\n\nArticle ",       # Article boundaries (most important for French law)
    "\n\nARTICLE ",       # Uppercase variant
    "\n\nArt. ",          # Abbreviated form
    "\n\nChapitre ",      # Chapter boundaries
    "\n\nCHAPITRE ",
    "\n\nSection ",       # Section boundaries
    "\n\nSECTION ",
    "\n\nTitre ",         # Title boundaries
    "\n\nTITRE ",
    "\n\nLivre ",         # Book boundaries (Code civil structure)
    "\n\nLIVRE ",
    "\n\nAnnexe ",        # Appendix boundaries
    "\n\nANNEXE ",
    # Numbered clauses
    "\n\n§ ",             # Section symbol
    "\n\n1° ",            # French ordinal numbering (1°, 2°, 3°...)
    "\n\n2° ",
    "\n\n3° ",
    # Standard separators (lower priority)
    "\n\n",               # Double newline (paragraph)
    "\n",                 # Single newline (alinéa within article)
    ". ",                 # Sentence boundary
    ", ",                 # Clause boundary
    " ",                  # Word boundary (last resort)
]


def chunk_text(text: str) -> list[str]:
    """
    Split text into chunks for vectorization.

    v1.6: Token-aware chunking with French legal separators.
    
    Key improvements over v1.5:
    - Measures in TOKENS (not characters) for precise embedding alignment
    - Uses tiktoken cl100k_base encoder (GPT-4/text-embedding-3 compatible)
    - Legal-specific separators preserve Article/Alinéa boundaries
    - 768 tokens optimal for French legal text (Chroma Research 2024)
    
    Why token-aware matters:
    - Character-based chunking can exceed embedding model limits
    - French legal text uses ~20-25% more tokens than English
    - Precise token counts enable accurate cost prediction
    
    Args:
        text: Input text to chunk
        
    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",  # Uses cl100k_base encoding
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        separators=LEGAL_SEPARATORS_FR,
    )
    docs = splitter.create_documents([text])
    return [d.page_content for d in docs]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=8))
def add_doc_chunks(doc_id: str, original_name: str, chunks: list[str]) -> list[str]:
    """
    Add document chunks to the vectorstore.

    Args:
        doc_id: Unique document identifier
        original_name: Original filename for metadata
        chunks: List of text chunks

    Returns:
        List of chunk IDs
    """
    global _bm25_docs
    
    vs = vectorstore()
    chunk_ids: list[str] = []
    docs: list[Document] = []

    for i, chunk in enumerate(chunks):
        cid = f"{doc_id}:{i}"
        chunk_ids.append(cid)
        docs.append(
            Document(
                page_content=chunk,
                metadata={"doc_id": doc_id, "source": original_name, "chunk_index": i},
            )
        )

    vs.add_documents(documents=docs, ids=chunk_ids)
    
    # Rebuild BM25 index with new documents
    _rebuild_bm25_index()
    
    logger.info("Indexed document", extra={"doc_id": doc_id, "chunks": len(chunk_ids)})
    return chunk_ids


# ============================================================================
# LLM IMPLEMENTATIONS
# ============================================================================


def _to_langchain_prompt(messages: list[dict[str, str]]) -> ChatPromptTemplate:
    """
    Convert generic message dicts to a LangChain ChatPromptTemplate.

    Expects exactly 2 messages: system and user.
    """
    if len(messages) != 2:
        raise ValueError(f"Expected 2 messages, got {len(messages)}")

    return ChatPromptTemplate.from_messages(
        [
            ("system", messages[0]["content"]),
            ("human", messages[1]["content"]),
        ]
    )


def llm_invoke_structured(messages: list[dict[str, str]]) -> tuple[str, list[int]]:
    """
    Invoke LLM with structured output parsing.

    This is the concrete implementation of LLMInvokeFn.

    Args:
        messages: List of message dicts (system, user)

    Returns:
        Tuple of (answer_text, citations_list)
    """
    parser = PydanticOutputParser(pydantic_object=RagAnswer)
    fmt = parser.get_format_instructions()

    # Append format instructions to user message
    messages = [
        messages[0],
        {
            "role": "user",
            "content": messages[1]["content"] + f"\n{fmt}",
        },
    ]

    prompt = _to_langchain_prompt(messages)
    chain = prompt | llm() | parser
    out: RagAnswer = chain.invoke({})
    return out.answer, out.citations


def llm_stream_tokens(messages: list[dict[str, str]]) -> Iterator[str]:
    """
    Stream LLM response tokens.

    This is the concrete implementation of LLMStreamFn.
    
    WARNING (v1.5): Streaming is deprecated for legal contexts.
    Use llm_invoke_structured() instead.

    Args:
        messages: List of message dicts (system, user)

    Yields:
        Token strings as they are generated
    """
    prompt = _to_langchain_prompt(messages)
    formatted = prompt.format_messages()

    for chunk in llm_streaming().stream(formatted):
        if getattr(chunk, "content", None):
            yield chunk.content
