"""
Centralized configuration for the Legal RAG PoC.

Design decisions:
- Frozen dataclass: immutable after creation, prevents accidental modification
- Environment variables: 12-factor app compliance, easy deployment configuration
- Sensible defaults: works out of the box for development

Key parameters explained:

LLM (v1.7):
- gpt-4.1-mini: Released April 2025, 1M token context window
  +17% improvement on legal cross-referencing tasks (Thomson Reuters)
  $0.40/$1.60 per 1M tokens (input/output)
  Ideal for processing entire legal documents without excessive chunking

Embeddings (v1.7 - Voyage AI):
- voyage-3-large: State-of-the-art embeddings released January 2025
  Surpasses voyage-law-2 on legal benchmarks (Voyage AI blog, Jan 2025)
  200M tokens FREE tier, then $0.05/M tokens
  32K context window (vs 8K for OpenAI)
  Recommended by Anthropic for RAG applications

RAG thresholds:
- min_relevance=0.35: Below this, documents are considered irrelevant
  (tuned empirically; legal docs often have lower similarity scores)
- keep_ratio=0.8: Secondary filter keeps docs scoring > 0.8 * min_relevance
- keep_floor=0.15: Absolute minimum to prevent noise
- top_k=6: Balance between context richness and token cost

Chunking (v1.6+):
- chunk_size_tokens=768: Measured in TOKENS, not characters
  Based on Chroma Research 2024 benchmarks showing optimal legal retrieval at 512-1024 tokens
  French legal text uses ~20-25% more tokens than English for same content
- chunk_overlap_tokens=115: ~15% overlap, preserves context across boundaries
- Token-aware via tiktoken ensures precise alignment with embedding model limits

Retrieval (v1.6+):
- hybrid_search=True: Combines BM25 (lexical) + dense (semantic) retrieval
- bm25_weight=0.6: BM25 favored for exact legal citations ("Article L.121-1")
  Based on MDPI 2025 study showing BM25 outperforms dense on legal text (ROUGE-L 0.8894)
- Anthropic Contextual Retrieval shows -49% retrieval failures with hybrid approach

Security (v1.5+):
- enable_streaming=False: DISABLED BY DEFAULT for legal contexts
  Streaming shows potentially invalid responses before validation
- max_answer_chars=4000: Prevents exfiltration via long responses

Rate limiting:
- 20 requests/60s: Prevents API abuse and cost overrun
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
SQLITE_PATH = DATA_DIR / "app.sqlite"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _env_bool(name: str, default: bool) -> bool:
    """Parse boolean from environment variable."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    """Parse float from environment variable."""
    v = os.getenv(name)
    if v is None:
        return default
    return float(v)


def _env_int(name: str, default: int) -> int:
    """Parse integer from environment variable."""
    v = os.getenv(name)
    if v is None:
        return default
    return int(v)


@dataclass(frozen=True)
class Settings:
    """
    Application settings.
    
    All settings can be overridden via environment variables.
    See .env.example for a complete list.
    
    Attributes:
        openai_chat_model: LLM model for chat completion (v1.7: gpt-4.1-mini)
        voyage_embed_model: Voyage AI model for embeddings (v1.7)
        voyage_embed_dimensions: Embedding dimensions (v1.7)
        rag_collection: Chroma collection name
        rag_min_relevance: Minimum similarity score to consider a doc relevant
        rag_keep_ratio: Keep docs scoring > min_relevance * keep_ratio
        rag_keep_floor: Absolute minimum score floor
        rag_top_k: Maximum number of sources to use
        chunk_size_tokens: Target chunk size in TOKENS
        chunk_overlap_tokens: Overlap between chunks in TOKENS
        hybrid_search: Enable BM25+Dense hybrid retrieval
        bm25_weight: Weight for BM25 in hybrid search (0.0-1.0)
        max_file_size_mb: Maximum upload file size
        enable_streaming: Whether to stream responses (DISABLED by default)
        rate_limit_max_requests: Max requests per window
        rate_limit_window_seconds: Rate limit window duration
        max_question_len: Maximum question length in characters
        max_answer_chars: Maximum answer length (anti-exfiltration)
    """
    
    # LLM (v1.7: GPT-4.1-mini with 1M context)
    # +17% improvement on legal cross-referencing (Thomson Reuters)
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    
    # Embeddings (v1.7: Voyage AI)
    # voyage-3-large surpasses voyage-law-2 on legal benchmarks (Voyage AI, Jan 2025)
    # 200M tokens FREE, then $0.05/M - 32K context window
    voyage_embed_model: str = os.getenv("VOYAGE_EMBED_MODEL", "voyage-3-large")
    voyage_embed_dimensions: int = _env_int("VOYAGE_EMBED_DIMENSIONS", 1024)

    # RAG collection
    rag_collection: str = os.getenv("RAG_COLLECTION", "legal_docs")

    # RAG thresholds (configurable)
    rag_min_relevance: float = _env_float("RAG_MIN_RELEVANCE", 0.35)
    rag_keep_ratio: float = _env_float("RAG_KEEP_RATIO", 0.8)
    rag_keep_floor: float = _env_float("RAG_KEEP_FLOOR", 0.15)
    rag_top_k: int = _env_int("RAG_TOP_K", 6)

    # Chunking (Token-aware with French legal separators)
    # 768 tokens optimal for French legal (512-1024 range per Chroma Research 2024)
    chunk_size_tokens: int = _env_int("CHUNK_SIZE_TOKENS", 768)
    # ~15% overlap preserves context (115/768 ≈ 15%)
    chunk_overlap_tokens: int = _env_int("CHUNK_OVERLAP_TOKENS", 115)
    
    # Hybrid search
    # BM25+Dense combination based on Anthropic Contextual Retrieval (-49% failures)
    hybrid_search: bool = _env_bool("HYBRID_SEARCH", True)
    # BM25 favored for exact legal citations (MDPI 2025: ROUGE-L 0.8894)
    bm25_weight: float = _env_float("BM25_WEIGHT", 0.6)

    # Upload limits
    max_file_size_mb: int = _env_int("MAX_FILE_SIZE_MB", 10)

    # Streaming - DISABLED BY DEFAULT for legal contexts (v1.5+)
    # Set ENABLE_STREAMING=true only if you accept the risk of
    # showing potentially invalid responses before validation
    enable_streaming: bool = _env_bool("ENABLE_STREAMING", False)

    # Rate limiting
    rate_limit_max_requests: int = _env_int("RATE_LIMIT_MAX_REQUESTS", 20)
    rate_limit_window_seconds: int = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)

    # Question length limit
    max_question_len: int = _env_int("MAX_QUESTION_LEN", 2000)
    
    # Answer length limit (anti-exfiltration)
    max_answer_chars: int = _env_int("MAX_ANSWER_CHARS", 4000)

    # Authentication (v1.8)
    auth_enabled: bool = _env_bool("AUTH_ENABLED", True)


settings = Settings()
