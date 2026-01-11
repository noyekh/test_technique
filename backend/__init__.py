"""
Backend package for Legal RAG PoC.

Modules:
- settings: Centralized configuration
- db: SQLite persistence layer
- rag_core: Pure RAG logic (testable without dependencies)
- rag_runtime: LangChain/Streamlit adapters
- rag: Public facade
- reranker: Voyage AI reranking
- multi_query: Query expansion
- citation_verifier: Post-LLM citation validation
- documents: Document lifecycle management
- files: File operations
- ingest: Document parsing
- auth: Authentication
- audit_log: GDPR-compliant logging
- security: Input sanitization
- rate_limit: Rate limiting
- mime_validation: MIME type validation
- logging_config: Logging setup
"""
