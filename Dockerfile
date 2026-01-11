# Legal RAG PoC v1.10 — Production-ready Dockerfile
#
# Build:   docker build -t legal-rag-poc .
# Run:     docker run -p 8501:8501 --env-file .env -v $(pwd)/.streamlit:/app/.streamlit:ro legal-rag-poc
#
# Security:
# - Non-root user (appuser)
# - Minimal base image (slim)
# - No secrets in image (mount at runtime)

FROM python:3.12-slim

# System dependencies for python-magic (libmagic) and lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/chroma logs

# Security: run as non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Streamlit default port
EXPOSE 8501

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "main.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
