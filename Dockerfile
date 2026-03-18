FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install system dependencies needed by PDF extraction and ripgrep
RUN apt-get update && apt-get install -y --no-install-recommends \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy application code plus the baked demo corpus under .rca/
COPY . .

ENV RCA_ENVIRONMENT=production \
    RCA_LLM_BACKEND=openai_compatible \
    RCA_ENABLE_FILESYSTEM_TOOLS=false \
    RCA_WORKSPACE_ROOT=/app \
    RCA_FILESYSTEM_ROOT=/app \
    RCA_DATA_DIR=/app/.rca \
    RCA_GRAPH_DB_PATH=/app/.rca/graph.sqlite3 \
    RCA_VECTOR_DIR=/app/.rca/vectors \
    RCA_EXPERIMENT_DB_PATH=/app/.rca/experiments.sqlite3 \
    ANONYMIZED_TELEMETRY=False

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5).status == 200 else 1)"

# Local Docker Compose overrides the backend back to Ollama. The image defaults
# to an external OpenAI-compatible endpoint so it can run unchanged on ECS.
CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
