FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install system dependencies needed by PDF extraction and ripgrep
RUN apt-get update && apt-get install -y --no-install-recommends \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless lockfile changes).
# The project itself depends on repo files like README.md and package sources,
# so we defer installing it until after the full copy step.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# Copy application code plus the baked demo corpus under .rca/
COPY . .

# Install the project itself once the package sources are present.
RUN uv sync --frozen

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

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5).status == 200 else 1)"

# Local Docker Compose overrides the backend back to Ollama. The image defaults
# to an external OpenAI-compatible endpoint so it can run unchanged on ECS.
# Serves the built React app (frontend/dist) and the API under /api.
CMD ["uv", "run", "uvicorn", "rca.api.main:create_site", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
