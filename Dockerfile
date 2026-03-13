FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install system dependencies needed by PDF extraction and ripgrep
RUN apt-get update && apt-get install -y --no-install-recommends \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy application code
COPY . .

EXPOSE 8501

# Ollama connectivity: RCA_EMBEDDING_BASE_URL is set in docker-compose.yml
# to http://host.docker.internal:11434 so the container reaches the host's Ollama.
CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
