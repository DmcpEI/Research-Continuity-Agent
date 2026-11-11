.PHONY: help install run demo eval test ablations

help:
	@echo "Usage:"
	@echo "  make install    Install Python dependencies via uv"
	@echo "  make run        React app + API on :8000 over ./.rca via Docker Compose (builds image on first run)"
	@echo "  make demo       Public demo via Docker Compose: React app + API on :8000, sample corpus"
	@echo "  make eval       Run the generation evaluation harness (local, no Docker)"
	@echo "  make ablations  Run retrieval ablation study (local, no Docker)"
	@echo "  make test       Run the test suite (local, no Docker)"

install:
	uv sync

run:
	docker compose up --build rca

demo:
	docker compose up --build demo

eval:
	uv run python eval/harness.py

ablations:
	uv run python eval/run_ablations.py

test:
	uv run pytest
