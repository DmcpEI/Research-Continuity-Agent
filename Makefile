.PHONY: help install run eval test ablations

help:
	@echo "Usage:"
	@echo "  make install    Install Python dependencies via uv"
	@echo "  make run        Start the app via Docker Compose (builds image on first run)"
	@echo "  make eval       Run the generation evaluation harness (local, no Docker)"
	@echo "  make ablations  Run retrieval ablation study (local, no Docker)"
	@echo "  make test       Run the test suite (local, no Docker)"

install:
	uv sync

run:
	docker compose up --build

eval:
	uv run python eval/harness.py

ablations:
	uv run python eval/run_ablations.py

test:
	uv run pytest
