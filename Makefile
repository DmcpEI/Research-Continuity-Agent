.PHONY: help install run eval test ablations aws-demo aws-deploy aws-down

help:
	@echo "Usage:"
	@echo "  make install    Install Python dependencies via uv"
	@echo "  make run        Start the app via Docker Compose (builds image on first run)"
	@echo "  make eval       Run the generation evaluation harness (local, no Docker)"
	@echo "  make ablations  Run retrieval ablation study (local, no Docker)"
	@echo "  make test       Run the test suite (local, no Docker)"
	@echo "  make aws-demo   Build, push, run a one-off ECS demo task, then tear it down"
	@echo "  make aws-deploy Build, push, and update an ECS service (expects precreated infra)"
	@echo "  make aws-down   Scale the ECS service back to zero"

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

aws-demo:
	./infra/demo.sh

aws-deploy:
	./infra/deploy.sh

aws-down:
	./infra/teardown_service.sh
