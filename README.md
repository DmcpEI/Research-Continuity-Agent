# Research Continuity Agent

Research Continuity Agent is a Python-first skeleton for ingesting research artifacts, storing them in graph and vector backends, and building retrieval-first workflows before generation.

## Layout

The repository is organized around a single package, `rca`, plus thin top-level layers for docs, CLI entrypoints, evaluation, and tests.

- `rca/contracts`: shared identifiers and DTOs.
- `rca/store`: graph, vector, and event-log persistence.
- `rca/extractors`: deterministic parsers for source materials.
- `rca/flows`: ingest and retrieval business logic.
- `rca/orchestrator`: routing and state management.
- `rca/mcp_servers`: demoable service adapters.

## Quick Start

1. Create a virtual environment.
2. Install the package in editable mode with `pip install -e .[dev]`.
3. Copy `.env.example` to `.env` and adjust paths if needed.
4. Ingest a note with `python -m cli.rca_ingest path/to/note.md`.
5. Query the local index with `python -m cli.rca_query "your question"`.

## Status

This repo now contains the target skeleton with working baseline implementations for:

- settings and tool policy loading
- contracts for IDs, nodes, citations, and artifacts
- SQLite graph storage
- Chroma-compatible vector storage with an in-process fallback
- deterministic note, PDF, git, and experiment extractors
- ingest and retrieval flows
- routing/orchestration scaffolding
- demo filesystem and experiment servers

Generation is intentionally left as a stub until retrieval quality is validated.
