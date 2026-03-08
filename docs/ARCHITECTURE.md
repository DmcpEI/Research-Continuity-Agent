# Architecture

The architecture is intentionally layered:

1. `contracts` defines identifiers and DTOs that every other layer shares.
2. `store` handles persistence only. No model calls belong here.
3. `extractors` turn local artifacts into deterministic text payloads.
4. `flows` compose extractors and storage into ingest and retrieval workflows.
5. `orchestrator` routes requests and enforces tool policy.
6. `llm` is isolated behind a small interface so providers can be swapped later.
7. `telemetry` records run IDs, prompt hashes, and tool-call traces.

This keeps retrieval quality measurable and generation optional.
