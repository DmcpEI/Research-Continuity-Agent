# Data Model

The repo uses two primary identifier families:

- `SourceID`: stable IDs for top-level artifacts such as notes, papers, and repos.
- `ChunkID`: stable IDs for derived text chunks attached to a source.

Graph storage persists:

- `Node`: `id`, `kind`, `title`, optional `text`, and arbitrary metadata.
- `Edge`: directed relationship between nodes with a typed edge kind.

Artifacts are DTOs that project the graph into application-facing records for papers, notes, and experiments.
