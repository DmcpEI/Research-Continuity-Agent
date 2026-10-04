# Constrained Decoding for Structured Output

Language models used as robot planners or tool callers must emit machine-readable output, usually JSON that follows a schema. Free sampling produces invalid JSON or missing fields at a noticeable rate.

Constrained decoding masks the token distribution at each step so that only tokens consistent with a grammar can be sampled. A JSON Schema is compiled into a grammar or finite-state machine, which guarantees syntactically valid output that matches the schema.

Benchmarks of structured output measure coverage (which schema features an engine supports), compliance with the schema, efficiency (decoding overhead), and whether constraining the output changes answer quality. Engines differ widely in how much of JSON Schema they support.
