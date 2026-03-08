# Evaluation

Evaluation is retrieval-first.

`eval/golden_set.json` stores representative queries and the source IDs that should be retrieved.

`eval/run_eval.py` executes the current retrieval flow against that set and writes a timestamped report into `eval/results/`.

Suggested metrics:

- hit rate at `k`
- whether expected source IDs appear in the top results
- citation coverage once answer generation exists
