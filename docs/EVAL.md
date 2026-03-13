# Evaluation

There are now two evaluation layers:

- `eval/run_eval.py` is retrieval-first and checks whether expected source IDs are surfaced.
- `eval/harness.py` is generation-first and scores groundedness, citation correctness, keyword coverage, and latency.

`eval/golden.json` is the default generation harness input. It stores 30 golden Q&A pairs with `id`, `question`, `expected_keywords`, `expected_source`, and `difficulty`.

Run the generation harness with:

```bash
uv run python eval/harness.py
```

The harness writes full per-question results to `eval/results/run_<timestamp>.json` and prints:

- overall grounded rate
- citation precision
- average keyword hit rate
- average latency
- difficulty breakdown
- top 5 failures by keyword hit rate
