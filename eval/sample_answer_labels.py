"""Draw saved harness answers for hand-labelling answer outcomes (#6).

Reads harness run artifacts (run_*.json plus their traces/ directories), drops
answers produced by the pre-generation gates, and samples answers per run with
hedged answers weighted up. Writes a JSON file with empty label fields; see
docs/EVAL.md, "Answer Outcome Labels".

    uv run python eval/sample_answer_labels.py \
        --run qwen3.5:4b/v1=eval/results/bakeoff_20261109/prompt/v1/qwen3_5_4b:35 ...
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# Fixed answers written by GenerateFlow before any LLM call.
GATE_ANSWERS = frozenset(
    {
        "No relevant context found in your knowledge base.",
        "I couldn't find evidence for this in your knowledge base.",
    }
)

# Frozen on purpose: the label files were drawn with this regex, and changing it would change
# which answers a re-run draws. Used only to weight the sample, never to label. The runtime
# rule, rca.flows.facts_check.is_hedged, matches everything this does and a little more.
_HEDGE = re.compile(
    r"\b(context|text|sources?|documents?|papers?)\b[^.]{0,40}?\b"
    r"(does not|doesn't|do not|don't)\s+"
    r"(contain|state|report|mention|provide|specify|include|describe|discuss|answer|detail)"
    r"|\bno (relevant )?information\b|\bcannot (find|answer|be determined)\b|\bnot mentioned\b",
    re.IGNORECASE,
)

LABEL_FIELDS = {
    "label": None,
    "unsupported": None,
    "contradictory": None,
    "missed_in_context": None,
    "unsure": None,
    "note": "",
    "labeled_by": None,
}


def is_hedged(answer: str) -> bool:
    return bool(_HEDGE.search(answer or ""))


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    """Return (run artifact, results) for the single run_*.json in run_dir."""
    files = sorted(run_dir.glob("run_*.json"))
    if len(files) != 1:
        raise ValueError(f"expected one run_*.json in {run_dir}, found {len(files)}")
    artifact = json.loads(files[0].read_text(encoding="utf-8"))
    return artifact, artifact["results"]


def context_node_ids(result: dict, run_dir: Path) -> list[str]:
    """Context node IDs from the case trace next to the run, else the recorded trace_path.

    The copy next to the run wins: trace_path is absolute and may point at a reused
    scratch directory holding another run. The trace directory named after the run
    timestamp is preferred when several exist.
    """
    stamp = run_stamp(run_dir)
    local = sorted(run_dir.glob(f"traces/*/{result['id']}.json"))
    local.sort(key=lambda path: path.parent.name != stamp)
    paths = local + ([Path(result["trace_path"])] if result.get("trace_path") else [])
    for path in paths:
        if path.exists():
            trace = json.loads(path.read_text(encoding="utf-8"))
            return list(trace.get("context_node_ids") or [])
    return []


def run_stamp(run_dir: Path) -> str:
    """Timestamp of the run artifact (run_<stamp>.json), or "" if there is none."""
    files = sorted(run_dir.glob("run_*.json"))
    return files[0].stem.removeprefix("run_") if files else ""


def candidates(results: list[dict]) -> list[dict]:
    """Answers the post-generation classifier actually decides on."""
    kept = []
    for result in results:
        answer = result.get("answer")
        if isinstance(answer, str) and answer.strip() and answer.strip() not in GATE_ANSWERS:
            kept.append(result)
    return kept


def weighted_sample(
    items: list[dict], k: int, rng: random.Random, hedge_weight: float = 3.0
) -> list[dict]:
    """Sample k items without replacement; hedged answers are hedge_weight times likelier."""
    pool = list(items)
    chosen: list[dict] = []
    while pool and len(chosen) < k:
        weights = [hedge_weight if is_hedged(r.get("answer", "")) else 1.0 for r in pool]
        index = rng.choices(range(len(pool)), weights=weights, k=1)[0]
        chosen.append(pool.pop(index))
    return chosen


def cited_ids(result: dict) -> list[str]:
    """Harness artifacts store citations as ID strings; tolerate {"source_id": ...} too."""
    ids = []
    for citation in result.get("citations") or []:
        value = citation.get("source_id") if isinstance(citation, dict) else citation
        if isinstance(value, str) and value:
            ids.append(value)
    return ids


def to_item(run_name: str, model: str, result: dict, run_dir: Path) -> dict:
    return {
        "item_id": f"{run_name}:{result['id']}",
        "run": run_name,
        "model": model,
        "question_id": result["id"],
        "question": result["question"],
        "answerable": result["answerable"],
        "negative_type": result.get("negative_type"),
        "expected_keywords": result.get("expected_keywords") or [],
        "answer": result["answer"],
        "cited_ids": cited_ids(result),
        "context_node_ids": context_node_ids(result, run_dir),
        "pipeline": {
            "abstained": result.get("abstained"),
            "grounded": result.get("grounded"),
            "failure_labels": result.get("failure_labels") or [],
        },
        "hedged": is_hedged(result["answer"]),
        **LABEL_FIELDS,
    }


def parse_run_spec(spec: str) -> tuple[str, Path, int]:
    """'name=path:count' -> (name, path, count)."""
    name, _, rest = spec.partition("=")
    path, _, count = rest.rpartition(":")
    if not name or not path or not count.isdigit():
        raise ValueError(f"bad --run {spec!r}; expected name=path:count")
    return name, Path(path), int(count)


def build_sample(specs: list[str], seed: int) -> list[dict]:
    rng = random.Random(seed)
    items: list[dict] = []
    for spec in specs:
        name, path, count = parse_run_spec(spec)
        artifact, results = load_run(path)
        model = artifact.get("run_config", {}).get("resolved_model", "")
        for result in weighted_sample(candidates(results), count, rng):
            items.append(to_item(name, model, result, path))
    rng.shuffle(items)
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw saved answers for outcome labelling.")
    parser.add_argument("--run", action="append", required=True, help="name=run_dir:count")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--output", type=Path, default=Path("eval/answer_labels.json"))
    args = parser.parse_args(argv)

    items = build_sample(args.run, args.seed)
    payload = {"seed": args.seed, "runs": args.run, "count": len(items), "items": items}
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(items)} items to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
