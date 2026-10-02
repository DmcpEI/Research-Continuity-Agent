"""Calibrate a retrieval-feature abstention rule on the dev split, report on test.

Input: a retrieval_features_*.json artifact from collect_retrieval_features.py.
A rule abstains when one feature crosses a threshold. Rules are chosen on dev
to maximize negative recall while keeping answerable false abstentions within a
budget, then evaluated once on test.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rca.telemetry.provenance import git_state

# (feature, direction): "low" abstains when value <= t, "high" when value >= t
FEATURES = [
    ("max_score", "low"),
    ("mean_score", "low"),
    ("rerank_max", "low"),
    ("rerank_mean", "low"),
    ("rerank_gap", "low"),
    ("hits_above_055", "low"),
    ("context_hits", "low"),
    ("top_source_share", "low"),
    ("unique_sources", "high"),
]


def value(row: dict, feature: str) -> float:
    v = row.get(feature)
    return float("-inf") if v is None else float(v)


def auroc(rows: list[dict], feature: str, direction: str) -> float:
    """P(negative looks more abstain-worthy than answerable); ties count half."""
    neg = [value(r, feature) for r in rows if not r["answerable"]]
    pos = [value(r, feature) for r in rows if r["answerable"]]
    sign = -1 if direction == "low" else 1
    wins = sum((sign * n > sign * p) + 0.5 * (n == p) for n in neg for p in pos)
    return wins / (len(neg) * len(pos))


def single_rule(feature: str, direction: str, t: float):
    if direction == "low":
        return lambda r: value(r, feature) <= t
    return lambda r: value(r, feature) >= t


def score(rows: list[dict], rule) -> dict:
    neg = [r for r in rows if not r["answerable"]]
    pos = [r for r in rows if r["answerable"]]
    neg_hit = sum(rule(r) for r in neg)
    pos_hit = sum(rule(r) for r in pos)
    return {
        "negative_recall": neg_hit / len(neg),
        "negatives_abstained": f"{neg_hit}/{len(neg)}",
        "answerable_false_abstention": pos_hit / len(pos),
        "answerables_abstained": f"{pos_hit}/{len(pos)}",
    }


def candidates(rows: list[dict]):
    # Single-feature thresholds only: pairwise rules overfit the small dev negative set.
    for feature, direction in FEATURES:
        for t in sorted({value(r, feature) for r in rows}):
            op = "<=" if direction == "low" else ">="
            yield f"{feature} {op} {t:.4g}", single_rule(feature, direction, t)


def pick(rows: list[dict], budget: float):
    best = None
    for name, rule in candidates(rows):
        s = score(rows, rule)
        if s["answerable_false_abstention"] > budget:
            continue
        key = (s["negative_recall"], -s["answerable_false_abstention"], -len(name))
        if best is None or key > best[0]:
            best = (key, name, rule, s)
    return best


def by_type(rows: list[dict], rule) -> dict:
    groups = defaultdict(list)
    for r in rows:
        if not r["answerable"]:
            groups[r.get("negative_type") or "untyped"].append(rule(r))
    return {k: f"{sum(v)}/{len(v)}" for k, v in sorted(groups.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_path")
    parser.add_argument("--splits-dir", default="eval/splits")
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--output", default="eval/results/abstention_calibration.json")
    args = parser.parse_args(argv)

    features = json.loads(Path(args.features_path).read_text())
    rows = [r for r in features["rows"] if "skipped" not in r]
    split_ids = {
        name: set(json.loads((Path(args.splits_dir) / f"{name}.json").read_text())["ids"])
        for name in ("dev", "test")
    }
    dev = [r for r in rows if r["id"] in split_ids["dev"]]
    test = [r for r in rows if r["id"] in split_ids["test"]]

    separation = {f: round(auroc(dev, f, d), 3) for f, d in FEATURES}
    print("Dev AUROC per feature (1.0 = perfectly separates negatives):")
    for f, a in sorted(separation.items(), key=lambda kv: -kv[1]):
        print(f"  {f:18} {a}")

    baseline = single_rule("max_score", "low", 0.4999)  # current gate: max < 0.50
    picked = pick(dev, args.budget)
    if picked is None:
        print(f"No rule keeps answerable false abstention within {args.budget:.0%} on dev.")
        return 1
    _, name, rule, dev_score = picked
    report = {
        "features_path": args.features_path,
        # Provenance of the retrieval run the features came from (None if older artifact).
        "features_provenance": features.get("provenance"),
        "calibration_code": git_state(),
        "budget": args.budget,
        "dev_n": len(dev),
        "test_n": len(test),
        "dev_auroc": separation,
        "baseline_rule": "max_score < 0.50 (retrieval gate only)",
        "baseline_dev": score(dev, baseline),
        "baseline_test": score(test, baseline),
        "baseline_test_by_type": by_type(test, baseline),
        "selected_rule": name,
        "selected_dev": dev_score,
        "selected_test": score(test, rule),
        "selected_test_by_type": by_type(test, rule),
        "selected_all_by_type": by_type(rows, rule),
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "dev_auroc"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
