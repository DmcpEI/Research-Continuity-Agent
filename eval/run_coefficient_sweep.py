"""Coefficient sweep for lexical reranker weights with held-out validation."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from rca.config.settings import get_settings
from rca.flows.retrieve_flow import TITLE_WORD_PATTERN, RetrieveFlow
from rca.store.graph_store import GraphStore
from rca.store.vector_store import VectorStore

TITLE_WEIGHTS = [0.08, 0.12, 0.16, 0.20, 0.24]
TEXT_WEIGHTS = [0.01, 0.02, 0.03, 0.04, 0.05]
BASE_SCORE = 0.45
SEED = 42
TEST_SIZE = 20
FETCH_K = 10
BASELINE_WEIGHTS = (0.12, 0.04)


def resolve_to_source(node_id: str) -> str:
    if node_id.startswith("src:"):
        return node_id
    base = node_id.rsplit(":", 1)[0]
    return "src:" + base.split(":", 1)[1]


def expected_source_ids(pair: dict[str, Any]) -> list[str]:
    sources = [source for source in pair.get("expected_sources", []) if source]
    if sources:
        return sources
    source = pair.get("expected_source")
    return [source] if source else []


def is_retrieval_case(pair: dict[str, Any]) -> bool:
    return pair.get("answerable", True) and bool(expected_source_ids(pair))


def hit_at_k(hits: list[Any], expected_sources: list[str], k: int = 5) -> bool:
    source_ids = {resolve_to_source(hit.node_id) for hit in hits[:k]}
    return all(source in source_ids for source in expected_sources)


def load_golden_pairs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload.get("pairs", [])
    return payload


def stratified_split_ids(pairs: list[dict[str, Any]], test_size: int, seed: int) -> tuple[list[str], list[str]]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        by_category[pair["category"]].append(pair["id"])

    rng = random.Random(seed)
    for ids in by_category.values():
        rng.shuffle(ids)

    category_counts = {category: len(ids) for category, ids in by_category.items()}
    quotas = {category: count * test_size / len(pairs) for category, count in category_counts.items()}
    test_counts = {category: math.floor(quota) for category, quota in quotas.items()}

    remaining = test_size - sum(test_counts.values())
    order = sorted(
        by_category.keys(),
        key=lambda category: (quotas[category] - test_counts[category], category_counts[category], category),
        reverse=True,
    )
    for category in order:
        if remaining <= 0:
            break
        if test_counts[category] >= category_counts[category]:
            continue
        test_counts[category] += 1
        remaining -= 1

    test_ids: list[str] = []
    dev_ids: list[str] = []
    for category in sorted(by_category):
        ids = by_category[category]
        n_test = test_counts[category]
        test_ids.extend(ids[:n_test])
        dev_ids.extend(ids[n_test:])

    assert len(dev_ids) + len(test_ids) == len(pairs)
    assert len(test_ids) == test_size
    assert not (set(dev_ids) & set(test_ids))
    return dev_ids, test_ids


def write_split(path: Path, ids: list[str], seed: int, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "seed": seed,
        "count": len(ids),
        "ids": ids,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class WeightedRetrieveFlow(RetrieveFlow):
    def __init__(
        self,
        *,
        title_weight: float,
        text_weight: float,
        base_score: float = BASE_SCORE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.title_weight = title_weight
        self.text_weight = text_weight
        self.base_score = base_score

    def _lexical_score(self, query_tokens: set[str], title: str, text: str) -> float:
        if not query_tokens:
            return 0.5

        title_word_tokens = set(TITLE_WORD_PATTERN.findall(title.lower()))
        text_word_tokens = set(TITLE_WORD_PATTERN.findall(text.lower()))
        title_overlap = sum(1 for token in query_tokens if token in title_word_tokens)
        text_overlap = sum(1 for token in query_tokens if token in text_word_tokens)

        if title_overlap == 0 and text_overlap == 0:
            return 0.0

        return min(0.95, self.base_score + (self.title_weight * title_overlap) + (self.text_weight * text_overlap))


def evaluate_subset(
    pairs_by_id: dict[str, dict[str, Any]],
    ids: list[str],
    graph_store: GraphStore,
    vector_store: VectorStore,
    title_weight: float,
    text_weight: float,
) -> dict[str, Any]:
    flow = WeightedRetrieveFlow(
        title_weight=title_weight,
        text_weight=text_weight,
        settings=get_settings(),
        graph_store=graph_store,
        vector_store=vector_store,
    )

    retrieval_cases = 0
    hits5 = 0
    hits10 = 0
    per_case: list[dict[str, Any]] = []

    for pair_id in ids:
        pair = pairs_by_id[pair_id]
        expected = expected_source_ids(pair)
        answerable = is_retrieval_case(pair)
        if answerable:
            retrieval_cases += 1
            bundle = flow.retrieve(pair["question"], limit=FETCH_K)
            case_hit5 = hit_at_k(bundle.hits, expected, k=5)
            case_hit10 = hit_at_k(bundle.hits, expected, k=10)
            hits5 += int(case_hit5)
            hits10 += int(case_hit10)
        else:
            case_hit5 = False
            case_hit10 = False

        per_case.append(
            {
                "id": pair_id,
                "category": pair["category"],
                "difficulty": pair["difficulty"],
                "answerable": pair.get("answerable", True),
                "hit_at_5": case_hit5,
                "hit_at_10": case_hit10,
            }
        )

    return {
        "questions": len(ids),
        "retrieval_cases": retrieval_cases,
        "hit_at_5": round(hits5 / retrieval_cases, 4) if retrieval_cases else 0.0,
        "hit_at_10": round(hits10 / retrieval_cases, 4) if retrieval_cases else 0.0,
        "hits_at_5_count": hits5,
        "hits_at_10_count": hits10,
        "per_case": per_case,
    }


def format_grid_table(results: dict[tuple[float, float], dict[str, Any]], best: tuple[float, float]) -> str:
    lines = []
    header = ["title\\text", *[f"{value:.2f}" for value in TEXT_WEIGHTS]]
    lines.append(" | ".join(header))
    lines.append(" | ".join(["---"] * len(header)))
    for title_weight in TITLE_WEIGHTS:
        row = [f"{title_weight:.2f}"]
        for text_weight in TEXT_WEIGHTS:
            metrics = results[(title_weight, text_weight)]
            marker = ""
            if (title_weight, text_weight) == BASELINE_WEIGHTS:
                marker += " baseline"
            if (title_weight, text_weight) == best:
                marker += " selected"
            cell = f"{metrics['hit_at_5']:.1%}{marker}"
            row.append(cell)
        lines.append(" | ".join(row))
    return "\n".join(lines)


def select_frontier(results: dict[tuple[float, float], dict[str, Any]]) -> list[tuple[float, float]]:
    best_hit5 = max(metrics["hit_at_5"] for metrics in results.values())
    best_hit10 = max(metrics["hit_at_10"] for metrics in results.values() if metrics["hit_at_5"] == best_hit5)
    return [
        key
        for key, metrics in results.items()
        if metrics["hit_at_5"] == best_hit5 and metrics["hit_at_10"] == best_hit10
    ]


def main() -> None:
    golden_path = Path("eval/golden.json")
    splits_dir = Path("eval/splits")
    results_path = Path("eval/results/coefficient_sweep.json")

    pairs = load_golden_pairs(golden_path)
    pairs_by_id = {pair["id"]: pair for pair in pairs}

    dev_ids, test_ids = stratified_split_ids(pairs, test_size=TEST_SIZE, seed=SEED)
    write_split(splits_dir / "dev.json", dev_ids, seed=SEED, split="dev")
    write_split(splits_dir / "test.json", test_ids, seed=SEED, split="test")

    settings = get_settings()
    graph_store = GraphStore(settings.graph_db_path)
    vector_store = VectorStore(settings.vector_dir, settings.default_collection)

    sweep_results: dict[tuple[float, float], dict[str, Any]] = {}

    print(f"Loaded {len(pairs)} golden questions")
    print(f"Dev split: {len(dev_ids)} questions | Test split: {len(test_ids)} questions")

    for title_weight in TITLE_WEIGHTS:
        for text_weight in TEXT_WEIGHTS:
            metrics = evaluate_subset(
                pairs_by_id,
                dev_ids,
                graph_store,
                vector_store,
                title_weight=title_weight,
                text_weight=text_weight,
            )
            sweep_results[(title_weight, text_weight)] = metrics

    frontier = select_frontier(sweep_results)
    baseline_key = BASELINE_WEIGHTS
    current_dev = sweep_results[baseline_key]
    current_test = evaluate_subset(
        pairs_by_id,
        test_ids,
        graph_store,
        vector_store,
        title_weight=baseline_key[0],
        text_weight=baseline_key[1],
    )

    frontier_test: dict[tuple[float, float], dict[str, Any]] = {}
    for key in frontier:
        frontier_test[key] = evaluate_subset(
            pairs_by_id,
            test_ids,
            graph_store,
            vector_store,
            title_weight=key[0],
            text_weight=key[1],
        )

    selected_key = max(
        frontier,
        key=lambda key: (
            frontier_test[key]["hit_at_5"],
            frontier_test[key]["hit_at_10"],
            -(abs(key[0] - baseline_key[0]) + abs(key[1] - baseline_key[1])),
        ),
    )
    selected_dev = sweep_results[selected_key]
    selected_test = frontier_test[selected_key]

    improvement = round(selected_test["hit_at_5"] - current_test["hit_at_5"], 4)
    should_update = improvement > 0.02

    grid_table = format_grid_table(sweep_results, selected_key)
    print()
    print("Dev hit@5 grid (config 3, answerable cases only)")
    print(grid_table)
    print()
    print(
        "Dev-optimal frontier: "
        + ", ".join(f"({title:.2f}, {text:.2f})" for title, text in frontier)
    )
    print(
        f"Baseline production: title={baseline_key[0]:.2f}, text={baseline_key[1]:.2f} | "
        f"dev hit@5={current_dev['hit_at_5']:.1%}, dev hit@10={current_dev['hit_at_10']:.1%}"
    )
    print(
        f"Selected weights after held-out validation: title={selected_key[0]:.2f}, text={selected_key[1]:.2f} | "
        f"dev hit@5={selected_dev['hit_at_5']:.1%}, dev hit@10={selected_dev['hit_at_10']:.1%}"
    )
    print(
        f"Held-out test (selected): hit@5={selected_test['hit_at_5']:.1%}, hit@10={selected_test['hit_at_10']:.1%}"
    )
    print(
        f"Held-out test (baseline): hit@5={current_test['hit_at_5']:.1%}, hit@10={current_test['hit_at_10']:.1%}"
    )
    print(
        "Decision: "
        + (
            f"update production weights (held-out gain {improvement:.1%} > 2.0%)"
            if should_update
            else f"keep current weights (held-out gain {improvement:.1%} <= 2.0%)"
        )
    )

    payload = {
        "seed": SEED,
        "split_strategy": "stratified_by_category_largest_remainder",
        "base_score": BASE_SCORE,
        "title_weights": TITLE_WEIGHTS,
        "text_weights": TEXT_WEIGHTS,
        "baseline_weights": {"title_weight": baseline_key[0], "text_weight": baseline_key[1]},
        "dev_frontier": [
            {"title_weight": title_weight, "text_weight": text_weight}
            for title_weight, text_weight in frontier
        ],
        "selected_weights": {"title_weight": selected_key[0], "text_weight": selected_key[1]},
        "dev_split": {"questions": len(dev_ids), "retrieval_cases": current_dev["retrieval_cases"]},
        "test_split": {"questions": len(test_ids), "retrieval_cases": current_test["retrieval_cases"]},
        "dev_grid": {
            f"title={title_weight:.2f}|text={text_weight:.2f}": metrics
            for (title_weight, text_weight), metrics in sweep_results.items()
        },
        "grid_table_markdown": grid_table,
        "dev_selected": selected_dev,
        "dev_baseline": current_dev,
        "test_selected": selected_test,
        "test_baseline": current_test,
        "held_out_hit_at_5_delta": improvement,
        "update_production": should_update,
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    main()
