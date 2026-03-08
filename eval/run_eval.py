"""Run a simple retrieval evaluation against a golden set."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rca.flows.retrieve_flow import RetrieveFlow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation against eval/golden_set.json.")
    parser.add_argument(
        "--golden-set",
        default="eval/golden_set.json",
        help="Path to a JSON file containing retrieval evaluation cases.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of retrieval hits per query.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    golden_set_path = Path(args.golden_set)
    cases = json.loads(golden_set_path.read_text(encoding="utf-8"))
    retrieve_flow = RetrieveFlow()

    results = []
    matches = 0

    for case in cases:
        bundle = retrieve_flow.retrieve(case["query"], limit=args.limit)
        actual_source_ids = sorted(
            {
                hit.metadata.get("source_id", hit.node_id)
                for hit in bundle.hits
            }
        )
        expected_source_ids = sorted(case.get("expected_source_ids", []))
        matched = any(source_id in actual_source_ids for source_id in expected_source_ids)
        matches += int(matched)
        results.append(
            {
                "query": case["query"],
                "expected_source_ids": expected_source_ids,
                "actual_source_ids": actual_source_ids,
                "matched": matched,
            }
        )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cases": len(cases),
        "matches": matches,
        "hit_rate": matches / len(cases) if cases else 0.0,
        "results": results,
    }

    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
