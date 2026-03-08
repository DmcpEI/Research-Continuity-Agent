"""Query the local RCA index."""

from __future__ import annotations

import argparse
import json

from rca.flows.retrieve_flow import RetrieveFlow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the local RCA retrieval layer.")
    parser.add_argument("question", help="Question or retrieval query.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of hits to return.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = RetrieveFlow().retrieve(args.question, limit=args.limit)
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
