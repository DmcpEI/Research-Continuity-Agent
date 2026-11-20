"""Ingest a local artifact into the Research Continuity Agent stores."""

from __future__ import annotations

import argparse
import json
import sys

from rca.flows.ingest_flow import EmptyDocumentError, IngestFlow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a local artifact into the RCA stores.")
    parser.add_argument("path", help="Path to a note, PDF, experiment file, or git repository.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = IngestFlow().ingest_path(args.path)
    except EmptyDocumentError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
