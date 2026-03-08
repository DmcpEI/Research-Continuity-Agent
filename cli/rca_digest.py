"""Placeholder digest entrypoint."""

from __future__ import annotations

import argparse

from rca.flows.generate_flow import GenerateFlow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a weekly digest once generation is enabled.")
    parser.add_argument("--week", action="store_true", help="Request the current weekly digest.")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    try:
        print(GenerateFlow().generate_answer("weekly digest"))
    except NotImplementedError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
