"""Run generation harness across multiple models and aggregate results.

Example:
    uv run python eval/run_model_matrix.py \
      --models gemma4:e4b qwen2.5:14b llama3.1:8b \
      --pull-missing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run eval/harness.py across a list of models and summarize outcomes."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to evaluate (Ollama naming expected for --pull-missing).",
    )
    parser.add_argument(
        "--pull-missing",
        action="store_true",
        help="Call 'ollama pull <model>' before each run.",
    )
    parser.add_argument(
        "--golden-path",
        default=str(Path("eval") / "golden.json"),
        help="Path to golden evaluation JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("eval") / "results"),
        help="Directory where harness artifacts are written.",
    )
    parser.add_argument(
        "--matrix-output",
        default=str(Path("eval") / "results" / "model_matrix.json"),
        help="Path for the aggregate matrix artifact.",
    )
    parser.add_argument(
        "--rewrite-model",
        default=None,
        help="Optional rewrite model to use during normal chat model runs.",
    )
    parser.add_argument(
        "--rewrite-models",
        nargs="+",
        default=None,
        help="Optional query-rewrite models to sweep against a fixed chat model.",
    )
    parser.add_argument(
        "--rewrite-chat-model",
        default="gemma4:e4b",
        help="Chat model used during rewrite-model sweep.",
    )
    parser.add_argument(
        "--rewrite-matrix-output",
        default=str(Path("eval") / "results" / "rewrite_matrix.json"),
        help="Path for the rewrite sweep aggregate artifact.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Optional per-model timeout in seconds (0 disables timeout).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining models if one fails or times out.",
    )
    return parser


def run_command(
    command: list[str],
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        check=False,
        timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )


def maybe_pull_model(model: str) -> None:
    print(f"[pull] {model}")
    pull = run_command(["ollama", "pull", model])
    if pull.returncode != 0:
        raise RuntimeError(
            f"ollama pull failed for {model}: {pull.stderr.strip() or pull.stdout.strip()}"
        )


def load_harness_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_run_files(output_dir: Path) -> set[Path]:
    return set(output_dir.glob("run_*.json"))


def latest_new_run_file(before: set[Path], after: set[Path]) -> Path:
    new_files = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if not new_files:
        raise ValueError("Harness completed but no new run artifact was found.")
    return new_files[-1]


def model_row(artifact: dict[str, Any]) -> dict[str, Any]:
    summary = artifact.get("summary", {})
    return {
        "model": artifact.get("model"),
        "rewrite_model": artifact.get("rewrite_model"),
        "run_file": str(artifact.get("_run_file")),
        "citation_precision": summary.get("citation_precision", 0.0),
        "citation_precision_cases": summary.get("citation_precision_cases", 0),
        "abstention_recall": summary.get("abstention_recall", 0.0),
        "abstention_cases": summary.get("abstention_cases", 0),
        "answerable_abstentions": summary.get("answerable_abstentions", 0),
        "average_keyword_hit_rate": summary.get("average_keyword_hit_rate", 0.0),
        "average_latency_ms": summary.get("average_latency_ms", 0.0),
        "overall_grounded_rate": summary.get("overall_grounded_rate", 0.0),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No runs completed.")
        return

    print("\nModel matrix summary")
    print(
        "{:<22} {:>8} {:>8} {:>8} {:>10} {:>9}".format(
            "model",
            "cit_prec",
            "abst_rec",
            "ans_abs",
            "kw_hit",
            "lat_s",
        )
    )
    print("-" * 74)
    for row in rows:
        print(
            "{:<22} {:>8.1%} {:>8.1%} {:>8} {:>10.3f} {:>9.1f}".format(
                str(row.get("model", "--"))[:22],
                float(row["citation_precision"]),
                float(row["abstention_recall"]),
                int(row["answerable_abstentions"]),
                float(row["average_keyword_hit_rate"]),
                float(row["average_latency_ms"]) / 1000.0,
            )
        )


def print_rewrite_table(rows: list[dict[str, Any]], chat_model: str) -> None:
    if not rows:
        print("No rewrite sweep runs completed.")
        return

    print(f"\nRewrite model sweep summary (chat model: {chat_model})")
    print("{:<22} {:>10} {:>8} {:>9}".format("rewrite_model", "kw_hit", "cit_pr", "lat_s"))
    print("-" * 58)
    for row in rows:
        print(
            "{:<22} {:>10.3f} {:>8.1%} {:>9.1f}".format(
                str(row.get("rewrite_model", "--"))[:22],
                float(row["average_keyword_hit_rate"]),
                float(row["citation_precision"]),
                float(row["average_latency_ms"]) / 1000.0,
            )
        )


def sort_chat_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["citation_precision"]),
            -float(row["abstention_recall"]),
            -float(row["average_keyword_hit_rate"]),
            float(row["average_latency_ms"]),
        ),
    )


def sort_rewrite_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["average_keyword_hit_rate"]),
            -float(row["citation_precision"]),
            float(row["average_latency_ms"]),
        ),
    )


def write_matrix(
    matrix_output: Path,
    models_requested: list[str],
    run_files: list[str],
    rows: list[dict[str, Any]],
) -> None:
    payload = {
        "models_requested": models_requested,
        "runs": run_files,
        "rows": sort_chat_rows(rows),
        "updated_at": int(time.time()),
    }
    matrix_output.parent.mkdir(parents=True, exist_ok=True)
    matrix_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_rewrite_matrix(
    rewrite_output: Path,
    rewrite_models_requested: list[str],
    chat_model: str,
    rewrite_run_files: list[str],
    rewrite_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "rewrite_models_requested": rewrite_models_requested,
        "chat_model": chat_model,
        "runs": rewrite_run_files,
        "rows": sort_rewrite_rows(rewrite_rows),
        "updated_at": int(time.time()),
    }
    rewrite_output.parent.mkdir(parents=True, exist_ok=True)
    rewrite_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.models and not args.rewrite_models:
        parser.error("at least one of --models or --rewrite-models is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, Any]] = []
    run_files: list[str] = []
    matrix_output = Path(args.matrix_output)

    for model in args.models or []:
        print(f"\n[run] model={model}")
        if args.pull_missing:
            maybe_pull_model(model)

        cmd = [
            "uv",
            "run",
            "python",
            "eval/harness.py",
            "--golden-path",
            args.golden_path,
            "--output-dir",
            args.output_dir,
            "--model",
            model,
        ]
        if args.rewrite_model:
            cmd.extend(["--rewrite-model", args.rewrite_model])
        before_files = list_run_files(output_dir)
        try:
            run = run_command(cmd, timeout_seconds=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            message = f"Harness timed out for model '{model}' after {args.timeout_seconds:.1f}s"
            if args.continue_on_error:
                print(f"[warn] {message}")
                continue
            raise RuntimeError(message) from None

        if run.returncode != 0:
            message = f"Harness failed for model '{model}' (exit={run.returncode})"
            if args.continue_on_error:
                print(f"[warn] {message}")
                continue
            raise RuntimeError(message)

        after_files = list_run_files(output_dir)
        output_path = latest_new_run_file(before_files, after_files)
        artifact = load_harness_artifact(output_path)
        artifact["_run_file"] = str(output_path)
        run_files.append(str(output_path))
        matrix_rows.append(model_row(artifact))

        # Persist after each completed model so interrupted runs keep prior progress.
        write_matrix(matrix_output, args.models or [], run_files, matrix_rows)

    if args.models:
        print_table(sort_chat_rows(matrix_rows))
        print(f"\nSaved matrix artifact to {matrix_output}")

    if args.rewrite_models:
        rewrite_rows: list[dict[str, Any]] = []
        rewrite_run_files: list[str] = []
        rewrite_output = Path(args.rewrite_matrix_output)

        if args.pull_missing:
            maybe_pull_model(args.rewrite_chat_model)

        for rewrite_model in args.rewrite_models:
            print(
                f"\n[rewrite-run] chat_model={args.rewrite_chat_model} "
                f"rewrite_model={rewrite_model}"
            )
            if args.pull_missing:
                maybe_pull_model(rewrite_model)

            cmd = [
                "uv",
                "run",
                "python",
                "eval/harness.py",
                "--golden-path",
                args.golden_path,
                "--output-dir",
                args.output_dir,
                "--model",
                args.rewrite_chat_model,
                "--rewrite-model",
                rewrite_model,
            ]
            before_files = list_run_files(output_dir)
            try:
                run = run_command(cmd, timeout_seconds=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                message = (
                    f"Harness timed out for rewrite model '{rewrite_model}' "
                    f"after {args.timeout_seconds:.1f}s"
                )
                if args.continue_on_error:
                    print(f"[warn] {message}")
                    continue
                raise RuntimeError(message) from None

            if run.returncode != 0:
                message = (
                    f"Harness failed for rewrite model '{rewrite_model}' (exit={run.returncode})"
                )
                if args.continue_on_error:
                    print(f"[warn] {message}")
                    continue
                raise RuntimeError(message)

            after_files = list_run_files(output_dir)
            output_path = latest_new_run_file(before_files, after_files)
            artifact = load_harness_artifact(output_path)
            artifact["_run_file"] = str(output_path)
            rewrite_run_files.append(str(output_path))
            rewrite_rows.append(model_row(artifact))

            # Persist after each completed rewrite run for resumability.
            write_rewrite_matrix(
                rewrite_output,
                args.rewrite_models,
                args.rewrite_chat_model,
                rewrite_run_files,
                rewrite_rows,
            )

        print_rewrite_table(sort_rewrite_rows(rewrite_rows), args.rewrite_chat_model)
        print(f"\nSaved rewrite matrix artifact to {rewrite_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
