from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HARNESS_PATH = Path(__file__).resolve().parents[2] / "eval" / "harness.py"


def _load_harness_module():
    spec = importlib.util.spec_from_file_location("eval_harness", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harness_parser_accepts_rewrite_model_argument() -> None:
    harness = _load_harness_module()
    args = harness.build_parser().parse_args(["--rewrite-model", "qwen2.5:3b"])
    assert args.rewrite_model == "qwen2.5:3b"


def test_harness_uses_split_models_for_generation_and_rewrite(tmp_path: Path) -> None:
    harness = _load_harness_module()

    golden_path = tmp_path / "golden.json"
    golden_path.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "results"

    class FakeLLM:
        def __init__(self, model: str) -> None:
            self.model = model

    class FakeFlow:
        last_instance = None

        def __init__(self, settings) -> None:
            self.settings = settings
            self.llm = FakeLLM(settings.generation_model)
            self.rewrite_llm = self.llm
            FakeFlow.last_instance = self

        def generate_answer(self, _query: str):  # pragma: no cover
            raise AssertionError("No pairs should be evaluated in this test")

    harness.GenerateFlow = FakeFlow
    harness.get_llm_client = lambda settings: FakeLLM(settings.generation_model)
    harness.fetch_available_ollama_models = lambda base_url: set()
    harness.validate_eval_llm = lambda flow: None
    harness.print_summary = lambda summary, total: None

    exit_code = harness.main(
        [
            "--golden-path",
            str(golden_path),
            "--output-dir",
            str(output_dir),
            "--model",
            "gemma4:e4b",
            "--rewrite-model",
            "qwen2.5:3b",
        ]
    )

    assert exit_code == 0
    assert FakeFlow.last_instance is not None
    assert FakeFlow.last_instance.llm.model == "gemma4:e4b"
    assert FakeFlow.last_instance.rewrite_llm.model == "qwen2.5:3b"

    run_files = list(output_dir.glob("run_*.json"))
    assert len(run_files) == 1
    payload = json.loads(run_files[0].read_text(encoding="utf-8"))
    assert payload["run_config"]["resolved_model"] == "gemma4:e4b"
    assert payload["run_config"]["resolved_rewrite_model"] == "qwen2.5:3b"
