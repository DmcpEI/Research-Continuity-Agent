from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HARNESS_PATH = Path(__file__).resolve().parents[2] / "eval" / "harness.py"


def _load_harness_module():
    spec = importlib.util.spec_from_file_location("eval_harness", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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


def test_classify_failure_stage_attributes_first_lost_stage() -> None:
    harness = _load_harness_module()
    pair = harness.GoldenPair(id="q1", question="q", difficulty="easy", expected_source="src:pdf/a")
    negative = harness.GoldenPair(id="n1", question="q", difficulty="easy", answerable=False)

    def stage(p, abstained=False, correct=False, retrieved=(), context=()):
        return harness.classify_failure_stage(
            p,
            abstained=abstained,
            source_correct=correct,
            retrieved_sources=list(retrieved),
            context_sources=list(context),
        )

    assert stage(pair, correct=True, retrieved=["src:pdf/a"], context=["src:pdf/a"]) == "ok"
    assert stage(pair, retrieved=["src:pdf/b"]) == "retrieval_miss"
    assert stage(pair, retrieved=["src:pdf/a"], context=["src:pdf/b"]) == "context_miss"
    assert (
        stage(pair, abstained=True, retrieved=["src:pdf/a"], context=["src:pdf/a"])
        == "false_abstention"
    )
    assert stage(pair, retrieved=["src:pdf/a"], context=["src:pdf/a"]) == "citation_miss"
    assert stage(negative, abstained=True) == "ok"
    assert stage(negative) == "false_answer"


def test_trace_sources_resolves_chunks_and_dedupes() -> None:
    harness = _load_harness_module()
    from rca.contracts.trace import HitProvenance, QueryTrace

    trace = QueryTrace(
        query="q",
        provenance=[
            HitProvenance(node_id="chk:pdf/a:0001", score=0.9, stage="vector", rank=1),
            HitProvenance(node_id="src:pdf/a", score=0.8, stage="expansion", rank=2),
            HitProvenance(node_id="chk:pdf/b:0003", score=0.7, stage="lexical", rank=3),
        ],
        context_node_ids=["chk:pdf/a:0001"],
    )

    assert harness.trace_sources(trace) == (["src:pdf/a", "src:pdf/b"], ["src:pdf/a"])
    assert harness.trace_sources(None) == ([], [])
