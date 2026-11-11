from __future__ import annotations

import hashlib
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


def test_harness_records_provenance_and_stamps_corpus_version_on_traces(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from rca.config.settings import Settings
    from rca.contracts.nodes import Node, NodeKind
    from rca.contracts.trace import QueryTrace
    from rca.flows.generate_flow import GeneratedAnswer
    from rca.store.graph_store import GraphStore

    harness = _load_harness_module()
    graph = GraphStore(tmp_path / "graph.sqlite3")
    graph.upsert_node(Node(id="src:pdf/a", kind=NodeKind.paper, title="A", text="abstract"))
    vectors = SimpleNamespace(
        backend="chroma", backend_warning=None, _collection=None, _documents={}
    )

    class FakeFlow:
        # harness helpers resolve chunk IDs through the GenerateFlow name this test replaces
        _resolve_source_id = harness.GenerateFlow._resolve_source_id

        def __init__(self, settings) -> None:
            self.settings = settings
            self.llm = SimpleNamespace(model="fake-gen")
            self.rewrite_llm = self.llm
            self.retrieve_flow = SimpleNamespace(graph_store=graph, vector_store=vectors)

        def generate_answer(self, query: str) -> GeneratedAnswer:
            return GeneratedAnswer(
                query=query, answer="answer", outcome="unverified", trace=QueryTrace(query=query)
            )

    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps([{"id": "q1", "question": "What is A?", "difficulty": "easy"}]),
        encoding="utf-8",
    )
    harness.GenerateFlow = FakeFlow
    harness.validate_eval_llm = lambda flow: None
    harness.fetch_available_ollama_models = lambda base_url: set()
    harness.build_settings = lambda model: Settings(llm_backend="openai_compatible")
    harness.print_summary = lambda summary, total: None

    assert harness.main(["--golden-path", str(golden), "--output-dir", str(tmp_path / "out")]) == 0

    payload = json.loads(next((tmp_path / "out").glob("run_*.json")).read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    version = provenance["corpus"]["corpus_version"]
    assert version and provenance["corpus"]["sources"] == 1
    assert provenance["index_degraded_during_run"] is False
    assert provenance["code"]["commit"]
    assert payload["results"][0]["error"] is None, payload["results"][0]["error"]
    trace = json.loads(Path(payload["results"][0]["trace_path"]).read_text(encoding="utf-8"))
    assert trace["corpus_version"] == version


def _write_split_fixture(tmp_path: Path) -> tuple[Path, Path]:
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            [
                {"id": f"q{i}", "question": f"Question {i}?", "difficulty": "easy"}
                for i in range(1, 5)
            ]
        ),
        encoding="utf-8",
    )
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "test.json").write_text(json.dumps({"ids": ["q3", "q1"]}), encoding="utf-8")
    return golden, splits


def test_select_split_keeps_golden_order_and_rejects_unknown_or_empty_ids(tmp_path: Path) -> None:
    harness = _load_harness_module()
    golden, splits = _write_split_fixture(tmp_path)
    pairs = harness.load_golden_pairs(golden)

    assert [p.id for p in harness.select_split(pairs, splits / "test.json")] == ["q1", "q3"]

    for bad in ({"ids": ["q1", "q9"]}, {"ids": []}, {}):
        (splits / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
        try:
            harness.select_split(pairs, splits / "bad.json")
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_harness_split_runs_only_that_split_and_records_it(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from rca.contracts.trace import QueryTrace
    from rca.flows.generate_flow import GeneratedAnswer

    harness = _load_harness_module()
    golden, splits = _write_split_fixture(tmp_path)
    asked: list[str] = []

    class FakeFlow:
        _resolve_source_id = harness.GenerateFlow._resolve_source_id

        def __init__(self, settings) -> None:
            self.llm = SimpleNamespace(model="fake-gen")
            self.rewrite_llm = self.llm

        def generate_answer(self, query: str) -> GeneratedAnswer:
            asked.append(query)
            trace = QueryTrace(query=query, warnings=[f"note for {query}"])
            return GeneratedAnswer(query=query, answer="answer", outcome="unverified", trace=trace)

    harness.GenerateFlow = FakeFlow
    harness.validate_eval_llm = lambda flow: None
    harness.fetch_available_ollama_models = lambda base_url: set()
    harness.print_summary = lambda summary, total: None
    harness.collect_provenance = lambda *args, **kwargs: {}
    harness.finish_provenance = lambda provenance, vector_store=None: provenance

    out = tmp_path / "out"
    argv = ["--golden-path", str(golden), "--split", "test", "--splits-dir", str(splits)]
    assert harness.main([*argv, "--output-dir", str(out)]) == 0

    payload = json.loads(next(out.glob("run_*.json")).read_text(encoding="utf-8"))
    assert asked == ["Question 1?", "Question 3?"]
    assert payload["split"] == "test"
    assert payload["cases"] == 2
    # the scorecard pins the question set and refuses runs where a component fell back
    assert payload["question_ids"] == ["q1", "q3"]
    assert payload["golden_sha256"] == hashlib.sha256(golden.read_bytes()).hexdigest()
    assert [r["error"] for r in payload["results"]] == [None, None]
    assert [r["warnings"] for r in payload["results"]] == [
        ["note for Question 1?"],
        ["note for Question 3?"],
    ]


def test_harness_refuses_split_with_external_file(tmp_path: Path, capsys) -> None:
    harness = _load_harness_module()
    golden, splits = _write_split_fixture(tmp_path)
    try:
        harness.main(["--external-golden-path", str(golden), "--split", "test"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("--split with --external-golden-path was accepted")
    assert "cannot be used with an external file" in capsys.readouterr().err
