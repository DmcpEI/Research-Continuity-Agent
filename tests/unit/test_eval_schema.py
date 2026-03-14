from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from rca.flows.retrieve_flow import RetrievalHit


ROOT = Path(__file__).resolve().parents[2]


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_harness_load_golden_pairs_supports_negative_and_cross_paper(tmp_path: Path) -> None:
    harness = load_module(ROOT / "eval" / "harness.py", "eval_harness")
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(
        json.dumps(
            [
                {
                    "id": "neg-001",
                    "question": "Unsupported question?",
                    "expected_keywords": ["cannot answer"],
                    "expected_source": None,
                    "difficulty": "hard",
                    "category": "negative",
                    "answerable": False,
                },
                {
                    "id": "cross-002",
                    "question": "Synthesize two sources",
                    "expected_keywords": ["scene graphs"],
                    "expected_sources": ["src:pdf/a", "src:pdf/b"],
                    "difficulty": "hard",
                    "category": "cross_paper",
                },
            ]
        ),
        encoding="utf-8",
    )

    pairs = harness.load_golden_pairs(golden_path)

    assert len(pairs) == 2
    assert pairs[0].expected_source_ids() == []
    assert pairs[0].answerable is False
    assert pairs[1].expected_source_ids() == ["src:pdf/a", "src:pdf/b"]


def test_ablation_helpers_handle_negative_and_multi_source_cases() -> None:
    run_ablations = load_module(ROOT / "eval" / "run_ablations.py", "eval_run_ablations")

    negative = {
        "id": "neg-001",
        "expected_source": None,
        "answerable": False,
    }
    cross = {
        "id": "cross-002",
        "expected_sources": ["src:pdf/a", "src:pdf/b"],
        "answerable": True,
    }

    assert run_ablations.expected_source_ids(negative) == []
    assert run_ablations.is_retrieval_case(negative) is False
    assert run_ablations.expected_source_ids(cross) == ["src:pdf/a", "src:pdf/b"]
    assert run_ablations.is_retrieval_case(cross) is True

    hits = [
        RetrievalHit(node_id="src:pdf/a", score=0.9, title="A", excerpt="", metadata={}),
        RetrievalHit(node_id="chk:pdf/b:0001", score=0.8, title="B", excerpt="", metadata={}),
    ]
    assert run_ablations.hit_at_k(hits, ["src:pdf/a"], k=5) is True
    assert run_ablations.hit_at_k(hits, ["src:pdf/a", "src:pdf/b"], k=5) is True
    assert run_ablations.hit_at_k(hits, ["src:pdf/a", "src:pdf/c"], k=5) is False
