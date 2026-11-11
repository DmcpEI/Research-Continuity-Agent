from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from rca.config.settings import Settings
from rca.telemetry.provenance import RESULT_SETTINGS

SCORECARD_PATH = Path(__file__).resolve().parents[2] / "eval" / "scorecard.py"


def _load():
    spec = importlib.util.spec_from_file_location("eval_scorecard", SCORECARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result(rid: str, *, answerable=True, abstained=False, correct=False, retrieved=(), **kw):
    return {
        "id": rid,
        "answerable": answerable,
        "abstained": abstained,
        "source_correct": correct,
        "expected_sources": kw.get("expected", ["src:a"] if answerable else []),
        "retrieved_sources": list(retrieved),
        "keyword_hits": kw.get("keywords", 0.5),
        "latency_ms": kw.get("ms", 1000.0),
        "outcome": kw.get("outcome", "not_found" if abstained else "answered"),
        "failure_labels": [],
        "warnings": ["rewrite skipped: proper_noun query"] if answerable else [],
    }


GOLDEN_BYTES = b'{"pairs": []}'


def _root(tmp_path: Path) -> Path:
    """A repo root whose committed golden file and test split match _run()."""
    (tmp_path / "eval" / "splits").mkdir(parents=True, exist_ok=True)
    (tmp_path / "eval" / "golden.json").write_bytes(GOLDEN_BYTES)
    ids = [r["id"] for r in _run()["results"]]
    (tmp_path / "eval" / "splits" / "test.json").write_text(json.dumps({"ids": ids[::-1]}))
    return tmp_path


def _run() -> dict:
    results = [
        _result("a1", correct=True, retrieved=["src:a", "src:b"], keywords=1.0, ms=2000.0),
        _result("a2", abstained=True, retrieved=["src:b"], keywords=0.0, ms=1000.0),
        _result("a3", retrieved=["src:a"], keywords=0.5, ms=4000.0),
        _result("a4", correct=True, expected=["src:a", "src:c"], retrieved=["src:a"], ms=3000.0),
        _result("n1", answerable=False, abstained=True, ms=500.0),
        _result("n2", answerable=False, ms=500.0),
    ]
    settings = {name: Settings.model_fields[name].default for name in RESULT_SETTINGS}
    return {
        "timestamp": "20261111T120000Z",
        "dataset_kind": "golden",
        "split": "test",
        "golden_sha256": hashlib.sha256(GOLDEN_BYTES).hexdigest(),
        "question_ids": [r["id"] for r in results],
        "cases": len(results),
        "run_config": {"resolved_model": "qwen3.5:4b", "resolved_rewrite_model": "qwen3.5:4b"},
        "provenance": {
            "corpus": {"corpus_version": "abcd1234abcd1234"},
            "index_start": {"backend": "chroma"},
            "index_degraded_during_run": False,
            "models": {"models": {"qwen3.5:4b": "digest1"}},
            "reranker": {"model": "cross-encoder/x", "revision": "rev1"},
            "settings": settings,
            "code": {"commit": "0123456789abcdef", "dirty": False},
        },
        "results": results,
    }


def test_wilson_interval_matches_known_values() -> None:
    sc = _load()
    lower, upper = sc.wilson_interval(12, 12)
    assert round(lower, 4) == 0.7575 and upper == 1.0  # 12 / (12 + 1.96**2)
    lower, upper = sc.wilson_interval(5, 10)
    assert (round(lower, 3), round(upper, 3)) == (0.237, 0.763)
    assert sc.wilson_interval(0, 0) == (0.0, 0.0)


def test_build_scorecard_counts_each_metric_and_names_the_misses() -> None:
    card = _load().build_scorecard(_run())
    m = card["metrics"]

    def counts(key: str) -> tuple[int, int]:
        return m[key]["hits"], m[key]["n"]

    assert counts("answered_with_expected_citation") == (2, 4)
    assert counts("answerable_refused") == (1, 4)
    assert counts("unanswerable_refused") == (1, 2)
    assert counts("citation_precision") == (2, 3)
    # a4 needs two papers and only one is in the top 5
    assert counts("retrieval_hit_at_5") == (2, 4)
    assert m["keyword_hit_rate"] == round((1.0 + 0.0 + 0.5 + 0.5) / 4, 3)
    assert m["mean_seconds"] == 1.8 and m["median_seconds"] == 1.5
    assert card["misses"] == {
        "answerable_refused": ["a2"],
        "answered_without_expected_citation": ["a3"],
        "unanswerable_answered": ["n2"],
        "retrieval_not_in_top5": ["a2", "a4"],
    }
    assert card["model_digest"] == "digest1" and card["facts_check_enabled"] is True


def test_check_run_accepts_a_clean_shipped_default_run(tmp_path: Path) -> None:
    assert _load().check_run(_run(), _root(tmp_path)) == []


@pytest.mark.parametrize(
    "mutate",
    [
        # settings grounded chat does not use on the default backend
        lambda run: run["provenance"]["settings"].update(openai_chat_model="gpt-4.1"),
        lambda run: run["provenance"]["settings"].update(agent_model="qwen3.5:9b"),
        # normal flow warnings
        lambda run: run["results"][0]["warnings"].extend(
            ["rerank gate: max rerank score -4.1", "failure:llm_abstained", "empty retrieval"]
        ),
    ],
)
def test_check_run_ignores_what_does_not_change_the_answers(mutate, tmp_path: Path) -> None:
    run = _run()
    mutate(run)
    assert _load().check_run(run, _root(tmp_path)) == []


def _set(path: tuple, value):
    def mutate(run: dict) -> None:
        target = run
        for key in path[:-1]:
            target = target[key]
        if value is KeyError:
            del target[path[-1]]
        else:
            target[path[-1]] = value

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        _set(("split",), "dev"),
        _set(("split",), None),
        _set(("dataset_kind",), "external"),
        _set(("cases",), 7),
        _set(("results",), []),
        _set(("provenance", "code", "dirty"), True),
        _set(("provenance", "code", "dirty"), None),
        _set(("provenance", "index_degraded_during_run"), True),
        _set(("provenance", "index_degraded_during_run"), None),
        _set(("provenance", "index_start", "backend"), "json"),
        _set(("results", 0, "outcome"), "error"),
        _set(("run_config", "resolved_rewrite_model"), "gemma3:12b"),
        _set(("provenance", "settings", "generation_model"), "gemma4:e4b"),
        _set(("provenance", "settings", "facts_check_enabled"), False),
        _set(("provenance", "settings", "facts_check_enabled"), KeyError),
        _set(("provenance", "settings"), {}),
        # degraded components (review finding: these passed before)
        _set(("results", 0, "warnings"), ["cross-encoder reranker unavailable: OSError: x"]),
        _set(("results", 0, "warnings"), ["query rewrite failed: ReadTimeout: x"]),
        _set(("results", 0, "warnings"), ["some new warning nobody listed"]),
        _set(("results", 0, "failure_labels"), ["llm_abstained", "facts_check_failed"]),
        _set(("results", 0, "warnings"), KeyError),
        # question set not pinned
        _set(("golden_sha256",), "0" * 64),
        _set(("golden_sha256",), KeyError),
        _set(("question_ids",), ["a1"]),
        _set(("question_ids",), KeyError),
        _set(("results", 0, "id"), "z9"),
        # missing fields that rendering needs
        _set(("run_config",), KeyError),
        _set(("provenance", "code", "commit"), KeyError),
        _set(("provenance", "corpus"), KeyError),
        _set(("provenance", "reranker"), KeyError),
    ],
)
def test_check_run_refuses_runs_that_do_not_stand_for_the_shipped_system(
    mutate, tmp_path: Path
) -> None:
    run = _run()
    mutate(run)
    assert _load().check_run(run, _root(tmp_path))


def test_replace_block_swaps_only_the_block_and_needs_exactly_one() -> None:
    sc = _load()
    text = f"intro\n{sc.START}\nold numbers\n{sc.END}\noutro\n"
    assert sc.replace_block(text, "NEW") == "intro\nNEW\noutro\n"
    for bad in ("no markers", f"{sc.END}\n{sc.START}", f"{sc.START}{sc.END}{sc.START}{sc.END}"):
        with pytest.raises(ValueError):
            sc.replace_block(bad, "NEW")


def _repo(tmp_path: Path, eval_doc: str | None = None) -> tuple[Path, Path]:
    sc = _load()
    marked = f"top\n{sc.START}\nold\n{sc.END}\nbottom\n"
    (tmp_path / "docs").mkdir(parents=True)
    _root(tmp_path)
    (tmp_path / "README.md").write_text(marked, encoding="utf-8")
    (tmp_path / "docs" / "EVAL.md").write_text(eval_doc or marked, encoding="utf-8")
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(_run()), encoding="utf-8")
    return run_path, tmp_path


def test_main_publishes_card_and_the_same_block_in_both_docs(tmp_path: Path) -> None:
    sc = _load()
    run_path, root = _repo(tmp_path)
    assert sc.main([str(run_path), "--root", str(root)]) == 0

    card = json.loads((root / "eval" / "scorecard.json").read_text(encoding="utf-8"))
    block = sc.render(card)
    assert "`2/4` (50%)" in block and "`1/2` (50%)" in block
    for doc in ("README.md", "docs/EVAL.md"):
        assert (root / doc).read_text(encoding="utf-8") == f"top\n{block}\nbottom\n"


def test_main_writes_nothing_for_a_refused_run_or_a_doc_without_markers(tmp_path: Path) -> None:
    sc = _load()
    run_path, root = _repo(tmp_path / "dirty")
    run = _run()
    run["provenance"]["code"]["dirty"] = True
    run_path.write_text(json.dumps(run), encoding="utf-8")
    before = (root / "README.md").read_text(encoding="utf-8")
    assert sc.main([str(run_path), "--root", str(root)]) == 1
    assert (root / "README.md").read_text(encoding="utf-8") == before
    assert not (root / "eval" / "scorecard.json").exists()

    run_path, root = _repo(tmp_path / "nomarker", eval_doc="no block here\n")
    before = (root / "README.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        sc.main([str(run_path), "--root", str(root)])
    assert (root / "README.md").read_text(encoding="utf-8") == before
    assert not (root / "eval" / "scorecard.json").exists()


def test_build_scorecard_does_not_modify_the_run() -> None:
    run = _run()
    snapshot = copy.deepcopy(run)
    _load().build_scorecard(run)
    assert run == snapshot


def test_main_finds_the_repo_from_any_working_directory(tmp_path: Path, monkeypatch) -> None:
    """Default --root is the repo, not the cwd: a refused run prints why instead of crashing."""
    sc = _load()
    run = _run()
    run["split"] = "dev"
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(run), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert sc.main([str(run_path)]) == 1


def test_published_docs_match_the_committed_scorecard() -> None:
    """README and EVAL.md carry exactly what eval/scorecard.json renders; no hand edits."""
    sc = _load()
    root = SCORECARD_PATH.parents[1]
    card = json.loads((root / sc.CARD).read_text(encoding="utf-8"))
    block = sc.render(card)
    for doc in sc.DOCS:
        text = (root / doc).read_text(encoding="utf-8")
        assert sc.replace_block(text, block) == text, f"{doc} differs from {sc.CARD}"
