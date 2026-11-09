from __future__ import annotations

import json
import random

import pytest

from eval import sample_answer_labels as sal


def _result(case_id: str, answer: str, *, answerable: bool = True, trace_path=None) -> dict:
    return {
        "id": case_id,
        "question": f"q {case_id}",
        "answerable": answerable,
        "negative_type": None if answerable else "unreported",
        "expected_keywords": ["k"],
        "answer": answer,
        "citations": ["src:pdf/a"],
        "abstained": False,
        "grounded": True,
        "failure_labels": [],
        "trace_path": trace_path,
    }


@pytest.mark.parametrize(
    "answer",
    [
        "The provided context does not contain the specific success rate.",
        "The text doesn't state the license [[src:pdf/a]].",
        "Based on the provided context, there is no information about it.",
        "The sources do not report how many annotators were used.",
    ],
)
def test_is_hedged_catches_gap_statements_written_by_small_models(answer: str) -> None:
    assert sal.is_hedged(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "JamPacker combines planning and recovery modules [[src:pdf/jampacker]].",
        "The method does not require depth input [[src:pdf/a]].",
        "",
    ],
)
def test_is_hedged_ignores_plain_answers(answer: str) -> None:
    assert not sal.is_hedged(answer)


def test_candidates_drop_gate_answers_and_empty_answers() -> None:
    results = [
        _result("a", "No relevant context found in your knowledge base."),
        _result("b", "I couldn't find evidence for this in your knowledge base."),
        _result("c", "   "),
        _result("d", "A real answer [[src:pdf/a]]."),
        _result("e", None),
        {**_result("f", ""), "answer": {"text": "not a string"}},
    ]
    assert [r["id"] for r in sal.candidates(results)] == ["d"]


def test_weighted_sample_is_deterministic_unique_and_capped() -> None:
    items = [_result(str(i), "answer") for i in range(10)]
    first = sal.weighted_sample(items, 4, random.Random(7))
    second = sal.weighted_sample(items, 4, random.Random(7))
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert len({r["id"] for r in first}) == 4
    assert len(sal.weighted_sample(items, 50, random.Random(7))) == 10
    assert sal.weighted_sample([], 3, random.Random(7)) == []


def test_weighted_sample_prefers_hedged_answers() -> None:
    hedged = [_result(f"h{i}", "The context does not mention it.") for i in range(50)]
    plain = [_result(f"p{i}", "Plain answer.") for i in range(50)]
    picked = sal.weighted_sample(hedged + plain, 40, random.Random(1), hedge_weight=3.0)
    assert sum(r["id"].startswith("h") for r in picked) > 25


def test_build_sample_reads_context_ids_from_trace_and_leaves_labels_empty(tmp_path) -> None:
    trace = tmp_path / "t.json"
    trace.write_text(json.dumps({"context_node_ids": ["chk:pdf/a:0001"]}), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_1.json").write_text(
        json.dumps(
            {
                "run_config": {"resolved_model": "m:1b"},
                "results": [
                    _result("x", "Answer [[src:pdf/a]].", trace_path=str(trace)),
                    _result("y", "No relevant context found in your knowledge base."),
                ],
            }
        ),
        encoding="utf-8",
    )

    items = sal.build_sample([f"m/v1={run_dir}:5"], seed=3)

    assert len(items) == 1
    item = items[0]
    assert item["item_id"] == "m/v1:x"
    assert item["model"] == "m:1b"
    assert item["context_node_ids"] == ["chk:pdf/a:0001"]
    assert item["label"] is None and item["labeled_by"] is None


@pytest.mark.parametrize("spec", ["noequals", "name=path", "name=path:x", "=path:3"])
def test_parse_run_spec_rejects_malformed_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        sal.parse_run_spec(spec)


def test_load_run_requires_exactly_one_artifact(tmp_path) -> None:
    with pytest.raises(ValueError):
        sal.load_run(tmp_path)


def test_context_ids_fall_back_to_traces_next_to_the_run(tmp_path) -> None:
    trace_dir = tmp_path / "traces" / "20261109T000000Z"
    trace_dir.mkdir(parents=True)
    (trace_dir / "x.json").write_text(
        json.dumps({"context_node_ids": ["src:pdf/b"]}), encoding="utf-8"
    )
    result = _result("x", "Answer.", trace_path=str(tmp_path / "gone" / "x.json"))

    assert sal.context_node_ids(result, tmp_path) == ["src:pdf/b"]
    assert sal.context_node_ids(_result("missing", "Answer."), tmp_path) == []


def test_cited_ids_accept_harness_strings_and_dicts_and_skip_junk() -> None:
    result = {"citations": ["src:pdf/a", {"source_id": "src:pdf/b"}, None, {"x": 1}, ""]}
    assert sal.cited_ids(result) == ["src:pdf/a", "src:pdf/b"]
    assert sal.cited_ids({}) == []


def test_context_ids_prefer_the_trace_next_to_the_run_over_trace_path(tmp_path) -> None:
    (tmp_path / "run_20261109T000002Z.json").write_text("{}", encoding="utf-8")
    for stamp, node in [("20261109T000001Z", "src:pdf/older"), ("20261109T000002Z", "src:pdf/own")]:
        trace_dir = tmp_path / "traces" / stamp
        trace_dir.mkdir(parents=True)
        (trace_dir / "x.json").write_text(
            json.dumps({"context_node_ids": [node]}), encoding="utf-8"
        )
    elsewhere = tmp_path / "other_run_trace.json"
    elsewhere.write_text(json.dumps({"context_node_ids": ["src:pdf/wrong"]}), encoding="utf-8")

    result = _result("x", "Answer.", trace_path=str(elsewhere))

    assert sal.context_node_ids(result, tmp_path) == ["src:pdf/own"]


def test_sampled_items_carry_every_label_field_empty(tmp_path) -> None:
    item = sal.to_item("m/v1", "m:1b", _result("x", "Answer."), tmp_path)
    assert {k: item[k] for k in sal.LABEL_FIELDS} == sal.LABEL_FIELDS
    assert {"contradictory", "missed_in_context"} <= set(sal.LABEL_FIELDS)
