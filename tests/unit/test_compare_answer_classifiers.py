from __future__ import annotations

import json

import pytest

from eval import compare_answer_classifiers as cac


def _item(item_id: str, label: str | None, answerable: bool = True, **extra) -> dict:
    return {
        "item_id": item_id,
        "question": "Q?",
        "answer": extra.pop("answer", "An answer."),
        "answerable": answerable,
        "label": label,
        "pipeline": {"abstained": extra.pop("abstained", False)},
        **extra,
    }


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"reason": "r", "status": "partial"}', "partial"),
        ('{"status": "not_found"}', "not_found"),
        ('{"status": "maybe"}', None),
        ('["not_found"]', None),
        ("not json", None),
        ("", None),
    ],
)
def test_parse_status_accepts_only_known_statuses(content, expected):
    assert cac.parse_status(content) == expected


def test_score_splits_by_answerable_and_counts_missing_as_wrong():
    items = [
        _item("a", "answered"),
        _item("b", "partial"),
        _item("c", "not_found"),
        _item("d", "not_found", answerable=False),
        _item("e", "partial", answerable=False),
    ]
    predictions = {"a": "answered", "b": "answered", "c": None, "d": "not_found", "e": "partial"}

    report = cac.score(items, predictions)

    assert report["all"]["three_class"] == 3  # a, d, e
    assert report["all"]["two_class"] == 4  # b counts: partial and answered are both useful
    assert report["answerable"]["not_found_caught"] == 0
    assert report["answerable"]["not_found_total"] == 1
    assert report["answerable"]["missing"] == 1
    assert report["unanswerable"]["not_found_caught"] == 1
    assert report["unanswerable"]["useful_kept"] == 1
    assert report["confusion"]["not_found->None"] == 1


def test_rule_classifiers_read_the_saved_decision_and_the_answer_text():
    hedged = _item("h", "not_found", answer="The provided context does not contain this.")
    plain = _item("p", "answered", abstained=True)

    assert cac.predict_hedge(hedged) == "not_found"
    assert cac.predict_hedge(plain) == "answered"
    assert cac.predict_current(plain) == "not_found"
    assert cac.predict_current(hedged) == "answered"


def test_load_items_rejects_unlabelled_files(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"items": [_item("a", "answered"), _item("b", None)]}))

    with pytest.raises(ValueError, match="b has no label"):
        cac.load_items([path])


def test_verifier_sends_question_and_answer_with_the_status_schema():
    class StubClient:
        def __init__(self):
            self.payload = None

        def _keep_alive_field(self):
            return {"keep_alive": "1m"}

        def _post_json(self, path, payload, timeout):
            self.payload = payload
            return {"message": {"content": '{"reason": "x", "status": "not_found"}'}}

    client = StubClient()
    item = _item("a", "not_found", answer="Nothing about that here.")

    status, _, raw = cac.predict_verifier(item, client, "small-model")

    assert status == "not_found"
    assert raw.endswith('"not_found"}')
    assert client.payload["model"] == "small-model"
    assert client.payload["format"]["properties"]["status"]["enum"] == list(cac.STATUSES)
    assert client.payload["think"] is False
    assert "Nothing about that here." in client.payload["messages"][-1]["content"]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            {
                "gap_stated": False,
                "facts": ["HM minimises pile volume."],
                "premise_corrected": False,
            },
            "answered",
        ),
        (
            {
                "gap_stated": True,
                "facts": ["Only the SayPlan side is given."],
                "premise_corrected": False,
            },
            "partial",
        ),
        ({"gap_stated": True, "facts": [], "premise_corrected": True}, "partial"),
        (
            {
                "gap_stated": False,
                "facts": ["It covers JSON schemas only."],
                "premise_corrected": True,
            },
            "partial",
        ),
        ({"gap_stated": True, "facts": ["  "], "premise_corrected": False}, "not_found"),
        ({"gap_stated": False, "facts": [], "premise_corrected": False}, "not_found"),
        ({"gap_stated": "yes", "facts": [], "premise_corrected": False}, None),
        ({"gap_stated": True, "facts": "none", "premise_corrected": False}, None),
        ({"gap_stated": True, "facts": []}, None),
    ],
)
def test_facts_reply_decides_status_from_what_the_answer_gives(reply, expected):
    assert cac.parse_facts(json.dumps(reply)) == expected


def test_facts_parser_rejects_non_json():
    assert cac.parse_facts("not json") is None


def test_useful_answer_with_no_prediction_is_not_counted_as_kept():
    items = [_item("a", "answered"), _item("b", "partial")]

    report = cac.score(items, {"a": None, "b": "partial"})

    assert report["all"]["useful_kept"] == 1
    assert report["all"]["missing"] == 1


class _StubClient:
    def __init__(self, content='{"facts": [], "gap_stated": true, "premise_corrected": false}'):
        self.content = content
        self.payload = None

    def _keep_alive_field(self):
        return {}

    def _post_json(self, path, payload, timeout):
        self.payload = payload
        if isinstance(self.content, Exception):
            raise self.content
        return {"message": {"content": self.content}}


def test_facts_verifier_sends_its_own_prompt_and_schema():
    client = _StubClient()

    status, _, _ = cac.predict_verifier(_item("a", "not_found"), client, "m", "verifier_facts")

    assert status == "not_found"
    assert client.payload["format"] == cac.FACTS_SCHEMA
    assert client.payload["messages"][0]["content"] == cac.FACTS_SYSTEM


def test_failed_verifier_call_is_a_missing_prediction_not_a_crash():
    client = _StubClient(RuntimeError("HTTP Error 500: boom"))

    status, _, raw = cac.predict_verifier(_item("a", "answered"), client, "m", "verifier_facts")

    assert status is None
    assert raw == "ERROR: HTTP Error 500: boom"


def test_gate_on_hedge_keeps_unhedged_answers_as_answers():
    items = [
        _item("h", "not_found", answer="The provided context does not contain this."),
        _item("p", "answered", answer="HM minimises pile volume."),
    ]

    gated = cac.gate_on_hedge(items, {"h": "not_found", "p": "not_found"})

    assert gated == {"h": "not_found", "p": "answered"}


def test_load_items_rejects_the_same_item_twice(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"items": [_item("a", "answered")]}))

    with pytest.raises(ValueError, match="duplicate item a"):
        cac.load_items([path, path])


def test_main_without_verifier_scores_rule_classifiers_and_writes_output(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "items": [
                    _item("a", "answered"),
                    _item("b", "not_found", answer="The context does not contain this."),
                ]
            }
        )
    )
    output = tmp_path / "out.json"

    assert cac.main(["--labels", str(labels), "--output", str(output)]) == 0

    payload = json.loads(output.read_text())
    assert set(payload["predictions"]) == {"current", "hedge"}
    assert payload["reports"]["hedge"]["all"]["two_class"] == 2
    assert payload["verifier_prompts"] == {}
