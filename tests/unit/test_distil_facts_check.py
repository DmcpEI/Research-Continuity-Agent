import json
from types import SimpleNamespace

import pytest

from eval import distil_facts_check as dfc


def test_parse_questions_keeps_each_kind_in_order():
    reply = json.dumps(
        {"answerable": " What is A? ", "unanswered": "What is B?", "false_premise": "Why C?"}
    )

    assert dfc.parse_questions(reply) == [
        ("answerable", "What is A?"),
        ("unanswered", "What is B?"),
        ("false_premise", "Why C?"),
    ]


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        "[]",
        json.dumps({"answerable": "What is A?"}),
        json.dumps({"answerable": "A?", "unanswered": 3, "false_premise": "C?"}),
    ],
)
def test_parse_questions_refuses_a_malformed_reply(reply):
    assert dfc.parse_questions(reply) is None


def test_parse_questions_drops_empty_questions():
    reply = json.dumps({"answerable": "What is A?", "unanswered": " ", "false_premise": "Why C?"})

    assert dfc.parse_questions(reply) == [("answerable", "What is A?"), ("false_premise", "Why C?")]


def test_too_close_catches_a_reworded_golden_question_only():
    golden = ["What success rate does AutoBag report for bag opening?"]

    assert dfc.too_close("What success rate does AutoBag report for opening bags?", golden)
    assert not dfc.too_close("Which robot arm does AutoBag use?", golden)
    assert not dfc.too_close("", golden)


def test_source_of_maps_a_chunk_to_its_paper():
    assert dfc.source_of("chk:pdf/autobag:0018") == "src:pdf/autobag"


def _stage(name, notes=None):
    return SimpleNamespace(name=name, notes=notes)


def test_teacher_status_reads_the_facts_check_stage():
    trace = SimpleNamespace(stages=[_stage("llm_generate"), _stage("facts_check", "partial")])

    assert dfc.teacher_status(trace) == "partial"


@pytest.mark.parametrize(
    "stages", [[_stage("llm_generate")], [_stage("facts_check", "failed")], []]
)
def test_teacher_status_is_none_without_a_usable_check(stages):
    assert dfc.teacher_status(SimpleNamespace(stages=stages)) is None
    assert dfc.teacher_status(None) is None


def test_student_input_drops_citation_markers_and_keeps_the_pair_apart():
    question, answer = dfc.student_input(
        "Part one.\n\nWhat is X? ",
        "The context does not say [[chk:pdf/a:0001]]. It is X [[src:pdf/b]].",
    )

    assert question == "Part one.\n\nWhat is X?"
    assert answer == "The context does not say . It is X ."


def _record(paper, teacher, hedged=True, question="Q?"):
    return {
        "paper": paper,
        "teacher": teacher,
        "hedged": hedged,
        "question": question,
        "answer": "A",
    }


def test_training_pairs_keep_hedged_teacher_labels_as_decline_or_not():
    records = [
        _record("src:a", "not_found"),
        _record("src:a", "partial"),
        _record("src:a", "answered"),
        _record("src:a", None),
        _record("src:a", "answered", hedged=False),
    ]

    assert [label for _, label in dfc.training_pairs(records)] == [1, 0, 0]


def test_split_by_paper_keeps_each_paper_on_one_side():
    records = [_record(f"src:{p}", "answered") for p in "abcdefgh" for _ in range(3)]

    train, held = dfc.split_by_paper(records, held_out=2, seed=7)

    held_papers = {r["paper"] for r in held}
    assert len(held_papers) == 2
    assert not held_papers & {r["paper"] for r in train}
    assert len(train) + len(held) == len(records)
    assert dfc.split_by_paper(records, held_out=2, seed=7) == (train, held)


def test_agreement_counts_caught_and_kept():
    labels = [1, 1, 0, 0, 0]
    predicted = [1, 0, 0, 0, 1]

    assert dfc.agreement(labels, predicted) == {
        "declines_caught": "1/2",
        "useful_kept": "2/3",
        "agree": "3/5",
    }


def test_pick_threshold_maximises_agreement_and_prefers_the_middle_on_ties():
    # Any cut in (0.3, 0.7] separates these; 0.5 is in the band, so 0.5 wins the tie.
    probs = [0.9, 0.7, 0.3, 0.1]
    labels = [1, 1, 0, 0]

    assert dfc.pick_threshold(probs, labels) == 0.5
    assert dfc.predict(probs, 0.5) == labels


def test_pick_threshold_cuts_between_points_not_on_the_lowest_decline():
    # The best band is (0.6, 0.8]: the cut must sit inside it, not on 0.8.
    probs = [0.2, 0.6, 0.8, 0.9]
    labels = [0, 0, 1, 1]

    threshold = dfc.pick_threshold(probs, labels)

    assert threshold == 0.7
    assert dfc.predict(probs, threshold) == labels


def test_pick_threshold_tie_break_prefers_the_cut_nearest_the_middle():
    # Cuts 0.05, 0.5 and 0.6 each get 2 of 3 right; 0.05 comes first, 0.5 is nearest 0.5.
    assert dfc.pick_threshold([0.1, 0.3, 0.9], [1, 0, 1]) == 0.5
    # Cuts 0.2 and 0.95 tie at 2 of 3; 0.2 is nearer the middle.
    assert dfc.pick_threshold([0.1, 0.3, 0.9], [0, 1, 0]) == 0.2


def test_hand_inputs_labels_mark_only_not_found_as_a_decline():
    items = [
        {"question": "Q1", "answer": "A [[chk:pdf/a:0001]]", "label": "not_found"},
        {"question": "Q2", "answer": "B", "label": "partial"},
        {"question": "Q3", "answer": "C", "label": "answered"},
    ]

    inputs, labels = dfc.hand_inputs_labels(items)

    assert labels == [1, 0, 0]
    assert inputs[0] == ("Q1", "A")


def test_nearest_returns_the_closest_question():
    assert dfc.nearest("a b c d", ["a b x y", "a b c z"]) == (0.6, "a b c z")
    assert dfc.nearest("", ["a"]) == (0.0, "")


def _write_lines(path, records, tail=""):
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + tail)


def test_load_checkpoint_drops_failed_and_cut_off_chunks(tmp_path):
    path = tmp_path / "teacher.jsonl"
    good = [{"chunk_id": "c1", "kind": k, "of": 2} for k in ("answerable", "unanswered")]
    failed = [{"chunk_id": "c2", "kind": None, "error": "ConnectionError", "of": 1}]
    cut_off = [{"chunk_id": "c3", "kind": "answerable", "of": 3}]
    old_format = [{"chunk_id": "c4", "kind": "answerable"}]
    _write_lines(path, good + failed + cut_off + old_format, tail='{"chunk_id": "c5", "ki')

    assert dfc.load_checkpoint(path) == good + old_format
    assert dfc.load_checkpoint(tmp_path / "missing.jsonl") == []


class _FailingThenWorkingLLM:
    def __init__(self):
        self.down = True

    def chat_json(self, messages, schema):
        if self.down:
            raise ConnectionError("ollama down")
        reply = {"answerable": "What is A?", "unanswered": "What is B?", "false_premise": ""}
        return SimpleNamespace(text=json.dumps(reply))


class _Flow:
    def __init__(self):
        self.llm = _FailingThenWorkingLLM()

    def generate_answer(self, question, trace):
        stages = [SimpleNamespace(name="llm_generate", notes=None, duration_ms=5.0)]
        if "B" in question:
            stages.append(SimpleNamespace(name="facts_check", notes="not_found", duration_ms=9.0))
        return SimpleNamespace(
            answer="The context does not contain B.",
            outcome=SimpleNamespace(value="not_found"),
            trace=SimpleNamespace(stages=stages),
        )


def test_build_asks_a_failed_chunk_again_on_resume_without_duplicates(tmp_path, capsys):
    path = tmp_path / "teacher.jsonl"
    flow = _Flow()
    chunks = [{"id": "chk:pdf/p:0001", "text": "x" * 500, "title": "P"}]

    first = dfc.build(flow, chunks, golden=[], checkpoint=path)
    flow.llm.down = False
    second = dfc.build(flow, chunks, golden=[], checkpoint=path)
    third = dfc.build(flow, chunks, golden=[], checkpoint=path)

    assert [r.get("error") for r in first] == ["ConnectionError: ollama down"]
    assert [(r["kind"], r["hedged"], r["teacher"]) for r in second] == [
        ("answerable", False, None),
        ("unanswered", True, "not_found"),
    ]
    assert third == second
    assert dfc.load_checkpoint(path) == second
    assert len(path.read_text().splitlines()) == 2
