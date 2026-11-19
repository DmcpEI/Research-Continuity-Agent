import json
from types import SimpleNamespace

import pytest

from eval import run_qasper as rq


def _q(*flags, evidence=(), spans=(), free="", yes_no=None, qid="q1"):
    return {
        "question": "What is it? ",
        "question_id": qid,
        "answers": [
            {
                "answer": {
                    "unanswerable": flag,
                    "extractive_spans": list(spans),
                    "free_form_answer": free,
                    "yes_no": yes_no,
                    "evidence": list(evidence),
                }
            }
            for flag in flags
        ],
    }


@pytest.mark.parametrize(
    ("flags", "kind"),
    [((False, False), "answerable"), ((True, True), "unanswerable"), ((True, False), "mixed")],
)
def test_classify_needs_annotators_to_agree(flags, kind):
    assert rq.classify(_q(*flags)) == kind


def test_sample_papers_is_seeded_and_sorted():
    data = {f"p{i:02d}": {} for i in range(50)}
    first = rq.sample_papers(data, 5, seed=1)
    assert first == rq.sample_papers(data, 5, seed=1)
    assert first == sorted(first) and len(set(first)) == 5
    assert first != rq.sample_papers(data, 5, seed=2)


def test_gold_answers_follow_qaspers_answer_types():
    question = _q(False, False, True, False, spans=("3,044 sentences",))
    question["answers"][1]["answer"] |= {"extractive_spans": [], "free_form_answer": "a corpus"}
    question["answers"][3]["answer"] |= {"extractive_spans": [], "yes_no": False}
    assert rq.gold_answers(question) == ["3,044 sentences", "a corpus", "Unanswerable", "No"]


def test_text_evidence_leaves_out_tables_and_figures():
    question = _q(False, evidence=("FLOAT SELECTED: Table 2", "We collected 220 dialogs.", " "))
    assert rq.text_evidence(question) == ["We collected 220 dialogs."]


def test_evidence_hit_tolerates_pdf_text_differences():
    evidence = ["We collected 220 human-human dialogs on the Amazon Mechanical Turk platform."]
    pdf_text = "Intro... We col-\nlected 220 human-human dialogs on the Amazon Mechanical Turk plat-\nform. More."

    assert rq.evidence_hit(evidence, ["unrelated chunk", pdf_text]) is True
    assert rq.evidence_hit(evidence, ["a different paragraph about scam dialogs entirely"]) is False
    assert rq.evidence_hit(["Too short"], [pdf_text]) is None
    assert rq.evidence_hit([], [pdf_text]) is None


@pytest.mark.parametrize(
    ("prediction", "golds", "expected"),
    [
        ("3,044 sentences in 100 dialogs", ["3,044 sentences in 100 dialogs"], 1.0),
        ("The dataset has 3,044 sentences.", ["3,044 sentences"], 2 / 3),
        ("Unanswerable", ["Unanswerable", "No"], 1.0),
        ("Something else", ["Unanswerable"], 0.0),
        ("", [""], 0.0),  # as the official scorer: nothing to compare is 0
        ("the", ["a"], 0.0),
    ],
)
def test_token_f1_takes_the_best_annotator(prediction, golds, expected):
    assert rq.token_f1(prediction, golds) == pytest.approx(expected)


def test_paced_spaces_out_only_arxiv_requests():
    now = [0.0]
    slept: list[float] = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    def clock():
        return now[0]

    get = rq.paced(lambda url: url.encode(), gap=3.0, clock=clock, sleep=sleep)
    get("https://export.arxiv.org/api/query?id_list=1")
    now[0] += 1.0
    get("https://arxiv.org/pdf/1")
    get("https://api.openalex.org/works/x")

    assert slept == [2.0]


def test_build_library_skips_finished_papers_and_records_failures(tmp_path):
    fetched: list[str] = []

    def fetch(paper_id):
        fetched.append(paper_id)
        if paper_id == "bad":
            raise RuntimeError("withdrawn")
        return SimpleNamespace(pdf=b"%PDF x", file_name=f"{paper_id}.pdf", origin={"kind": "arxiv"})

    written: dict[str, dict] = {}
    ingest = SimpleNamespace(
        ingest_path=lambda path: SimpleNamespace(source_id=f"src:pdf/{path.stem}", chunk_ids=["c"]),
        graph_store=SimpleNamespace(
            set_metadata_field=lambda sid, key, value: written.update({sid: value})
        ),
    )

    manifest = rq.build_library(["good", "bad"], tmp_path, fetch, ingest)
    again = rq.build_library(["good", "bad"], tmp_path, fetch, ingest)

    assert manifest["good"]["source_id"] == "src:pdf/good"
    assert manifest["bad"] == {"error": "RuntimeError: withdrawn"}
    assert fetched == ["good", "bad", "bad"]  # the finished paper is not fetched again
    assert again == manifest
    assert written == {"src:pdf/good": {"kind": "arxiv"}}


def test_summarize_counts_each_group():
    records = [
        {
            "kind": "answerable",
            "outcome": "answered",
            "own_paper_cited": True,
            "evidence_hit": True,
            "model_ran": True,
            "answer_f1": 0.5,
            "seconds": 8,
        },
        {
            "kind": "answerable",
            "outcome": "not_found",
            "own_paper_cited": False,
            "evidence_hit": True,
            "model_ran": False,
            "answer_f1": 0.0,
            "seconds": 2,
        },
        {"kind": "answerable", "outcome": "error", "error": "TimeoutError: x"},
        {
            "kind": "unanswerable",
            "outcome": "not_found",
            "own_paper_cited": False,
            "evidence_hit": None,
            "model_ran": False,
            "answer_f1": 1.0,
            "seconds": 3,
        },
        {
            "kind": "mixed",
            "outcome": "answered",
            "own_paper_cited": True,
            "evidence_hit": None,
            "model_ran": True,
            "answer_f1": 0.5,
            "seconds": 9,
        },
    ]

    summary = rq.summarize(records)

    assert summary["questions"] == 5 and summary["errors"] == 1 and summary["mixed"] == 1
    assert summary["evidence_in_selected_context"]["hits"] == 2
    assert summary["evidence_reached_model"] | {"lower": 0, "upper": 0} == {
        "hits": 1,
        "n": 1,
        "lower": 0,
        "upper": 0,
    }
    assert summary["answerable_answered_citing_its_paper"]["hits"] == 1
    assert summary["answerable_answered_citing_its_paper"]["n"] == 2  # the errored one is out
    assert summary["answerable_refused"]["hits"] == 1
    assert summary["unanswerable_refused"]["hits"] == 1
    assert summary["answer_f1_answerable_mean"] == 0.25
    # Official-style: every question, an error scores 0, mixed included.
    assert summary["answer_f1_all_questions_mean"] == pytest.approx((0.5 + 0 + 0 + 1.0 + 0.5) / 5)
    assert summary["median_seconds"] == 5.5


def test_token_f1_ignores_citation_markers():
    answer = "BLEU and ROUGE [[chk:pdf/some_paper:0003]]."
    assert rq.token_f1(rq.answer_text(answer), ["BLEU, ROUGE"]) == pytest.approx(0.8)


class _Flow:
    """GenerateFlow stand-in: answers from a script; "boom" raises like a timed-out model call."""

    def __init__(self, graph, script):
        self.retrieve_flow = SimpleNamespace(graph_store=graph)
        self.script = script

    def generate_answer(self, question, trace=None):
        outcome, text, context, model_ran = self.script[question.strip()]
        if outcome == "boom":
            raise TimeoutError("timed out")
        trace.context_node_ids = context
        if model_ran:
            trace.stages.append(SimpleNamespace(name="llm_generate"))
        return SimpleNamespace(
            outcome=SimpleNamespace(value=outcome),
            answer=text,
            citations=[SimpleNamespace(source_id="src:pdf/p")] if outcome == "answered" else [],
            trace=trace,
        )


def _paper_data():
    evidence = ["We collected 220 human-human dialogs on the Amazon Mechanical Turk platform."]
    return {
        "p1": {
            "qas": [
                _q(False, evidence=evidence, spans=("220 dialogs",), qid="a1") | {"question": "Q1"},
                _q(False, evidence=evidence, spans=("220 dialogs",), qid="a2") | {"question": "Q2"},
                _q(True, False, qid="m1") | {"question": "Q3"},
                _q(False, evidence=evidence, qid="a3") | {"question": "Q4"},
            ]
        }
    }


def test_ask_records_a_failed_question_and_keeps_going(tmp_path):
    graph = SimpleNamespace(
        get_node=lambda node_id: SimpleNamespace(
            text="We collected 220 human-human dialogs on the Amazon Mechanical Turk platform."
        )
    )
    flow = _Flow(
        graph,
        {
            "Q1": ("answered", "220 dialogs [[chk:pdf/p:0001]].", ["c1"], True),
            "Q2": ("boom", "", [], False),
            "Q3": ("answered", "220 dialogs.", ["c1"], True),  # mixed questions are asked too
            "Q4": (
                "not_found",
                "I couldn't find evidence for this in your knowledge base.",
                ["c1"],
                False,
            ),
        },
    )
    manifest = {"p1": {"source_id": "src:pdf/p"}}
    checkpoint = tmp_path / "answers.jsonl"

    records = rq.ask(flow, ["p1"], _paper_data(), manifest, checkpoint)

    by_id = {r["question_id"]: r for r in records}
    assert by_id["a2"]["outcome"] == "error" and "timed out" in by_id["a2"]["error"]
    assert by_id["a1"]["answer_f1"] == pytest.approx(1.0)  # marker stripped
    assert by_id["m1"]["kind"] == "mixed" and by_id["m1"]["outcome"] == "answered"
    # The rerank gate refused before the model ran: the evidence was selected, never read.
    assert by_id["a3"]["evidence_hit"] is True and by_id["a3"]["model_ran"] is False
    assert by_id["a1"]["context_node_ids"] == ["c1"]
    assert len(checkpoint.read_text().splitlines()) == 4


def test_ask_resumes_from_its_checkpoint(tmp_path):
    graph = SimpleNamespace(get_node=lambda node_id: SimpleNamespace(text="x"))
    checkpoint = tmp_path / "answers.jsonl"
    checkpoint.write_text(
        json.dumps({"question_id": "a1", "kind": "answerable", "outcome": "answered"}) + "\n"
    )
    asked: list[str] = []

    class Recording(_Flow):
        def generate_answer(self, question, trace=None):
            asked.append(question)
            return super().generate_answer(question, trace)

    flow = Recording(graph, {q: ("not_found", "No.", [], False) for q in ("Q1", "Q2", "Q3", "Q4")})

    records = rq.ask(flow, ["p1"], _paper_data(), {"p1": {"source_id": "src:pdf/p"}}, checkpoint)

    assert asked == ["Q2", "Q3", "Q4"]
    assert [r["question_id"] for r in records] == ["a1", "a2", "m1", "a3"]


def test_check_library_refuses_a_flow_that_reads_another_library(tmp_path):
    library = tmp_path / "qasper-library"
    settings = SimpleNamespace(graph_db_path=library / "graph.sqlite3")

    def flow(path):
        return SimpleNamespace(
            retrieve_flow=SimpleNamespace(graph_store=SimpleNamespace(db_path=path))
        )

    rq.check_library(flow(library / "graph.sqlite3"), settings)  # no error
    with pytest.raises(SystemExit, match="not the QASPER library"):
        rq.check_library(flow(tmp_path / ".rca" / "graph.sqlite3"), settings)


def test_asked_text_names_the_paper_only_when_asked_to():
    assert rq.asked_text(" What data? ", "A Paper", name_paper=False) == "What data?"
    assert (
        rq.asked_text(" What data? ", "A Paper", name_paper=True)
        == 'In the paper "A Paper": What data?'
    )


def test_ask_sends_the_named_question_and_records_it(tmp_path):
    graph = SimpleNamespace(get_node=lambda node_id: SimpleNamespace(text="x"))
    sent: list[str] = []

    class Recording(_Flow):
        def generate_answer(self, question, trace=None):
            sent.append(question)
            return super().generate_answer("Q1", trace)

    flow = Recording(graph, {"Q1": ("not_found", "No.", [], False)})
    data = {"p1": {"title": "A Paper", "qas": [_paper_data()["p1"]["qas"][0]]}}

    records = rq.ask(
        flow, ["p1"], data, {"p1": {"source_id": "s"}}, tmp_path / "a.jsonl", name_paper=True
    )

    assert sent == ['In the paper "A Paper": Q1']
    assert records[0]["question"] == 'In the paper "A Paper": Q1'
