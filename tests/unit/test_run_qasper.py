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
        ("", [""], 1.0),
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


def test_summarize_counts_each_group_and_keeps_mixed_out():
    records = [
        {
            "kind": "answerable",
            "outcome": "answered",
            "own_paper_cited": True,
            "evidence_hit": True,
            "answer_f1": 0.5,
            "seconds": 8,
        },
        {
            "kind": "answerable",
            "outcome": "not_found",
            "own_paper_cited": False,
            "evidence_hit": None,
            "answer_f1": 0.0,
            "seconds": 2,
        },
        {
            "kind": "unanswerable",
            "outcome": "not_found",
            "own_paper_cited": False,
            "evidence_hit": None,
            "answer_f1": 1.0,
            "seconds": 3,
        },
        {"kind": "mixed"},
    ]

    summary = rq.summarize(records)

    assert summary["questions"] == 4 and summary["mixed_dropped"] == 1
    assert summary["evidence_in_context"]["hits"] == 1 and summary["evidence_in_context"]["n"] == 1
    assert summary["answerable_answered_citing_its_paper"]["hits"] == 1
    assert summary["answerable_refused"] | {"lower": 0, "upper": 0} == {
        "hits": 1,
        "n": 2,
        "lower": 0,
        "upper": 0,
    }
    assert summary["unanswerable_refused"]["hits"] == 1
    assert summary["answer_f1_answerable_mean"] == 0.25
    assert summary["median_seconds"] == 3
