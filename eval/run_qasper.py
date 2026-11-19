"""Ask RCA QASPER's questions about papers it has never seen (step 17a).

Builds a separate library from a seeded sample of QASPER test papers, fetching each paper's
PDF from arXiv (step 18's fetch_paper, paced to one arXiv request every 3 s), then asks every
question about those papers with the shipped defaults. The user's library is never touched; the
data, PDFs and library live under eval/external/ (git-ignored). See docs/EVAL.md, "QASPER".

    uv run python eval/run_qasper.py --data eval/external/qasper/data/qasper-test-v0.3.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import string
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.compare_support_checks import quote_in  # noqa: E402
from eval.scorecard import wilson_interval  # noqa: E402

REFUSAL = "Unanswerable"  # QASPER's gold string for an unanswerable question
ARXIV_GAP_SECONDS = 3.0


def classify(question: dict) -> str:
    """answerable (no annotator says unanswerable), unanswerable (all do), or mixed."""
    flags = [annotation["answer"]["unanswerable"] for annotation in question["answers"]]
    if all(flags):
        return "unanswerable"
    return "mixed" if any(flags) else "answerable"


def sample_papers(data: dict, count: int, seed: int) -> list[str]:
    return sorted(random.Random(seed).sample(sorted(data), count))


def gold_answers(question: dict) -> list[str]:
    """Each annotator's answer as a string, as QASPER's evaluator builds them."""
    answers = []
    for annotation in question["answers"]:
        answer = annotation["answer"]
        if answer["unanswerable"]:
            answers.append(REFUSAL)
        elif answer["extractive_spans"]:
            answers.append(", ".join(answer["extractive_spans"]))
        elif answer["free_form_answer"]:
            answers.append(answer["free_form_answer"])
        elif answer["yes_no"] is not None:
            answers.append("Yes" if answer["yes_no"] else "No")
    return answers


def text_evidence(question: dict) -> list[str]:
    """Evidence paragraphs from the text; tables and figures ("FLOAT SELECTED") are left out,
    because RCA reads the PDF's text and cannot match them."""
    return [
        paragraph
        for annotation in question["answers"]
        for paragraph in annotation["answer"]["evidence"]
        if paragraph.strip() and not paragraph.startswith("FLOAT SELECTED")
    ]


def evidence_hit(evidence: list[str], context_texts: list[str]) -> bool | None:
    """True when the text RCA gave the model holds half of one evidence paragraph's 4-word runs.

    Fuzzy on purpose: QASPER's text comes from LaTeX, RCA's from the PDF. None when the question
    has no text evidence to look for.
    """
    usable = [paragraph for paragraph in evidence if len(paragraph.split()) >= 4]
    if not usable:
        return None
    joined = "\n".join(context_texts)
    return any(quote_in(paragraph, joined, coverage=0.5) for paragraph in usable)


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def token_f1(prediction: str, golds: list[str]) -> float:
    """Best token F1 against any annotator's answer (SQuAD-style, as QASPER's Answer F1)."""
    best = 0.0
    predicted = _normalize(prediction)
    for gold in golds:
        reference = _normalize(gold)
        common = sum((Counter(predicted) & Counter(reference)).values())
        if not predicted or not reference:
            score = float(predicted == reference)
        elif common == 0:
            score = 0.0
        else:
            precision, recall = common / len(predicted), common / len(reference)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def paced(
    get: Callable[[str], bytes],
    gap: float = ARXIV_GAP_SECONDS,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Wrap a getter so arXiv requests are at least `gap` seconds apart (arXiv's API terms)."""
    last = [float("-inf")]

    def wrapped(url: str) -> bytes:
        if "arxiv.org" in url:
            wait = last[0] + gap - clock()
            if wait > 0:
                sleep(wait)
            last[0] = clock()
        return get(url)

    return wrapped


def library_settings(library: Path):
    from rca.config.settings import Settings

    return Settings(
        data_dir=library,
        graph_db_path=library / "graph.sqlite3",
        vector_dir=library / "vectors",
        telemetry_log_path=library / "telemetry.jsonl",
        experiment_db_path=library / "experiments.sqlite3",
        filesystem_root=library,
        workspace_root=library,
    )


def build_library(paper_ids: list[str], library: Path, fetch, ingest) -> dict[str, dict]:
    """Fetch and ingest each paper once; the manifest makes a rerun skip finished papers."""
    manifest_path = library / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    papers_dir = library / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    for paper_id in paper_ids:
        if "source_id" in manifest.get(paper_id, {}):
            continue
        try:
            paper = fetch(paper_id)
            path = papers_dir / paper.file_name
            path.write_bytes(paper.pdf)
            result = ingest.ingest_path(path)
            ingest.graph_store.set_metadata_field(result.source_id, "origin", paper.origin)
            manifest[paper_id] = {
                "source_id": result.source_id,
                "chunks": len(result.chunk_ids),
                "origin": paper.origin,
            }
        except Exception as exc:  # one paper that cannot be fetched or read is skipped
            manifest[paper_id] = {"error": f"{type(exc).__name__}: {exc}"}
        print(
            f"{paper_id}: {manifest[paper_id].get('source_id') or manifest[paper_id]['error']}",
            flush=True,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def ask(flow, paper_ids: list[str], data: dict, manifest: dict) -> list[dict]:
    """One record per non-mixed question on a paper that made it into the library."""
    from rca.contracts.trace import QueryTrace

    graph = flow.retrieve_flow.graph_store
    records = []
    for paper_id in paper_ids:
        if "source_id" not in manifest.get(paper_id, {}):
            continue
        for question in data[paper_id]["qas"]:
            kind = classify(question)
            record = {"paper_id": paper_id, "question_id": question["question_id"], "kind": kind}
            if kind == "mixed":
                records.append(record)
                continue
            started = time.perf_counter()
            answer = flow.generate_answer(
                question["question"].strip(), trace=QueryTrace(query=question["question"])
            )
            seconds = time.perf_counter() - started
            context_ids = answer.trace.context_node_ids if answer.trace else []
            texts = [
                node.text or "" for node_id in context_ids if (node := graph.get_node(node_id))
            ]
            refused = answer.outcome.value == "not_found"
            record |= {
                "question": question["question"].strip(),
                "outcome": answer.outcome.value,
                "answer": answer.answer,
                "cited_sources": [citation.source_id for citation in answer.citations],
                "own_paper_cited": manifest[paper_id]["source_id"]
                in {citation.source_id for citation in answer.citations},
                "evidence_hit": evidence_hit(text_evidence(question), texts),
                "answer_f1": token_f1(
                    REFUSAL if refused else answer.answer, gold_answers(question)
                ),
                "seconds": round(seconds, 2),
            }
            print(
                f"{record['question_id'][:10]} {kind:12} {record['outcome']:12} ev={record['evidence_hit']}",
                flush=True,
            )
            records.append(record)
    return records


def summarize(records: list[dict]) -> dict[str, Any]:
    def share(hits: int, n: int) -> dict[str, Any]:
        low, high = wilson_interval(hits, n) if n else (None, None)
        return {"hits": hits, "n": n, "lower": low, "upper": high}

    answerable = [r for r in records if r["kind"] == "answerable"]
    unanswerable = [r for r in records if r["kind"] == "unanswerable"]
    with_evidence = [r for r in answerable if r["evidence_hit"] is not None]
    asked = answerable + unanswerable
    return {
        "questions": len(records),
        "mixed_dropped": sum(r["kind"] == "mixed" for r in records),
        "evidence_in_context": share(
            sum(r["evidence_hit"] for r in with_evidence), len(with_evidence)
        ),
        "answerable_answered_citing_its_paper": share(
            sum(r["outcome"] == "answered" and r["own_paper_cited"] for r in answerable),
            len(answerable),
        ),
        "answerable_refused": share(
            sum(r["outcome"] == "not_found" for r in answerable), len(answerable)
        ),
        "unanswerable_refused": share(
            sum(r["outcome"] == "not_found" for r in unanswerable), len(unanswerable)
        ),
        "answer_f1_answerable_mean": round(statistics.mean(r["answer_f1"] for r in answerable), 3)
        if answerable
        else None,
        "median_seconds": statistics.median(r["seconds"] for r in asked) if asked else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="qasper-test-v0.3.json")
    parser.add_argument("--papers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--library", type=Path, default=Path("eval/external/qasper/library"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    args = parser.parse_args(argv)

    from rca.flows.fetch_flow import fetch_paper, http_get
    from rca.flows.generate_flow import GenerateFlow
    from rca.flows.ingest_flow import IngestFlow
    from rca.telemetry.provenance import collect_provenance, finish_provenance

    data = json.loads(args.data.read_text(encoding="utf-8"))
    paper_ids = sample_papers(data, args.papers, args.seed)
    library = args.library.resolve()
    settings = library_settings(library)
    getter = paced(http_get)
    ingest = IngestFlow(settings=settings)
    manifest = build_library(
        paper_ids, library, lambda paper_id: fetch_paper(paper_id, get=getter), ingest
    )

    flow = GenerateFlow(settings=settings)
    provenance = collect_provenance(
        settings, flow.retrieve_flow.graph_store, flow.retrieve_flow.vector_store
    )
    records = ask(flow, paper_ids, data, manifest)
    summary = summarize(records)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"qasper_{timestamp}.json"
    output.write_text(
        json.dumps(
            {
                "dataset": str(args.data),
                "split": "test",
                "seed": args.seed,
                "paper_ids": paper_ids,
                "not_fetched": {k: v["error"] for k, v in manifest.items() if "error" in v},
                "provenance": finish_provenance(provenance, flow.retrieve_flow.vector_store),
                "summary": summary,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(json.dumps(summary, indent=2))
    print(f"-> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
