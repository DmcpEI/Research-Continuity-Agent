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
        if common == 0:  # also when either side is empty, as QASPER's scorer
            score = 0.0
        else:
            precision, recall = common / len(predicted), common / len(reference)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


_MARKER = re.compile(r"\[\[(?:src|chk):[^\]]*\]\]")


def answer_text(answer: str) -> str:
    """The answer as a reader sees it: citation markers would each count as a wrong token."""
    return _MARKER.sub("", answer)


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


def asked_text(question: str, title: str, name_paper: bool) -> str:
    """QASPER questions are written about one paper ("what data do they use?") and do not name
    it; a person asking RCA, which searches a whole library, would."""
    question = question.strip()
    return f'In the paper "{title}": {question}' if name_paper else question


def check_library(flow, settings) -> None:
    """Stop before asking anything if the flow would search another library (the user's)."""
    searched = Path(flow.retrieve_flow.graph_store.db_path).resolve()
    if searched != Path(settings.graph_db_path).resolve():
        raise SystemExit(f"RCA would search {searched}, not the QASPER library; nothing asked.")


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


def ask(
    flow,
    paper_ids: list[str],
    data: dict,
    manifest: dict,
    checkpoint: Path,
    name_paper: bool = False,
) -> list[dict]:
    """One record per question on a paper that made it into the library, mixed ones included.

    Each record is appended to `checkpoint` as soon as it exists, and questions already there
    are not asked again, so a crash or a stop loses nothing. A question that raises is recorded
    as an error and the run goes on.
    """
    from rca.contracts.trace import QueryTrace

    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["question_id"]] = record
    graph = flow.retrieve_flow.graph_store
    records = []
    with checkpoint.open("a", encoding="utf-8") as sink:
        for paper_id in paper_ids:
            if "source_id" not in manifest.get(paper_id, {}):
                continue
            for question in data[paper_id]["qas"]:
                if question["question_id"] in done:
                    records.append(done[question["question_id"]])
                    continue
                text = asked_text(question["question"], data[paper_id].get("title", ""), name_paper)
                record = _ask_one(
                    flow, graph, question, text, manifest[paper_id]["source_id"], QueryTrace
                )
                record = {"paper_id": paper_id, **record}
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                print(
                    f"{record['question_id'][:10]} {record['kind']:12} {record['outcome']:12}"
                    f" ev={record.get('evidence_hit')}",
                    flush=True,
                )
                records.append(record)
    return records


def _ask_one(flow, graph, question: dict, text: str, source_id: str, trace_type) -> dict:
    record = {"question_id": question["question_id"], "kind": classify(question), "question": text}
    started = time.perf_counter()
    try:
        answer = flow.generate_answer(text, trace=trace_type(query=text))
    except Exception as exc:  # one failed question, e.g. a model timeout, is not a lost run
        return record | {"outcome": "error", "error": f"{type(exc).__name__}: {exc}"}
    seconds = time.perf_counter() - started
    trace = answer.trace
    context_ids = trace.context_node_ids if trace else []
    texts = [node.text or "" for node_id in context_ids if (node := graph.get_node(node_id))]
    refused = answer.outcome.value == "not_found"
    return record | {
        "outcome": answer.outcome.value,
        "answer": answer.answer,
        "cited_sources": [citation.source_id for citation in answer.citations],
        "context_node_ids": list(context_ids),
        "own_paper_cited": source_id in {citation.source_id for citation in answer.citations},
        # False when a gate refused before the model call: the context was selected, not read.
        "model_ran": any(stage.name == "llm_generate" for stage in (trace.stages if trace else [])),
        "evidence_hit": evidence_hit(text_evidence(question), texts),
        "answer_f1": token_f1(
            REFUSAL if refused else answer_text(answer.answer), gold_answers(question)
        ),
        "seconds": round(seconds, 2),
    }


def select_context(flow, text: str) -> tuple[list[str], float | None]:
    """The context GenerateFlow would give the model, without calling it: (node IDs, top
    rerank score). Mirrors generate_answer: classify, rewrite unless a proper noun, retrieve 5,
    select, and retry with the raw question if the rewrite left nothing."""
    from rca.retrieval.query_classifier import QueryType, classify_query

    query_type = classify_query(text)
    query = text if query_type is QueryType.proper_noun else flow._rewrite_query(text)
    bundle = flow.retrieve_flow.retrieve(query, limit=5, query_type=query_type)
    context = flow._select_context_hits(bundle)
    if not context and query != text:
        bundle = flow.retrieve_flow.retrieve(text, limit=5, query_type=query_type)
        context = flow._select_context_hits(bundle)
    scores = [hit.metadata["rerank_score"] for hit in bundle.hits if "rerank_score" in hit.metadata]
    return [hit.node_id for hit in context], (max(scores) if scores else None)


def retrieval_only(flow, paper_ids, data, manifest, name_paper: bool) -> dict[str, Any]:
    """Evidence in the selected context for answerable questions, no answers generated."""
    graph = flow.retrieve_flow.graph_store
    records = []
    for paper_id in paper_ids:
        if "source_id" not in manifest.get(paper_id, {}):
            continue
        source_id = manifest[paper_id]["source_id"]
        own = source_id.replace("src:", "chk:", 1) + ":"
        for question in data[paper_id]["qas"]:
            if classify(question) != "answerable":
                continue
            text = asked_text(question["question"], data[paper_id].get("title", ""), name_paper)
            context, top = select_context(flow, text)
            texts = [node.text or "" for node_id in context if (node := graph.get_node(node_id))]
            records.append(
                {
                    "question_id": question["question_id"],
                    "context_node_ids": context,
                    "evidence_hit": evidence_hit(text_evidence(question), texts),
                    "own_share": sum(n == source_id or n.startswith(own) for n in context)
                    / max(1, len(context)),
                    "opening_first": bool(context)
                    and (context[0] == source_id or context[0] == own + "0000"),
                    "rerank_max": top,
                }
            )
    scored = [r for r in records if r["evidence_hit"] is not None]
    return {
        "summary": {
            "questions": len(records),
            "evidence_in_selected_context": f"{sum(r['evidence_hit'] for r in scored)}/{len(scored)}",
            "own_paper_share_mean": round(statistics.mean(r["own_share"] for r in records), 3),
            "opening_ranked_first": f"{sum(r['opening_first'] for r in records)}/{len(records)}",
        },
        "records": records,
    }


def summarize(records: list[dict]) -> dict[str, Any]:
    def share(hits: int, n: int) -> dict[str, Any]:
        low, high = wilson_interval(hits, n) if n else (None, None)
        return {"hits": hits, "n": n, "lower": low, "upper": high}

    asked = [r for r in records if r.get("outcome") != "error"]
    answerable = [r for r in asked if r["kind"] == "answerable"]
    unanswerable = [r for r in asked if r["kind"] == "unanswerable"]
    with_evidence = [r for r in answerable if r["evidence_hit"] is not None]
    read = [r for r in with_evidence if r["model_ran"]]
    return {
        "questions": len(records),
        "errors": len(records) - len(asked),
        "mixed": sum(r["kind"] == "mixed" for r in records),
        "evidence_in_selected_context": share(
            sum(r["evidence_hit"] for r in with_evidence), len(with_evidence)
        ),
        "evidence_reached_model": share(sum(r["evidence_hit"] for r in read), len(read)),
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
        # As QASPER's own evaluator: every question, mixed included, an error scoring 0.
        "answer_f1_all_questions_mean": round(
            statistics.mean(r.get("answer_f1", 0.0) for r in records), 3
        )
        if records
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
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only retrieve (no answers): evidence in the selected context, for ablations.",
    )
    parser.add_argument(
        "--name-paper",
        action="store_true",
        help='Ask "In the paper "<title>": <question>" instead of the question as written.',
    )
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
    check_library(flow, settings)
    provenance = collect_provenance(
        settings, flow.retrieve_flow.graph_store, flow.retrieve_flow.vector_store
    )
    if args.retrieval_only:
        result = retrieval_only(flow, paper_ids, data, manifest, args.name_paper)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / f"qasper_retrieval_{stamp}.json"
        result["provenance"] = finish_provenance(provenance, flow.retrieve_flow.vector_store)
        result["question_form"] = "paper named" if args.name_paper else "as written"
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result["summary"], indent=2))
        print(f"-> {output}")
        return 0
    commit = (provenance.get("code") or {}).get("commit") or "unknown"
    # One checkpoint per commit, so answers from different code are never mixed.
    form = "-named" if args.name_paper else ""
    records = ask(
        flow,
        paper_ids,
        data,
        manifest,
        library / f"answers-{commit[:12]}{form}.jsonl",
        name_paper=args.name_paper,
    )
    summary = summarize(records)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"qasper_{timestamp}.json"
    output.write_text(
        json.dumps(
            {
                "dataset": str(args.data),
                "split": "test",
                "question_form": "paper named" if args.name_paper else "as written",
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
