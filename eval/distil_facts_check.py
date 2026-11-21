"""Distil the facts check (step 15): train a small cross-encoder to make its decision.

The teacher is the shipped facts check (rca/flows/facts_check.py): on a hedged reply, a second
qwen3.5:4b call decides whether the reply gives anything the question asked for. The student
reads the question and the reply and predicts the same useful-or-decline decision.

    build  questions generated from library chunks, answered by GenerateFlow on a copy of the
           library; each hedged reply keeps the teacher's status (checkpointed JSONL)
    train  fine-tune the reranker's MiniLM cross-encoder on the teacher labels; dev hand labels
           pick the epoch and the threshold; 5 papers are held out for teacher agreement
    test   score the chosen student once on the hedged test hand labels, with latency

Design and ship bar: docs/PLAN.md, "Step 15: distil the facts check, design".
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_qasper import check_library, library_settings  # noqa: E402
from rca.flows.facts_check import is_hedged, json_object  # noqa: E402

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
KINDS = ("answerable", "unanswered", "false_premise")
TEACHER_STATUSES = ("answered", "partial", "not_found")
_MARKER = re.compile(r"\[\[((?:src|chk):[^\]]+)\]\]")
_WORD = re.compile(r"[a-z0-9]+")

QUESTION_SYSTEM = """You write test questions for a research assistant that answers only from a library of research papers. You get one passage from one paper.
- answerable: a specific question that this passage answers.
- unanswered: a specific question about the same paper that a reader could ask but this passage does not answer.
- false_premise: a question about the same paper that assumes something the passage contradicts or that the paper does not do.
Name the paper's method or topic in every question so it can be found among many papers. Never write "this passage", "the passage" or "the text". One sentence each. Reply with JSON."""

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {kind: {"type": "string"} for kind in KINDS},
    "required": list(KINDS),
}


def parse_questions(content: str) -> list[tuple[str, str]] | None:
    """(kind, question) for each non-empty question; None when the reply is malformed."""
    payload = json_object(content)
    if not payload or not all(isinstance(payload.get(kind), str) for kind in KINDS):
        return None
    return [(kind, payload[kind].strip()) for kind in KINDS if payload[kind].strip()]


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def nearest(question: str, others: list[str]) -> tuple[float, str]:
    """(word-set Jaccard, question) of the closest of `others`; (0.0, "") when none overlap."""
    words = _words(question)
    best = (0.0, "")
    for other in others:
        union = words | _words(other)
        score = len(words & _words(other)) / len(union) if union else 0.0
        if score > best[0]:
            best = (score, other)
    return best


def too_close(question: str, others: list[str], threshold: float = 0.5) -> bool:
    """Word-set Jaccard against each golden question, so no hand-labelled question is trained on."""
    return bool(_words(question)) and nearest(question, others)[0] >= threshold


def source_of(chunk_id: str) -> str:
    return "src:" + chunk_id.removeprefix("chk:").rsplit(":", 1)[0]


def teacher_status(trace) -> str | None:
    """The facts check's status from GenerateFlow's trace; None if it did not run or failed."""
    for stage in getattr(trace, "stages", None) or []:
        if stage.name == "facts_check":
            return stage.notes if stage.notes in TEACHER_STATUSES else None
    return None


def student_input(question: str, answer: str) -> tuple[str, str]:
    """What the student reads, as a (question, answer) pair: the teacher's input without
    citation IDs, which GenerateFlow strips from a declined reply but not from a kept one."""
    return question.strip(), " ".join(_MARKER.sub("", answer).split())


def training_pairs(records: list[dict]) -> list[tuple[tuple[str, str], int]]:
    """(input, 1 if the teacher declines else 0) for hedged replies the teacher labelled."""
    return [
        (student_input(r["question"], r["answer"]), int(r["teacher"] == "not_found"))
        for r in records
        if r.get("hedged") and r.get("teacher") in TEACHER_STATUSES
    ]


def split_by_paper(records: list[dict], held_out: int, seed: int) -> tuple[list[dict], list[dict]]:
    papers = sorted({r["paper"] for r in records})
    held = set(random.Random(seed).sample(papers, held_out))
    return (
        [r for r in records if r["paper"] not in held],
        [r for r in records if r["paper"] in held],
    )


def agreement(labels: list[int], predicted: list[int]) -> dict[str, str]:
    """1 means decline. Caught: declines predicted as declines; kept: useful kept."""
    pairs = list(zip(labels, predicted, strict=True))
    declines = [p for label, p in pairs if label == 1]
    useful = [p for label, p in pairs if label == 0]
    return {
        "declines_caught": f"{sum(declines)}/{len(declines)}",
        "useful_kept": f"{len(useful) - sum(useful)}/{len(useful)}",
        "agree": f"{sum(label == p for label, p in pairs)}/{len(pairs)}",
    }


def predict(probs: list[float], threshold: float) -> list[int]:
    return [int(p >= threshold) for p in probs]


def pick_threshold(probs: list[float], labels: list[int]) -> float:
    """The cut on P(decline) with the most agreement; among ties, the one nearest 0.5.

    Candidates sit halfway between neighbouring probabilities (and at 0.5), never on a dev
    point, so a test answer just below the lowest dev decline is not missed by construction.
    """

    def correct(threshold: float) -> int:
        return sum(p == label for p, label in zip(predict(probs, threshold), labels, strict=True))

    points = sorted(set(probs) | {0.0, 1.0})
    candidates = [(low + high) / 2 for low, high in zip(points, points[1:])] + [0.5]
    return max(candidates, key=lambda t: (correct(t), -abs(t - 0.5)))


def first_chunks(chunks: list[dict], count: int, seed: int) -> list[dict]:
    """A seeded order of all chunks, cut at `count`: a bigger run extends a smaller one."""
    order = sorted(chunks, key=lambda chunk: chunk["id"])
    random.Random(seed).shuffle(order)
    return order[:count]


def load_checkpoint(path: Path) -> list[dict]:
    """Records of the chunks that finished cleanly. A chunk with an error record (Ollama down,
    a timeout) or with fewer records than it announced (killed mid-write) is left out, so a
    resumed build asks it again; a half-written last line is ignored."""
    by_chunk: dict[str, list[dict]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and "chunk_id" in record:
                by_chunk.setdefault(record["chunk_id"], []).append(record)
    return [
        record
        for records in by_chunk.values()
        if not any("error" in r for r in records)
        and all(r.get("of", len(records)) == len(records) for r in records)
        for record in records
    ]


def hand_labelled(path: Path) -> list[dict]:
    """The hedged hand-labelled answers: the only ones the facts check, and the student, see."""
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    return [i for i in items if is_hedged(i["answer"])]


# --- build ------------------------------------------------------------------------------------


def _chunks(graph, min_chars: int = 400) -> list[dict]:
    with graph.connect() as connection:
        titles = dict(connection.execute("SELECT id, title FROM nodes WHERE kind = 'paper'"))
        rows = connection.execute(
            "SELECT id, text FROM nodes WHERE kind = 'chunk' ORDER BY id"
        ).fetchall()
    return [
        {"id": chunk_id, "text": text, "title": titles.get(source_of(chunk_id), "")}
        for chunk_id, text in rows
        if text and len(text) >= min_chars
    ]


def build(flow, chunks: list[dict], golden: list[str], checkpoint: Path) -> list[dict]:
    from rca.contracts.trace import QueryTrace
    from rca.llm.client import ChatMessage

    kept = load_checkpoint(checkpoint)
    if checkpoint.exists():  # drop failed or cut-off chunks, so their retry is not a duplicate
        tmp = checkpoint.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
        tmp.replace(checkpoint)
    done: dict[str, list[dict]] = {}
    for record in kept:
        done.setdefault(record["chunk_id"], []).append(record)
    records: list[dict] = []
    with checkpoint.open("a", encoding="utf-8") as sink:
        for number, chunk in enumerate(chunks, 1):
            if chunk["id"] in done:
                records.extend(done[chunk["id"]])
                continue
            base = {"chunk_id": chunk["id"], "paper": source_of(chunk["id"])}
            title = chunk["title"].replace("_", " ")
            messages = [
                ChatMessage(role="system", content=QUESTION_SYSTEM),
                ChatMessage(role="user", content=f"Paper: {title}\n\nPassage:\n{chunk['text']}"),
            ]
            try:
                questions = parse_questions(flow.llm.chat_json(messages, QUESTION_SCHEMA).text)
            except Exception as exc:  # one failed call skips one chunk, not the run
                questions, error = None, f"{type(exc).__name__}: {exc}"
            else:
                error = "malformed question reply"
            if questions is None:
                chunk_records = [base | {"kind": None, "error": error}]
            elif not questions:
                chunk_records = [base | {"kind": None, "skipped": "no questions"}]
            else:
                chunk_records = [
                    _answer(flow, base, kind, q, golden, QueryTrace) for kind, q in questions
                ]
            chunk_records = [r | {"of": len(chunk_records)} for r in chunk_records]
            for record in chunk_records:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"{number}/{len(chunks)} {record.get('kind')} hedged={record.get('hedged')}"
                    f" teacher={record.get('teacher')} {record.get('seconds')}s",
                    flush=True,
                )
            sink.flush()
            records.extend(chunk_records)
    return records


def _answer(flow, base: dict, kind: str, question: str, golden: list[str], trace_type) -> dict:
    record = base | {"kind": kind, "question": question}
    if too_close(question, golden):
        return record | {"skipped": "close to a golden question"}
    started = time.perf_counter()
    try:
        answer = flow.generate_answer(question, trace=trace_type(query=question))
    except Exception as exc:
        return record | {"error": f"{type(exc).__name__}: {exc}"}
    stages = answer.trace.stages if answer.trace else []
    return record | {
        "answer": answer.answer,
        "outcome": answer.outcome.value,
        # A gate refused before the model call: the facts check never sees those.
        "model_ran": any(stage.name == "llm_generate" for stage in stages),
        "hedged": any(stage.name == "facts_check" for stage in stages),
        "teacher": teacher_status(answer.trace),
        "teacher_ms": next((round(s.duration_ms) for s in stages if s.name == "facts_check"), None),
        "seconds": round(time.perf_counter() - started, 2),
    }


# --- train and test ---------------------------------------------------------------------------


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(path: str, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1).to(device)
    return tokenizer, model


def _encode(tokenizer, inputs: list[tuple[str, str]], device: str):
    questions, answers = zip(*inputs, strict=True)
    batch = tokenizer(
        list(questions),
        list(answers),
        truncation="only_second",
        max_length=512,
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in batch.items()}


def probabilities(tokenizer, model, inputs: list[tuple[str, str]], device: str, batch: int = 16):
    """P(decline) per input. The model's logit means "useful", the direction of the
    reranker's "relevant" it starts from, so P(decline) is 1 - sigmoid."""
    import torch

    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch):
            logits = model(**_encode(tokenizer, inputs[start : start + batch], device)).logits
            out.extend((1 - torch.sigmoid(logits.squeeze(-1))).tolist())
    return out


def train_epochs(pairs, epochs: int, device: str, seed: int, lr: float = 2e-5, batch: int = 16):
    """Yield (epoch, tokenizer, model) after each epoch of binary cross-entropy training."""
    import torch

    torch.manual_seed(seed)
    tokenizer, model = load_model(BASE_MODEL, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    order = random.Random(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = pairs[:]
        order.shuffle(shuffled)
        for start in range(0, len(shuffled), batch):
            inputs, declines = zip(*shuffled[start : start + batch], strict=True)
            logits = model(**_encode(tokenizer, list(inputs), device)).logits.squeeze(-1)
            useful = torch.tensor([1 - d for d in declines], dtype=torch.float32, device=device)
            loss = loss_fn(logits, useful)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        yield epoch, tokenizer, model


def hand_inputs_labels(items: list[dict]) -> tuple[list[tuple[str, str]], list[int]]:
    """Student inputs and labels (1 = `not_found`; `answered` and `partial` are useful)."""
    return (
        [student_input(i["question"], i["answer"]) for i in items],
        [int(i["label"] == "not_found") for i in items],
    )


def train(args) -> int:
    hand = hand_labelled(args.dev_labels) + hand_labelled(args.test_labels)
    hand_questions = [item["question"] for item in hand]
    hedged = [r for r in load_checkpoint(args.data) if r.get("hedged")]
    # The build already dropped near-copies of golden questions; this stricter cut also
    # drops paraphrases of the hand-labelled ones (the review found 0.5 lets them through).
    close = [r for r in hedged if too_close(r["question"], hand_questions, args.close)]
    train_records, held_records = split_by_paper(
        [r for r in hedged if r not in close], args.held_out, args.seed
    )
    train_pairs, held_pairs = training_pairs(train_records), training_pairs(held_records)
    dev_inputs, dev_labels = hand_inputs_labels(hand_labelled(args.dev_labels))
    device = _device()
    print(
        f"train {len(train_pairs)} ({sum(label for _, label in train_pairs)} declines),"
        f" held out {len(held_pairs)}, dropped as close to a hand-labelled question"
        f" {len(close)}, dev hand labels {len(dev_inputs)}, device {device}"
    )
    trained_questions = [r["question"] for r in train_records]
    nearest_trained = {
        item["question"]: nearest(item["question"], trained_questions)
        for item in hand_labelled(args.test_labels)
    }
    epochs, best = [], None
    started = time.perf_counter()
    for epoch, tokenizer, model in train_epochs(train_pairs, args.epochs, device, args.seed):
        dev_probs = probabilities(tokenizer, model, dev_inputs, device)
        threshold = pick_threshold(dev_probs, dev_labels)
        held_probs = probabilities(tokenizer, model, [i for i, _ in held_pairs], device)
        row = {
            "epoch": epoch,
            "threshold": threshold,  # unrounded: the one these rows were scored with
            "dev": agreement(dev_labels, predict(dev_probs, threshold)),
            "held_out_teacher": agreement(
                [label for _, label in held_pairs], predict(held_probs, threshold)
            ),
            "seconds": round(time.perf_counter() - started, 1),
        }
        print(json.dumps(row), flush=True)
        epochs.append(row)
        # Dev alone picks the epoch (ties: the earlier one); held-out stays a clean number.
        key = _count(row["dev"]["agree"])
        if best is None or key > best[0]:
            best = (key, row)
            args.out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
    chosen = best[1]
    (args.out / "student.json").write_text(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "trained_on": f"{platform.node()} ({device})",
                "data": str(args.data),
                "seed": args.seed,
                "held_out_papers": sorted({r["paper"] for r in held_records}),
                "train_pairs": len(train_pairs),
                "held_out_pairs": len(held_pairs),
                "dropped_close_to_hand_labels": len(close),
                "close_threshold": args.close,
                "nearest_trained_question_per_test_item": nearest_trained,
                "epochs": epochs,
                "chosen": chosen,
            },
            indent=2,
        )
    )
    print(f"chosen epoch {chosen['epoch']}, threshold {chosen['threshold']} -> {args.out}")
    return 0


def _count(fraction: str) -> int:
    return int(fraction.split("/")[0])


def test(args) -> int:
    meta = json.loads((args.model / "student.json").read_text())
    threshold = meta["chosen"]["threshold"]
    inputs, labels = hand_inputs_labels(hand_labelled(args.test_labels))
    result: dict[str, Any] = {"threshold": threshold, "n": len(inputs)}
    for device in dict.fromkeys(("cpu", _device())):
        tokenizer, model = load_model(str(args.model), device)
        probabilities(tokenizer, model, inputs[:2], device)  # warm-up
        times, probs = [], []
        for one in inputs:
            started = time.perf_counter()
            probs.extend(probabilities(tokenizer, model, [one], device))
            times.append((time.perf_counter() - started) * 1000)
        result[device] = {
            "median_ms": round(statistics.median(times), 1),
            "test": agreement(labels, predict(probs, threshold)),
        }
    meta["test"] = result
    (args.model / "student.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--library", type=Path, required=True, help="a copy of the library")
    b.add_argument("--chunks", type=int, default=30)
    b.add_argument("--seed", type=int, default=20261121)
    b.add_argument("--out", type=Path, required=True, help="checkpoint JSONL")
    t = commands.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True, help="directory for the chosen model")
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--held-out", type=int, default=5)
    t.add_argument("--seed", type=int, default=20261121)
    t.add_argument(
        "--close", type=float, default=0.35, help="Jaccard cut to hand-labelled questions"
    )
    t.add_argument("--dev-labels", type=Path, default=ROOT / "eval/answer_labels_fulltext_dev.json")
    t.add_argument(
        "--test-labels", type=Path, default=ROOT / "eval/answer_labels_fulltext_test.json"
    )
    s = commands.add_parser("test")
    s.add_argument("--model", type=Path, required=True)
    s.add_argument(
        "--test-labels", type=Path, default=ROOT / "eval/answer_labels_fulltext_test.json"
    )
    args = parser.parse_args(argv)

    if args.command == "train":
        return train(args)
    if args.command == "test":
        return test(args)

    from rca.flows.generate_flow import GenerateFlow

    settings = library_settings(args.library.resolve())
    flow = GenerateFlow(settings=settings)
    check_library(flow, settings)
    if not settings.facts_check_enabled:
        raise SystemExit("The facts check is off; it is the teacher, so nothing was asked.")
    chunks = _chunks(flow.retrieve_flow.graph_store)
    sample = first_chunks(chunks, args.chunks, args.seed)
    golden = [pair["question"] for pair in json.loads((ROOT / "eval/golden.json").read_text())]
    records = build(flow, sample, golden, args.out)
    answered = [r for r in records if r.get("model_ran")]
    print(
        f"{len(records)} records, {len(answered)} reached the model,"
        f" {sum(1 for r in answered if r['hedged'])} hedged,"
        f" {sum(1 for r in answered if r.get('teacher'))} teacher-labelled"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
