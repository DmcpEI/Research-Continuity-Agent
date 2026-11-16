"""Draw answer sentences for hand-labelling citation support (step 12).

Reads an answer-outcome label file (eval/answer_labels_fulltext_*.json), keeps the answers
labelled useful (not `not_found`), splits each into sentences, and adds one swapped
citation per answer: a cited sentence re-paired with another block the model saw. Writes the
items with empty label fields; see docs/EVAL.md, "Citation Support Labels". The file stores node
IDs, not paper text.

    uv run python eval/sample_support_labels.py eval/answer_labels_fulltext_test.json \
        --output eval/support_labels_test.json
    uv run python eval/sample_support_labels.py eval/answer_labels_fulltext_dev.json \
        --answers 25 --output eval/support_labels_dev.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from pydantic import BaseModel, Field

_MARKER = re.compile(r"\[\[((?:src|chk):[^\]]+)\]\]")
# A marker group right after the full stop ("claim. [[id]]") moves in front of it.
_MARKERS_AFTER_END = re.compile(r"([.!?])((?:[ \t]*\[\[(?:src|chk):[^\]]+\]\])+)")
_BULLET = re.compile(r"^(?:[*+-]|\d+[.)])\s+")
_SENTENCE_END = re.compile(r"[.!?]\s+(?=[A-Z\"'(*_`])")
_ABBREVIATIONS = {"e.g", "i.e", "al", "fig", "figs", "eq", "eqs", "vs", "cf", "sec", "tab", "no"}


class Sentence(BaseModel):
    text: str  # markers removed
    cited_ids: list[str] = Field(default_factory=list)  # in order, without repeats


def split_sentences(answer: str) -> list[Sentence]:
    """One sentence per line or per sentence end, with the IDs it cites."""
    text = _MARKERS_AFTER_END.sub(r"\2\1", answer or "")
    sentences = []
    for line in text.splitlines():
        line = _BULLET.sub("", line.strip())
        for piece in _split_line(line):
            ids = list(dict.fromkeys(_MARKER.findall(piece)))
            clean = re.sub(r"\s+([.!?,;:])", r"\1", _MARKER.sub("", piece))
            clean = re.sub(r"\s{2,}", " ", clean).strip()
            if clean.strip(".") == "" and ids and sentences:
                # Markers alone on a line cite the sentence before them.
                last = sentences[-1]
                last.cited_ids = list(dict.fromkeys(last.cited_ids + ids))
            elif clean or ids:
                sentences.append(Sentence(text=clean, cited_ids=ids))
    return sentences


def _split_line(line: str) -> list[str]:
    pieces, start = [], 0
    for match in _SENTENCE_END.finditer(line):
        words = line[start : match.start()].split()
        if words and words[-1].lower() in _ABBREVIATIONS:
            continue
        pieces.append(line[start : match.start() + 1])
        start = match.end()
    pieces.append(line[start:])
    return [piece.strip() for piece in pieces if piece.strip()]


LABELS = ("supported", "partial", "unsupported", "no_claim")
LABEL_FIELDS = {"label": None, "unsure": False, "note": "", "labeled_by": None}


def paper_of(node_id: str) -> str:
    """src:pdf/x for both src:pdf/x and chk:pdf/x:0003."""
    return "src:" + node_id[4:].rsplit(":", 1)[0] if node_id.startswith("chk:") else node_id


def swap_target(cited_ids: list[str], context_ids: list[str], rng: random.Random) -> str | None:
    """A block the model saw from another paper; else another chunk of the same paper.

    Never the cited paper's src: node or a cited block, whose text the sentence came from.
    """
    cited_papers = {paper_of(node_id) for node_id in cited_ids}
    other_paper = [n for n in context_ids if paper_of(n) not in cited_papers]
    same_paper = [
        n
        for n in context_ids
        if n.startswith("chk:") and n not in cited_ids and n not in other_paper
    ]
    options = other_paper or same_paper
    return rng.choice(options) if options else None


def items_for_answer(answer: dict, rng: random.Random) -> list[dict]:
    base = {
        "question_id": answer["question_id"],
        "question": answer["question"],
        "answerable": answer["answerable"],
        "answer_item_id": answer["item_id"],
        "context_node_ids": answer["context_node_ids"],
    }
    sentences = split_sentences(answer["answer"])
    items = [
        {
            **base,
            "item_id": f"{answer['item_id']}#{index}",
            "kind": "natural",
            "sentence_index": index,
            "sentence": sentence.text,
            "cited_ids": sentence.cited_ids,
            **LABEL_FIELDS,
        }
        for index, sentence in enumerate(sentences)
    ]
    cited = [item for item in items if item["cited_ids"]]
    rng.shuffle(cited)
    for item in cited:
        target = swap_target(item["cited_ids"], answer["context_node_ids"], rng)
        if target:
            items.append(
                {
                    **item,
                    "item_id": f"{item['item_id']}/swap",
                    "kind": "swapped",
                    "original_cited_ids": item["cited_ids"],
                    "cited_ids": [target],
                    **LABEL_FIELDS,
                }
            )
            break
    return items


def build(answers: list[dict], count: int | None, seed: int) -> list[dict]:
    rng = random.Random(seed)
    useful = [answer for answer in answers if answer["label"] != "not_found"]
    if count is not None:
        useful = sorted(rng.sample(useful, count), key=lambda answer: answer["item_id"])
    return [item for answer in useful for item in items_for_answer(answer, rng)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw sentences for citation-support labels.")
    parser.add_argument("answers", type=Path, help="An answer_labels_fulltext_*.json file.")
    parser.add_argument("--answers", dest="count", type=int, default=None, help="Sample size.")
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    source = json.loads(args.answers.read_text(encoding="utf-8"))
    items = build(source["items"], args.count, args.seed)
    payload = {
        "source": str(args.answers),
        "seed": args.seed,
        "answers": len({item["answer_item_id"] for item in items}),
        "labels": list(LABELS),
        "context_note": "Judged against the full text of the cited blocks, as the model saw "
        "them (RetrievalHit.text); uncited sentences against every block in context_node_ids.",
        "items": items,
    }
    args.output.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(items)} items from {payload['answers']} answers -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
