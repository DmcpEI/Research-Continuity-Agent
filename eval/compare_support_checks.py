"""Compare citation-support checks against hand labels (step 12).

Reads label files written by eval/sample_support_labels.py and labelled by hand (docs/EVAL.md,
"Citation Support Labels"), runs each check on the cited sentences, and reports how many
not-supported sentences it catches and how many supported ones it keeps, per kind (natural,
swapped), with Cohen's kappa and the added time per answer.

Checks:
- overlap: GenerateFlow's word-overlap rule (two shared content words), per sentence.
- llm_quote: one call per cited sentence with the cited text; the model copies a quote, then
  gives a verdict. Supported only if the verdict says so and the quote is in the text.
- llm_quote_verdict_only: the same replies, verdict only (no quote check).
- llm_verdict: one call per cited sentence that asks for the verdict alone (shorter replies).

    uv run python eval/compare_support_checks.py --labels eval/support_labels_dev.json \
        --graph-db <library copy>/graph.sqlite3 --check llm_quote --check llm_verdict
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca.flows.facts_check import json_object  # noqa: E402
from rca.flows.generate_flow import GenerateFlow  # noqa: E402
from rca.llm.client import ChatMessage  # noqa: E402

LABELS = ("supported", "partial", "unsupported", "no_claim")
# What the model may answer: nothing_to_check is a sentence that only says something is missing.
VERDICTS = ("supported", "partial", "unsupported", "nothing_to_check")

_RULES = """Use only the passage, not what you know. Judge only what the sentence says the passage states: ignore any part saying that something is not stated, not described or missing.
- verdict: "supported" if the passage states everything the sentence says (paraphrase is fine); "partial" if it states only part of it (a number, name, cause, comparison or attribution in the sentence is missing from the passage or different); "unsupported" if the passage does not state it or says something else; "nothing_to_check" if the sentence only says that something is not stated."""

QUOTE_SYSTEM = f"""You check one sentence written by a research assistant against the passage it cites. {_RULES}
- quote (before the verdict): copy, word for word, the shortest part of the passage (at most 30 words) that states it. Use an empty string if the passage does not state it.
Reply with JSON."""

VERDICT_SYSTEM = f"""You check one sentence written by a research assistant against the passage it cites. {_RULES}
Reply with JSON."""

QUOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "quote": {"type": "string"},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
    },
    "required": ["quote", "verdict"],
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": list(VERDICTS)}},
    "required": ["verdict"],
}


def _tokens(text: str) -> list[str]:
    """Words, lowercased, ligatures undone and PDF line-break hyphens ("pack-\\ning") joined."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.findall(r"\w+", re.sub(r"(\w)-\s+(\w)", r"\1\2", text))


def quote_in(quote: str, passage: str, coverage: float = 0.8) -> bool:
    """Most of the quote's 4-word runs appear in the passage, so a copied (or stitched) quote
    passes and an invented one does not. Quotes under four words show nothing and fail."""
    words = _tokens(quote)
    if len(words) < 4:
        return False
    text = f" {' '.join(_tokens(passage))} "
    runs = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
    return sum(f" {run} " in text for run in runs) / len(runs) >= coverage


def parse_reply(content: str, with_quote: bool) -> tuple[str, str] | None:
    """(quote, verdict); quote is "" for the verdict-only schema. None when malformed."""
    payload = json_object(content)
    if not payload or payload.get("verdict") not in VERDICTS:
        return None
    quote = payload.get("quote") if with_quote else ""
    return (quote, payload["verdict"]) if isinstance(quote, str) else None


def decide(reply: tuple[str, str] | None, passage: str, check_quote: bool) -> bool | None:
    """True when the sentence does not block "Supported"; None when the reply was unusable."""
    if reply is None:
        return None
    quote, verdict = reply
    if verdict == "nothing_to_check":
        return True
    return verdict == "supported" and (not check_quote or quote_in(quote, passage))


def passage_for(cited_ids: list[str], nodes: dict[str, tuple[str, str]]) -> str:
    """The cited blocks as the answer model saw them: id and title, then the full text."""
    return "\n\n".join(
        f"[{node_id}] {nodes[node_id][0]}\n{nodes[node_id][1]}"
        for node_id in cited_ids
        if node_id in nodes
    )


def messages(system: str, sentence: str, passage: str) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=f"Passage:\n{passage}\n\nSentence: {sentence}"),
    ]


def predict_overlap(item: dict, nodes: dict[str, tuple[str, str]]) -> bool:
    tokens = GenerateFlow._content_tokens(item["sentence"])
    return bool(tokens) and GenerateFlow._has_basic_support(
        tokens, passage_for(item["cited_ids"], nodes)
    )


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    observed = sum(a == b for a, b in pairs) / n
    p_true = sum(a for a, _ in pairs) / n
    q_true = sum(b for _, b in pairs) / n
    expected = p_true * q_true + (1 - p_true) * (1 - q_true)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def score(items: list[dict], predictions: dict[str, bool | None]) -> dict:
    """Agreement on cited sentences that make a claim; a missing prediction is not supported."""
    report: dict = {}
    claims = [i for i in items if i["cited_ids"] and i["label"] != "no_claim"]
    for name, group in (
        ("all", claims),
        ("natural", [i for i in claims if i["kind"] == "natural"]),
        ("swapped", [i for i in claims if i["kind"] == "swapped"]),
    ):
        truth = [(i["label"] == "supported", bool(predictions[i["item_id"]])) for i in group]
        bad = [p for t, p in truth if not t]
        good = [p for t, p in truth if t]
        report[name] = {
            "n": len(group),
            "caught": sum(not p for p in bad),
            "not_supported": len(bad),
            "kept": sum(good),
            "supported": len(good),
            "kappa": cohen_kappa(truth),
            "missing": sum(predictions[i["item_id"]] is None for i in group),
        }
    no_claim = [i for i in items if i["cited_ids"] and i["label"] == "no_claim"]
    report["cited_no_claim_flagged"] = sum(not predictions[i["item_id"]] for i in no_claim)
    report["cited_no_claim"] = len(no_claim)
    return report


def load_items(paths: list[Path]) -> list[dict]:
    items, seen = [], set()
    for path in paths:
        for item in json.loads(path.read_text(encoding="utf-8"))["items"]:
            if item.get("label") not in LABELS:
                raise ValueError(f"{path}: {item['item_id']} has no label")
            if item["item_id"] in seen:
                raise ValueError(f"{path}: duplicate item {item['item_id']}")
            seen.add(item["item_id"])
            items.append(item)
    return items


def load_nodes(graph_db: Path, items: list[dict]) -> dict[str, tuple[str, str]]:
    from rca.store.graph_store import GraphStore

    store = GraphStore(graph_db)
    nodes = {}
    for node_id in {i for item in items for i in item["cited_ids"]}:
        node = store.get_node(node_id)
        if node is not None:
            nodes[node_id] = (node.title, node.text or "")
    return nodes


def timed(call):
    started = time.perf_counter()
    try:
        return call().text, time.perf_counter() - started
    except Exception as exc:  # one failed call is a missing prediction, not a lost run
        return f"ERROR: {exc}", time.perf_counter() - started


# check -> (system prompt, schema, the reply carries a quote)
CHECKS = {
    "llm_quote": (QUOTE_SYSTEM, QUOTE_SCHEMA, True),
    "llm_verdict": (VERDICT_SYSTEM, VERDICT_SCHEMA, False),
}


def run_llm(items: list[dict], nodes, client, checks: list[str]) -> tuple[dict, dict, dict]:
    """Predictions, raw replies and seconds per answer (natural sentences only) per check."""
    predictions: dict[str, dict] = defaultdict(dict)
    replies: dict[str, dict] = defaultdict(dict)
    seconds: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for check in checks:
        system, schema, with_quote = CHECKS[check]
        for item in (i for i in items if i["cited_ids"]):
            passage = passage_for(item["cited_ids"], nodes)
            raw, took = timed(
                lambda: client.chat_json(messages(system, item["sentence"], passage), schema)
            )
            reply = parse_reply(raw, with_quote)
            predictions[check][item["item_id"]] = decide(reply, passage, with_quote)
            if with_quote:
                predictions[f"{check}_verdict_only"][item["item_id"]] = decide(
                    reply, passage, False
                )
            replies[check][item["item_id"]] = {"raw": raw, "seconds": took}
            if item["kind"] == "natural":
                seconds[check][item["answer_item_id"]] += took
    return predictions, replies, seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score citation-support checks against labels.")
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--graph-db", type=Path, required=True, help="The library's graph.sqlite3.")
    parser.add_argument("--check", action="append", choices=sorted(CHECKS), default=[])
    parser.add_argument("--output", type=Path, default=None, help="Write predictions + report.")
    args = parser.parse_args(argv)

    items = load_items(args.labels)
    nodes = load_nodes(args.graph_db, items)
    missing = sorted({i for item in items for i in item["cited_ids"]} - set(nodes))
    if missing:
        parser.error(f"cited blocks not in {args.graph_db}: {missing[:3]}")
    predictions: dict[str, dict] = {
        "overlap": {i["item_id"]: predict_overlap(i, nodes) for i in items if i["cited_ids"]}
    }
    replies: dict = {}
    seconds: dict = {}
    if args.check:
        from rca.config.settings import Settings
        from rca.llm import factory

        client = factory.get_llm_client(Settings())
        llm_predictions, replies, seconds = run_llm(items, nodes, client, args.check)
        predictions.update(llm_predictions)

    reports = {name: score(items, preds) for name, preds in predictions.items()}
    for name, report in reports.items():
        per_answer = list(seconds.get(name.removesuffix("_verdict_only"), {}).values())
        median = f"  median {statistics.median(per_answer):.1f}s/answer" if per_answer else ""
        print(f"\n{name}{median}")
        for group in ("all", "natural", "swapped"):
            r = report[group]
            kappa = "n/a" if r["kappa"] is None else f"{r['kappa']:.2f}"
            print(
                f"  {group:8} n={r['n']:3}  caught {r['caught']}/{r['not_supported']}"
                f"  kept {r['kept']}/{r['supported']}  kappa {kappa}  missing {r['missing']}"
            )
        print(
            f"  cited no_claim flagged {report['cited_no_claim_flagged']}/{report['cited_no_claim']}"
        )
    if args.output:
        payload = {
            "labels": [str(p) for p in args.labels],
            "prompts": {name: {"system": s, "schema": sc} for name, (s, sc, _) in CHECKS.items()},
            "predictions": predictions,
            "replies": replies,
            "seconds_per_answer": seconds,
            "reports": reports,
        }
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
