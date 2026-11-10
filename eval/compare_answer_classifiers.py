"""Compare answer-outcome classifiers against hand labels (#6).

Reads label files written by eval/sample_answer_labels.py and labelled by hand
(docs/EVAL.md, "Answer Outcome Labels"), runs each classifier on the saved
answers, and reports agreement split by answerable and unanswerable questions.

Classifiers:
- current: GenerateFlow's own decision (pipeline.abstained in the saved run).
- hedge: any "context does not contain"-style phrase means not_found.
- verifier: a second LLM call that reads the question and the answer and returns
  answered / partial / not_found as JSON (Ollama structured output).
- verifier_facts: a second LLM call that lists the statements in the answer that
  address the question, whether it states a gap, and whether it corrects the
  premise; the status is derived from those fields in code.
- <verifier>_on_hedge: that verifier's prediction only where the hedge rule fires.

    uv run python eval/compare_answer_classifiers.py \
        --labels eval/answer_labels_fulltext_dev.json --model qwen3.5:4b
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.sample_answer_labels import is_hedged  # noqa: E402

STATUSES = ("answered", "partial", "not_found")

VERIFIER_SYSTEM = """You grade answers written by a research assistant that may only use the user's papers.
Read the question and the answer, then classify the answer:
- "answered": it gives the information the question asks for.
- "partial": it says part of what was asked is not in the papers AND gives information that answers another part of the question; or it states plainly that the question's premise is wrong and gives the correct fact.
- "not_found": it says it cannot answer, and everything else it says is background that does not answer any part of the question, even when that background is cited. A hint the reader would have to interpret is not a premise correction. A real fact about a different setting or a different paper than the one asked about is background.
Judge only what the answer says, not whether it is true. Reply with JSON."""

VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "status": {"type": "string", "enum": list(STATUSES)},
    },
    "required": ["reason", "status"],
}


FACTS_SYSTEM = """You check an answer written by a research assistant that may only use the user's papers. Do not judge whether the answer is true.
- facts: copy each statement from the answer that gives information the question asks for, about the exact paper, method and setting named in the question. Include such statements even when the answer also says that other parts are missing. Leave out background: facts about other papers, other settings or other tasks, and general descriptions that do not answer what was asked. Use an empty list when there are none.
- gap_stated: does the answer say that some or all of what was asked is not in the papers?
- premise_corrected: does the answer state plainly that something the question assumes is wrong, and say what is true instead?
Reply with JSON."""

FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "items": {"type": "string"}},
        "gap_stated": {"type": "boolean"},
        "premise_corrected": {"type": "boolean"},
    },
    "required": ["facts", "gap_stated", "premise_corrected"],
}


def verifier_messages(question: str, answer: str, system: str = VERIFIER_SYSTEM) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n\nAnswer:\n{answer}"},
    ]


def parse_status(content: str) -> str | None:
    """Status from the verifier's JSON reply; None when it is missing or not a known status."""
    payload = _json_object(content)
    status = payload.get("status") if payload else None
    return status if status in STATUSES else None


def parse_facts(content: str) -> str | None:
    """Status derived in code from the facts reply; None when the reply is malformed.

    Facts that answer the question make it useful: partial when the answer also states
    a gap or corrects the premise, else answered. With no such facts, only a plain
    premise correction is partial.
    """
    payload = _json_object(content)
    if not payload:
        return None
    facts, gap, premise = (
        payload.get("facts"),
        payload.get("gap_stated"),
        payload.get("premise_corrected"),
    )
    if not isinstance(facts, list) or not isinstance(gap, bool) or not isinstance(premise, bool):
        return None
    if any(isinstance(fact, str) and fact.strip() for fact in facts):
        return "partial" if gap or premise else "answered"
    return "partial" if premise else "not_found"


def _json_object(content: str) -> dict | None:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


# name -> (system prompt, JSON schema, reply parser)
VERIFIERS = {
    "verifier": (VERIFIER_SYSTEM, VERIFIER_SCHEMA, parse_status),
    "verifier_facts": (FACTS_SYSTEM, FACTS_SCHEMA, parse_facts),
}


def predict_current(item: dict) -> str:
    return "not_found" if item["pipeline"].get("abstained") else "answered"


def predict_hedge(item: dict) -> str:
    return "not_found" if is_hedged(item["answer"]) else "answered"


def predict_verifier(
    item: dict, client, model: str, variant: str = "verifier"
) -> tuple[str | None, float, str]:
    """(status, latency in seconds, raw reply). Uses the client's HTTP helper for Ollama."""
    system, schema, parse = VERIFIERS[variant]
    payload = {
        "model": model,
        "messages": verifier_messages(item["question"], item["answer"], system),
        "stream": False,
        "think": False,
        "format": schema,
        "options": {"temperature": 0},
        **client._keep_alive_field(),
    }
    started = time.perf_counter()
    try:
        result = client._post_json("/api/chat", payload, timeout=120)
    except Exception as exc:  # one failed call is a missing prediction, not a lost run
        return None, time.perf_counter() - started, f"ERROR: {exc}"
    content = result.get("message", {}).get("content", "")
    return parse(content), time.perf_counter() - started, content


def gate_on_hedge(items: list[dict], predictions: dict[str, str | None]) -> dict[str, str | None]:
    """Use a verifier's prediction only where the hedge rule fires; other answers stay answers.

    Models running the verifier only on hedged answers (about 40% of them here).
    """
    return {
        i["item_id"]: predictions[i["item_id"]] if is_hedged(i["answer"]) else "answered"
        for i in items
    }


def two_class(status: str | None) -> str | None:
    """Merge partial into answered: the abstain-or-show decision GenerateFlow makes."""
    if status is None:
        return None
    return "not_found" if status == "not_found" else "useful"


def score(items: list[dict], predictions: dict[str, str | None]) -> dict:
    """Agreement of predictions with labels, overall and per answerable/unanswerable.

    A missing prediction (None) counts as a disagreement, never silently skipped.
    """
    report: dict = {}
    groups = {
        "all": items,
        "answerable": [i for i in items if i["answerable"]],
        "unanswerable": [i for i in items if not i["answerable"]],
    }
    for name, group in groups.items():
        not_found = [i for i in group if i["label"] == "not_found"]
        useful = [i for i in group if i["label"] != "not_found"]
        report[name] = {
            "n": len(group),
            "three_class": sum(predictions[i["item_id"]] == i["label"] for i in group),
            "two_class": sum(
                two_class(predictions[i["item_id"]]) == two_class(i["label"]) for i in group
            ),
            "not_found_caught": sum(predictions[i["item_id"]] == "not_found" for i in not_found),
            "not_found_total": len(not_found),
            "useful_kept": sum(two_class(predictions[i["item_id"]]) == "useful" for i in useful),
            "useful_total": len(useful),
            "missing": sum(predictions[i["item_id"]] is None for i in group),
        }
    report["confusion"] = dict(Counter(f"{i['label']}->{predictions[i['item_id']]}" for i in items))
    return report


def load_items(paths: list[Path]) -> list[dict]:
    items = []
    seen: set[str] = set()
    for path in paths:
        for item in json.loads(path.read_text(encoding="utf-8"))["items"]:
            if item.get("label") not in STATUSES:
                raise ValueError(f"{path}: {item['item_id']} has no label")
            if item["item_id"] in seen:
                raise ValueError(f"{path}: duplicate item {item['item_id']}")
            seen.add(item["item_id"])
            items.append(item)
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score answer classifiers against labels.")
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--model", default=None, help="Verifier model (default: chat model).")
    parser.add_argument(
        "--verifier",
        action="append",
        choices=sorted(VERIFIERS),
        help="LLM verifier variant(s) to run; none means rule classifiers only.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write predictions + report.")
    args = parser.parse_args(argv)

    items = load_items(args.labels)
    predictions: dict[str, dict[str, str | None]] = {
        "current": {i["item_id"]: predict_current(i) for i in items},
        "hedge": {i["item_id"]: predict_hedge(i) for i in items},
    }
    replies: dict[str, dict[str, dict]] = {}
    if args.verifier:
        from rca.config.settings import Settings
        from rca.llm import factory
        from rca.llm.client import OllamaLLMClient

        settings = Settings()
        model = args.model or settings.generation_model
        client = factory.get_llm_client(settings)
        # predict_verifier posts Ollama's native /api/chat with structured output.
        if not (isinstance(client, OllamaLLMClient) and client.api_style == "ollama"):
            parser.error(
                "--verifier needs the Ollama backend: RCA_LLM_BACKEND=ollama and a non-empty "
                f"RCA_GENERATION_MODEL (got {type(client).__name__})"
            )
        for variant in args.verifier:
            predictions[variant], replies[variant] = {}, {}
            for item in items:
                status, latency, raw = predict_verifier(item, client, model, variant)
                predictions[variant][item["item_id"]] = status
                replies[variant][item["item_id"]] = {
                    "status": status,
                    "latency_s": latency,
                    "raw": raw,
                }
            predictions[f"{variant}_on_hedge"] = gate_on_hedge(items, predictions[variant])
            latency = statistics.median(r["latency_s"] for r in replies[variant].values())
            print(f"{variant} {model}: median {latency:.2f}s per answer")

    reports = {name: score(items, preds) for name, preds in predictions.items()}
    for name, report in reports.items():
        print(f"\n{name}")
        for group in ("all", "answerable", "unanswerable"):
            r = report[group]
            print(
                f"  {group:12} n={r['n']:3}  3-class {r['three_class']}/{r['n']}"
                f"  2-class {r['two_class']}/{r['n']}"
                f"  not_found caught {r['not_found_caught']}/{r['not_found_total']}"
                f"  useful kept {r['useful_kept']}/{r['useful_total']}"
                f"  missing {r['missing']}"
            )
    if args.output:
        payload = {
            "labels": [str(p) for p in args.labels],
            "predictions": predictions,
            "verifier_prompts": {
                name: {"system": VERIFIERS[name][0], "schema": VERIFIERS[name][1]}
                for name in args.verifier or []
            },
            "verifier_replies": replies,
            "reports": reports,
        }
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
