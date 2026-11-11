"""Facts check (#6): does a hedged reply give anything the question asked for?

The prompt, schema and parser are the ones measured as `verifier_facts` in docs/EVAL.md
("Answer classifiers"); eval/compare_answer_classifiers.py imports them from here, so the
eval measures what ships.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from rca.llm.client import ChatMessage

# The one definition of a hedged reply ("the context does not contain ...", "cannot answer").
# It is also the eval sampler's weighting rule. It covers every phrase of the earlier list in
# GenerateFlow ("not provided in", "does not contain information", ...) plus the sampler's
# "<the context> does not <state>" form, which on the 211 labelled answers adds 41 replies.
_HEDGE = re.compile(
    r"\b(context|text|sources?|documents?|papers?|passages?|excerpts?|chunks?|library"
    r"|knowledge base)\b[^.]{0,40}?\b"
    r"(does not|doesn't|do not|don't|did not|didn't)\s+"
    r"(contain|state|report|mention|provide|specify|include|describe|discuss|answer|detail)"
    r"|\bno (relevant )?information\b|\b(cannot|can't|could not|couldn't) (find|answer|be determined)\b"
    r"|\bnot (mentioned|provided)\b|\bdoes not contain information\b",
    re.IGNORECASE,
)

FactsStatus = Literal["answered", "partial", "not_found"]

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


def is_hedged(answer: str) -> bool:
    return bool(_HEDGE.search(answer or ""))


def facts_messages(question: str, answer: str) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=FACTS_SYSTEM),
        ChatMessage(role="user", content=f"Question: {question}\n\nAnswer:\n{answer}"),
    ]


def parse_facts(content: str) -> FactsStatus | None:
    """Status derived in code from the facts reply; None when the reply is malformed.

    Facts that answer the question make it useful: partial when the answer also states
    a gap or corrects the premise, else answered. With no such facts, only a plain
    premise correction is partial.
    """
    payload = json_object(content)
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


def json_object(content: str) -> dict | None:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None
