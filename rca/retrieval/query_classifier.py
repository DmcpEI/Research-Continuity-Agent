"""Query type classifier for retrieval routing."""

from __future__ import annotations

import re
from enum import StrEnum


class QueryType(StrEnum):
    proper_noun = "proper_noun"
    conceptual = "conceptual"
    hybrid = "hybrid"


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+_.-]*")
AUTHOR_LIKE_PATTERN = re.compile(r"\b[A-Z][a-z]+ et al\.?", re.IGNORECASE)
ARXIV_ID_PATTERN = re.compile(r"\b\d{4}\.\d{4,5}\b")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
VERSION_PATTERN = re.compile(r"\bv\d+(?:\.\d+)*\b", re.IGNORECASE)
QUESTION_WORDS = {
    "are",
    "can",
    "compare",
    "describe",
    "does",
    "explain",
    "how",
    "is",
    "list",
    "results",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


def classify_query(query: str) -> QueryType:
    """Classify a query as proper_noun, conceptual, or hybrid.

    Strong lexical anchors such as acronyms, mixed-case model names, versions,
    years, and author/year references push the query toward `proper_noun`.
    Plain capitalized words are treated as weaker signals so sentence casing
    alone does not dominate the decision.
    """

    strong_signals = 0
    weak_signals = 0
    tokens = TOKEN_PATTERN.findall(query)

    if AUTHOR_LIKE_PATTERN.search(query):
        strong_signals += 1
    if ARXIV_ID_PATTERN.search(query):
        strong_signals += 1

    for token in tokens:
        normalized = token.casefold()
        alpha_only = "".join(char for char in token if char.isalpha())

        if YEAR_PATTERN.fullmatch(token) or VERSION_PATTERN.fullmatch(token):
            strong_signals += 1
            continue
        if alpha_only and alpha_only.isupper() and len(alpha_only) >= 2:
            strong_signals += 1
            continue
        if any(char.isupper() for char in token[1:]):
            strong_signals += 1
            continue
        if any(char.isdigit() for char in token) and any(char.isalpha() for char in token):
            strong_signals += 1
            continue
        if token[:1].isupper() and token[1:].islower() and normalized not in QUESTION_WORDS:
            weak_signals += 1

    if strong_signals >= 1 or weak_signals >= 2:
        return QueryType.proper_noun
    if weak_signals == 1:
        return QueryType.hybrid
    return QueryType.conceptual
