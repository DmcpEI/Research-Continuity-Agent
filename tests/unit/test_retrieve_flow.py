from __future__ import annotations

from rca.flows.retrieve_flow import RetrieveFlow


def test_lexical_score_rewards_title_substrings_over_generic_text_overlap() -> None:
    flow = object.__new__(RetrieveFlow)
    query = "What are the two main components of JamPacker and what problem does each solve?"
    query_tokens = flow._tokenize(query)

    jampacker_score = flow._lexical_score(
        query_tokens,
        "Jampacker_An_Efficient_and_Reliable_Robotic_Bin_Packing_System_for_Cuboid_Objects",
        "This paper studies the bin packing problem and explains how each module solves reliability issues.",
    )
    generic_score = flow._lexical_score(
        query_tokens,
        "Comprehensive_Review_of_Robotized_Freight_Packing",
        "This survey covers components, main problem formulations, and how each solver addresses them in packing.",
    )

    assert jampacker_score > generic_score


def test_lexical_score_does_not_match_partial_title_words() -> None:
    flow = object.__new__(RetrieveFlow)
    query_tokens = {"mode"}

    score = flow._lexical_score(
        query_tokens,
        "Vision_Language_Models_A_Survey",
        "",
    )

    assert score == 0.0


def test_lexical_score_does_not_match_partial_text_words() -> None:
    flow = object.__new__(RetrieveFlow)
    query_tokens = {"jampack"}

    score = flow._lexical_score(
        query_tokens,
        "",
        "This review mentions Jampacker systems in passing.",
    )

    assert score == 0.0
