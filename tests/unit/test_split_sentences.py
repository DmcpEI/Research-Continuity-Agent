from eval.sample_support_labels import Sentence, split_sentences

A = "chk:pdf/paper_a:0003"
B = "src:pdf/paper_b"


def test_split_sentences_attaches_markers_to_their_sentence() -> None:
    answer = f"Jampack packs objects densely [[{A}]]. It uses two stages [[{B}]]."

    assert split_sentences(answer) == [
        Sentence(text="Jampack packs objects densely.", cited_ids=[A]),
        Sentence(text="It uses two stages.", cited_ids=[B]),
    ]


def test_split_sentences_marker_after_the_full_stop_belongs_to_that_sentence() -> None:
    answer = f"The first claim. [[{A}]] The second claim is uncited."

    assert split_sentences(answer) == [
        Sentence(text="The first claim.", cited_ids=[A]),
        Sentence(text="The second claim is uncited.", cited_ids=[]),
    ]


def test_split_sentences_keeps_abbreviations_and_figure_numbers_whole() -> None:
    answer = (
        f"As shown by Wang et al. in Fig. 4(a), packing improves, e.g. for boxes [[{A}]]. "
        "It is compared with LCGA vs. EP-FFD."
    )

    assert [s.text for s in split_sentences(answer)] == [
        "As shown by Wang et al. in Fig. 4(a), packing improves, e.g. for boxes.",
        "It is compared with LCGA vs. EP-FFD.",
    ]


def test_split_sentences_one_per_list_item_without_bullets_or_numbers() -> None:
    answer = (
        "SGVL uses two mechanisms:\n\n"
        f"1. **Scene graph tokens** adapt the model [[{A}]]\n"
        f"*   A graph loss aligns patches [[{B}]]\n"
        "- Plain item"
    )

    assert split_sentences(answer) == [
        Sentence(text="SGVL uses two mechanisms:", cited_ids=[]),
        Sentence(text="**Scene graph tokens** adapt the model", cited_ids=[A]),
        Sentence(text="A graph loss aligns patches", cited_ids=[B]),
        Sentence(text="Plain item", cited_ids=[]),
    ]


def test_split_sentences_several_markers_keep_order_without_duplicates() -> None:
    answer = f"Both papers agree [[{A}]][[{B}]] [[{A}]]."

    assert split_sentences(answer) == [
        Sentence(text="Both papers agree.", cited_ids=[A, B]),
    ]


def test_split_sentences_question_and_exclamation_end_sentences() -> None:
    assert [s.text for s in split_sentences("Why? Because it works! Done.")] == [
        "Why?",
        "Because it works!",
        "Done.",
    ]


def test_split_sentences_empty_and_marker_only_input() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n\n ") == []
    assert split_sentences(f"[[{A}]]") == [Sentence(text="", cited_ids=[A])]


def test_split_sentences_markers_on_their_own_line_cite_the_sentence_before() -> None:
    answer = f"First claim.\n\nLast claim of the paragraph.\n\n[[{A}]] [[{B}]].\n"

    assert split_sentences(answer) == [
        Sentence(text="First claim.", cited_ids=[]),
        Sentence(text="Last claim of the paragraph.", cited_ids=[A, B]),
    ]


def test_split_sentences_lowercase_after_full_stop_does_not_split() -> None:
    # "approx. two" and "No. of items" are not sentence ends; a capital or marker-free
    # lowercase continuation stays in one sentence.
    answer = "It takes approx. two seconds per item."

    assert [s.text for s in split_sentences(answer)] == ["It takes approx. two seconds per item."]
