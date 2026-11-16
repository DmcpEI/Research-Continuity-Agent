import random

from eval import sample_support_labels as sampler

A0 = "chk:pdf/a:0000"
A3 = "chk:pdf/a:0003"
A = "src:pdf/a"
B1 = "chk:pdf/b:0001"


def test_swap_prefers_a_block_from_another_paper() -> None:
    for seed in range(20):
        assert sampler.swap_target([A3], [A, A0, A3, B1], random.Random(seed)) == B1


def test_swap_falls_back_to_another_chunk_never_the_cited_papers_src_node() -> None:
    for seed in range(20):
        assert sampler.swap_target([A3], [A, A0, A3], random.Random(seed)) == A0


def test_swap_none_when_only_cited_blocks_were_shown() -> None:
    assert sampler.swap_target([A], [A], random.Random(0)) is None
    assert sampler.swap_target([A3], [A3, A], random.Random(0)) is None


def _answer(item_id: str, label: str, answer: str) -> dict:
    return {
        "item_id": item_id,
        "question_id": item_id,
        "question": "q?",
        "answerable": True,
        "answer": answer,
        "context_node_ids": [A3, B1],
        "label": label,
    }


def test_build_skips_not_found_answers_and_adds_one_swap_per_answer() -> None:
    answers = [
        _answer("x", "answered", f"One [[{A3}]]. Two [[{A3}]]. Uncited."),
        _answer("y", "not_found", f"No [[{A3}]]."),
    ]

    items = sampler.build(answers, None, seed=1)

    assert [i["kind"] for i in items] == ["natural", "natural", "natural", "swapped"]
    swap = items[-1]
    assert swap["cited_ids"] == [B1]
    assert swap["original_cited_ids"] == [A3]
    assert swap["label"] is None
    assert {i["answer_item_id"] for i in items} == {"x"}
