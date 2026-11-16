import pytest

from eval import compare_support_checks as csc

PASSAGE = (
    "[chk:pdf/a:0001] A\nThe robot reaches 98% success in ﬁve-item pack-\ning, up from 83%. "
    "It models the historical features with an explicit policy head."
)


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("98% success in five-item packing", True),  # ligature and line-break hyphen undone
        ('"The robot reaches 98% success"', True),  # quotes and case ignored
        ("in five-item pack- ing, up from 83%", True),  # the model kept the PDF hyphen
        # Two copied sentences stitched together still pass.
        ("up from 83%. It models the historical features with an explicit policy head", True),
        ("reaches 100% success in every packing task", False),
        ("the robot reaches 99% success", False),
        ("98% success", False),  # too short to show anything
        ("", False),
    ],
)
def test_quote_in_accepts_copied_text_only(quote, expected):
    assert csc.quote_in(quote, PASSAGE) is expected


@pytest.mark.parametrize(
    ("content", "with_quote", "expected"),
    [
        ('{"quote": "q", "verdict": "supported"}', True, ("q", "supported")),
        ('{"quote": "", "verdict": "nothing_to_check"}', True, ("", "nothing_to_check")),
        ('{"verdict": "partial"}', False, ("", "partial")),
        ('{"verdict": "partial"}', True, None),  # the quote schema needs a quote
        ('{"quote": 3, "verdict": "supported"}', True, None),
        ('{"quote": "q", "verdict": "yes"}', True, None),
        ("not json", False, None),
        ("[]", False, None),
    ],
)
def test_parse_reply_accepts_only_the_schema(content, with_quote, expected):
    assert csc.parse_reply(content, with_quote) == expected


def test_decide_needs_a_supported_verdict_and_a_real_quote():
    real = ("reaches 98% success in five-item packing", "supported")
    invented = ("reaches 100% success in every packing task", "supported")
    assert csc.decide(real, PASSAGE, check_quote=True) is True
    assert csc.decide(invented, PASSAGE, check_quote=True) is False
    assert csc.decide(invented, PASSAGE, check_quote=False) is True
    assert csc.decide((real[0], "partial"), PASSAGE, check_quote=True) is False
    assert csc.decide(("", "nothing_to_check"), PASSAGE, check_quote=True) is True
    assert csc.decide(None, PASSAGE, check_quote=True) is None


def _item(item_id, label, kind="natural", cited=("chk:pdf/a:0001",)):
    return {"item_id": item_id, "label": label, "kind": kind, "cited_ids": list(cited)}


def test_score_counts_caught_and_kept_and_treats_missing_as_not_supported():
    items = [
        _item("s1", "supported"),
        _item("s2", "supported"),
        _item("p1", "partial"),
        _item("u1", "unsupported", kind="swapped"),
        _item("n1", "no_claim"),
        _item("x1", "supported", cited=()),  # uncited: not scored
    ]
    predictions = {"s1": True, "s2": None, "p1": False, "u1": True, "n1": False}

    report = csc.score(items, predictions)

    assert report["all"] | {"kappa": None} == {
        "n": 4,
        "caught": 1,
        "not_supported": 2,
        "kept": 1,
        "supported": 2,
        "kappa": None,
        "missing": 1,
    }
    assert (report["natural"]["n"], report["swapped"]["n"]) == (3, 1)
    assert report["swapped"]["caught"] == 0
    assert (report["cited_no_claim_flagged"], report["cited_no_claim"]) == (1, 1)


def test_cohen_kappa_known_values():
    assert csc.cohen_kappa([(True, True), (False, False)]) == 1.0
    assert csc.cohen_kappa([(True, False), (False, True)]) == -1.0
    assert csc.cohen_kappa([(True, True), (True, True)]) is None  # chance agreement is 1
    assert csc.cohen_kappa([]) is None
