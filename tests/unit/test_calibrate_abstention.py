from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "eval" / "calibrate_abstention.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("calibrate_abstention", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pick_finds_threshold_within_false_abstention_budget() -> None:
    calib = _load_module()
    answerable = [{"answerable": True, "rerank_max": v} for v in (2.0, 3.0, 4.0, -5.0)]
    negatives = [
        {"answerable": False, "rerank_max": v, "negative_type": "fabricated"}
        for v in (-6.0, -7.0, 1.0)
    ]
    rows = answerable + negatives

    assert calib.auroc(rows, "rerank_max", "low") > 0.5

    picked = calib.pick(rows, budget=0.0)
    assert picked is not None
    _, name, rule, dev_score = picked
    assert name.startswith("rerank_max <=")
    assert dev_score["negatives_abstained"] == "2/3"
    assert dev_score["answerables_abstained"] == "0/4"
    assert calib.by_type(rows, rule) == {"fabricated": "2/3"}
