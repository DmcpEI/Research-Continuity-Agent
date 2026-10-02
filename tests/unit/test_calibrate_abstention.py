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


def test_calibration_report_carries_feature_run_provenance(tmp_path) -> None:
    import json

    calib = _load_module()
    rows = [{"id": f"a{i}", "answerable": True, "rerank_max": 3.0 + i} for i in range(4)] + [
        {"id": f"n{i}", "answerable": False, "rerank_max": -8.0 - i} for i in range(4)
    ]
    features = tmp_path / "features.json"
    features.write_text(
        json.dumps({"provenance": {"corpus": {"corpus_version": "abc123"}}, "rows": rows})
    )
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "dev.json").write_text(json.dumps({"ids": ["a0", "a1", "n0", "n1"]}))
    (splits / "test.json").write_text(json.dumps({"ids": ["a2", "a3", "n2", "n3"]}))
    output = tmp_path / "report.json"

    assert calib.main([str(features), "--splits-dir", str(splits), "--output", str(output)]) == 0

    report = json.loads(output.read_text())
    assert report["features_provenance"] == {"corpus": {"corpus_version": "abc123"}}
    assert report["calibration_code"]["commit"]
