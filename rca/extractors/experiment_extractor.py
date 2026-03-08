"""Deterministic extraction for experiment result payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


class ExperimentExtractor:
    """Load experiment metadata from JSON or YAML files."""

    def extract(self, path: str | Path) -> dict[str, Any]:
        experiment_path = Path(path)
        suffix = experiment_path.suffix.lower()
        raw_text = experiment_path.read_text(encoding="utf-8")

        if suffix == ".json":
            data = json.loads(raw_text)
        elif suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("YAML experiment extraction requires the optional 'PyYAML' dependency.")
            data = yaml.safe_load(raw_text) or {}
        else:
            raise ValueError(f"Unsupported experiment file type: {experiment_path.suffix}")

        title = data.get("name") or data.get("title") or experiment_path.stem
        metrics = data.get("metrics", {}) if isinstance(data, dict) else {}

        return {
            "title": title,
            "content": json.dumps(data, indent=2, sort_keys=True),
            "sections": [{"heading": "Experiment", "text": json.dumps(data, indent=2, sort_keys=True)}],
            "metadata": {
                "kind": "experiment",
                "path": str(experiment_path),
                "metric_names": sorted(metrics.keys()) if isinstance(metrics, dict) else [],
            },
        }
