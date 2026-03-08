"""Minimal in-process metrics registry."""

from __future__ import annotations

from collections import defaultdict


class MetricsRegistry:
    """Keep simple counters and duration samples in memory."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = defaultdict(int)
        self.durations: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def record_duration(self, name: str, seconds: float) -> None:
        self.durations[name].append(seconds)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            "counters": dict(self.counters),
            "durations": {
                name: {
                    "count": float(len(samples)),
                    "avg_seconds": sum(samples) / len(samples) if samples else 0.0,
                }
                for name, samples in self.durations.items()
            },
        }
