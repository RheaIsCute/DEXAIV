"""Rolling per-stage statistics over a fixed ring of hops (docs/research/04 §5.2).

The hot path only appends float32 samples. Percentiles are computed off the hot path
(report time, or a 1-2 Hz stats flush once live audio exists).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Summary:
    count: int
    p50: float
    p95: float
    p99: float
    max: float
    mean: float

    @staticmethod
    def empty() -> Summary:
        nan = float("nan")
        return Summary(0, nan, nan, nan, nan, nan)


class StatsRing:
    """``n_series`` independent rings of ``capacity`` float32 samples each."""

    def __init__(self, n_series: int, capacity: int = 1024) -> None:
        self.capacity = capacity
        self._data = np.zeros((n_series, capacity), dtype=np.float32)
        self._count = np.zeros(n_series, dtype=np.int64)

    def push(self, series: int, value: float) -> None:
        self._data[series, self._count[series] % self.capacity] = value
        self._count[series] += 1

    def total(self, series: int) -> int:
        return int(self._count[series])

    def samples(self, series: int, skip: int = 0) -> np.ndarray:
        """Samples in chronological order, dropping the first ``skip`` ever pushed."""
        count = int(self._count[series])
        if count <= self.capacity:
            return self._data[series, min(skip, count):count].copy()
        head = count % self.capacity
        ordered = np.concatenate((self._data[series, head:], self._data[series, :head]))
        dropped = count - self.capacity
        return ordered[max(skip - dropped, 0):]

    def summary(self, series: int, skip: int = 0) -> Summary:
        x = self.samples(series, skip)
        if x.size == 0:
            return Summary.empty()
        p50, p95, p99 = np.percentile(x, [50, 95, 99])
        return Summary(int(x.size), float(p50), float(p95), float(p99), float(x.max()),
                       float(x.mean()))

    def reset(self) -> None:
        self._count[:] = 0


def headroom(hop_ms: float, p99_compute_ms: float) -> float:
    """``hop / p99``: the stability gauge. <1.2 raise the hop, >2 it may shrink."""
    if not p99_compute_ms or p99_compute_ms != p99_compute_ms:
        return float("nan")
    return hop_ms / p99_compute_ms
