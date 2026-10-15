"""Per-stage timers: host ``perf_counter_ns`` plus CUDA events read one hop late.

Host timers answer the scheduling question (wall time, including GIL and launch overhead).
CUDA events answer the optimisation question (device time). ``Event.elapsed_time`` blocks
until the event has completed, so a hop's events are only read after the *next* hop ends,
by which point they are long done: profiling never adds a ``synchronize()`` to the hot
path (docs/research/04 §5.1).

Everything is preallocated in ``__init__``; stamping a stage touches no allocator.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch

from vox.profiling.stats import StatsRing

TOTAL = "total"


class Counters:
    """Audio-health counters. Single writer per field; plain ints are GIL-atomic."""

    __slots__ = (
        "deadline_misses",
        "dropped_hops",
        "hops",
        "input_overflows",
        "output_underruns",
        "vad_transitions",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        for name in self.__slots__:
            setattr(self, name, 0)

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__slots__}


EVENT_KINDS = ("gate_open", "gate_close", "underrun")


class EventLog:
    """Timestamped audio events, kept apart so a cut can be attributed to its cause.

    ``gate_close`` / ``gate_open``: the silence gate changed state (``value`` = the block's
    gate level in dBFS). ``underrun``: the output ring ran dry (``value`` = samples the
    device got as zeros). Each event carries the hop index and a time in seconds: stream
    time (hop * hop duration) in the simulator, ``perf_counter`` since ``reset`` live.
    Preallocated ring of ``capacity`` events; the counts never wrap.
    """

    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = capacity
        self._t = np.zeros(capacity, dtype=np.float64)
        self._hop = np.zeros(capacity, dtype=np.int64)
        self._kind = np.zeros(capacity, dtype=np.int8)
        self._value = np.zeros(capacity, dtype=np.float64)
        self.total = 0  # events ever logged; the ring keeps the last ``capacity``
        self.counts = dict.fromkeys(EVENT_KINDS, 0)
        self.t0 = time.perf_counter()

    def reset(self) -> None:
        self.total = 0
        self.counts = dict.fromkeys(EVENT_KINDS, 0)
        self.t0 = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def add(self, kind: str, hop: int, t: float, value: float = 0.0, count: int = 1) -> None:
        """``count``: occurrences this one entry stands for (several underruns seen at once)."""
        i = self.total % self.capacity
        self._t[i], self._hop[i], self._value[i] = t, hop, value
        self._kind[i] = EVENT_KINDS.index(kind)
        self.total += 1
        self.counts[kind] += count

    def since(self, seq: int = 0) -> list[dict]:
        """Events with sequence number >= ``seq`` still in the ring, oldest first."""
        first = max(seq, self.total - self.capacity)
        out = []
        for n in range(first, self.total):
            i = n % self.capacity
            out.append({"seq": n, "kind": EVENT_KINDS[self._kind[i]], "hop": int(self._hop[i]),
                        "t": float(self._t[i]), "value": float(self._value[i])})
        return out


class _Stage:
    """Reusable context manager for one stage, created once so ``with`` allocates nothing."""

    __slots__ = ("_index", "_timers")

    def __init__(self, timers: StageTimers, index: int) -> None:
        self._timers = timers
        self._index = index

    def __enter__(self) -> None:
        self._timers._start(self._index)

    def __exit__(self, *exc) -> None:
        self._timers._stop(self._index)


class _NullStage:
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc) -> None:
        return None


_NULL_STAGE = _NullStage()


class StageTimers:
    """Timers for a fixed list of stages plus an implicit whole-hop ``total``.

    Usage per hop::

        timers.begin_hop()
        with timers.stage("hubert"):
            ...
        timers.end_hop()
    """

    def __init__(
        self,
        stages: Sequence[str],
        device: torch.device | str = "cpu",
        capacity: int = 1024,
        enabled: bool = True,
    ) -> None:
        self.stages: tuple[str, ...] = (*stages, TOTAL)
        self.enabled = enabled
        self.counters = Counters()
        self.events = EventLog()  # logged even when timing is disabled
        self._index = {name: i for i, name in enumerate(self.stages)}
        self._total = len(self.stages) - 1
        n = len(self.stages)
        self.host = StatsRing(n, capacity)
        self.gpu = StatsRing(n, capacity)
        self._t0 = np.zeros(n, dtype=np.int64)
        self._host_ms = np.zeros(n, dtype=np.float64)
        self._used = np.zeros((2, n), dtype=bool)
        self._ctx = tuple(_Stage(self, i) for i in range(n))
        self._slot = 0
        self._pending = False
        self.cuda = enabled and torch.device(device).type == "cuda"
        self._events: list[list[tuple[torch.cuda.Event, torch.cuda.Event]]] = []
        if self.cuda:
            self._events = [
                [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                 for _ in range(n)]
                for _ in range(2)
            ]

    # ------------------------------------------------------------------ hot path

    def stage(self, name: str):
        if not self.enabled:
            return _NULL_STAGE
        return self._ctx[self._index[name]]

    def begin_hop(self) -> None:
        if not self.enabled:
            return
        self._host_ms[:] = 0.0
        self._used[self._slot, :] = False
        self._start(self._total)

    def end_hop(self) -> None:
        """Close the hop: store host times now, harvest the previous hop's CUDA events."""
        if not self.enabled:
            return
        self._stop(self._total)
        for i in range(len(self.stages)):
            if self._used[self._slot, i]:
                self.host.push(i, self._host_ms[i])
        self.counters.hops += 1
        if self.cuda:
            if self._pending:
                self._harvest(1 - self._slot)
            self._pending = True
            self._slot = 1 - self._slot

    def _start(self, i: int) -> None:
        self._used[self._slot, i] = True
        if self.cuda:
            self._events[self._slot][i][0].record()
        self._t0[i] = time.perf_counter_ns()

    def _stop(self, i: int) -> None:
        self._host_ms[i] += (time.perf_counter_ns() - self._t0[i]) * 1e-6
        if self.cuda:
            self._events[self._slot][i][1].record()

    # --------------------------------------------------------------- control path

    def _harvest(self, slot: int) -> None:
        for i in range(len(self.stages)):
            if self._used[slot, i]:
                start, end = self._events[slot][i]
                # query() is a non-blocking completion check: an event still in flight is
                # skipped rather than waited for, so harvesting can never stall the hop.
                if end.query():
                    self.gpu.push(i, start.elapsed_time(end))

    def flush(self) -> None:
        """Harvest the last hop's CUDA events. Control path only: this synchronizes."""
        if self.cuda and self._pending:
            torch.cuda.synchronize()
            self._harvest(1 - self._slot)
            self._pending = False

    def reset(self) -> None:
        self.flush()
        self.host.reset()
        self.gpu.reset()
        self.counters.reset()
        self.events.reset()

    def index(self, name: str) -> int:
        return self._index[name]


_ALLOC_KEYS = (
    "allocation.all.allocated",  # allocator requests served (pool hits included)
    "segment.all.allocated",  # cudaMalloc calls into the driver
    "reserved_bytes.all.current",
    "allocated_bytes.all.current",
)


class AllocProbe:
    """Per-hop CUDA allocator deltas, to verify the steady-state hop is allocation-stable.

    Reading ``torch.cuda.memory_stats`` only inspects host-side allocator counters; it does
    not synchronize. What the probe distinguishes:

    * ``driver_mallocs`` - new segments requested from the driver (a latency spike, must be 0),
    * ``reserved_growth_bytes`` / ``live_drift_bytes`` - persistent growth (a leak, must be 0),
    * ``pool_allocs_per_hop`` - temporaries served from the caching pool. Eager PyTorch
      modules always create these; they disappear under CUDA-graph replay (Phase 6).
    """

    def __init__(self, device: torch.device | str, capacity: int = 4096) -> None:
        self.device = torch.device(device)
        self.enabled = self.device.type == "cuda"
        self._rows = np.zeros((capacity, len(_ALLOC_KEYS)), dtype=np.int64)
        self._before = np.zeros(len(_ALLOC_KEYS), dtype=np.int64)
        self._n = 0

    def _read(self, out: np.ndarray) -> None:
        stats = torch.cuda.memory_stats(self.device)
        for i, key in enumerate(_ALLOC_KEYS):
            out[i] = stats.get(key, 0)

    def begin_hop(self) -> None:
        if self.enabled:
            self._read(self._before)

    def end_hop(self) -> None:
        if not self.enabled or self._n >= len(self._rows):
            return
        row = self._rows[self._n]
        self._read(row)
        row[:2] -= self._before[:2]  # counters become per-hop deltas; byte gauges stay absolute
        self._n += 1

    def summary(self, skip: int = 0) -> dict[str, float] | None:
        if not self.enabled or self._n <= skip:
            return None
        rows = self._rows[skip:self._n]
        return {
            "hops": len(rows),
            "pool_allocs_per_hop": float(np.median(rows[:, 0])),
            "driver_mallocs": int(rows[:, 1].sum()),
            "reserved_growth_bytes": int(rows[-1, 2] - rows[0, 2]),
            "live_drift_bytes": int(rows[:, 3].max() - rows[:, 3].min()),
        }
