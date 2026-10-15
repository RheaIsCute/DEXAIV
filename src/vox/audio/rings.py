"""SPSC ring buffers crossing the callback <-> worker boundary (numpy only, P14).

One writer, one reader, each owning its own monotonic position (plain Python ints: a
single attribute store is atomic under the GIL). The writer never touches the reader's
position and vice versa, so there is no lock to contend in an audio callback.

    InputRing   callback writes (<= 10 ms blocks), worker reads one hop at a time
    OutputRing  worker writes one hop, callback reads its block size

Policies (docs/research/03 §10): a reader that finds fewer samples than it needs gets zeros
for the rest and an *underrun* is counted; a writer that laps the reader destroyed the
oldest unread audio, which the reader detects on its next read, skips past, and counts as
an *overrun*. Nothing ever blocks and nothing allocates after construction.
"""

from __future__ import annotations

import time

import numpy as np


class SPSCRing:
    def __init__(self, capacity: int, guard: int = 0) -> None:
        self.capacity = 1 << max(capacity - 1, 1).bit_length()  # power of two
        self._mask = self.capacity - 1
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        # ``guard`` samples stay between the reader and the writer's in-flight region after an
        # overrun recovery: at least one maximum write size.
        self.guard = guard
        self.write_pos = 0  # writer-owned
        self.read_pos = 0  # reader-owned
        self.overruns = 0  # reader-owned
        self.overrun_samples = 0
        self.underruns = 0  # reader-owned
        self.underrun_samples = 0
        self.last_underrun_at = 0.0  # perf_counter of the latest underrun (reader-owned)
        self._drop_request = 0  # writer -> reader, single slot (see ``request_drop``)

    # ------------------------------------------------------------------ either side

    def available(self) -> int:
        """Unread samples (may exceed capacity after an overrun until the reader recovers)."""
        return self.write_pos - self.read_pos

    def free(self) -> int:
        return self.capacity - self.available()

    def fill(self) -> int:
        return min(self.available(), self.capacity)

    # ----------------------------------------------------------------------- writer

    def write(self, x: np.ndarray) -> None:
        """Append ``x`` (len <= capacity). Always succeeds; the reader detects a lap."""
        n = x.shape[0]
        w = self.write_pos & self._mask
        first = min(n, self.capacity - w)
        self._buf[w:w + first] = x[:first]
        if first < n:
            self._buf[:n - first] = x[first:]
        self.write_pos += n  # publish last

    def request_drop(self, n: int) -> None:
        """Writer side: have the reader discard ``n`` of the oldest unread samples at its
        next read. The reader owns ``read_pos``; a ``drop`` from the writer's thread races
        with a read in progress and one of the two position updates is lost."""
        self._drop_request = max(0, int(n))

    @property
    def drop_pending(self) -> bool:
        return self._drop_request > 0

    # ----------------------------------------------------------------------- reader

    def _recover(self) -> None:
        """Writer lapped us: jump to the oldest sample that is still intact."""
        target = self.write_pos - self.capacity + self.guard
        if target > self.read_pos:
            self.overruns += 1
            self.overrun_samples += target - self.read_pos
            self.read_pos = target

    def read(self, out: np.ndarray) -> int:
        """Fill ``out`` with the oldest unread samples; zeros past the end on an underrun.

        Returns the number of real samples delivered.
        """
        pending = self._drop_request
        if pending:
            self._drop_request = 0
            self.read_pos += min(pending, self.available())
        if self.available() > self.capacity - self.guard:
            self._recover()
        n_want = out.shape[0]
        n = min(n_want, self.available())
        r = self.read_pos & self._mask
        first = min(n, self.capacity - r)
        out[:first] = self._buf[r:r + first]
        if first < n:
            out[first:n] = self._buf[:n - first]
        if n < n_want:
            out[n:] = 0.0
            self.underrun_samples += n_want - n
            self.last_underrun_at = time.perf_counter()
            self.underruns += 1  # last: a reader of the count sees the time and samples
        self.read_pos += n
        return n

    def drop(self, n: int) -> int:
        """Discard up to ``n`` unread samples (latency trimming). Returns how many."""
        n = max(0, min(n, self.available()))
        self.read_pos += n
        return n

    def keep_latest(self, keep: int) -> int:
        """Discard everything but the newest ``keep`` samples. Returns how many were dropped."""
        return self.drop(self.available() - keep)

    def reset_counters(self) -> None:
        self.overruns = self.overrun_samples = 0
        self.underruns = self.underrun_samples = 0
