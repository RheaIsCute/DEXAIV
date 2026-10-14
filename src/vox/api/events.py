"""Outbound event stream: the only data a UI ever sees (docs/research/12 §3.12).

Three kinds of event leave the controller, all plain JSON-able dicts:

    state   the full state document, sent whenever anything in it changes
    meters  input / output level and gate state, about 15 times a second
    perf    profiler snapshot (latency estimate, per-stage percentiles, system), ~2 Hz

Plus ``toast`` for one-line notices. Subscribers get a bounded queue each; a subscriber
that falls behind loses its oldest events rather than slowing the publisher, so a stalled
UI can never back up into the control path (and nothing here is ever called from a hop).
"""

from __future__ import annotations

import queue
import threading
from typing import Any

STATE, METERS, PERF, TOAST = "state", "meters", "perf", "toast"


class EventBus:
    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = maxsize
        self._subscribers: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(self.maxsize)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    @property
    def subscribers(self) -> int:
        return len(self._subscribers)

    def publish(self, kind: str, data: Any) -> None:
        with self._lock:
            targets = list(self._subscribers)
        for q in targets:
            try:
                q.put_nowait((kind, data))
            except queue.Full:
                try:
                    q.get_nowait()  # drop the oldest; snapshots are replaceable
                except queue.Empty:
                    pass
                try:
                    q.put_nowait((kind, data))
                except queue.Full:
                    pass
