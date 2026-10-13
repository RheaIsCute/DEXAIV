"""Inference worker thread decoupling <=10 ms callbacks from the 20-32 ms hop.

Contract: docs/research/12 §3.6, 03 §10.

Not implemented yet: scheduled for Phase 3 (docs/research/13).
"""

from __future__ import annotations
