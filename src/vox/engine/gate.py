"""Silence / VAD state machine: RMS gate + Silero VAD with hangover, tail drain, onset fade.

Contract: docs/research/12 §3.6, 03 §5.

Not implemented yet: scheduled for Phase 4 (docs/research/13).
"""

from __future__ import annotations
