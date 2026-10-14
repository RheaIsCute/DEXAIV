"""Hardcoded engine defaults: the left end of the resolution chain (docs/research/08 §7).

    engine defaults  <-  profile globals (voice_profile / runtime)  <-  per-model override
"""

from __future__ import annotations

from vox.config.schemas import PrecisionParams, RuntimeParams, ShapeParams

# Sizes snap to 10 ms units at the device rate (the chunking law, engine/chunker.py), so the
# research docs' "32 ms crossfade" is 30 ms here. See docs/DEVIATIONS.md.
DEFAULT_SHAPE = ShapeParams(
    block_ms=20.0,
    crossfade_ms=30.0,
    sola_search_ms=10.0,
    extra_ms=1000.0,  # was 500: pauses were ~2.8 dB too loud (docs/DEVIATIONS.md R4)
    device_sr=48000,
    decoder_context_ms=0.0,
)

DEFAULT_RUNTIME = RuntimeParams(
    f0_up_key=0.0,
    index_rate=0.0,
    protect=0.33,
    volume_envelope=1.0,
    f0_method="rmvpe",
)

# TF32 on; HuBERT FP16 autocast exists but stays off until Phase 6 validates it against the
# FP32 baseline (docs/research/13 Phase 6). Per-stage CUDA graphs are the default fast path
# (docs/DEVIATIONS.md R1/R2): "auto" captures, verifies a replay against eager, else eager.
DEFAULT_PRECISION = PrecisionParams(
    tf32=True, hubert_fp16=False, decoder_fp16=False, graphs="auto"
)

ANALYSIS_SR = 16000  # ContentVec / F0 / index domain (docs/research/02 §1)
FRAME_16K = 160  # one 10 ms feature frame at 16 kHz
PROFILER_RING = 1024  # hops kept for percentile statistics
PROFILER_WARMUP_HOPS = 50  # discarded before reporting (docs/research/10 §3.2)
