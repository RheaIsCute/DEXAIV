"""Hardcoded engine defaults: the left end of the resolution chain (docs/research/08 §7).

    engine defaults  <-  profile globals (voice_profile / runtime)  <-  per-model override
"""

from __future__ import annotations

from vox.config.schemas import PrecisionParams, RuntimeParams, ShapeParams

# Sizes snap to 10 ms units at the device rate (the chunking law, engine/chunker.py), so the
# research docs' "32 ms crossfade" is 30 ms here. The overlap may not exceed the hop, so at
# the 20 ms hop it is 20 ms (R8: -10 ms of latency, ~1% on the Q7 metric). See
# docs/DEVIATIONS.md.
DEFAULT_SHAPE = ShapeParams(
    block_ms=20.0,
    crossfade_ms=20.0,
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
    # RMVPE right-edge guard (docs/DEVIATIONS.md §8.4): the seam pitch step on held notes
    # halves (p95 21 -> 10 cents); F0 lands 10 ms later against the features.
    f0_trim_tail=2,
)

# Blackwell defaults (TARGET.md rule 2): TF32 on; HuBERT autocast "auto" tries FP16 then
# BF16 at warmup and keeps the first whose eager render matches FP32 (engine/warmup.py).
# Per-stage CUDA graphs are the default fast path (docs/DEVIATIONS.md R1/R2): "auto"
# captures, verifies a replay against eager, else eager.
DEFAULT_PRECISION = PrecisionParams(
    tf32=True, hubert_fp16=False, decoder_fp16=False, graphs="auto", hubert_autocast="auto"
)

ANALYSIS_SR = 16000  # ContentVec / F0 / index domain (docs/research/02 §1)
FRAME_16K = 160  # one 10 ms feature frame at 16 kHz
PROFILER_RING = 1024  # hops kept for percentile statistics
PROFILER_WARMUP_HOPS = 50  # discarded before reporting (docs/research/10 §3.2)
