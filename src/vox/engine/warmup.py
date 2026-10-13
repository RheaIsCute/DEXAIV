"""Precision flags and warmup before going live (docs/research/06 §5, 12 §3.6).

Order of business: eager warmup (cuDNN heuristics, allocator, resamplers, SOLA) on a
synthetic voiced signal, then CUDA-graph capture of every stage, then one verification hop
replayed against eager, then every buffer zeroed. ``graphs="auto"`` falls back to eager
when capture or verification fails and records why in ``StageGraphs.report``.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from vox.config.schemas import PrecisionParams
from vox.engine.graphs import GraphCaptureError, GraphMode, StageGraphs
from vox.engine.pipeline import HopPipeline
from vox.engine.sola import Sola

VERIFY_HOPS = 10  # hops compared eager vs replay after capture (200 ms at a 20 ms hop)


def apply_precision(precision: PrecisionParams) -> None:
    """Set global backend flags once, before any warmup or capture; never toggle later."""
    torch.backends.cuda.matmul.allow_tf32 = precision.tf32
    torch.backends.cudnn.allow_tf32 = precision.tf32
    # Autotuning re-benchmarks on any new shape: a multi-ms stall mid-stream.
    torch.backends.cudnn.benchmark = False


def test_signal(n: int, sr: int, start: int = 0, f0: float = 140.0) -> np.ndarray:
    """A voiced, vibrato'd harmonic tone with a soft envelope: enough for every stage of
    the hop to do real work (RMVPE finds a pitch, the gate would call it speech)."""
    t = (np.arange(start, start + n, dtype=np.float64)) / sr
    pitch = f0 * (1.0 + 0.03 * np.sin(2 * math.pi * 5.0 * t))
    phase = 2 * math.pi * np.cumsum(pitch) / sr
    tone = sum(np.sin(k * phase) / k for k in range(1, 9))
    envelope = 0.6 + 0.4 * np.sin(2 * math.pi * 1.3 * t)
    return (0.08 * tone * envelope).astype(np.float32)


def run(pipeline: HopPipeline, sola: Sola, hops: int = 5, graphs: StageGraphs | None = None,
        log=print) -> float:
    """Warm up, capture graphs when asked, verify, zero all state.

    Returns the eager steady-state estimate in ms (``max`` of the last three warmup hops,
    the Applio convention).
    """
    timers_enabled = pipeline.timers.enabled
    pipeline.timers.enabled = False
    cuda = pipeline.device.type == "cuda"
    plan = pipeline.plan
    # Enough hops to fill the whole analysis window with signal before anything is timed.
    fill = -(-plan.history_16k // plan.block_16k) + 2
    hops = max(hops, 3)
    signal = test_signal((fill + hops + VERIFY_HOPS) * plan.block, plan.device_sr)
    timings: list[float] = []
    try:
        pipeline.reset()
        sola.reset()
        for n in range(fill + hops):
            pipeline.upload(signal[n * plan.block:(n + 1) * plan.block])
            if n < fill:
                # Fill the 16 kHz window with signal cheaply (resample stage only, ~0.5 ms
                # eager): a full eager hop costs ~45 ms and a live rebuild waits on this.
                pipeline.advance_context()
                continue
            start = time.perf_counter()
            sola.stitch(pipeline.process_hop())
            if cuda:
                torch.cuda.synchronize(pipeline.device)
            timings.append((time.perf_counter() - start) * 1000.0)
        eager_ms = max(timings[-3:])
        if graphs is not None and graphs.supported():
            rest = signal[(fill + hops) * plan.block:].reshape(VERIFY_HOPS, plan.block)
            _capture_and_verify(pipeline, sola, graphs, rest, log)
    finally:
        pipeline.timers.enabled = timers_enabled
    pipeline.reset()
    sola.reset()
    return eager_ms


def _capture_and_verify(pipeline: HopPipeline, sola: Sola, graphs: StageGraphs,
                        blocks: np.ndarray, log) -> None:
    mode: GraphMode = graphs.mode
    try:
        report = graphs.capture()
    except GraphCaptureError as err:
        if mode == "on":
            raise
        graphs.report.fallback_reason = f"capture failed: {err}"
        log(f"graphs: capture failed, running eager ({err})")
        return
    blocks_t = torch.from_numpy(np.ascontiguousarray(blocks)).to(pipeline.device)
    result = graphs.verify(blocks_t)
    if not result["passed"]:
        message = (f"replay/eager mismatch: log-mel L1 {result['log_mel_l1']:.4f} > "
                   f"{result['tolerance']}")
        if mode == "on":
            raise GraphCaptureError(message)
        graphs.release()
        graphs.report.fallback_reason = message
        log(f"graphs: {message}; running eager")
        return
    stages = ", ".join(f"{k}={v}" for k, v in report.stages.items())
    log(f"graphs: captured in {report.capture_ms:.0f} ms ({stages}); verify log-mel L1 "
        f"{result['log_mel_l1']:.4f}, rel rms {result['rel_rms_diff']:.2e}")
