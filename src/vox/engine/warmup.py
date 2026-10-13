"""Precision flags and warmup before going live (docs/research/06 §5, 12 §3.6).

Phase 2 scope: eager warmup only. CUDA-graph capture and auto hop sizing join in Phase 6.
"""

from __future__ import annotations

import time

import torch

from vox.config.schemas import PrecisionParams
from vox.engine.pipeline import HopPipeline
from vox.engine.sola import Sola


def apply_precision(precision: PrecisionParams) -> None:
    """Set global backend flags once, before any warmup or capture; never toggle later."""
    torch.backends.cuda.matmul.allow_tf32 = precision.tf32
    torch.backends.cudnn.allow_tf32 = precision.tf32
    # Autotuning re-benchmarks on any new shape: a multi-ms stall mid-stream.
    torch.backends.cudnn.benchmark = False


def run(pipeline: HopPipeline, sola: Sola, hops: int = 5) -> float:
    """Run full silent hops, then zero all state. Returns the steady-state estimate in ms.

    Covers cuDNN heuristics, allocator growth, resampler kernels and SOLA. The estimate is
    ``max`` of the last three hops, the Applio convention.
    """
    timers_enabled = pipeline.timers.enabled
    pipeline.timers.enabled = False
    cuda = pipeline.device.type == "cuda"
    timings: list[float] = []
    pipeline.in_block.zero_()
    try:
        for _ in range(max(hops, 3)):
            start = time.perf_counter()
            sola.stitch(pipeline.process_hop())
            if cuda:
                torch.cuda.synchronize(pipeline.device)
            timings.append((time.perf_counter() - start) * 1000.0)
    finally:
        pipeline.timers.enabled = timers_enabled
    pipeline.reset()
    sola.reset()
    return max(timings[-3:])
