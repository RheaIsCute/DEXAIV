"""Precision flags and warmup before going live (docs/research/06 §5, 12 §3.6).

Order of business: eager warmup (cuDNN heuristics, allocator, resamplers, SOLA) on a
synthetic voiced signal, then the two one-time tuning decisions of TARGET.md rules 2 and 4
(when ``tune=True``), then CUDA-graph capture of every stage, then verification hops
replayed against eager, then every buffer zeroed. ``graphs="auto"`` falls back to eager
when capture or verification fails and records why in ``StageGraphs.report``.

The tuning decisions, persisted by the caller under the engine's shape key so a later
startup applies them without measuring again (``tuned`` argument):

* **HuBERT autocast ladder** (rule 2): try the precision candidates in order (FP16, BF16 by
  default) and keep the first whose eager render of ``VERIFY_HOPS`` hops matches the FP32
  eager render within ``MATCH_TOLERANCE`` log-mel L1 (the same bar the graph replay has to
  clear). Nothing passes: FP32.
* **cuDNN autotune** (rule 4): capture the graphs with cuDNN's heuristic algorithms and
  with ``cudnn.benchmark`` during the capture warmup, time ``TIME_HOPS`` replays of each,
  keep the faster by more than ``AUTOTUNE_MARGIN``. The captured graphs hold the chosen
  kernels; ``cudnn.benchmark`` itself is never on outside a capture warmup.

PyTorch cannot export cuDNN's algorithm cache, so what is persisted is the *decision*
(autotune on or off) and the measurements behind it; a startup that applies it captures
once, exactly as before, and gets the same kernels because the shapes are the same.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from vox.config.schemas import PrecisionParams
from vox.engine.graphs import GraphCaptureError, GraphMode, StageGraphs, _log_mel, _Snapshot
from vox.engine.pipeline import GPU_STAGES, HopPipeline
from vox.engine.sola import Sola

VERIFY_HOPS = 10  # hops compared eager vs replay after capture (200 ms at a 20 ms hop)
MATCH_HOPS = 30  # hops compared FP32 eager vs an autocast candidate (600 ms at 20 ms)
TIME_HOPS = 60  # replayed hops timed per cuDNN candidate (one-time, persisted)
NOISE_SEED = 0  # the decoder noise of every comparison: the decisions must be reproducible
MATCH_TOLERANCE = 0.05  # log-mel L1, eager FP32 vs candidate (docs/DEVIATIONS.md R5)
# Autotuned kernels must beat the heuristics by this fraction to be kept. Measured on the
# target: the two differ by 1-3% in either direction between runs (docs/DEVIATIONS.md
# R10), so a smaller margin would make the persisted decision a coin toss.
AUTOTUNE_MARGIN = 0.05


def apply_precision(precision: PrecisionParams) -> None:
    """Set global backend flags once, before any warmup or capture; never toggle later."""
    torch.backends.cuda.matmul.allow_tf32 = precision.tf32
    torch.backends.cudnn.allow_tf32 = precision.tf32
    # Autotuning re-benchmarks on any new shape: a multi-ms stall mid-stream. It is turned
    # on only inside a graph capture's warmup runs (graphs._capture), never here.
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
        log=print, precision: PrecisionParams | None = None, tune: bool = False,
        tuned: dict[str, Any] | None = None) -> float:
    """Warm up, tune when asked, capture graphs when asked, verify, zero all state.

    Returns the eager steady-state estimate in ms (``max`` of the last three warmup hops,
    the Applio convention). With ``tune=True`` the HuBERT autocast and cuDNN autotune
    decisions are made here (or applied from ``tuned``) and left in ``graphs.tuned`` for
    the caller to persist; without it the embedder's and graphs' current settings stand
    (the live rebuild path, which must not spend seconds measuring).
    """
    timers_enabled = pipeline.timers.enabled
    pipeline.timers.enabled = False
    cuda = pipeline.device.type == "cuda"
    plan = pipeline.plan
    # Enough hops to fill the whole analysis window with signal before anything is timed.
    fill = -(-plan.history_16k // plan.block_16k) + 2
    hops = max(hops, 3)
    tail = VERIFY_HOPS + (MATCH_HOPS + TIME_HOPS if tune else 0)
    signal = test_signal((fill + hops + tail) * plan.block, plan.device_sr)
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
        rest = signal[(fill + hops) * plan.block:].reshape(tail, plan.block)
        blocks = torch.from_numpy(np.ascontiguousarray(rest)).to(pipeline.device)
        verify_blocks = blocks[:VERIFY_HOPS]
        match_blocks = blocks[VERIFY_HOPS:VERIFY_HOPS + MATCH_HOPS]
        time_blocks = blocks[VERIFY_HOPS + MATCH_HOPS:]
        record: dict[str, Any] = dict(tuned or {})
        if tune and cuda:
            _tune_hubert(pipeline, sola, match_blocks, precision, record, log)
        if graphs is not None and graphs.supported():
            if tune and cuda:
                _tune_cudnn(pipeline, sola, graphs, verify_blocks, time_blocks, record, log)
            else:
                _capture_and_verify(pipeline, sola, graphs, verify_blocks, log)
        if graphs is not None:
            graphs.tuned = record
    finally:
        pipeline.timers.enabled = timers_enabled
    pipeline.reset()
    sola.reset()
    return eager_ms


# ------------------------------------------------------------------------- measurement


def eager_render(pipeline: HopPipeline, sola: Sola, blocks: torch.Tensor,
                 noise: tuple[list, list | None], impl: dict | None = None,
                 stitch=None) -> torch.Tensor:
    """Run ``blocks`` from the *current* state with the given noise draws; the caller
    snapshots and restores around it. Returns the stitched stream."""
    impl = impl or pipeline.stage_fns
    stitch = stitch or sola._stitch_eager
    noise_z, noise_src = noise
    out = []
    with torch.inference_mode():
        for n in range(blocks.shape[0]):
            pipeline.in_block.copy_(blocks[n])
            for name in GPU_STAGES:
                if name == "synth":
                    pipeline.noise_z.copy_(noise_z[n])
                    if noise_src is not None:
                        pipeline.noise_src.copy_(noise_src[n])
                impl[name]()
            out.append(stitch(pipeline.out).clone())
    if pipeline.device.type == "cuda":
        torch.cuda.synchronize(pipeline.device)
    return torch.cat(out)


def _draw_noise(pipeline: HopPipeline, hops: int) -> tuple[list, list | None]:
    """Seeded decoder noise: the same draws on every build, so a persisted decision is
    the decision any later build would have made."""
    gen = torch.Generator(device=pipeline.device).manual_seed(NOISE_SEED)

    def draw(like: torch.Tensor) -> torch.Tensor:
        return torch.randn(like.shape, dtype=like.dtype, device=like.device, generator=gen)

    noise_z = [draw(pipeline.noise_z) for _ in range(hops)]
    noise_src = ([draw(pipeline.noise_src) for _ in range(hops)]
                 if pipeline.noise_src is not None else None)
    return noise_z, noise_src


def log_mel_l1(a: torch.Tensor, b: torch.Tensor, sr: int) -> float:
    return (_log_mel(a, sr) - _log_mel(b, sr)).abs().mean().item()


def _tune_hubert(pipeline: HopPipeline, sola: Sola, blocks: torch.Tensor,
                 precision: PrecisionParams | None, record: dict[str, Any], log) -> None:
    """The autocast ladder: first candidate whose eager render matches FP32 eager."""
    embedder = pipeline.embedder
    if not hasattr(embedder, "autocast"):
        return
    if "hubert_autocast" in record:
        embedder.autocast = record["hubert_autocast"]
        log(f"hubert autocast: {embedder.autocast} (from profile)")
        return
    candidates = (precision or PrecisionParams()).hubert_candidates
    if candidates == ("off",):
        embedder.autocast = "off"
        record["hubert_autocast"] = "off"
        return
    snapshot = _Snapshot(pipeline.state_tensors() + sola.state_tensors())
    noise = _draw_noise(pipeline, blocks.shape[0])
    sr = pipeline.plan.device_sr
    embedder.autocast = "off"
    snapshot.restore()
    reference = eager_render(pipeline, sola, blocks, noise)
    chosen = "off"
    matches: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        embedder.autocast = candidate
        snapshot.restore()
        try:
            rendered = eager_render(pipeline, sola, blocks, noise)
            l1 = log_mel_l1(reference, rendered, sr)
            error = ""
        except RuntimeError as err:  # a dtype the kernels reject: not a candidate
            l1, error = float("inf"), f"{type(err).__name__}: {err}"[:120]
        passed = l1 <= MATCH_TOLERANCE
        matches[candidate] = {"log_mel_l1": l1, "passed": passed}
        if error:
            matches[candidate]["error"] = error
        if passed:
            chosen = candidate
            break
    embedder.autocast = chosen
    snapshot.restore()
    record["hubert_autocast"] = chosen
    record["hubert_match"] = matches
    tried = ", ".join(f"{k} L1 {v['log_mel_l1']:.4f}" for k, v in matches.items())
    log(f"hubert autocast: {chosen} (eager-match vs FP32, tolerance {MATCH_TOLERANCE}: {tried})")


def time_replays(pipeline: HopPipeline, sola: Sola, blocks: torch.Tensor) -> dict[str, float]:
    """Host ms per hop through the current ``stage_impl`` (graphs or eager), synchronised
    per hop, over ``blocks``; the state is restored afterwards."""
    snapshot = _Snapshot(pipeline.state_tensors() + sola.state_tensors())
    times = []
    with torch.inference_mode():
        for n in range(blocks.shape[0]):
            start = time.perf_counter()
            pipeline.in_block.copy_(blocks[n])
            sola.stitch(pipeline.process_hop())
            torch.cuda.synchronize(pipeline.device)
            times.append((time.perf_counter() - start) * 1000.0)
    snapshot.restore()
    skip = min(10, len(times) // 4)
    p50, p99 = np.percentile(times[skip:], [50, 99])
    return {"p50": float(p50), "p99": float(p99), "hops": len(times) - skip}


def _tune_cudnn(pipeline: HopPipeline, sola: Sola, graphs: StageGraphs,
                verify_blocks: torch.Tensor, time_blocks: torch.Tensor,
                record: dict[str, Any], log) -> None:
    """Heuristic vs autotuned kernels, decided once per shape and persisted."""
    if "cudnn_autotune" in record:
        graphs.autotune = bool(record["cudnn_autotune"])
        log(f"cudnn autotune: {'on' if graphs.autotune else 'off'} (from profile)")
        _capture_and_verify(pipeline, sola, graphs, verify_blocks, log)
        return
    timings: dict[str, dict[str, float]] = {}
    for autotune in (False, True):
        graphs.autotune = autotune
        _capture_and_verify(pipeline, sola, graphs, verify_blocks, log)
        if not graphs.active:  # capture or verify failed: eager, nothing to choose
            record["cudnn_autotune"] = False
            record["cudnn_timings"] = timings
            graphs.autotune = False
            return
        timings["autotune" if autotune else "heuristic"] = time_replays(pipeline, sola,
                                                                        time_blocks)
    heuristic, autotuned = timings["heuristic"]["p50"], timings["autotune"]["p50"]
    keep = autotuned < heuristic * (1.0 - AUTOTUNE_MARGIN)
    record["cudnn_autotune"] = keep
    record["cudnn_timings"] = timings
    log(f"cudnn autotune: heuristic p50 {heuristic:.2f} ms, autotuned {autotuned:.2f} ms -> "
        f"{'autotune' if keep else 'heuristic'} kernels")
    if not keep:  # the graphs currently hold the autotuned kernels: capture once more
        graphs.autotune = False
        _capture_and_verify(pipeline, sola, graphs, verify_blocks, log)


def _capture_and_verify(pipeline: HopPipeline, sola: Sola, graphs: StageGraphs,
                        blocks: torch.Tensor, log) -> None:
    mode: GraphMode = graphs.mode
    try:
        report = graphs.capture()
    except GraphCaptureError as err:
        if mode == "on":
            raise
        graphs.report.fallback_reason = f"capture failed: {err}"
        log(f"graphs: capture failed, running eager ({err})")
        return
    result = graphs.verify(blocks)
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
        f"{result['log_mel_l1']:.4f}, rel rms {result['rel_rms_diff']:.2e}"
        f"{' [cudnn autotune]' if graphs.autotune else ''}")
