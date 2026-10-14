"""Per-stage CUDA graph capture / replay / invalidation (docs/research/06 §2.4, 12 §3.6).

Each GPU stage of ``HopPipeline`` (``GPU_STAGES``) and the SOLA stitch is captured as its
own ``torch.cuda.CUDAGraph``. Every stage reads and writes only pointer-stable buffers, so
a replay is a drop-in for the eager method: ``pipeline.stage_impl[name]`` is pointed at the
replay and ``process_hop`` is unchanged. Timers and CUDA events stay outside the captured
regions (events cannot live inside a graph); a graphed stage is timed around its replay.

Capture follows the Applio pattern: snapshot the mutable state, run the stage eagerly
``WARMUP_ITERS`` times on a side stream so cuDNN heuristics and the allocator settle at the
exact shapes, capture one run, restore the state. Randomness is never captured: the
pipeline pre-fills its decoder noise buffers outside the graphs each hop.

Modes (``GraphMode``): ``off`` eager; ``on`` capture, raise on failure; ``auto`` capture,
verify one replayed hop against eager, fall back to eager on any failure and record why.
Whole-hop single-graph capture (``whole_hop=True``) exists for the Q16 investigation and is
not the default: its replay hung on the reference machine (docs/DEVIATIONS.md R2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from vox.engine.pipeline import GPU_STAGES, HopPipeline
from vox.engine.sola import Sola

GraphMode = Literal["off", "on", "auto"]
MODES = ("off", "on", "auto")
WARMUP_ITERS = 3
SOLA = "sola"
HOP = "hop"  # the whole-hop single graph (experimental, Q16)


class GraphCaptureError(RuntimeError):
    """Capture failed. A failed capture can poison the CUDA context; the caller decides
    whether to run eager in this process or restart."""


@dataclass
class GraphReport:
    mode: str
    active: bool = False
    stages: dict[str, str] = field(default_factory=dict)  # name -> "graph" | "eager" | why
    capture_ms: float = 0.0
    verify: dict[str, float] = field(default_factory=dict)
    fallback_reason: str = ""


def _log_mel(audio: Tensor, sr: int, n_mels: int = 80) -> Tensor:
    """Natural-log mel spectrogram on the device (same recipe as ``offline.metrics``, which
    the engine must not import). 40 ms window, 10 ms hop."""
    import torchaudio

    hop, win = sr // 100, sr // 25
    n_fft = 1 << (win - 1).bit_length()
    spec = torch.stft(audio, n_fft, hop_length=hop, win_length=win,
                      window=torch.hann_window(win, device=audio.device), center=True,
                      return_complex=True).abs()
    fbank = torchaudio.functional.melscale_fbanks(
        n_fft // 2 + 1, 50.0, min(sr / 2, 12000.0), n_mels, sr, norm="slaney",
        mel_scale="htk").to(audio.device)
    return torch.log(torch.clamp(spec.T @ fbank, min=1e-5))


class _Snapshot:
    """Clone a list of tensors now, copy them back later."""

    def __init__(self, tensors: list[Tensor]) -> None:
        self.tensors = tensors
        self.saved = [t.clone() for t in tensors]

    def restore(self) -> None:
        for t, s in zip(self.tensors, self.saved):
            t.copy_(s)


def _capture(fn, device: torch.device, autotune: bool = False) -> torch.cuda.CUDAGraph:
    """Warm up on a side stream, then capture ``fn`` (which must allocate no new inputs).

    ``autotune`` turns ``cudnn.benchmark`` on for the warmup runs only: cuDNN then times
    its conv algorithms for these exact shapes and the capture records the winners. The
    flag is back to False before the capture (and before the graph key is computed), so
    no eager stage can ever re-benchmark mid-stream.
    """
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    torch.backends.cudnn.benchmark = autotune
    try:
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(WARMUP_ITERS):
                fn()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
    finally:
        torch.backends.cudnn.benchmark = False
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize(device)
    return graph


class _Replay:
    """Callable replacing an eager stage. Replays under inference mode (the capture mode)."""

    __slots__ = ("graph",)

    def __init__(self, graph: torch.cuda.CUDAGraph) -> None:
        self.graph = graph

    def __call__(self, *_args) -> None:
        self.graph.replay()


class StageGraphs:
    """Owns the captured graphs of one pipeline + SOLA pair and keeps them valid.

    ``refresh()`` (control path, between hops) recaptures whenever the pipeline's static
    key changed: a new plan, index load/unload, a ``f0_unvoiced`` switch, new buffers.
    """

    def __init__(self, pipeline: HopPipeline, sola: Sola, mode: GraphMode = "auto",
                 whole_hop: bool | tuple[str, ...] = False, autotune: bool = False) -> None:
        # ``whole_hop``: True = all GPU stages as one graph; a tuple = that consecutive run
        # of stages as one graph, the rest per stage (the Q16 bisection).
        # ``autotune``: cuDNN benchmark during the capture warmup (TARGET.md rule 4); the
        # decision is made once per shape by warmup.py and persisted with the profile.
        if mode not in MODES:
            raise ValueError(f"graphs mode must be one of {MODES}")
        self.pipeline = pipeline
        self.sola = sola
        self.mode = mode
        self.whole_hop = whole_hop
        self.autotune = autotune
        self.tuned: dict = {}  # warmup's persisted decisions for this shape (warmup.run)
        self.device = pipeline.device
        self.graphs: dict[str, torch.cuda.CUDAGraph] = {}
        self._key: tuple | None = None
        self.report = GraphReport(mode)
        self._sola_static: Tensor | None = None

    # ------------------------------------------------------------------ control path

    @property
    def active(self) -> bool:
        return bool(self.graphs)

    def supported(self) -> bool:
        return self.mode != "off" and self.device.type == "cuda"

    def key(self) -> tuple:
        return (self.pipeline.static_key(), self.sola.mode, self.sola.buffer.data_ptr(),
                self.whole_hop, self.autotune)

    def stale(self) -> bool:
        return self.supported() and self._key != self.key()

    def release(self) -> None:
        """Back to eager; drop the graphs (their private memory pools go with them)."""
        self.pipeline.stage_impl.update(self.pipeline.stage_fns)
        self.sola._impl = self.sola._stitch_eager
        self.graphs.clear()
        self._sola_static = None
        self._key = None
        self.report = GraphReport(self.mode)

    def capture(self) -> GraphReport:
        """Capture every graph-safe stage. Raises ``GraphCaptureError`` on failure, after
        releasing anything captured so far (the pipeline is then fully eager again)."""
        self.release()
        if not self.supported():
            self.report.fallback_reason = "graphs off" if self.mode == "off" else "no CUDA"
            return self.report
        pipeline, sola = self.pipeline, self.sola
        snapshot = _Snapshot(pipeline.state_tensors() + sola.state_tensors())
        start = time.perf_counter()
        try:
            if self.whole_hop:
                self._capture_whole_hop()
            else:
                for name in GPU_STAGES:
                    if pipeline.stage_fns[name] is pipeline._stage_noop:
                        self.report.stages[name] = "eager (no-op)"
                        continue
                    if not pipeline.stage_graph_safe(name):
                        self.report.stages[name] = "eager (stage not graph-safe)"
                        continue
                    graph = _capture(pipeline.stage_fns[name], self.device, self.autotune)
                    self.graphs[name] = graph
                    pipeline.stage_impl[name] = _Replay(graph)
                    self.report.stages[name] = "graph"
            self._capture_sola()
        except Exception as err:  # noqa: BLE001 - any failure means "run eager"
            self.release()
            snapshot.restore()
            raise GraphCaptureError(f"{type(err).__name__}: {err}") from err
        finally:
            torch.cuda.synchronize(self.device)
        snapshot.restore()
        self._key = self.key()
        self.report.active = True
        self.report.capture_ms = (time.perf_counter() - start) * 1000.0
        return self.report

    def _capture_sola(self) -> None:
        sola, pipeline = self.sola, self.pipeline
        static = pipeline.out  # the only tensor the hop ever stitches
        graph = _capture(lambda: sola._stitch_eager(static), self.device, self.autotune)
        self.graphs[SOLA] = graph
        self._sola_static = static

        def replay(infer: Tensor) -> Tensor:
            if infer.data_ptr() != static.data_ptr():
                static.copy_(infer)
            graph.replay()
            return sola.out

        sola._impl = replay
        self.report.stages[SOLA] = "graph"

    def _capture_whole_hop(self) -> None:
        pipeline = self.pipeline
        fused = GPU_STAGES if self.whole_hop is True else tuple(self.whole_hop)
        first = GPU_STAGES.index(fused[0])
        assert GPU_STAGES[first:first + len(fused)] == fused, "stages must be consecutive"

        def run() -> None:
            for name in fused:
                pipeline.stage_fns[name]()

        for name in GPU_STAGES:
            if name in fused:
                continue
            if pipeline.stage_fns[name] is pipeline._stage_noop or                     not pipeline.stage_graph_safe(name):
                self.report.stages[name] = "eager"
                continue
            graph = _capture(pipeline.stage_fns[name], self.device, self.autotune)
            self.graphs[name] = graph
            pipeline.stage_impl[name] = _Replay(graph)
            self.report.stages[name] = "graph"
        graph = _capture(run, self.device, self.autotune)
        self.graphs[HOP] = graph
        # One graph stands in for the fused run: its first stage replays it, the others
        # become no-ops. (With synth inside, the noise fill in process_hop happens before
        # the synth slot, i.e. after the replay: the fused graph uses the previous draw.)
        pipeline.stage_impl[fused[0]] = _Replay(graph)
        for name in fused[1:]:
            pipeline.stage_impl[name] = pipeline._stage_noop
        for name in fused:
            self.report.stages[name] = f"graph (fused {'+'.join(fused)})"

    def refresh(self) -> bool:
        """Recapture if anything invalidated the graphs. Returns True when it did."""
        if not self.stale():
            return False
        self.capture()
        return True

    def verify(self, blocks: Tensor, tolerance: float = 0.05) -> dict[str, float]:
        """Run ``blocks`` (``[hops, block]``) eager and replayed from identical state and
        noise; compare the stitched streams.

        Both runs start from a snapshot of the current state and consume the same noise
        draws, so the outputs should agree to kernel-level rounding. The distance is the
        log-mel L1 over the stitched output; ``tolerance`` is set from the same-seed repeat
        measurement in docs/DEVIATIONS.md (0.005) with a wide margin.
        """
        pipeline, sola = self.pipeline, self.sola
        if not self.active:
            raise RuntimeError("nothing captured")
        snapshot = _Snapshot(pipeline.state_tensors() + sola.state_tensors())
        hops = blocks.shape[0]
        noise_z = [torch.randn_like(pipeline.noise_z) for _ in range(hops)]
        noise_src = ([torch.randn_like(pipeline.noise_src) for _ in range(hops)]
                     if pipeline.noise_src is not None else None)

        def run(impl: dict, stitch) -> Tensor:
            snapshot.restore()
            out = []
            with torch.inference_mode():
                for n in range(hops):
                    pipeline.in_block.copy_(blocks[n])
                    for name in GPU_STAGES:
                        if name == "synth":
                            pipeline.noise_z.copy_(noise_z[n])
                            if noise_src is not None:
                                pipeline.noise_src.copy_(noise_src[n])
                        impl[name]()
                    out.append(stitch(pipeline.out).clone())
            torch.cuda.synchronize(self.device)
            return torch.cat(out)

        eager = run(pipeline.stage_fns, sola._stitch_eager)
        graphed = run(pipeline.stage_impl, sola.stitch)
        snapshot.restore()
        diff = (eager - graphed).abs()
        rms = eager.pow(2).mean().sqrt().item() + 1e-9
        sr = pipeline.plan.device_sr
        mel_a, mel_b = _log_mel(eager, sr), _log_mel(graphed, sr)
        result = {
            "max_abs_diff": diff.max().item(),
            "rel_rms_diff": diff.pow(2).mean().sqrt().item() / rms,
            "log_mel_l1": (mel_a - mel_b).abs().mean().item(),
            "tolerance": tolerance,
            "passed": float((mel_a - mel_b).abs().mean().item() <= tolerance),
        }
        self.report.verify = result
        return result
