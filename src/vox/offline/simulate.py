"""Stream simulator: WAV in, hop by hop, through the real engine (docs/research/10 §2).

``load_engine`` builds the same ``HopPipeline`` and ``Sola`` objects the live worker will
use; ``StreamSimulator`` only replaces the audio streams and rings with a file feeder and
an accumulator (principle P13: there is no second pipeline to diverge).

Two clock modes: throughput (hops back to back, for cost and RTF) and paced (hop N becomes
available at ``N * block``, deadline misses counted, for stability verdicts).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vox.config import defaults
from vox.config.schemas import PrecisionParams, RuntimeParams, ShapeParams
from vox.engine import chunker, warmup
from vox.engine.chunker import ChunkPlan
from vox.engine.pipeline import STAGES, HopPipeline
from vox.engine.sola import Sola
from vox.features.contentvec import ContentVec
from vox.models.registry import LoadedModel, ModelRegistry
from vox.pitch import make_extractor
from vox.profiling.timers import AllocProbe, StageTimers
from vox.retrieval import index_loader


@dataclass
class Engine:
    plan: ChunkPlan
    pipeline: HopPipeline
    sola: Sola
    timers: StageTimers
    device: torch.device
    loaded: LoadedModel
    params: RuntimeParams
    precision: PrecisionParams
    warmup_ms: float = 0.0


def load_engine(models_dir: Path, model_name: str, shape: ShapeParams | None = None,
                params: RuntimeParams | None = None,
                precision: PrecisionParams | None = None,
                device: torch.device | str | None = None, warmup_hops: int = 5,
                registry: ModelRegistry | None = None,
                cache: dict | None = None) -> Engine:
    """Control-path construction: load models, plan, allocate, warm up.

    ``cache`` lets a sweep reuse the loaded embedder and extractors across engine builds.
    """
    shape = shape or defaults.DEFAULT_SHAPE
    precision = precision or defaults.DEFAULT_PRECISION
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    warmup.apply_precision(precision)

    registry = registry or ModelRegistry(models_dir)
    loaded = registry.get(model_name, device)
    model = loaded.model
    model.decoder_fp16 = precision.decoder_fp16 and device.type == "cuda"
    params = params or RuntimeParams(sid=loaded.config.sid)

    cache = cache if cache is not None else {}
    key = ("embedder", loaded.config.version, precision.hubert_fp16, str(device))
    if key not in cache:
        cache[key] = ContentVec(loaded.config.version, fp16=precision.hubert_fp16)
        cache[key].load(device, Path(models_dir))
    embedder = cache[key]
    extractor = None
    f0_tail = 0
    if model.has_f0:
        key = ("extractor", params.f0_method, str(device))
        if key not in cache:
            cache[key] = make_extractor(params.f0_method)
            cache[key].load(device, Path(models_dir))
        extractor = cache[key]
        block_16k = chunker.plan(shape.block_ms, shape.crossfade_ms, shape.extra_ms,
                                 shape.sola_search_ms, shape.device_sr, model.tgt_sr).block_16k
        f0_tail = extractor.tail_samples(block_16k)
    plan = chunker.plan(shape.block_ms, shape.crossfade_ms, shape.extra_ms,
                        shape.sola_search_ms, shape.device_sr, model.tgt_sr, f0_tail,
                        shape.decoder_context_ms)

    index = None
    if params.index_rate > 0 and loaded.index_path is not None:  # lazy: P6
        index = index_loader.load(loaded.index_path, model.emb_dim, device)

    timers = StageTimers(STAGES, device, capacity=defaults.PROFILER_RING)
    pipeline = HopPipeline(plan, model, embedder, extractor, device, params, timers, index)
    sola = Sola(plan, device, params.sola_mode)
    engine = Engine(plan, pipeline, sola, timers, device, loaded, params, precision)
    if warmup_hops:
        engine.warmup_ms = warmup.run(pipeline, sola, warmup_hops)
    return engine


@dataclass
class SimResult:
    audio: np.ndarray  # stitched stream, exactly as emitted (device rate)
    plan: ChunkPlan
    hops: int
    audio_seconds: float
    wall_seconds: float
    rtf: float
    deadline_misses: int
    sola_offsets_ms: np.ndarray
    alloc: dict[str, float] | None
    pointers_stable: bool
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def aligned(self) -> np.ndarray:
        """The stream advanced by the nominal SOLA delay, so it lines up with the input."""
        return self.audio[self.plan.stream_delay:]


class StreamSimulator:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.alloc = AllocProbe(engine.device)

    def run(self, audio: np.ndarray, realtime: bool = False, jitter_ms: float = 0.0,
            sync_debug: bool = False, reset: bool = True, seed: int | None = 0) -> SimResult:
        """Feed ``audio`` (mono float32 at the device rate) through the engine.

        ``sync_debug`` makes any implicit host<->device synchronisation inside the hop raise
        (after a few settling hops), which is how the no-``.item()`` rule is enforced.
        """
        engine = self.engine
        plan, pipeline, sola, timers = engine.plan, engine.pipeline, engine.sola, engine.timers
        cuda = engine.device.type == "cuda"
        block = plan.block
        # Whole hops, plus enough trailing silence to flush the SOLA delay.
        flush = -(-plan.stream_delay // block)
        hops = -(-len(audio) // block) + flush
        padded = np.zeros(hops * block, dtype=np.float32)
        padded[:len(audio)] = audio
        out = np.zeros(hops * block, dtype=np.float32)
        host_out = torch.zeros(block, pin_memory=cuda)
        host_out_np = host_out.numpy()

        if reset:
            pipeline.reset()
            sola.reset()
        timers.reset()
        self.alloc = AllocProbe(engine.device, capacity=hops)
        if seed is not None:
            torch.manual_seed(seed)
        pointers = {k: v.data_ptr() for k, v in pipeline.buffers().items()}
        rng = np.random.default_rng(0)
        hop_s = block / plan.device_sr
        settle = 8

        compute = 0.0
        start = time.perf_counter()
        for n in range(hops):
            if realtime:
                available = start + (n + 1) * hop_s
                if jitter_ms:
                    available += rng.uniform(-jitter_ms, jitter_ms) / 1000.0
                while (remaining := available - time.perf_counter()) > 0:
                    if remaining > 0.002:
                        time.sleep(remaining - 0.002)
            strict = sync_debug and cuda and n >= settle
            hop_start = time.perf_counter()
            timers.begin_hop()
            self.alloc.begin_hop()
            with timers.stage("upload"):
                pipeline.upload(padded[n * block:(n + 1) * block])
            if strict:
                torch.cuda.set_sync_debug_mode("error")
            try:
                infer = pipeline.process_hop()
                with timers.stage("sola"):
                    stitched = sola.stitch(infer)
            finally:
                if strict:
                    torch.cuda.set_sync_debug_mode("default")
            with timers.stage("download"):
                host_out.copy_(stitched)  # the one device -> host copy; syncs the hop
            self.alloc.end_hop()
            timers.end_hop()
            compute += time.perf_counter() - hop_start
            out[n * block:(n + 1) * block] = host_out_np
            if realtime and time.perf_counter() > available + hop_s:
                timers.counters.deadline_misses += 1
        wall = time.perf_counter() - start

        timers.flush()
        stable = pointers == {k: v.data_ptr() for k, v in pipeline.buffers().items()}
        seconds = hops * hop_s
        return SimResult(
            audio=out, plan=plan, hops=hops, audio_seconds=seconds, wall_seconds=wall,
            rtf=compute / seconds,
            deadline_misses=timers.counters.deadline_misses,
            sola_offsets_ms=sola.offsets_ms(plan.device_sr),
            alloc=self.alloc.summary(skip=min(defaults.PROFILER_WARMUP_HOPS, hops // 2)),
            pointers_stable=stable,
        )
