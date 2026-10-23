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

from vox.audio.conditioning import ConditioningConfig, InputConditioner
from vox.config import defaults, target
from vox.config.schemas import PrecisionParams, RuntimeParams, ShapeParams
from vox.config.store import load_tuned, save_tuned
from vox.engine import chunker, warmup
from vox.engine.chunker import ChunkPlan
from vox.engine.gate import GateConfig, GateDecision, SilenceGate, SileroVad
from vox.engine.graphs import StageGraphs
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
    graphs: StageGraphs
    gate: SilenceGate | None = None
    warmup_ms: float = 0.0
    conditioner: InputConditioner | None = None  # the live mic's input chain

    @property
    def mode(self) -> str:
        """``graphs`` when replaying captured graphs, else ``eager``."""
        return "graphs" if self.graphs.active else "eager"


def tune_key(model_name: str, plan: ChunkPlan, params: RuntimeParams,
             precision: PrecisionParams) -> str:
    """Everything that fixes the hop's kernel shapes: the persisted decisions' key."""
    return (f"{model_name}|hop{plan.block_ms:g}|extra{plan.extra_ms:g}|"
            f"xf{plan.overlap_ms:g}|search{plan.search_ms:g}|"
            f"{f'right{plan.right_ms:g}|' if plan.right else ''}sr{plan.device_sr}|"
            f"{params.f0_method}|dec{'16' if precision.decoder_fp16 else '32'}|"
            f"torch{torch.__version__}")


def load_engine(models_dir: Path, model_name: str, shape: ShapeParams | None = None,
                params: RuntimeParams | None = None,
                precision: PrecisionParams | None = None,
                device: torch.device | str | None = None, warmup_hops: int = 5,
                registry: ModelRegistry | None = None,
                cache: dict | None = None, graphs: str | None = None,
                gate: GateConfig | None = None, log=print,
                profile_dir: Path | None = None,
                conditioning: ConditioningConfig | None = None) -> Engine:
    """Control-path construction: load models, plan, allocate, warm up, capture graphs.

    The device is resolved and asserted against TARGET.md first (``config.target``); the
    check line is logged. ``cache`` lets a sweep reuse the loaded embedder and extractors
    across engine builds. ``graphs`` overrides ``precision.graphs`` ("on" | "off" |
    "auto"). ``gate`` configures the silence gate (default: enabled, Silero VAD, engine
    defaults); the Silero model is loaded on the CPU here, never in the hot loop.
    ``profile_dir`` is where warmup's one-time decisions (HuBERT autocast, cuDNN
    autotune) are read from and written to (``runtime.json``, keyed by shape); without it
    they are measured on every build and not stored. ``conditioning`` configures the live
    microphone's input chain (high-pass, auto level, trim, soft limit; default on), which
    the worker and the simulator run on each block before the upload and the gate.
    """
    shape = shape or defaults.DEFAULT_SHAPE
    precision = precision or defaults.DEFAULT_PRECISION
    if graphs is not None:
        import dataclasses

        precision = dataclasses.replace(precision, graphs=graphs)
    device, check = target.select_device(device)
    log(check)
    warmup.apply_precision(precision)

    registry = registry or ModelRegistry(models_dir)
    loaded = registry.get(model_name, device)
    model = loaded.model
    model.decoder_fp16 = precision.decoder_fp16 and device.type == "cuda"
    params = params or RuntimeParams(sid=loaded.config.sid)

    cache = cache if cache is not None else {}
    key = ("embedder", loaded.config.version, str(device))
    if key not in cache:
        cache[key] = ContentVec(loaded.config.version)
        cache[key].load(device, Path(models_dir))
    embedder = cache[key]
    # Off until warmup's ladder picks a candidate (or the profile already did). A build
    # without warmup (the full-file converter, ``vox say``) therefore runs HuBERT in FP32.
    embedder.autocast = "off"
    if embedder.has_group_norm:
        stats = None
        if params.contentvec_norm == "frozen":
            stats = ContentVec.load_norm_stats(Path(models_dir))
            if stats is None:
                raise FileNotFoundError(
                    f"contentvec_norm='frozen' needs {ContentVec.norm_stats_path(models_dir)} "
                    "(vox bench drift --store-norm measures it on a take)")
        embedder.set_norm(params.contentvec_norm, stats)
    extractor = None
    f0_tail = 0
    if model.has_f0:
        key = ("extractor", params.f0_method, str(device))
        if key not in cache:
            cache[key] = make_extractor(params.f0_method)
            cache[key].load(device, Path(models_dir))
        extractor = cache[key]
        f0_tail = extractor.tail_samples(chunker.from_shape(shape, model.tgt_sr).block_16k)
    plan = chunker.from_shape(shape, model.tgt_sr, f0_tail)
    if plan.overlap_note:
        log(f"warning: {plan.overlap_note}")

    index = None
    if params.index_rate > 0 and loaded.index_path is not None:  # lazy: P6
        index = index_loader.load(loaded.index_path, model.emb_dim, device)

    timers = StageTimers(STAGES, device, capacity=defaults.PROFILER_RING)
    pipeline = HopPipeline(plan, model, embedder, extractor, device, params, timers, index)
    sola = Sola(plan, device, params.sola_mode, params.sola_hold_eps)
    stage_graphs = StageGraphs(pipeline, sola, precision.graphs)
    gate_config = gate or GateConfig()
    vad = None
    if gate_config.use_vad:
        if "vad" not in cache:
            cache["vad"] = SileroVad()
            cache["vad"].load(Path(models_dir))
        vad = cache["vad"]
    silence_gate = SilenceGate(plan, gate_config, vad)
    conditioner = InputConditioner(plan.device_sr, plan.block,
                                   conditioning or ConditioningConfig())
    engine = Engine(plan, pipeline, sola, timers, device, loaded, params, precision,
                    stage_graphs, silence_gate, conditioner=conditioner)
    if warmup_hops:
        key = tune_key(model_name, plan, params, precision)
        tuned = load_tuned(profile_dir, key)
        engine.warmup_ms = warmup.run(pipeline, sola, warmup_hops, stage_graphs, log=log,
                                      precision=precision, tune=True, tuned=tuned)
        if stage_graphs.tuned and stage_graphs.tuned != tuned:
            save_tuned(profile_dir, key, stage_graphs.tuned)
            if profile_dir is not None:
                log(f"tuning: saved to {Path(profile_dir) / 'runtime.json'} [{key}]")
        log(target.memory_line(device))
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
    gate_open: np.ndarray | None = None  # per hop: True when the hop ran through the model
    gate_transitions: int = 0
    closed_hop_ms: np.ndarray | None = None  # host ms of hops the gate skipped
    # Paced runs: underruns of a modelled output ring (``_PlayoutModel``); None otherwise.
    underruns: int | None = None
    sola_jumps_ms: np.ndarray | None = None  # hop-to-hop |offset change| per hop
    events: list[dict] = field(default_factory=list)  # timers.events: gate / underrun log
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def aligned(self) -> np.ndarray:
        """The stream advanced by the nominal SOLA delay, so it lines up with the input."""
        return self.audio[self.plan.stream_delay:]


class _PlayoutModel:
    """The live output ring, modelled for a paced run: playback starts when the first hop
    is written, on top of a ``fill`` prefill, and drains at the device rate. A hop that
    lands after the ring would have run dry is an underrun; the ring is then topped back up
    to the target, as ``InferenceWorker`` does after one."""

    def __init__(self, fill: int, block: int, sr: int) -> None:
        self.fill, self.block, self.sr = fill, block, sr
        self.t0: float | None = None
        self.written = 0  # samples handed to the "device" since t0

    def hop_written(self, now: float) -> int:
        """Returns the samples the device got as zeros before this hop (0 = on time)."""
        if self.t0 is None:
            self.t0, self.written = now, self.fill + self.block
            return 0
        missing = int((now - self.t0) * self.sr) - self.written
        if missing > 0:
            self.t0, self.written = now, self.fill + self.block
            return missing
        self.written += self.block
        return 0


class StreamSimulator:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.alloc = AllocProbe(engine.device)

    def run(self, audio: np.ndarray, realtime: bool = False, jitter_ms: float = 0.0,
            sync_debug: bool = False, reset: bool = True, seed: int | None = 0,
            gate: bool = False, fill_ms: float | None = None,
            condition: bool = True, fill_rule: str = "legacy",
            fill_margin_ms: float = 20.0, on_hop=None) -> SimResult:
        """Feed ``audio`` (mono float32 at the device rate) through the engine.

        ``sync_debug`` makes any implicit host<->device synchronisation inside the hop raise
        (after a few settling hops), which is how the no-``.item()`` rule is enforced.
        ``gate`` runs the engine's silence gate exactly as the live worker does, on the
        block after ``engine.conditioner`` (the live input chain; ``condition=False``
        feeds the raw audio, as the engine did before it existed).
        ``realtime`` paces the input and models the output ring (fill target ``fill_ms``,
        default ``fill_rule`` / ``fill_margin_ms`` at a 10 ms callback, as ``LiveSession``:
        ``chunker.fill_target_ms``) to count underruns.
        Gate transitions and underruns are logged in ``engine.timers.events`` with stream
        time (hop index x hop duration) and returned in ``SimResult.events``.
        ``on_hop(n)`` is called after every hop the model ran (the pipeline's buffers hold
        that hop's features and pitch): the drift measurement's probe (``offline/drift.py``).
        It syncs with the device, so a run with it is not a timing run.
        """
        engine = self.engine
        plan, pipeline, sola, timers = engine.plan, engine.pipeline, engine.sola, engine.timers
        silence_gate = engine.gate if gate else None
        conditioner = engine.conditioner if condition else None
        always = GateDecision(True)
        cuda = engine.device.type == "cuda"
        block = plan.block
        zeros_block = torch.zeros(block, device=engine.device)
        # Whole hops, plus enough trailing silence to flush the SOLA delay.
        flush = -(-plan.stream_delay // block)
        hops = -(-len(audio) // block) + flush
        padded = np.zeros(hops * block, dtype=np.float32)
        padded[:len(audio)] = audio
        out = np.zeros(hops * block, dtype=np.float32)
        host_out = torch.zeros(block, pin_memory=cuda)
        host_out_np = host_out.numpy()

        if seed is not None:
            torch.manual_seed(seed)  # before the reset: it draws the window's first noise
        if reset:
            pipeline.reset()
            sola.reset()
            if silence_gate is not None:
                silence_gate.reset()
            if conditioner is not None:
                conditioner.reset()
        timers.reset()
        self.alloc = AllocProbe(engine.device, capacity=hops)
        pointers = {k: v.data_ptr() for k, v in pipeline.buffers().items()}
        rng = np.random.default_rng(0)
        hop_s = block / plan.device_sr
        settle = 8
        gate_open = np.ones(hops, dtype=bool)
        closed_ms: list[float] = []
        transitions = 0
        events = timers.events
        if fill_ms is None:
            fill_ms = chunker.fill_target_ms(plan.block_ms, 10.0, fill_rule, fill_margin_ms)
        fill = round(plan.device_sr * fill_ms / 1000.0)
        playout = _PlayoutModel(fill, block, plan.device_sr) if realtime else None
        underruns = 0

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
            host_block = padded[n * block:(n + 1) * block]
            with timers.stage("upload"):
                if conditioner is not None:
                    host_block = conditioner.process(host_block)
                    if conditioner.retro is not None:  # de-clicker look-back (§15)
                        pipeline.apply_retro(conditioner.retro)
                pipeline.upload(host_block)
            with timers.stage("gate"):
                decision = silence_gate.decide(host_block) if silence_gate else always
            if strict:
                torch.cuda.set_sync_debug_mode("error")
            try:
                if decision.speech:
                    if decision.onset and not sola.draining:  # as the live worker does
                        sola.mark_onset(decision.energy_only)
                        pipeline.on_onset()
                    infer = pipeline.process_hop()
                    with timers.stage("sola"):
                        stitched = sola.stitch(infer)
                    if on_hop is not None:
                        on_hop(n)
                else:
                    # Gate closed: the Applio protocol (engine/gate.py module docstring).
                    if decision.drain:
                        sola.begin_drain()
                        pipeline.forget_speech()
                    pipeline.advance_context()
                    with timers.stage("sola"):
                        stitched = sola.drain_step() if sola.draining else zeros_block
                    gate_open[n] = False
            finally:
                if strict:
                    torch.cuda.set_sync_debug_mode("default")
            if decision.onset or decision.drain:
                transitions += 1
                timers.counters.vad_transitions += 1
                events.add("gate_open" if decision.onset else "gate_close", n, n * hop_s,
                           decision.rms_dbfs)
            with timers.stage("download"):
                host_out.copy_(stitched)  # the one device -> host copy; syncs the hop
            pipeline.register.publish()
            self.alloc.end_hop()
            timers.end_hop()
            if playout is not None:
                missing = playout.hop_written(time.perf_counter())
                if missing:
                    underruns += 1
                    timers.counters.output_underruns += 1
                    events.add("underrun", n, n * hop_s, missing)
            hop_ms = time.perf_counter() - hop_start
            compute += hop_ms
            if not decision.speech:
                closed_ms.append(hop_ms * 1000.0)
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
            sola_jumps_ms=sola.jumps_ms(plan.device_sr),
            alloc=self.alloc.summary(skip=min(defaults.PROFILER_WARMUP_HOPS, hops // 2)),
            pointers_stable=stable,
            gate_open=gate_open if silence_gate is not None else None,
            gate_transitions=transitions,
            closed_hop_ms=np.asarray(closed_ms, dtype=np.float64),
            underruns=underruns if realtime else None,
            events=events.since(0),
        )
