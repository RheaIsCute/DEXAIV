"""LiveSession: the headless start sequence and command surface (docs/research/12 §3.12).

Control path only. Builds the same engine objects the stream simulator uses (P17), adds the
rings, the worker thread and the PortAudio streams, and exposes hot parameter changes,
restart-class changes and a once-per-second stats snapshot. ``vox run`` is a thin shell
around it; ``scripts/measure_rtt.py`` drives it with a file-backed input.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vox.audio import devices as audio_devices
from vox.audio import suppression
from vox.audio.conditioning import ConditioningConfig
from vox.audio.recorder import SessionRecorder
from vox.audio.rings import SPSCRing
from vox.audio.streams import DuplexAudio, FileInput, OutputTap, StreamConfig
from vox.config import defaults
from vox.config.schemas import PrecisionParams, RuntimeParams, ShapeParams
from vox.engine.gate import GateConfig
from vox.engine.worker import InferenceWorker, ShapeRequest
from vox.offline.simulate import Engine, load_engine
from vox.pitch import EXTRACTORS, make_extractor
from vox.profiling.timers import TOTAL

INPUT_RING_S = 2.0
OUTPUT_RING_S = 1.0
STATS_WINDOW_HOPS = 50  # the 1 Hz line summarises about the last second at a 20 ms hop


@dataclass
class LiveConfig:
    model: str
    models_dir: Path
    shape: ShapeParams = field(default_factory=lambda: defaults.DEFAULT_SHAPE)
    params: RuntimeParams = field(default_factory=RuntimeParams)
    precision: PrecisionParams = field(default_factory=lambda: defaults.DEFAULT_PRECISION)
    gate: GateConfig = field(default_factory=GateConfig)
    # The microphone's input chain (high-pass, auto level, trim): audio/conditioning.py
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    input: str | int | None = None  # device index / name fragment / None = default
    output: str | int | None = None
    exclusive: bool = False  # WASAPI exclusive on the output stream
    exclusive_input: bool = False  # and on the input (check the delivered rate: R9)
    callback_ms: float = 10.0
    fill_target_ms: float | None = None  # default: two hops + one callback block
    input_file: Path | None = None  # software microphone (see streams.FileInput)
    loop_file: bool = False
    device: str | None = None
    preload_extractors: bool = True  # so a live F0-method switch never loads a model
    registry: Any = None  # a ModelRegistry to reuse (keeps the model loaded across sessions)
    cache: dict | None = None  # loaded embedder / extractors / VAD to reuse likewise
    profile_dir: Path | None = None  # where warmup's tuning decisions are persisted
    # Noise suppression (the engine's RNNoise path): None = resolve at start from the
    # profile's setting, except that the NVIDIA Broadcast virtual mic (already denoised)
    # defaults it OFF (TARGET.md: double denoising hurts). True/False = explicit.
    suppression: bool | None = None
    suppression_profile: bool = False  # calibration.json's rnnoise_enabled
    record_output_s: float = 0.0  # > 0: tap what the device plays (streams.OutputTap)


BROADCAST_MARKER = "nvidia broadcast"


def resolve_suppression(requested: bool | None, profile_default: bool,
                        input_name: str | None) -> tuple[bool, str]:
    """The effective suppression setting and a one-line reason."""
    if requested is not None:
        return requested, "explicit"
    if input_name and BROADCAST_MARKER in input_name.lower():
        return False, "input is the NVIDIA Broadcast virtual mic (already denoised)"
    return profile_default, "profile"


class LiveSession:
    def __init__(self, cfg: LiveConfig, log=print) -> None:
        self.cfg = cfg
        self.log = log
        self.engine: Engine | None = None
        self.worker: InferenceWorker | None = None
        self.audio: DuplexAudio | None = None
        self.file_input: FileInput | None = None
        self.in_ring: SPSCRing | None = None
        self.out_ring: SPSCRing | None = None
        self.hop_event = threading.Event()
        self.started_at = 0.0
        self.warnings: list[str] = []
        self.in_device = None
        self.out_device = None
        self.suppression = False
        self.suppression_why = ""
        self._event_seq = 0  # first event the next snapshot reports
        self._record_events = 0  # first event of the current take
        self._record_started = 0.0

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        cfg = self.cfg
        sr = cfg.shape.device_sr
        self.out_device = audio_devices.resolve_device(cfg.output, "output")
        if cfg.input_file is None:
            self.in_device = audio_devices.resolve_device(cfg.input, "input")
        for dev, kind in ((self.in_device, "input"), (self.out_device, "output")):
            if dev is not None:
                warning = audio_devices.check_rate(dev, sr, kind)
                if warning:
                    self.warnings.append(warning)
                    self.log(f"warning: {warning}")

        input_name = self.in_device.name if self.in_device is not None else None
        self.suppression, why = resolve_suppression(cfg.suppression, cfg.suppression_profile,
                                                    input_name)
        reason = suppression.unavailable_reason(sr)
        if self.suppression and reason is not None:
            self.suppression, why = False, f"unavailable: {reason}"
        self.suppression_why = why
        self.log(f"suppression: {'on' if self.suppression else 'off'} ({why})")
        cfg.conditioning = dataclasses.replace(cfg.conditioning, suppression=self.suppression)

        cache: dict = cfg.cache if cfg.cache is not None else {}
        engine = load_engine(cfg.models_dir, cfg.model, cfg.shape, cfg.params, cfg.precision,
                             cfg.device, registry=cfg.registry, cache=cache, gate=cfg.gate,
                             log=self.log, profile_dir=cfg.profile_dir,
                             conditioning=cfg.conditioning)
        self.engine = engine
        extractors = {}
        if cfg.preload_extractors and engine.pipeline.extractor is not None:
            for name in EXTRACTORS:
                key = ("extractor", name, str(engine.device))
                if key not in cache:
                    cache[key] = make_extractor(name)
                    cache[key].load(engine.device, Path(cfg.models_dir))
                extractors[name] = cache[key]
        plan = engine.plan
        block = plan.block
        callback_block = int(round(sr * cfg.callback_ms / 1000.0))
        # The first hop is ready one hop of accumulation plus compute (<= one hop under
        # graphs) after its first sample; the callback pulls in callback-sized steps.
        fill_ms = cfg.fill_target_ms if cfg.fill_target_ms is not None else \
            2 * plan.block_ms + cfg.callback_ms
        fill_target = int(round(sr * fill_ms / 1000.0))
        self.in_ring = SPSCRing(int(sr * INPUT_RING_S), guard=4 * callback_block)
        self.out_ring = SPSCRing(int(sr * OUTPUT_RING_S), guard=2 * block)
        self.worker = InferenceWorker(engine.pipeline, engine.sola, engine.gate, engine.graphs,
                                      engine.timers, self.in_ring, self.out_ring, self.hop_event,
                                      fill_target, extractors, log=self.log,
                                      conditioner=engine.conditioner, shape=cfg.shape)
        self.worker.start()
        self.worker.ready.wait(timeout=10.0)

        stream_cfg = StreamConfig(self.in_device, self.out_device, sr, cfg.callback_ms,
                                  cfg.exclusive, cfg.exclusive_input)
        tap = OutputTap(sr, cfg.record_output_s) if cfg.record_output_s > 0 else None
        self.audio = DuplexAudio(stream_cfg, self.in_ring, self.out_ring, block, self.hop_event,
                                 tap)
        if cfg.input_file is not None:
            from vox.audio.resample import load_audio

            audio, _ = load_audio(cfg.input_file, sr)
            self.file_input = FileInput(audio, sr, callback_block, self.in_ring, block,
                                        self.hop_event, loop=cfg.loop_file)
        # Event times (gate closures, underruns) count from the moment audio starts.
        engine.timers.events.reset()
        self._event_seq = 0
        # Input first; the output stream opens once the first hop (and the prefill under
        # it) is in the output ring, so the device never starts on an empty ring.
        self.audio.start_input()
        if self.file_input is not None:
            self.file_input.start()
        if not self.worker.first_hop.wait(timeout=10.0):
            self.log("warning: no input arrived within 10 s; starting the output anyway")
        self.audio.start_output()
        self.started_at = time.perf_counter()
        self.log(self.describe())

    def stop(self) -> None:
        if self.file_input is not None:
            self.file_input.stop()
        if self.audio is not None:
            self.audio.stop()
        if self.worker is not None:
            self.worker.stop()

    def describe(self) -> str:
        e, cfg = self.engine, self.cfg
        plan = self.worker.plan  # the active plan: ``engine.plan`` is the one at start
        src = (f"file {cfg.input_file.name}" if cfg.input_file is not None
               else f"{self.in_device.index} {self.in_device.name}")
        return (f"live: {cfg.model} | in: {src} | out: {self.out_device.index} "
                f"{self.out_device.name} | {plan.device_sr} Hz, callback {cfg.callback_ms:.0f} ms"
                f" | hop {plan.block_ms:.0f} ms, overlap {plan.overlap_ms:.0f}, search "
                f"{plan.search_ms:.0f}, extra {plan.extra_ms:.0f} | f0 "
                f"{self.worker.params.f0_method} | "
                f"mode {e.mode} | gate {'on' if cfg.gate.enabled else 'off'} | "
                f"suppression {'on' if self.suppression else 'off'} | "
                f"fill target {1000 * self.worker.fill_target / plan.device_sr:.0f} ms | "
                f"device latency in {self.audio.health.input_latency_ms:.1f} / out "
                f"{self.audio.health.output_latency_ms:.1f} ms")

    # ------------------------------------------------------------------- commands

    def set_params(self, **changes: Any) -> RuntimeParams:
        """Hot parameters (pitch, index_rate, protect, volume_envelope, f0_unvoiced, sid)."""
        params = dataclasses.replace(self.worker.params, **changes)
        self.worker.apply_params(params)
        return params

    def set_gate(self, enabled: bool | None = None, **changes: Any) -> GateConfig:
        current = self.worker.gate.config
        if enabled is not None:
            changes["enabled"] = enabled
        config = dataclasses.replace(current, **changes)
        self.worker.apply_gate(config)
        self.cfg.gate = config
        return config

    # ---------------------------------------------------------------- recording

    @property
    def recording(self) -> SessionRecorder | None:
        return self.worker.recorder if self.worker is not None else None

    def start_recording(self) -> None:
        """Start a take: raw mic, conditioned mic and output, aligned (audio/recorder.py)."""
        if self.worker.recorder is not None:
            return
        self._record_events = self.engine.timers.events.total
        self._record_started = time.time()
        self.worker.recorder = SessionRecorder(self.engine.plan.device_sr)

    def stop_recording(self, folder: Path) -> dict[str, Path] | None:
        """End the take and write it to ``folder``; None when nothing was recording."""
        recorder, self.worker.recorder = self.worker.recorder, None
        if recorder is None:
            return None
        time.sleep(2 * self.worker.plan.block_ms / 1000.0)  # let a hop in flight finish
        e, w = self.engine, self.worker
        plan = w.plan  # what ran: a live rebuild replaces it, ``engine.plan`` stays stale
        events = e.timers.events
        first = self._record_events
        meta = {
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._record_started)),
            "model": self.cfg.model, "input": self.describe(),
            "params": dataclasses.asdict(w.params),
            "shape": dataclasses.asdict(w.shape),
            "gate": dataclasses.asdict(w.gate.config),
            "conditioning": dataclasses.asdict(self.cfg.conditioning),
            # RNNoise delays the conditioned track (and the output) against mic_raw
            "input_chain_delay_ms": (e.conditioner.delay_ms
                                     if getattr(e, "conditioner", None) is not None else 0.0),
            "plan": {"hop_ms": plan.block_ms, "overlap_ms": plan.overlap_ms,
                     "search_ms": plan.search_ms, "extra_ms": plan.extra_ms},
            "mode": e.mode,
            "event_counts_in_take": {k: sum(1 for ev in events.since(first) if ev["kind"] == k)
                                     for k in ("gate_open", "gate_close", "underrun")},
            "events": events.since(first),
            "underruns_session": self.out_ring.underruns,
            "note": "tracks are sample-aligned; output trails the mic by the SOLA delay "
                    f"({plan.stream_delay * 1000 / plan.device_sr:.0f} ms)",
        }
        stem = time.strftime("take_%Y%m%d_%H%M%S", time.localtime(self._record_started))
        return recorder.save(folder, stem, meta)

    def set_conditioning(self, **changes: Any) -> ConditioningConfig:
        """Input chain, hot: ``auto_level``, ``trim_db`` (and the AGC's bounds),
        ``suppression`` and ``suppression_mix`` (crossfaded in the conditioner)."""
        config = dataclasses.replace(self.cfg.conditioning, **changes)
        self.cfg.conditioning = config
        self.worker.apply_conditioning(config)
        return config

    def set_shape(self, f0_method: str | None = None, sola_mode: str | None = None,
                  **changes: Any) -> ShapeParams:
        """Restart-class: hop / extra / crossfade / F0 method. Brief gated pause. Returns
        the shape requested; ``worker.shape`` becomes it (as the chunker rounds it) once
        the worker has rebuilt."""
        shape = dataclasses.replace(self.cfg.shape, **changes)
        self.cfg.shape = shape
        self.worker.apply_shape(ShapeRequest(shape, f0_method, sola_mode))
        return shape

    # ---------------------------------------------------------------------- stats

    def snapshot(self) -> dict[str, Any]:
        w, e, a = self.worker, self.engine, self.audio
        timers = e.timers
        plan = w.plan
        w.stats.window()
        stages = {}
        for i, name in enumerate(timers.stages):
            # Each stage keeps its own count: hops the gate skipped never ran the model
            # stages, so a window measured in total hops would come up empty for them.
            s = timers.host.summary(i, max(0, timers.host.total(i) - STATS_WINDOW_HOPS))
            if s.count:
                stages[name] = {"p50": s.p50, "p95": s.p95, "p99": s.p99}
                # Device time from CUDA events: under graphs the host only launches a replay.
                g = timers.gpu.summary(i, max(0, timers.gpu.total(i) - STATS_WINDOW_HOPS))
                if g.count:
                    stages[name].update(gpu_p50=g.p50, gpu_p99=g.p99)
        total = stages.get(TOTAL, {"p50": float("nan"), "p95": float("nan"),
                                   "p99": float("nan")})
        health = a.health
        device_ms = health.input_latency_ms + health.output_latency_ms
        if self.file_input is not None:
            device_ms = health.output_latency_ms
        estimate = w.latency_estimate_ms(device_ms, total["p50"])
        events = timers.events
        new_events = events.since(self._event_seq)
        self._event_seq = events.total
        return {
            "t": time.perf_counter() - self.started_at,
            "hops": w.stats.hops, "gated_hops": w.stats.gated_hops,
            "gate_open": w.stats.gate_open, "gate_rms_dbfs": w.stats.gate_rms_dbfs,
            "gate_vad": w.stats.gate_vad, "gate_transitions": w.stats.gate_transitions,
            "auto_pitch_st": e.pipeline.register.offset_st,  # nan: fixed or not warmed up
            "input_gain_db": w.stats.input_gain_db,
            "stages": stages, "total": total,
            "headroom": plan.block_ms / total["p99"] if total["p99"] else float("nan"),
            "in_fill_ms": 1000.0 * self.in_ring.fill() / plan.device_sr,
            "out_fill_ms": 1000.0 * self.out_ring.fill() / plan.device_sr,
            "out_fill_target_ms": 1000.0 * w.fill_target / plan.device_sr,
            "underruns": self.out_ring.underruns,
            "underrun_samples": self.out_ring.underrun_samples,
            "status_output_underflow": health.output_underflows,
            "status_input_overflow": health.input_overflows,
            "input_overruns": self.in_ring.overruns,
            "underrun_refills": w.stats.underrun_refills, "drift_trims": w.stats.drift_trims,
            "rebuilds": w.stats.rebuilds,
            "device_in_ms": health.input_latency_ms, "device_out_ms": health.output_latency_ms,
            "latency": estimate, "mode": e.mode,
            "input_peak": health.input_peak, "output_peak": health.output_peak,
            "input_rate_hz": a.input_rate_hz(),
            # Timestamped since the previous snapshot (seconds from audio start), and the
            # running totals: gate closures and underruns are logged apart (cut attribution).
            "events": new_events, "event_counts": dict(events.counts),
            "errors": list(w.stats.errors) + list(health.errors),
        }

    def stats_line(self, snap: dict[str, Any] | None = None) -> str:
        s = snap or self.snapshot()
        st, lat = s["stages"], s["latency"]

        def pair(name: str) -> str:
            row = st.get(name)
            return f"{row['p50']:.1f}/{row['p95']:.1f}" if row else "-"

        health = self.audio.health
        health.input_peak = health.output_peak = 0.0
        gate = "open" if s["gate_open"] else "shut"
        rate = s.get("input_rate_hz", 0.0)
        sr = self.engine.plan.device_sr
        rate_note = (f" | INPUT RATE {rate:.0f} Hz != {sr} (device/driver mismatch, R9)"
                     if rate and abs(rate - sr) > 0.02 * sr else "")
        rnnoise = f" + rnnoise {lat['input_chain']:.0f}" if lat.get("input_chain") else ""
        return (f"[{s['t']:6.1f}s] hop {s['total']['p50']:.1f}/{s['total']['p95']:.1f}/"
                f"{s['total']['p99']:.1f} ms  f0 {pair('f0')} emb {pair('embed')} syn "
                f"{pair('synth')} sola {pair('sola')} | ring in {s['in_fill_ms']:.0f} out "
                f"{s['out_fill_ms']:.0f}/{s['out_fill_target_ms']:.0f} ms | underruns "
                f"{s['underruns']} (flags {s['status_output_underflow']}) refills "
                f"{s['underrun_refills']} trims {s['drift_trims']} | gate {gate} "
                f"{s['gate_rms_dbfs']:.0f} dB vad {s['gate_vad']:.2f} closes "
                f"{s['event_counts']['gate_close']} | in gain {s['input_gain_db']:+.0f} dB | est "
                f"{lat['total']:.0f} ms (hop {lat['hop']:.0f} + in {lat['input_backlog']:.0f} + "
                f"cmp {lat['compute']:.0f} + out {lat['output_ring']:.0f} + sola {lat['sola']:.0f}"
                f" + dev {lat['devices']:.0f}{rnnoise}) | peak in {s['input_peak']:.2f} out "
                f"{s['output_peak']:.2f}{rate_note}")
