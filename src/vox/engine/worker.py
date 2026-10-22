"""Inference worker thread decoupling <=10 ms callbacks from the 20-30 ms hop.

Contract: docs/research/12 §3.6, 03 §10. The loop per hop:

    wait hop_event -> read one hop from the input ring -> condition -> upload -> gate.decide
      -> process_hop (graph replay) + sola.stitch | drain / zeros
      -> one device -> host copy -> write exactly one hop into the output ring

Everything the worker owns on the GPU was built by the control path before ``start``;
nothing here loads a model or allocates in steady state. Hot parameter changes arrive in a
single-slot mailbox and are applied between hops. Shape changes (hop, extra, F0 method)
go through ``_rebuild``: the gate is closed and drained, the plan replaced, graphs
recaptured and verified, then the input backlog is dropped so latency does not pile up.

The output ring is kept near ``fill_target``: zeros are inserted on an underrun, and once
the gate has been closed long enough that everything still queued is silence, the fill is
trimmed back if clock drift between the two devices (or an underrun refill) let it grow.
Those corrections are counted separately from underruns.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from vox.audio.conditioning import ConditioningConfig, InputConditioner
from vox.audio.recorder import SessionRecorder
from vox.audio.rings import SPSCRing
from vox.audio.volume import SpeechLevel
from vox.config.schemas import RuntimeParams, ShapeParams
from vox.engine import chunker, warmup
from vox.engine.gate import GateConfig, GateDecision, SilenceGate
from vox.engine.graphs import StageGraphs
from vox.engine.override import ClipOverride
from vox.engine.pipeline import HopPipeline
from vox.engine.sola import Sola
from vox.pitch.base import PitchExtractor
from vox.profiling.timers import StageTimers
from vox.synthesis.formant import snap_formant

HOLD_CUSHION_S = 0.5  # silence queued while a hold shares the GPU with a clip render


def _applied(shape: ShapeParams, plan: chunker.ChunkPlan) -> ShapeParams:
    """``shape`` with the sizes the chunker actually planned (whole 10 ms units)."""
    return dataclasses.replace(shape, block_ms=plan.block_ms,
                               crossfade_ms=plan.ms(plan.crossfade),
                               sola_search_ms=plan.search_ms, extra_ms=plan.extra_ms,
                               extra_right_ms=plan.right_ms, device_sr=plan.device_sr)


@dataclass
class ShapeRequest:
    shape: ShapeParams
    f0_method: str | None = None
    sola_mode: str | None = None


@dataclass
class WorkerStats:
    hops: int = 0
    gated_hops: int = 0
    gate_transitions: int = 0
    gate_open: bool = False
    dropped_input_samples: int = 0
    underrun_refills: int = 0
    drift_trims: int = 0
    rebuilds: int = 0
    last_rebuild_ms: float = 0.0
    # latency terms, running means over the last stats window (ms)
    in_backlog_ms: float = 0.0
    out_fill_ms: float = 0.0
    _sum_in: float = 0.0
    _sum_out: float = 0.0
    _n: int = 0
    gate_rms_dbfs: float = -120.0  # of the conditioned block: what the gate measured
    gate_vad: float = 0.0
    input_gain_db: float = 0.0  # input conditioning (auto level + trim) on the last block
    errors: list[str] = field(default_factory=list)

    def window(self) -> None:
        """Fold the running sums into the public means and start a new window."""
        if self._n:
            self.in_backlog_ms = self._sum_in / self._n
            self.out_fill_ms = self._sum_out / self._n
        self._sum_in = self._sum_out = 0.0
        self._n = 0


class InferenceWorker(threading.Thread):
    # The drift trim waits until the queued output is all silence (docs/DEVIATIONS.md
    # 14.1). False = the pre-fix rule (trim on the first closed hop), for the A/B soak only.
    trim_waits_for_silence = True

    def __init__(self, pipeline: HopPipeline, sola: Sola, gate: SilenceGate,
                 graphs: StageGraphs, timers: StageTimers, in_ring: SPSCRing,
                 out_ring: SPSCRing, hop_event: threading.Event, fill_target: int,
                 extractors: dict[str, PitchExtractor] | None = None,
                 log=print, conditioner: InputConditioner | None = None,
                 shape: ShapeParams | None = None,
                 fill_for_hop: Callable[[float], float] | None = None) -> None:
        super().__init__(name="vox-inference", daemon=True)
        self.pipeline = pipeline
        # The shape the hop runs at, as the chunker rounded it: replaced by ``_rebuild``
        # together with the plan, so a report never pairs a new shape with an old plan.
        self.shape = _applied(shape or ShapeParams(device_sr=pipeline.plan.device_sr),
                              pipeline.plan)
        self.sola = sola
        self.gate = gate
        # The microphone's input chain (high-pass, auto level, trim, soft limit) runs on
        # each block before the upload and the gate. None: raw blocks.
        self.conditioner = conditioner
        self.graphs = graphs
        self.timers = timers
        self.in_ring = in_ring
        self.out_ring = out_ring
        self.hop_event = hop_event
        self.fill_target = fill_target
        # hop ms -> fill target ms (the session's fill rule), for a live hop change
        self.fill_for_hop = fill_for_hop
        self.extractors = extractors or {}
        self.log = log
        self.stats = WorkerStats()
        self._stop = threading.Event()
        self._pending_params: RuntimeParams | None = None
        self._pending_gate: GateConfig | None = None
        self._pending_conditioning: ConditioningConfig | None = None
        self._pending_shape: ShapeRequest | None = None
        self.params = pipeline.params
        self._seen_underruns = 0
        self._seen_underrun_samples = 0
        self.ready = threading.Event()
        self.first_hop = threading.Event()  # set once the first hop is in the output ring
        # typed clips over the live voice
        self.override = ClipOverride(self.plan.block, self.plan.device_sr)
        # Level of the converted voice while it speaks: typed clips are matched to it.
        self.voice_level = SpeechLevel()
        self.hold = False  # control path: treat every hop as gated (see ``_hop``)
        # Control path sets it to record a take (audio/recorder.py): raw mic, conditioned
        # mic and output, one hop at a time. None: not recording.
        self.recorder: SessionRecorder | None = None
        self._cushion = 0  # extra silence queued for the duration of a hold, in samples
        self._silent_run = 0  # consecutive hops that wrote nothing but zeros (drift trim)
        self._fill_before_hold = 0  # output ring fill when the cushion went in
        self._alloc()

    # ------------------------------------------------------------------ control path

    @property
    def plan(self) -> chunker.ChunkPlan:
        return self.pipeline.plan

    def _alloc(self) -> None:
        block = self.plan.block
        cuda = self.pipeline.device.type == "cuda"
        self._host_block = np.zeros(block, dtype=np.float32)
        self._host_out = torch.zeros(block, pin_memory=cuda)
        self._host_out_np = self._host_out.numpy()
        self._zeros = torch.zeros(block, device=self.pipeline.device)
        self._zeros_np = np.zeros(block, dtype=np.float32)
        self.override.resize(block)

    def apply_params(self, params: RuntimeParams) -> None:
        """Hot parameters: applied before the next hop (single-slot mailbox)."""
        self._pending_params = params

    def apply_gate(self, config: GateConfig) -> None:
        self._pending_gate = config

    def apply_conditioning(self, config: ConditioningConfig) -> None:
        """Input chain: auto level on/off, trim. Applied before the next hop."""
        self._pending_conditioning = config

    def apply_shape(self, request: ShapeRequest) -> None:
        """Restart-class change: the worker rebuilds between hops (brief gated pause)."""
        self._pending_shape = request

    def play_clip(self, clip: np.ndarray) -> None:
        """Device-rate mono clip that takes over the output from the next hop; the live
        voice is faded out for its duration and back in after (engine/override.py)."""
        self.override.submit(clip)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.hop_event.set()
        self.join(timeout=timeout)

    def latency_estimate_ms(self, device_ms: float, compute_p50_ms: float) -> dict[str, float]:
        """End-to-end estimate: hop accumulation + input backlog + compute + output ring wait
        + SOLA delay + device buffers (docs/research/04 §2)."""
        plan = self.plan
        s = self.stats
        parts = {
            "hop": plan.block_ms, "input_backlog": s.in_backlog_ms, "compute": compute_p50_ms,
            "output_ring": s.out_fill_ms, "sola": plan.overlap_ms + plan.search_ms,
            "lookahead": plan.right_ms,  # extra_right_ms: the kept region's right context
            "devices": device_ms,
            # RNNoise's own delay while suppression runs (audio/suppression.py)
            "input_chain": self.conditioner.delay_ms if self.conditioner is not None else 0.0,
        }
        parts["total"] = sum(parts.values())
        return parts

    # ---------------------------------------------------------------------- hot loop

    def run(self) -> None:
        try:
            self.ready.set()
            while not self._stop.is_set():
                if not self.hop_event.wait(timeout=0.05):
                    continue
                self.hop_event.clear()
                self._apply_pending()
                while not self._stop.is_set() and self.in_ring.available() >= self.plan.block:
                    self._hop()
        except Exception as err:  # noqa: BLE001 - surface, never die silently
            self.stats.errors.append(f"{type(err).__name__}: {err}")
            self.log(f"worker: fatal {type(err).__name__}: {err}")
            raise

    def _prefill(self, n: int) -> None:
        """Zeros into the output ring so the callback has audio while the first hop runs."""
        if n > 0:
            self.out_ring.write(np.zeros(n, dtype=np.float32))

    def _apply_pending(self) -> None:
        shape = self._pending_shape
        if shape is not None:
            self._pending_shape = None
            self._rebuild(shape)
        params = self._pending_params
        if params is not None:
            self._pending_params = None
            self.params = params
            # A first visit to a formant step builds its warp (tens of ms) and captures its
            # synth graph (~250 ms): fade the voice out first so neither is heard as a stall.
            st = snap_formant(params.formant_st)
            heavy = st != self.pipeline.formant_st and (
                not self.pipeline.formant_is_built(st) or self.graphs.would_capture(st))
            start = time.perf_counter()
            if heavy:
                self._fade_out()
            if self.pipeline.set_params(params):
                self._recapture("f0_unvoiced changed")
            self.sola.set_hold_eps(params.sola_hold_eps)
            outcome = self.graphs.sync_formant()
            if heavy:
                self._resume()
                self.log(f"worker: formant {st:+g} st: synth graph {outcome} in "
                         f"{(time.perf_counter() - start) * 1000:.0f} ms")
        gate = self._pending_gate
        if gate is not None:
            self._pending_gate = None
            self.gate.set_config(gate)
        conditioning = self._pending_conditioning
        if conditioning is not None and self.conditioner is not None:
            self._pending_conditioning = None
            self.conditioner.set_config(conditioning)

    def _hop(self) -> None:
        plan, pipeline, sola, gate, timers = (self.plan, self.pipeline, self.sola, self.gate,
                                              self.timers)
        block = plan.block
        s = self.stats
        # Backlog beyond one hop means we are late; it is latency, so measure it.
        backlog = self.in_ring.available() - block
        timers.begin_hop()
        with timers.stage("upload"):
            self.in_ring.read(self._host_block)
            block_in = self._host_block
            if self.conditioner is not None:
                block_in = self.conditioner.process(block_in)
                s.input_gain_db = self.conditioner.gain_db
            pipeline.upload(block_in)
        with timers.stage("gate"):
            decision: GateDecision = gate.decide(block_in)
        events = timers.events
        if decision.onset or decision.drain:  # the gate's own decision, before any hold
            events.add("gate_open" if decision.onset else "gate_close", s.hops, events.now(),
                       decision.rms_dbfs)
        # Read the flag once: it is set from the control thread and must mean the same
        # thing to the gate decision above and to the cushion logic below.
        hold = self.hold
        if hold or self.override.active:
            # A typed clip is being rendered or is playing: the live voice is silent
            # anyway, so close it the way the gate would (SOLA tail faded out once, the
            # context ring kept current) and leave the GPU to the clip. When the hold
            # ends, speech resumes exactly like a gate onset.
            decision = GateDecision(False, drain=s.gate_open, rms_dbfs=decision.rms_dbfs,
                                    vad_prob=decision.vad_prob)
        silent = False  # this hop puts nothing but zeros in the ring
        if decision.speech:
            if decision.onset and not sola.draining:
                # Gate onset: short fade-in (keeps the decoded pre-onset consonant) and the
                # sine phase restarts, as the first hop of a stream does.
                sola.mark_onset(decision.energy_only)
                pipeline.on_onset()
            infer = pipeline.process_hop()
            with timers.stage("sola"):
                stitched = sola.stitch(infer)
        else:
            if decision.drain:
                sola.begin_drain()
                pipeline.forget_speech()
            pipeline.advance_context()
            silent = not sola.draining  # else this hop still carries the fading tail
            with timers.stage("sola"):
                stitched = self._zeros if silent else sola.drain_step()
            s.gated_hops += 1
        if decision.onset or decision.drain:
            s.gate_transitions += 1
            timers.counters.vad_transitions += 1
        s.gate_open = decision.speech
        s.gate_rms_dbfs = decision.rms_dbfs
        s.gate_vad = decision.vad_prob
        with timers.stage("download"):
            self._host_out.copy_(stitched)  # syncs the hop
        if decision.speech:
            self.voice_level.add(self._host_out_np, block / plan.device_sr)
        pipeline.register.publish()  # the GUI reads auto pitch from this host mirror
        fill_before = self.out_ring.fill()
        if s.hops == 0:
            # The output stream starts once this hop is in the ring (LiveSession waits for
            # ``first_hop``), so the prefill goes in now, right under the first real hop.
            self._prefill(self.fill_target)
            fill_before = self.fill_target
        if self.out_ring.underruns > self._seen_underruns:
            # The callback ran dry: top the ring back up to the target so one late hop
            # does not become a run of underruns. Counted as a refill, audible as a gap.
            ring = self.out_ring
            missing = ring.underrun_samples - self._seen_underrun_samples
            events.add("underrun", s.hops, ring.last_underrun_at - events.t0, missing,
                       count=ring.underruns - self._seen_underruns)
            self._seen_underruns = ring.underruns
            self._seen_underrun_samples = ring.underrun_samples
            self._prefill(self.fill_target - fill_before)
            s.underrun_refills += 1
            fill_before = self.fill_target
        releasing = False
        if hold:
            if not self._cushion and silent:
                # The GPU is about to be shared with a clip render and hops may run late.
                # The fade-out is in the ring and this hop is silence, so queue more
                # silence in front of it: a late hop then delays silence instead of
                # running the device dry. (Never in front of a hop that still carries
                # the tail: that would play as a blip after the cushion.)
                self._cushion = int(HOLD_CUSHION_S * plan.device_sr)
                self._fill_before_hold = fill_before
                self._prefill(self._cushion)
        elif self._cushion:
            # Hold over: what is queued is still all silence, so trim back to the fill the
            # ring had before (never thinner than one hop under the target, where a clip
            # would start on the edge of an underrun) and the old latency is restored.
            # Asked of the reader, which owns the read position (a drop from this thread
            # can be lost against a callback read, leaving the cushion in as latency).
            keep = min(self.fill_target, max(self._fill_before_hold, self.fill_target - block))
            self.out_ring.request_drop(self.out_ring.fill() - keep)
            self._cushion = 0
            releasing = True  # the reader trims; do not also drift-trim this hop
        overriding = self.override.mix(self._host_out_np)
        self.out_ring.write(self._host_out_np)
        recorder = self.recorder  # read once: the control thread starts and stops takes
        if recorder is not None:
            recorder.add(self._host_block, block_in, self._host_out_np, decision.speech)
        self._silent_run = self._silent_run + 1 if silent and not overriding else 0
        if (not decision.speech and not overriding and not hold and not releasing
                and not self.out_ring.drop_pending
                and fill_before > self.fill_target + block
                and (not self.trim_waits_for_silence
                     or self._silent_run * block >= fill_before + block)):
            # Everything queued was written by silent hops (the drain is spent and the
            # speech before it has played out), so trimming clock-drift growth drops
            # nothing anyone hears. Trimming on the first closed hop dropped the oldest
            # queued output, which was still the end of the sentence (docs/DEVIATIONS.md
            # 14.2): the ring then held ``fill`` ms of the hangover's render.
            self.out_ring.drop(fill_before - self.fill_target)
            s.drift_trims += 1
        timers.end_hop()
        s.hops += 1
        if s.hops == 1:
            self.first_hop.set()
        s._sum_in += 1000.0 * backlog / plan.device_sr
        # The hold cushion is silence that is taken back out: it is not latency.
        s._sum_out += 1000.0 * max(0, fill_before - self._cushion) / plan.device_sr
        s._n += 1
        timers.counters.output_underruns = self.out_ring.underruns
        timers.counters.input_overflows = self.in_ring.overruns

    # ------------------------------------------------------- rebuild (between hops)

    def _recapture(self, why: str) -> None:
        if not self.graphs.supported():
            return
        start = time.perf_counter()
        try:
            self.graphs.capture()
        except Exception as err:  # noqa: BLE001
            self.log(f"worker: recapture after {why} failed ({err}); running eager")
        self.pipeline.on_onset()  # carried sine phase restarts with the new graphs
        self.log(f"worker: recaptured graphs ({why}) in "
                 f"{(time.perf_counter() - start) * 1000:.0f} ms")

    def _fade_out(self) -> None:
        """Close the voice cleanly: fade the SOLA tail into the output ring, reset the gate
        (it reopens with the usual onset fade once the capture is done)."""
        if self.gate.open:
            self.sola.begin_drain()
            while self.sola.draining:
                self._host_out.copy_(self.sola.drain_step())
                self.out_ring.write(self._host_out_np)
            self.gate.reset()

    def _resume(self) -> None:
        """After a stall: drop the input that piled up (keep one hop) and refill the
        output to the target, so the stall does not stay behind as latency."""
        dropped = self.in_ring.keep_latest(self.plan.block)
        self.stats.dropped_input_samples += dropped
        self.out_ring.drop(self.out_ring.available())
        self._prefill(self.fill_target)
        self.pipeline.on_onset()

    def _rebuild(self, request: ShapeRequest) -> None:
        """Apply a restart-class change in place: drain, replan, recapture, resume."""
        start = time.perf_counter()
        pipeline, old_plan = self.pipeline, self.plan
        # 1. close the gate audibly cleanly: fade the SOLA tail out into the output ring
        if self.gate.open:
            self.sola.begin_drain()
            while self.sola.draining:
                self._host_out.copy_(self.sola.drain_step())
                self.out_ring.write(self._host_out_np)
            self.gate.reset()
        # 2. swap the extractor if asked (pre-loaded objects only: no loading here)
        if request.f0_method and request.f0_method in self.extractors:
            pipeline.extractor = self.extractors[request.f0_method]
            self.params = dataclasses.replace(self.params, f0_method=request.f0_method)
        shape = request.shape
        model = pipeline.model
        f0_tail = 0
        if pipeline.extractor is not None:
            block_16k = chunker.from_shape(shape, model.tgt_sr).block_16k
            f0_tail = pipeline.extractor.tail_samples(block_16k)
        plan = chunker.from_shape(shape, model.tgt_sr, f0_tail)
        if plan.overlap_note:
            self.log(f"worker: {plan.overlap_note}")
        self.shape = _applied(shape, plan)
        # 3. replan everything that depends on the plan
        pipeline.set_plan(plan)
        pipeline.set_params(self.params)
        mode = request.sola_mode or self.sola.mode
        self.sola = Sola(plan, pipeline.device, mode, self.params.sola_hold_eps)
        self.graphs.sola = self.sola
        self.gate = SilenceGate(plan, self.gate.config, self.gate.vad)
        if self.conditioner is not None:
            self.conditioner.resize(plan.block)  # filter state and level carry over
        self._alloc()
        if plan.block != old_plan.block:
            if self.fill_for_hop is not None:
                self.fill_target = int(round(plan.device_sr * self.fill_for_hop(plan.block_ms)
                                             / 1000.0))
            else:
                self.fill_target = int(round(self.fill_target * plan.block / old_plan.block))
        # 4. warm up and capture at the new shapes (resets pipeline + SOLA state)
        warmup.run(pipeline, self.sola, 3, self.graphs, log=self.log)
        # 5. the input piled up meanwhile: keep one hop, refill the output to target
        dropped = self.in_ring.keep_latest(plan.block)
        self.stats.dropped_input_samples += dropped
        self.out_ring.drop(self.out_ring.available())
        self._prefill(self.fill_target)
        self.stats.rebuilds += 1
        self.stats.last_rebuild_ms = (time.perf_counter() - start) * 1000.0
        self.log(f"worker: rebuilt for hop {plan.block_ms:.0f} ms / extra {plan.extra_ms:.0f} ms"
                 f" / f0 {self.params.f0_method} in {self.stats.last_rebuild_ms:.0f} ms "
                 f"({self.graphs.report.mode}: {'graphs' if self.graphs.active else 'eager'})")
