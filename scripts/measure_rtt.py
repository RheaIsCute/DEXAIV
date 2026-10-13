"""Loopback round-trip latency of the live engine (Phase 3 acceptance).

Setup: the engine runs with a *software microphone* (``FileInput``: a known signal fed into
the input ring at real-time pace, each block stamped with ``perf_counter``) and renders to
a virtual cable's playback end. This script records the cable's capture end and finds the
onset of each burst in the recording. Round trip = recorded onset time - injected onset
time, minus the recorder's own input latency as PortAudio reports it. That covers the input
ring wait, the gate, compute, the output ring, the output device buffer and the cable; the
one term it cannot see is a physical microphone's own capture buffer, which PortAudio
reports separately (``vox run`` prints it as ``dev``).

    python scripts/measure_rtt.py --cable "Virtual Audio Cable" [--gate on|off]
        [--signal speech|burst] [--reps 8] [--hop-ms 20] [--extra-ms 1000]

A click into an RVC model is not a click out of it: the signal is either a speech excerpt
(gate on, the realistic case) or a voiced synthetic burst (``burst``, gate off).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SR = 48000


def build_signal(kind: str, reps: int, gap_s: float, burst_s: float) -> tuple[np.ndarray, list[int]]:
    """``reps`` bursts separated by silence. Returns the signal and the onset sample of each."""
    from vox.audio.resample import load_audio
    from vox.engine.warmup import test_signal
    from vox.offline.metrics import frame_rms_dbfs

    n_burst = int(burst_s * SR)
    if kind == "speech":
        clip, _ = load_audio(ROOT / "benchmarks" / "corpus" / "libri_1272.wav", SR)
        level = frame_rms_dbfs(clip, SR)
        first = int(np.argmax(level > -25.0)) * (SR // 100)  # first loud frame
        burst = clip[first:first + n_burst].copy()
    else:
        burst = 3.0 * test_signal(n_burst, SR)  # ~ -10 dBFS voiced tone
    fade = int(0.005 * SR)
    burst[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    gap = np.zeros(int(gap_s * SR), dtype=np.float32)
    pieces, onsets, pos = [gap], [], len(gap)
    for _ in range(reps):
        pieces += [burst, gap]
        onsets.append(pos)
        pos += len(burst) + len(gap)
    return np.concatenate(pieces).astype(np.float32), onsets


class Recorder:
    """Captures a device into memory, stamping every callback with perf_counter."""

    def __init__(self, device_index: int, sr: int, seconds: float, block: int = 480) -> None:
        self.sr = sr
        self.block = block
        self.buffer = np.zeros(int(seconds * sr) + 10 * block, dtype=np.float32)
        self.stamps: list[tuple[int, int, float]] = []  # (start sample, frames, perf_counter)
        self.n = 0
        self.latency_ms = 0.0
        self.device = device_index
        self._stream = None

    def _cb(self, indata, frames, time_info, status) -> None:
        if self.n + frames > len(self.buffer):
            return
        np.mean(indata, axis=1, out=self.buffer[self.n:self.n + frames])
        self.stamps.append((self.n, frames, time.perf_counter()))
        self.n += frames

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(device=self.device, samplerate=self.sr, blocksize=self.block,
                                      channels=2, dtype="float32", latency="low",
                                      extra_settings=sd.WasapiSettings(auto_convert=True),
                                      callback=self._cb)
        self._stream.start()
        self.latency_ms = 1000.0 * float(self._stream.latency)

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    def sample_time(self, i: int) -> float:
        """perf_counter at which sample ``i`` was captured (callback stamp minus its age)."""
        for start, frames, stamp in self.stamps:
            if start <= i < start + frames:
                # the callback fires after the block is complete; sample i is (end - i) old,
                # plus the device's own capture latency
                return stamp - (start + frames - i) / self.sr - self.latency_ms / 1000.0
        raise LookupError(i)


def onsets_in(audio: np.ndarray, sr: int, expected: int, min_gap_s: float,
              threshold_db: float) -> list[int]:
    from vox.offline.metrics import frame_rms_dbfs

    hop = sr // 100
    level = frame_rms_dbfs(audio, sr, hop_ms=10.0, win_ms=20.0)
    # With the gate off the model renders digital silence as a -30 dBFS hiss (DEVIATIONS
    # R4), so an onset must also clear the recording's own floor by a margin.
    floor = float(np.percentile(level[np.isfinite(level)], 40))
    threshold_db = max(threshold_db, floor + 12.0)
    found, last = [], -10 ** 9
    for k, db in enumerate(level):
        i = k * hop
        if db > threshold_db and i - last > min_gap_s * sr:
            # refine to the first sample above a tenth of the frame's RMS inside the frame
            seg = np.abs(audio[max(0, i - hop):i + hop])
            amp = 10 ** (db / 20.0) * 0.3
            j = int(np.argmax(seg > amp)) + max(0, i - hop) if (seg > amp).any() else i
            found.append(j)
            last = j
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="base_f0_48k_v2")
    parser.add_argument("--cable", default="Virtual Audio Cable",
                        help="name fragment of the virtual cable (both ends)")
    parser.add_argument("--signal", choices=("speech", "burst"), default="speech")
    parser.add_argument("--gate", choices=("on", "off"), default="on")
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--gap-s", type=float, default=1.5)
    parser.add_argument("--burst-s", type=float, default=1.0)
    parser.add_argument("--hop-ms", type=float, default=20.0)
    parser.add_argument("--extra-ms", type=float, default=1000.0)
    parser.add_argument("--crossfade-ms", type=float, default=30.0)
    parser.add_argument("--callback-ms", type=float, default=10.0)
    parser.add_argument("--fill-ms", type=float, default=None)
    parser.add_argument("--graphs", choices=("on", "off", "auto"), default="auto")
    parser.add_argument("--exclusive", action="store_true")
    parser.add_argument("--threshold-db", type=float, default=-35.0)
    args = parser.parse_args()

    import dataclasses

    from vox.api.control import LiveConfig, LiveSession
    from vox.audio import devices
    from vox.audio.resample import save_audio
    from vox.config import defaults
    from vox.config.schemas import PrecisionParams
    from vox.engine.gate import GateConfig
    from vox.profiling.report import environment_fingerprint, write_jsonl

    out_dev = devices.resolve_device(args.cable, "output")
    in_dev = devices.resolve_device(args.cable, "input")
    signal, onsets = build_signal(args.signal, args.reps, args.gap_s, args.burst_s)
    wav = ROOT / "out" / "rtt_input.wav"
    save_audio(wav, signal, SR)
    seconds = len(signal) / SR + 3.0

    shape = dataclasses.replace(defaults.DEFAULT_SHAPE, block_ms=args.hop_ms,
                                extra_ms=args.extra_ms, crossfade_ms=args.crossfade_ms)
    cfg = LiveConfig(model=args.model, models_dir=ROOT / "models", shape=shape,
                     precision=PrecisionParams(graphs=args.graphs),
                     gate=GateConfig(enabled=args.gate == "on"), output=out_dev.index,
                     exclusive=args.exclusive, callback_ms=args.callback_ms,
                     fill_target_ms=args.fill_ms, input_file=wav)
    recorder = Recorder(in_dev.index, SR, seconds)
    session = LiveSession(cfg)
    recorder.start()
    session.start()
    feeder = session.file_input
    snaps = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        time.sleep(1.0)
        snaps.append(session.snapshot())
        print(session.stats_line(snaps[-1]), flush=True)
    session.stop()
    recorder.stop()

    # injected onset times: the FileInput block that carried each onset sample
    entered = dict(feeder.entered)  # position -> perf_counter when written
    block = feeder.block
    inject_times = []
    for onset in onsets:
        start = onset - onset % block
        if start not in entered:
            continue
        # the block entered the ring at ``entered[start]``; its samples are "live" from then
        inject_times.append(entered[start])
    recorded = recorder.buffer[:recorder.n]
    save_audio(ROOT / "out" / "rtt_recorded.wav", recorded, SR)
    # one onset per burst: anything closer than a burst plus half a gap is the same burst
    found = onsets_in(recorded, SR, len(inject_times), args.burst_s + 0.5 * args.gap_s,
                      args.threshold_db)
    found_times = []
    for i_out in found:
        try:
            found_times.append(recorder.sample_time(i_out))
        except LookupError:
            pass
    t0 = inject_times[0] if inject_times else 0.0
    print("injected onsets (s):", [round(t - t0, 3) for t in inject_times])
    print("recorded onsets (s):", [round(t - t0, 3) for t in found_times])
    rtts = []
    for t_in in inject_times:  # the first recorded onset within a second after injection
        later = [t for t in found_times if 0 < t - t_in < 1.0]
        if later:
            rtts.append(1000.0 * (min(later) - t_in))
    rtts = np.array(rtts)
    print("round trips (ms):", [round(float(r), 1) for r in rtts])
    snap = snaps[-1]
    est = snap["latency"]
    print(f"\nbursts injected {len(inject_times)}, onsets found {len(found)}, usable {len(rtts)}")
    if rtts.size:
        print(f"round trip (injection -> cable capture): median {np.median(rtts):.1f} ms, "
              f"p95 {np.percentile(rtts, 95):.1f}, min {rtts.min():.1f}, max {rtts.max():.1f}")
    print(f"engine estimate at the end: {est['total']:.1f} ms = hop {est['hop']:.0f} + input "
          f"backlog {est['input_backlog']:.1f} + compute {est['compute']:.1f} + output ring "
          f"{est['output_ring']:.1f} + sola {est['sola']:.0f} + device {est['devices']:.1f}")
    if rtts.size:
        delta = float(np.median(rtts) - est["total"])
        verdict = "within +-15 ms" if abs(delta) <= 15 else "OUTSIDE +-15 ms"
        print(f"measured - estimate = {delta:+.1f} ms  -> {verdict}")
    print(f"underruns {snap['underruns']} (flags {snap['status_output_underflow']}), gated hops "
          f"{snap['gated_hops']}/{snap['hops']}, mode {snap['mode']}")
    write_jsonl(ROOT / "benchmarks" / "results" / "rtt.jsonl", {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": environment_fingerprint(),
        "args": vars(args) | {"cable": args.cable}, "rtt_ms": rtts.tolist(),
        "estimate": est, "recorder_latency_ms": recorder.latency_ms, "snapshot": snap})
    return 0


if __name__ == "__main__":
    sys.exit(main())
