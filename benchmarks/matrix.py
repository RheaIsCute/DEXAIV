"""The raw-microphone take protocol (docs/research/15, Phase 0.3): ``vox bench take``.

The dominant complaint ("every mic render stutters") has never been heard on a raw
microphone: every take so far went through the NVIDIA Broadcast virtual mic. This command
sequences the runs; the user does the talking. For every cell of

    input {Shure MV6 direct, NVIDIA Broadcast} x gate {off, on} x AGC {off, on} x hop {20, 40}

it configures the live engine (the same ``LiveSession`` the GUI drives), shows the fixed
script, records with the Record-button pipeline (raw mic, conditioned mic and output,
sample-aligned, plus the JSON of settings and gate events) and names the files after the
cell. Afterwards it scores every output (``quality.py``: WER against the script and the
microphone's own WER, speaker cosine, UTMOS, F0 statistics, click ratio, gate behaviour,
speech lost) and writes blind pairs for the user's ear. Nothing here decides what the
stutter is: that is the user's verdict on the pairs, run by run.

Pure parts (the run plan, labels, thresholds, pair choice, take analysis) are importable
and unit-tested; ``record_matrix`` needs the devices and the GPU.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from quality import Pair, QualityKit, Scores, load_script, wer

INPUTS = {"shure": "Shure MV6", "broadcast": "NVIDIA Broadcast"}  # key -> device name fragment
GATE_THRESHOLD_AGC = -40.0  # dBFS after auto level (docs/DEVIATIONS.md 7.3)
GATE_THRESHOLD_RAW = -55.0  # dBFS on the raw mic (speech sits near -42 on this one)
TRACKS = ("output", "mic_raw", "mic_conditioned")


@dataclass(frozen=True)
class Run:
    input_key: str
    input_spec: str  # device name fragment
    gate: bool
    agc: bool
    hop_ms: float

    @property
    def label(self) -> str:
        return (f"{self.input_key}_gate{'on' if self.gate else 'off'}"
                f"_agc{'on' if self.agc else 'off'}_hop{self.hop_ms:g}")

    def gate_threshold(self, agc_thr: float = GATE_THRESHOLD_AGC,
                       raw_thr: float = GATE_THRESHOLD_RAW) -> float:
        """The gate measures the conditioned level: with auto level the mic is lifted to
        about -20 dBFS, without it the threshold must sit under the raw speech level."""
        return agc_thr if self.agc else raw_thr

    def describe(self) -> str:
        return (f"{self.input_spec}, gate {'ON' if self.gate else 'OFF'}, auto level "
                f"{'ON' if self.agc else 'OFF'}, hop {self.hop_ms:g} ms")


def plan_runs(inputs: dict[str, str] | None = None, gates=(False, True), agcs=(False, True),
              hops=(20.0, 40.0)) -> list[Run]:
    """Every cell, ordered input -> hop -> (gate, agc): one session per input, one rebuild
    per hop, the gate and auto level switched hot between runs."""
    inputs = inputs or INPUTS
    return [Run(key, spec, gate, agc, float(hop))
            for key, spec in inputs.items() for hop in hops for gate in gates for agc in agcs]


def default_pairs(runs: list[Run]) -> list[Pair]:
    """One change per pair, on the smallest hop with everything else on: gate, auto level,
    hop, and the two inputs against each other. Only pairs whose both runs exist."""
    have = {r.label: r for r in runs}
    hops = sorted({r.hop_ms for r in runs})
    keys = list(dict.fromkeys(r.input_key for r in runs))
    pairs: list[Pair] = []

    def add(a: Run, b: Run, what: str) -> None:
        if a.label in have and b.label in have and a.label != b.label:
            pairs.append(Pair(a.label, b.label, what))

    for key in keys:
        spec = have[next(r.label for r in runs if r.input_key == key)].input_spec
        base = Run(key, spec, True, True, hops[0])
        add(base, dataclasses.replace(base, gate=False), f"{spec}: gate on vs off (stutter?)")
        add(base, dataclasses.replace(base, agc=False),
            f"{spec}: auto level on vs off (gain ramps audible?)")
        if len(hops) > 1:
            add(base, dataclasses.replace(base, hop_ms=hops[1]),
                f"{spec}: hop {hops[0]:g} vs {hops[1]:g} ms (seam density)")
        add(Run(key, spec, False, False, hops[0]), Run(key, spec, True, True, hops[0]),
            f"{spec}: everything off vs everything on")
    if len(keys) > 1:
        a = next(r for r in runs if r.input_key == keys[0])
        b = next(r for r in runs if r.input_key == keys[1])
        add(Run(keys[0], a.input_spec, True, True, hops[0]),
            Run(keys[1], b.input_spec, True, True, hops[0]),
            f"{a.input_spec} vs {b.input_spec} (the denoiser in front of the engine)")
    return pairs


# ------------------------------------------------------------------------ take analysis


def take_files(folder: Path, label: str) -> dict[str, Path]:
    folder = Path(folder)
    return {**{t: folder / f"{label}_{t}.wav" for t in TRACKS}, "meta": folder / f"{label}.json"}


def gate_summary(meta: dict, mic: np.ndarray, output: np.ndarray, sr: int,
                 hop_ms: float) -> dict[str, float]:
    """Gate behaviour of one take from its JSON and tracks: openings per minute, closes
    inside the talker's phrases (chatter), speech lost to digital-zero runs inside
    phrases (ms), and the share of hops closed. The output trails the mic by the SOLA
    delay noted in the JSON (``stream_delay_ms``)."""
    from vox.offline.metrics import phrases, zero_runs

    events = meta.get("events", [])
    opens = [e for e in events if e["kind"] == "gate_open"]
    closes = [e for e in events if e["kind"] == "gate_close"]
    seconds = max(len(mic) / sr, 1e-9)
    delay = round(sr * meta.get("stream_delay_ms", 0.0) / 1000.0)
    aligned = output[delay:]
    n = min(len(aligned), len(mic))
    spans = phrases(mic[:n], sr)
    in_phrase = np.zeros(n, dtype=bool)
    for s, e in spans:
        in_phrase[s:e] = True
    lost = 0
    for a, b in zero_runs(aligned[:n], sr // 100):
        lost += int(in_phrase[a:b].sum())
    # closes while the talker was inside a phrase (100 ms earlier too, so a close at the
    # very end of a phrase does not count): the chatter count of DEVIATIONS 7.2
    t0 = events[0]["t"] - events[0]["hop"] * hop_ms / 1000.0 if events else 0.0
    chatter = 0
    for e in closes:
        at = round((e["t"] - t0) * sr)
        if 0 < at < n and in_phrase[at] and in_phrase[max(0, at - sr // 10)]:
            chatter += 1
    gate_per_hop = meta.get("gate_open_per_hop", "")
    closed_share = gate_per_hop.count("0") / len(gate_per_hop) if gate_per_hop else float("nan")
    return {"openings": len(opens), "closures": len(closes),
            "openings_per_min": 60.0 * len(opens) / seconds, "closes_in_phrases": chatter,
            "speech_lost_ms": 1000.0 * lost / sr, "phrases": len(spans),
            "speech_s": float(in_phrase.sum()) / sr, "closed_share": closed_share,
            "underruns": meta.get("event_counts_in_take", {}).get("underrun", 0)}


def analyze_take(folder: Path, label: str, kit: QualityKit | None, script_text: str | None,
                 targets=None, mic_transcript: str | None = None) -> tuple[Scores, dict]:
    """Scores of one take's output track plus its gate summary. ``mic_transcript`` (the
    raw mic through Whisper) is computed here when not given; the mic WER is the
    reference every output WER is read against."""
    from vox.audio.resample import load_audio
    from vox.offline.metrics import active_frames, boundary_click_ratio

    files = take_files(folder, label)
    meta = json.loads(files["meta"].read_text(encoding="utf-8"))
    output, sr = load_audio(files["output"])
    mic, _ = load_audio(files["mic_raw"], sr)
    conditioned, _ = load_audio(files["mic_conditioned"], sr)
    hop_ms = float(meta.get("plan", {}).get("hop_ms", 20.0))
    block = round(sr * hop_ms / 1000.0)
    gate = gate_summary(meta, conditioned, output, sr, hop_ms)
    mic_wer = float("nan")
    if kit is not None and script_text is not None and "wer" in kit.enable:
        mic_transcript = mic_transcript if mic_transcript is not None else \
            kit.transcript_of(mic, sr)
        mic_wer = wer(script_text, mic_transcript)
    scores = kit.score(output, sr, reference_text=script_text, mic_wer=mic_wer,
                       targets=targets, mask=active_frames(output, sr)) \
        if kit is not None else Scores()
    scores.extra.update(click=boundary_click_ratio(output, block), **gate)
    scores.extra["mic_transcript"] = mic_transcript or ""
    return scores, {"label": label, "meta": {k: meta.get(k) for k in
                                             ("model", "params", "shape", "gate",
                                              "conditioning", "plan", "seconds")}}


# ------------------------------------------------------------------------- the recorder


class _Meter(threading.Thread):
    """One line per second while a take runs: gate state, conditioned level, auto gain."""

    def __init__(self, session, log=print) -> None:
        super().__init__(name="vox-matrix-meter", daemon=True)
        self.session, self.log = session, log
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(1.0):
            try:
                s = self.session.snapshot()
            except Exception as err:  # noqa: BLE001 - a meter must never end a take
                self.log(f"    (meter: {type(err).__name__})")
                continue
            self.log(f"    [{s['t']:5.1f}s] gate {'open' if s['gate_open'] else 'shut'} "
                     f"level {s['gate_rms_dbfs']:6.1f} dBFS vad {s['gate_vad']:.2f} "
                     f"in gain {s['input_gain_db']:+.0f} dB | hop p99 {s['total']['p99']:.1f} ms"
                     f" | underruns {s['underruns']} | closes {s['event_counts']['gate_close']}")

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=2.0)


def _wait_rebuilt(session, before: int, timeout: float = 15.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if session.worker.stats.rebuilds > before:
            time.sleep(0.5)  # the first hops after a rebuild
            return True
        time.sleep(0.1)
    return False


def record_matrix(runs: list[Run], folder: Path, *, model: str, models_dir: Path,
                  profile_dir: Path, pitch: float, script_text: str, output: str | None,
                  seconds: float, agc_thr: float, raw_thr: float, extra_ms: float,
                  log=print, prompt=input) -> list[Run]:
    """Record every run in order; the user reads the script each time. Returns the runs
    that were recorded. One ``LiveSession`` per input device; hop changes rebuild in
    place (~3 s mute), gate and auto level switch hot."""
    import shutil

    from vox.api.control import LiveConfig, LiveSession
    from vox.audio.conditioning import ConditioningConfig
    from vox.config.schemas import ShapeParams
    from vox.config.store import load_profile, resolve_runtime_params
    from vox.engine.gate import GateConfig
    from vox.models.registry import ModelRegistry

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    profile = load_profile(profile_dir)
    config = ModelRegistry(models_dir).config(model)
    base_params = resolve_runtime_params(profile, config, {"f0_up_key": pitch})
    cal = profile.calibration
    done: list[Run] = []
    (folder / "script.txt").write_text(script_text + "\n", encoding="utf-8")
    registry, cache = ModelRegistry(models_dir), {}
    for input_key in dict.fromkeys(r.input_key for r in runs):
        mine = [r for r in runs if r.input_key == input_key]
        first = mine[0]
        shape = ShapeParams(block_ms=first.hop_ms, extra_ms=extra_ms)
        cond = ConditioningConfig(auto_level=first.agc, trim_db=cal.input_gain_db,
                                  noise_floor_dbfs=cal.noise_floor_dbfs)
        gate = GateConfig(enabled=first.gate, threshold_dbfs=first.gate_threshold(agc_thr, raw_thr),
                          vad_threshold=cal.vad.threshold, hangover_ms=cal.vad.release_ms,
                          use_vad=cal.vad.engine == "silero")
        cfg = LiveConfig(model=model, models_dir=models_dir, shape=shape,
                         params=dataclasses.replace(base_params, pitch_mode="fixed"),
                         gate=gate, conditioning=cond, input=first.input_spec, output=output,
                         suppression=False, profile_dir=profile_dir, registry=registry,
                         cache=cache)
        log(f"\n=== input: {first.input_spec} ===")
        session = LiveSession(cfg, log=log)
        try:
            session.start()
        except Exception as err:  # noqa: BLE001
            log(f"could not start on {first.input_spec!r}: {type(err).__name__}: {err}; "
                "skipping this input")
            continue
        try:
            current_hop = first.hop_ms
            for run in mine:
                if run.hop_ms != current_hop:
                    before = session.worker.stats.rebuilds
                    session.set_shape(block_ms=run.hop_ms)
                    if not _wait_rebuilt(session, before):
                        log("warning: the hop change did not complete in 15 s")
                    current_hop = run.hop_ms
                session.set_gate(enabled=run.gate,
                                 threshold_dbfs=run.gate_threshold(agc_thr, raw_thr))
                session.set_conditioning(auto_level=run.agc)
                time.sleep(0.3)
                log(f"\n--- run {len(done) + 1}/{len(runs)}: {run.label}\n    {run.describe()}"
                    f", gate threshold {run.gate_threshold(agc_thr, raw_thr):g} dBFS\n")
                log("Read this when recording starts:\n")
                for line in script_text.split(". "):
                    log("    " + line.strip() + ("." if not line.endswith(".") else ""))
                log("")
                answer = prompt("Press Enter to start recording (s = skip this run, q = quit): ")
                if answer.strip().lower() == "q":
                    return done
                if answer.strip().lower() == "s":
                    continue
                session.start_recording()
                meter = _Meter(session, log)
                meter.start()
                try:
                    if seconds > 0:
                        log(f"    recording for {seconds:g} s ...")
                        time.sleep(seconds)
                    else:
                        prompt("    recording: press Enter when you have finished reading ")
                finally:
                    meter.stop()
                paths = session.stop_recording(folder)
                if paths is None:
                    continue
                stem = paths["meta"].stem
                meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
                plan = session.worker.plan
                meta.update(label=run.label, run=dataclasses.asdict(run),
                            stream_delay_ms=plan.stream_delay * 1000.0 / plan.device_sr,
                            script=script_text, take_stem=stem)
                for track in TRACKS:
                    shutil.move(paths[track], folder / f"{run.label}_{track}.wav")
                (folder / f"{run.label}.json").write_text(json.dumps(meta, indent=1, default=str),
                                                          encoding="utf-8")
                paths["meta"].unlink(missing_ok=True)
                log(f"    saved {folder / (run.label + '_output.wav')} ({meta['seconds']:.1f} s;"
                    f" gate opens {meta['event_counts_in_take']['gate_open']}, closes "
                    f"{meta['event_counts_in_take']['gate_close']}, underruns "
                    f"{meta['event_counts_in_take']['underrun']})")
                done.append(run)
        finally:
            session.stop()
    return done


# ---------------------------------------------------------------------------- analysis


def recorded_runs(folder: Path) -> list[Run]:
    """The runs a matrix folder holds (from the JSON sidecars), in recording order."""
    runs = []
    for meta_path in sorted(Path(folder).glob("*.json"), key=lambda p: p.stat().st_mtime):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "run" in meta:
            runs.append(Run(**meta["run"]))
    return runs


def analyze_matrix(folder: Path, kit: QualityKit | None, script_text: str | None,
                   targets=None, log=print) -> dict[str, Scores]:
    """Score every recorded run; the microphone transcript is made once per input device
    (its WER is the floor every output WER is read against)."""
    from vox.audio.resample import load_audio

    folder = Path(folder)
    rows: dict[str, Scores] = {}
    mic_text: dict[str, str] = {}
    for run in recorded_runs(folder):
        if kit is not None and script_text is not None and run.input_key not in mic_text \
                and "wer" in kit.enable:
            # one raw-mic transcript per input: the gate/AGC/hop runs share the microphone
            mic, sr = load_audio(take_files(folder, run.label)["mic_raw"])
            mic_text[run.input_key] = kit.transcript_of(mic, sr)
        scores, _ = analyze_take(folder, run.label, kit, script_text, targets,
                                 mic_text.get(run.input_key))
        rows[run.label] = scores
        g = scores.extra
        log(f"[{run.label}] WER {scores.wer:.1%} (mic {scores.wer_mic:.1%}) spk "
            f"{scores.speaker_cos:.3f} UTMOS {scores.utmos:.2f} F0 "
            f"{scores.f0.get('median_hz', float('nan')):.0f} Hz flips "
            f"{scores.f0.get('flips_per_min', float('nan')):.1f}/min click {g['click']:.2f} "
            f"gate opens {g['openings']} closes-in-phrase {g['closes_in_phrases']} lost "
            f"{g['speech_lost_ms']:.0f} ms underruns {g['underruns']}")
    return rows


MATRIX_EXTRA = (("click", "click", "{:.2f}"), ("opens/min", "openings_per_min", "{:.1f}"),
                ("close-in-ph", "closes_in_phrases", "{:.0f}"),
                ("lost ms", "speech_lost_ms", "{:.0f}"), ("underruns", "underruns", "{:.0f}"))


def script_or_default(path: Path | None) -> str:
    return load_script(path or ROOT / "benchmarks" / "corpus" / "speech_script.txt")
