"""Report 19 A/B harness (``vox bench paraling``): any take through the user's live chain,
once per paralinguistics flag, so a new recording plugs straight in.

    vox bench paraling --audio TAKE.wav [--variants base,declick,...] [--preset natural]
                       [--proxy-clicks] [--quality] [--out-dir DIR]

Variants (docs/DEVIATIONS.md §15; every flag is off in the code defaults):

    base         the user's chain as the GUI runs it (hop 80, extra 2400, right 40, gate
                 -28 dBFS / hangover 500, RNNoise on, auto level +6 dB, Mom +12, protect
                 0.35, ceiling 330, formant -0.5)
    declick      + the in-speech de-clicker (conditioning)
    confirm      + context-based probation, 30 ms (gate)
    laugh        + laugh guard and ceiling lift (f0 stage)
    passthrough  + breath / click passthrough (post chain)
    all          all four

Writes ``<out>/<variant>.wav`` and ``<out>/report.json``: gate opens (energy-only ones
split out), the gate's onset log (crest factor, HF ratio, verdict per flagged onset), the
de-clicker's clamps, the output's sibilant centroid and level, and WER / UTMOS with
``--quality``. ``--proxy-clicks`` first builds a keystroke proxy of the take (its phrases
spaced by 1.2 s of its own room tone, synthetic two-impulse keystrokes mixed in during
speech, inside the hangover and in closed pauses) and scores each click's leak against
the same render without clicks.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

SR = 48000
USER_PARAMS = {"f0_up_key": 12.0, "protect": 0.35, "f0_ceiling_hz": 330.0, "formant_st": -0.5}
NATURAL_PARAMS = {**USER_PARAMS, "pitch_mode": "auto", "target_f0_hz": 194.5,
                  "note_detect": True, "note_ceiling_hz": 360.0}
USER_SHAPE = {"block_ms": 80.0, "extra_ms": 2400.0}
USER_GATE = {"threshold_dbfs": -28.0, "hangover_ms": 500.0}
USER_COND = {"auto_level": True, "trim_db": 6.0, "suppression": True}

VARIANTS: dict[str, dict[str, dict]] = {
    "base": {},
    "declick": {"cond": {"declick": True}},
    "confirm": {"gate": {"onset_confirm_ms": 30.0}},
    "laugh": {"params": {"laugh_guard": True}},
    "passthrough": {"params": {"passthrough": True}},
    "all": {"cond": {"declick": True}, "gate": {"onset_confirm_ms": 30.0},
            "params": {"laugh_guard": True, "passthrough": True}},
}


# ------------------------------------------------------------------------- the engine


def load_variant(variant: str, models_dir: Path, profile_dir: Path, model: str = "Mom",
                 preset: str = "speak", extra: dict[str, dict] | None = None):
    """The user's chain plus one variant's flags (plus ``extra`` overrides by owner:
    shape / params / gate / cond)."""
    from vox.audio.conditioning import ConditioningConfig
    from vox.config import defaults
    from vox.config.schemas import RuntimeParams
    from vox.engine.gate import GateConfig
    from vox.offline.simulate import load_engine

    v = VARIANTS[variant]
    extra = extra or {}
    pick = lambda owner, base: {**base, **v.get(owner, {}), **extra.get(owner, {})}  # noqa
    params = RuntimeParams(**pick("params", NATURAL_PARAMS if preset == "natural"
                                  else USER_PARAMS))
    shape = dataclasses.replace(defaults.DEFAULT_SHAPE, **pick("shape", USER_SHAPE))
    gate = GateConfig(**pick("gate", USER_GATE))
    cond = ConditioningConfig(**pick("cond", USER_COND))
    return load_engine(models_dir, model, shape, params, gate=gate, conditioning=cond,
                       profile_dir=profile_dir, log=lambda *a: None)


def render(engine, audio: np.ndarray, seed: int = 0):
    from vox.offline.simulate import StreamSimulator

    if engine.conditioner is not None:
        engine.conditioner.reset()
    if engine.gate is not None:
        engine.gate.onset_log.clear()
    return StreamSimulator(engine).run(audio, gate=True, seed=seed)


def output_delay(engine) -> int:
    """Samples from an input event to its place in the stream (SOLA delay + the chain's
    RNNoise delay)."""
    chain = 0
    if engine.conditioner is not None:
        chain = round(engine.conditioner.delay_ms * SR / 1000.0)
    return engine.plan.stream_delay + chain


def gate_summary(engine, result) -> dict:
    opens = [e for e in result.events if e["kind"] == "gate_open"]
    log = list(engine.gate.onset_log) if engine.gate is not None else []
    verdicts: dict[str, int] = {}
    for e in log:
        verdicts[e["verdict"]] = verdicts.get(e["verdict"], 0) + 1
    return {"opens": len(opens), "closes": sum(e["kind"] == "gate_close" for e in result.events),
            "open_share": float(result.gate_open.mean()) if result.gate_open is not None else 1.0,
            "onset_verdicts": verdicts, "onset_log": log}


def passthrough_summary(engine) -> dict:
    """Share of decoded frames the passthrough mixed the mic into (counted on the device;
    a frame is decoded once per hop it is in the kept tail, so this is a share, not a
    count of distinct frames)."""
    post = engine.pipeline.post
    if not post.passthrough_on:
        return {}
    seen = float(post.passthrough.frames_seen)
    return {"passthrough_share": float(post.passthrough.frames_on) / seen if seen else 0.0}


def declick_summary(engine) -> dict:
    d = engine.conditioner.declicker if engine.conditioner is not None else None
    if d is None:
        return {"clamps": None}
    return {"clamps": len(d.clicks), "clamp_times_s": [round(t, 3) for t, _ in d.clicks],
            "clamp_db": [round(g, 1) for _, g in d.clicks]}


def sibilant_centroid(audio: np.ndarray, sr: int = SR) -> dict:
    """Spectral centroid of the 10 % most HF-heavy active frames (4-10 kHz energy share),
    and the active level: the passthrough must not move either (sibilants carry sex)."""
    from vox.offline.metrics import frame_rms_dbfs

    n = len(audio) // 480
    frames = audio[:n * 480].reshape(n, 480).astype(np.float64)
    level = frame_rms_dbfs(audio[:n * 480], sr, hop_ms=10.0, win_ms=10.0)[:n]
    active = level >= np.percentile(level, 95) - 30.0
    frames = frames[active]
    spec = np.abs(np.fft.rfft(frames * np.hanning(480), n=1024, axis=1))
    freqs = np.fft.rfftfreq(1024, 1 / sr)
    hf = spec[:, (freqs >= 4000) & (freqs <= 10000)].sum(1) / (spec.sum(1) + 1e-12)
    sib = spec[hf >= np.percentile(hf, 90)]
    return {"sibilant_centroid_hz": float((sib * freqs).sum() / (sib.sum() + 1e-12)),
            "active_level_dbfs": float(10 * np.log10(np.mean(frames ** 2) + 1e-12))}


# --------------------------------------------------------------------- keystroke proxy


def keystroke(rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """One key: push (two 2-3 ms sub-events, touch and hit, broadband with most energy
    below ~4 kHz) and, ~90 ms later, a release impulse 6 dB down (report 19 §1.1)."""
    from scipy.signal import butter, sosfilt

    def impulse(n_ms: float, tau_ms: float) -> np.ndarray:
        n = int(sr * n_ms / 1000)
        return rng.standard_normal(n) * np.exp(-np.arange(n) / (sr * tau_ms / 1000))

    push = np.zeros(int(0.012 * sr))
    touch, hit = impulse(4, 0.8), impulse(8, 1.5)
    push[:len(touch)] += 0.5 * touch
    at = int(0.0025 * sr)
    push[at:at + len(hit)] += hit
    gap = int(rng.uniform(0.08, 0.1) * sr)
    out = np.zeros(gap + len(push))
    out[:len(push)] += push
    out[gap:gap + len(push)] += 0.5 * push * rng.uniform(0.8, 1.0)
    sos = butter(2, [400, 12000], btype="bandpass", fs=sr, output="sos")
    low = butter(2, 2500, btype="lowpass", fs=sr, output="sos")
    shaped = sosfilt(sos, out) + 1.5 * sosfilt(low, sosfilt(sos, out))
    return shaped


def build_click_proxy(audio: np.ndarray, seed: int = 2026, pause_s: float = 1.2,
                      levels_db=(-6.0, 0.0, 6.0)) -> tuple[np.ndarray, np.ndarray, dict]:
    """(proxy with clicks, the same without, meta). Phrases of the take spaced by
    ``pause_s`` of its own room tone; per phrase one keystroke inside the phrase, and per
    pause one at 0.2 s (inside the 500 ms hangover) and one at 0.85 s (gate closed). Push
    impulse 10 ms RMS at ``levels_db`` re the take's speech p95 (10 ms frames), in turn."""
    from vox.offline.metrics import frame_rms_dbfs, phrases

    rng = np.random.default_rng(seed)
    x = np.asarray(audio, np.float64)
    spans = phrases(x, SR)
    level = frame_rms_dbfs(x, SR, hop_ms=10.0, win_ms=10.0)
    speech_p95 = float(np.percentile(level, 95))
    win = 150
    quiet_at = int(np.argmin(np.convolve(10 ** (level / 10), np.ones(win) / win, mode="valid")))
    room = x[quiet_at * 480:(quiet_at + win) * 480]
    margin = int(0.15 * SR)
    pieces, events, pos = [np.tile(room, 2)[:int(0.5 * SR)]], [], int(0.5 * SR)
    k = {"speech": 0, "hangover": 1, "pause": 2}  # each place cycles through the levels
    for a, b in spans:
        a, b = max(0, a - margin), min(len(x), b + margin)
        seg = x[a:b]
        mid = int(rng.uniform(0.3, 0.7) * len(seg)) if len(seg) > 0.6 * SR else None
        pieces.append(seg)
        if mid is not None:
            events.append({"where": "speech", "at": pos + mid,
                           "level": levels_db[k["speech"] % len(levels_db)]})
            k["speech"] += 1
        pos += len(seg)
        gap = np.tile(room, int(np.ceil(pause_s * SR / len(room))) + 1)[:int(pause_s * SR)]
        pieces.append(gap.copy())
        for where, t in (("hangover", 0.2), ("pause", 0.85)):
            events.append({"where": where, "at": pos + int(t * SR),
                           "level": levels_db[k[where] % len(levels_db)]})
            k[where] += 1
        pos += len(gap)
    clean = np.concatenate(pieces)
    proxy = clean.copy()
    for e in events:
        s = keystroke(rng)
        push = s[:int(0.01 * SR)]
        s = s / (np.sqrt(np.mean(push ** 2)) + 1e-12) * 10 ** ((speech_p95 + e["level"]) / 20)
        end = min(len(proxy), e["at"] + len(s))
        proxy[e["at"]:end] += s[:end - e["at"]]
        e["t"] = e["at"] / SR
        e["dbfs"] = speech_p95 + e["level"]
    meta = {"speech_p95_dbfs": speech_p95, "events": events, "seconds": len(clean) / SR,
            "phrases": [(a / SR, b / SR) for a, b in phrases(clean, SR)]}
    return proxy.astype(np.float32), clean.astype(np.float32), meta


def click_leaks(proxy_out: np.ndarray, clean_out: np.ndarray, meta: dict, delay: int,
                block: int, gate_open: np.ndarray | None) -> dict:
    """Per click, over [-10, +150] ms around where it lands in the output:

    * ``excess_db``: output energy, proxy vs the same render without clicks (floor 40 dB
      under the speech level): what a click adds in a pause;
    * ``mel_change``: mean |log-mel difference| (nats, 10 ms frames, 80 mels) between the
      two renders there: how much the click changed the output even where it adds no
      energy (a click rendered as a consonant inside a word). A waveform difference does
      not work: a click nudges the SOLA alignment and every later hop shifts by a few
      samples, so the renders differ everywhere by as much as the signal itself.
      ``mel_floor`` is the same measure on click-free windows of speech (what the shifted
      alignment alone gives).

    For pause clicks, also whether the gate opened. A leak: excess > 3 dB, or a mel change
    above 1.5 x the floor's p90."""
    n = min(len(proxy_out), len(clean_out))
    p, c = proxy_out[:n].astype(np.float64), clean_out[:n].astype(np.float64)
    ref = np.percentile(np.convolve(c * c, np.ones(480) / 480, mode="valid"), 95)
    eps = ref * 1e-4
    from vox.offline.metrics import log_mel

    diff = (log_mel(p, SR) - log_mel(c, SR)).abs().mean(1).numpy()  # per 10 ms frame
    hop = SR // 100
    win = int(0.16 * SR)
    taken = np.zeros(n, bool)
    for e in meta["events"]:
        taken[max(0, e["at"] + delay - SR // 2):e["at"] + delay + SR // 2] = True
    floor = []
    for a in range(SR, n - win, SR // 4):
        if not taken[a:a + win].any() and np.mean(c[a:a + win] ** 2) > eps * 100:
            floor.append(float(diff[a // hop:(a + win) // hop].mean()))
    floor_p90 = float(np.percentile(floor, 90)) if floor else 0.0
    rows = []
    for e in meta["events"]:
        lo = e["at"] + delay - int(0.01 * SR)
        hi = e["at"] + delay + int(0.15 * SR)
        if hi > n:
            continue
        ep, ec = float(np.mean(p[lo:hi] ** 2)), float(np.mean(c[lo:hi] ** 2))
        row = {"where": e["where"], "t": round(e["t"], 3), "level": e["level"],
               "excess_db": round(10 * np.log10((ep + eps) / (ec + eps)), 2),
               "mel_change": round(float(diff[lo // hop:hi // hop].mean()), 3)}
        if gate_open is not None and e["where"] == "pause":
            h0, h1 = e["at"] // block, (e["at"] + int(0.15 * SR)) // block + 1
            row["gate_opened"] = bool(gate_open[h0:h1].any())
        rows.append(row)
    out: dict = {"clicks": rows, "mel_floor": {
        "n": len(floor), "p50": float(np.percentile(floor, 50)) if floor else None,
        "p90": floor_p90}}
    for where in ("speech", "hangover", "pause"):
        sel = [r for r in rows if r["where"] == where]
        if sel:
            out[where] = {"n": len(sel),
                          "leaks": sum(r["excess_db"] > 3.0 or r["mel_change"] > 1.5 * floor_p90
                                       for r in sel),
                          "excess_db_mean": round(float(np.mean([r["excess_db"] for r in sel])), 2),
                          "mel_change_median": round(float(np.median([r["mel_change"]
                                                                      for r in sel])), 3)}
            if where == "pause":
                out[where]["gate_opened"] = sum(r.get("gate_opened", False) for r in sel)
    return out


def write_report(path: Path, report: dict) -> None:
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    path.write_text(json.dumps(clean(report), indent=1), encoding="utf-8")
