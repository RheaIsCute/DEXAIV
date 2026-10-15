"""Benchmark harness (docs/research/10). Every number comes from the real engine, driven by
the stream simulator; there is no separate batch path.

    vox bench accept                  Phase 2 acceptance: streamed vs full-file, clicks,
                                      sync-free hop, allocation stability; plus the live
                                      path (paced, gated): gate closures, underruns, cuts
    vox bench ab [--variants a,b]     one setting per variant -> out/ab/<name>.wav + metrics
    vox bench typed                   typed-speech prosody variants -> out/ab/typed_*.wav
    vox bench sing                    singing A/Bs (phase carry, trim tail, ceiling): SOLA
                                      offsets, seam pitch steps -> out/ab/sing_*.wav
    vox bench onset                   phrase entrances through the gate, before/after the
                                      onset fixes, per hop -> out/ab/onset_*.wav
    vox bench suppress                RNNoise: delay, CPU per hop, attenuation, sung-vowel
                                      band loss, toggle (no GPU)
    vox bench underrun                paced runs over hop x extra: compute vs budget and
                                      underruns (optionally with NVIDIA Broadcast running)
    vox bench perf  [--sweep FILE]    perf sweep: block x extra x f0 method   (open question Q1)
    vox bench extra                   quality vs EXTRA context                (Q3)
    vox bench sola                    overlap x {plain, pv} stitching         (Q7)
    vox bench block                   quality vs hop size
    vox bench decctx                  quality vs decoder left context
    vox bench envelope                quality vs volume_envelope (RMS mix)
    vox bench graph-probe --stage S   capture one stage in a CUDA graph: eager vs replay cost

Results are appended to benchmarks/results/<name>.jsonl with the environment fingerprint.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
SKIP = 50  # warmup hops discarded before statistics (docs/research/10 §3.2)


def _setup(args):
    from vox.audio.resample import load_audio
    from vox.models.registry import ModelRegistry

    audio, _ = load_audio(args.audio, 48000)
    start = int(getattr(args, "start", 0.0) * 48000)
    audio = audio[start:start + int(args.seconds * 48000)]
    return audio, ModelRegistry(args.models_dir), {}


def _overrides(args) -> dict[str, dict]:
    """``--set key=value`` (repeatable) sorted by the dataclass that owns each key, plus the
    explicit convenience flags. Values are parsed as JSON where they can be. Input
    conditioning fields take a ``cond_`` prefix (``--set cond_target_dbfs=-24``)."""
    import json

    from vox.audio.conditioning import ConditioningConfig
    from vox.config.schemas import RuntimeParams, ShapeParams
    from vox.engine.gate import GateConfig

    owners = {"shape": ShapeParams, "params": RuntimeParams, "gate": GateConfig}
    out: dict[str, dict] = {k: {} for k in (*owners, "conditioning")}
    cond_fields = {f.name for f in dataclasses.fields(ConditioningConfig)}
    for item in getattr(args, "set", None) or []:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        if key.startswith("cond_") and key[5:] in cond_fields:
            out["conditioning"][key[5:]] = value
            continue
        for name, cls in owners.items():
            if key in {f.name for f in dataclasses.fields(cls)}:
                out[name][key] = value
                break
        else:
            raise SystemExit(f"--set: unknown key {key!r}")
    if getattr(args, "crossfade_ms", None) is not None:
        out["shape"]["crossfade_ms"] = args.crossfade_ms
    if getattr(args, "index_rate", None) is not None:
        out["params"]["index_rate"] = args.index_rate
    if getattr(args, "gate_threshold", None) is not None:
        out["gate"]["threshold_dbfs"] = args.gate_threshold
    if getattr(args, "conditioning", None) is not None:
        out["conditioning"]["enabled"] = args.conditioning == "on"
    if getattr(args, "auto_level", None) is not None:
        out["conditioning"]["auto_level"] = args.auto_level == "on"
    return out


def _conditioning_config(args):
    """The live input chain as ``vox run`` builds it (calibration.json), then overrides."""
    from vox.audio.conditioning import ConditioningConfig
    from vox.config.store import load_profile

    cal = load_profile(args.profile).calibration
    base = ConditioningConfig(trim_db=cal.input_gain_db, noise_floor_dbfs=cal.noise_floor_dbfs)
    return dataclasses.replace(base, **_overrides(args)["conditioning"])


def _gate_config(args):
    """The profile's gate (calibration.json), as ``vox run`` builds it, then overrides."""
    from vox.config.store import load_profile
    from vox.engine.gate import GateConfig

    cal = load_profile(args.profile).calibration
    base = GateConfig(threshold_dbfs=cal.silence_gate_dbfs, vad_threshold=cal.vad.threshold,
                      hangover_ms=cal.vad.release_ms, use_vad=cal.vad.engine == "silero")
    return dataclasses.replace(base, **_overrides(args)["gate"])


def _engine(args, registry, cache, shape_kw=None, cond_kw=None, gate_kw=None, **param_kw):
    from vox.config import defaults
    from vox.config.schemas import PrecisionParams, RuntimeParams
    from vox.offline.simulate import load_engine

    extra = _overrides(args)
    shape = dataclasses.replace(defaults.DEFAULT_SHAPE, **{**extra["shape"], **(shape_kw or {})})
    params = RuntimeParams(f0_up_key=args.pitch, **{**extra["params"], **param_kw})
    precision = PrecisionParams(hubert_fp16=args.hubert_fp16, graphs=args.graphs,
                                hubert_autocast=args.hubert_autocast)
    conditioning = dataclasses.replace(_conditioning_config(args), **(cond_kw or {}))
    gate = dataclasses.replace(_gate_config(args), **(gate_kw or {}))
    return load_engine(args.models_dir, args.model, shape, params, precision,
                       registry=registry, cache=cache, profile_dir=args.profile,
                       gate=gate, conditioning=conditioning)


def _record(name: str, record: dict) -> None:
    from vox.profiling.report import environment_fingerprint, write_jsonl

    record = {"bench": name, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "env": environment_fingerprint(), **record}
    write_jsonl(RESULTS / f"{name}.jsonl", record)


def _stage_p50(engine, skip: int) -> dict[str, float]:
    from vox.profiling.report import stage_table

    return {name: row["host"]["p50"] for name, row in stage_table(engine.timers, skip).items()}


def _settle_s(plan) -> float:
    """Seconds until a fresh stream has real audio (not zeros) throughout its context."""
    return plan.context_ms / 1000.0 + 0.5


def _quality(reference: np.ndarray, result, sr: int, start_s: float) -> dict[str, float]:
    from vox.offline.metrics import boundary_click_ratio, compare

    stream = result.aligned[:len(reference)]
    out = compare(reference, stream, sr, lag_search_ms=15.0, start_s=start_s)
    out["click_ratio"] = boundary_click_ratio(result.audio, result.plan.block)
    return out


def _f0_100hz(engine, audio: np.ndarray) -> np.ndarray:
    """Pitch track of device-rate audio with the engine's own extractor (RMVPE), 100 Hz."""
    from vox.audio.resample import resample_array

    x = resample_array(audio, engine.plan.device_sr, 16000)
    with torch.inference_mode():
        f0 = engine.pipeline.extractor.extract(torch.from_numpy(x).to(engine.device))
    return f0.float().cpu().numpy()


def _voice_metrics(engine, audio: np.ndarray, mask: np.ndarray | None = None
                   ) -> dict[str, float]:
    """Voiced ratio, 90th-pct HF/LF and level of one rendered signal over ``mask`` (10 ms
    frames; default: its own active frames)."""
    from vox.offline.metrics import active_frames, frame_rms_dbfs, hf_lf_p90, voiced_ratio

    sr = engine.plan.device_sr
    mask = active_frames(audio, sr) if mask is None else mask
    level = frame_rms_dbfs(audio, sr)[:len(mask)]
    power = np.mean(10 ** (level[mask[:len(level)]] / 10.0)) if mask.any() else 1e-12
    out = {"hf_lf_p90_db": hf_lf_p90(audio, sr, mask),
           "active_rms_dbfs": float(10 * np.log10(power + 1e-12)),
           "active_share": float(mask.mean())}
    if engine.pipeline.extractor is not None:
        out["voiced_ratio"] = voiced_ratio(_f0_100hz(engine, audio), mask)
    return out


def _cuts(result, audio_in: np.ndarray) -> dict:
    """Digital-zero runs >= 10 ms that overlap the input's phrases by >= 10 ms, each
    attributed to the gate (>= 95% of its samples left the engine in gated hops; the
    onset hop's fade-in starts at exactly 0) or an underrun (an event within 2 hops)."""
    from vox.offline.metrics import phrases, zero_runs

    plan = result.plan
    sr, block = plan.device_sr, plan.block
    stream = result.aligned[:len(audio_in)]
    spans = phrases(audio_in, sr)
    in_phrase = np.zeros(len(stream), dtype=bool)
    for s, e in spans:
        in_phrase[s:e] = True
    under_hops = np.array([e["hop"] for e in result.events if e["kind"] == "underrun"])
    cuts = []
    for a, b in zero_runs(stream, sr // 100):
        lost = int(in_phrase[a:b].sum())
        if lost < sr // 100:
            continue
        # aligned sample k left the engine in hop (k + stream_delay) // block
        hops = (np.arange(a, b) + plan.stream_delay) // block
        closed = (np.mean(~result.gate_open[np.minimum(hops, len(result.gate_open) - 1)])
                  if result.gate_open is not None else 0.0)
        h0, h1 = int(hops[0]), int(hops[-1])
        under = under_hops.size > 0 and bool(np.any((under_hops >= h0 - 2)
                                                    & (under_hops <= h1 + 2)))
        cause = "gate" if closed >= 0.95 else "underrun" if under else "unattributed"
        cuts.append({"t": a / sr, "ms": 1000.0 * (b - a) / sr,
                     "speech_ms": 1000.0 * lost / sr, "cause": cause})
    speech_s = float(in_phrase.sum()) / sr
    # Chatter: gate transitions while the input is inside a phrase (an honest gate opens
    # and closes once per phrase, at its edges, so these should be ~0). The hop's audio is
    # [n * block, (n + 1) * block) of the input.
    inside = [e for e in result.events if e["kind"] in ("gate_open", "gate_close")
              and 0 < e["hop"] * block < len(in_phrase) and in_phrase[e["hop"] * block]
              and in_phrase[max(0, e["hop"] * block - sr // 10)]]
    return {"cuts": cuts, "speech_s": speech_s, "phrases": len(spans),
            "toggles_in_phrases": len(inside),
            "by_cause": {c: sum(1 for x in cuts if x["cause"] == c)
                         for c in ("gate", "underrun", "unattributed")},
            "cut_ms": float(sum(x["ms"] for x in cuts)),
            "speech_lost_ms": float(sum(x["speech_ms"] for x in cuts))}


def _print_events(result, limit: int = 40) -> None:
    """The two logs the cut attribution rests on, kept apart (Step 0 of the live fixes)."""
    closes = [e for e in result.events if e["kind"] == "gate_close"]
    unders = [e for e in result.events if e["kind"] == "underrun"]
    print(f"gate closures: {len(closes)}" + ("" if not closes else "   at " + ", ".join(
        f"{e['t']:.2f}s" for e in closes[:limit]) + (" ..." if len(closes) > limit else "")))
    shown = "n/a (throughput run)" if result.underruns is None else str(result.underruns)
    print(f"output-ring underruns: {shown}" + ("" if not unders else "   at " + ", ".join(
        f"{e['t']:.2f}s ({e['value'] / result.plan.device_sr * 1000:.0f} ms)"
        for e in unders[:limit])))


def live_report(engine, audio: np.ndarray, full: np.ndarray, result,
                typed: np.ndarray | None = None) -> dict:
    """The report's live-vs-typed measures for one gated, paced stream of ``audio``.

    Stream and full-file render are compared on the frames active in *both* (the gate
    silences pauses the full render still voices, which would bias either measure); each
    is also reported over its own active frames, and the typed clip over its own.
    """
    from vox.offline.metrics import active_frames

    sr = engine.plan.device_sr
    stream = result.aligned[:len(audio)]
    cuts = _cuts(result, audio)
    seconds = len(audio) / sr
    own = active_frames(stream, sr)
    shared = own & active_frames(full, sr)[:len(own)]
    report = {
        "seconds": seconds, "gate_transitions": result.gate_transitions,
        "toggles_per_8s": result.gate_transitions / (seconds / 8.0),
        "gate_closures": sum(1 for e in result.events if e["kind"] == "gate_close"),
        "underruns": result.underruns, "closed_share": float(1.0 - result.gate_open.mean())
        if result.gate_open is not None else 0.0,
        "mid_speech_cuts": cuts, "stream": _voice_metrics(engine, stream, own),
        "stream_shared": _voice_metrics(engine, stream, shared),
        "full_shared": _voice_metrics(engine, full, shared),
        "full": _voice_metrics(engine, full),
    }
    if typed is not None:
        report["typed"] = _voice_metrics(engine, typed)
    return report


def _print_live(report: dict) -> None:
    c = report["mid_speech_cuts"]
    print(f"gate: {report['gate_transitions']} transitions in {report['seconds']:.1f} s = "
          f"{report['toggles_per_8s']:.1f} per 8 s ({c['phrases']} input phrases: 2 each is "
          f"{2 * c['phrases'] / (report['seconds'] / 8):.1f} per 8 s); "
          f"{c['toggles_in_phrases']} inside phrases; {report['closed_share'] * 100:.0f}% "
          f"of hops closed")
    print(f"mid-speech zero cuts: {len(c['cuts'])} ({c['speech_lost_ms']:.0f} ms of the "
          f"{c['speech_s']:.1f} s of input phrases lost) by cause {c['by_cause']}"
          + ("" if not c["cuts"] else "\n  " + ", ".join(
              f"{x['t']:.2f}s {x['speech_ms']:.0f}ms {x['cause']}" for x in c["cuts"][:30])))
    rows = [("stream, own frames", report["stream"]),
            ("stream, shared frames", report["stream_shared"]),
            ("full-file, shared frames", report["full_shared"]),
            ("full-file, own frames", report["full"])]
    if "typed" in report:
        rows.append(("typed clip (TTS), own", report["typed"]))
    print(f"{'render':<26} {'frames':>7} {'voiced':>7} {'HF/LF p90':>10} {'active dBFS':>12}")
    for label, m in rows:
        voiced = m.get("voiced_ratio", float("nan")) * 100
        print(f"{label:<26} {m['active_share'] * 100:>6.0f}% {voiced:>6.1f}% "
              f"{m['hf_lf_p90_db']:>+9.2f} {m['active_rms_dbfs']:>12.1f}")


def _typed_clip(engine, text: str, models_dir: Path) -> np.ndarray | None:
    """``text`` through Kokoro and the engine's model, as the GUI's Speak does, at the
    device rate (the typed-path reference). None when no TTS backend is installed."""
    from vox.audio.resample import resample_array
    from vox.offline.convert import FullFileConverter
    from vox.tts.backends import pick_backend
    from vox.tts.cli import TTS_DIR
    from vox.tts.speak import convert_rendered, render_tts, split_pieces

    try:
        backend = pick_backend("kokoro", cache_dir=models_dir / TTS_DIR)
        backend.load()
    except Exception as err:  # noqa: BLE001
        print(f"typed reference skipped: {err}")
        return None
    pieces = split_pieces(text)
    said = convert_rendered(FullFileConverter.from_engine(engine), engine.device,
                            engine.loaded.model.tgt_sr, render_tts(backend, pieces),
                            engine.params, None, seed=0, pieces=pieces)
    return resample_array(said.audio, said.sr, engine.plan.device_sr)


# ------------------------------------------------------------------------------- accept


def cmd_accept(args) -> int:
    from vox.audio.resample import save_audio
    from vox.offline.convert import render_full
    from vox.offline.metrics import boundary_click_ratio, compare
    from vox.offline.simulate import StreamSimulator
    from vox.profiling.report import format_report

    audio, registry, cache = _setup(args)
    engine = _engine(args, registry, cache)
    plan, sr = engine.plan, engine.plan.device_sr
    sim = StreamSimulator(engine)

    start_s = _settle_s(plan)
    full_a = render_full(engine, audio, seed=0)
    full_b = render_full(engine, audio, seed=1)
    floor = compare(full_a, full_b, sr, start_s=start_s)

    result = sim.run(audio, sync_debug=True, seed=0)
    report = format_report(
        engine.timers, hop_ms=plan.block_ms, skip=SKIP, title="rt-profiler (throughput)",
        notes={"overlap": f"{plan.overlap_ms:.0f} ms", "search": f"{plan.search_ms:.0f} ms",
               "extra": f"{plan.extra_ms:.0f} ms", "model": args.model},
        alloc=result.alloc, sola_offsets_ms=result.sola_offsets_ms)
    quality = _quality(full_a, result, sr, start_s)
    again = sim.run(audio, seed=1)
    stream_floor = compare(result.aligned[:len(audio)], again.aligned[:len(audio)], sr,
                           start_s=start_s)
    full_click = boundary_click_ratio(full_a, plan.block)
    # The live path: paced input, the modelled output ring, the gate as configured. Its
    # event log separates gate closures from underruns (cut attribution).
    live = sim.run(audio, realtime=True, seed=0, gate=args.gate == "on")
    typed = _typed_clip(engine, args.tts_text, args.models_dir) if args.tts_text else None
    live_rep = live_report(engine, audio, full_a, live, typed)

    out = ROOT / "out"
    save_audio(out / "accept_full.wav", full_a, sr)
    save_audio(out / "accept_stream.wav", result.aligned[:len(audio)], sr)
    save_audio(out / "accept_live.wav", live.aligned[:len(audio)], sr)
    if args.save_as:
        save_audio(out / "ab" / f"{args.save_as}.wav", live.aligned[:len(audio)], sr)
        if typed is not None:
            save_audio(out / "ab" / f"{args.save_as}_typed.wav", typed, sr)

    ratio = quality["l1_speech"] / floor["l1_speech"]
    alloc = result.alloc or {}
    checks = {
        "streamed within the regression guard (speech-frame log-mel L1 <= 2.2x floor)":
            ratio <= 2.2,
        "no clicks at hop boundaries (ratio <= 1.25)": quality["click_ratio"] <= 1.25,
        "hop is sync-free (sync debug mode raised nothing)": True,
        "buffers pointer-stable": result.pointers_stable,
        "no driver mallocs / growth in steady state":
            alloc.get("driver_mallocs", 0) == 0 and alloc.get("reserved_growth_bytes", 0) == 0
            and alloc.get("live_drift_bytes", 0) == 0,
    }
    print(report)
    print(f"\nRTF {result.rtf:.3f}   mode {engine.mode}   eager warmup estimate "
          f"{engine.warmup_ms:.1f} ms   algorithmic latency {plan.algorithmic_latency_ms:.0f} ms")
    if engine.graphs.report.verify:
        v = engine.graphs.report.verify
        print(f"graph replay vs eager (same state, same noise): log-mel L1 {v['log_mel_l1']:.4f}"
              f"  rel rms {v['rel_rms_diff']:.2e}  max |diff| {v['max_abs_diff']:.2e}")
    print(f"\nlog-mel L1 after {start_s:.1f} s settle   {'all':>7} {'speech':>7} {'quiet':>7} "
          f"{'bias':>7}")
    for label, m in (("stream vs full", quality), ("full vs full (noise floor)", floor),
                     ("stream vs stream (new seed)", stream_floor)):
        print(f"  {label:<30} {m['l1']:>7.3f} {m['l1_speech']:>7.3f} {m['l1_quiet']:>7.3f} "
              f"{m['bias']:>+7.3f}")
    print(f"  speech-frame ratio stream/floor = {ratio:.2f}   (best lag "
          f"{quality['lag_ms']:+.1f} ms)")
    print(f"boundary click ratio  stream {quality['click_ratio']:.3f}   "
          f"full-file (control) {full_click:.3f}")
    gate = engine.gate.config
    print(f"\nlive path (paced, gate {args.gate}: threshold {gate.threshold_dbfs:g} dBFS, VAD "
          f"{gate.vad_threshold:g}, hangover {gate.hangover_ms:g} ms; latency estimate "
          f"{plan.algorithmic_latency_ms:.0f} ms algorithmic + fill + devices)")
    cond = engine.conditioner
    if cond is not None and cond.config.enabled:
        level = cond.level_db
        print(f"input chain: high-pass {cond.config.highpass_hz:g} Hz, auto level "
              f"{'on' if cond.config.auto_level else 'off'}, trim {cond.config.trim_db:+g} dB; "
              f"gain at the end {cond.gain_db:+.1f} dB (speech level "
              f"{'-' if level is None else f'{level:.1f}'} dBFS -> target "
              f"{cond.config.target_dbfs:g})")
    else:
        print("input chain: off (raw microphone)")
    _print_events(live)
    _print_live(live_rep)
    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    met = "met" if ratio <= 1.5 else "NOT met"
    print(f"  [info] strict equivalence target (<= 1.5x floor): {met} at {ratio:.2f}x; "
          "see docs/research/14 Q14")
    _record("accept", {"model": args.model, "audio": str(args.audio), "pitch": args.pitch,
                       "label": args.save_as, "overrides": _overrides(args),
                       "plan": dataclasses.asdict(plan), "rtf": result.rtf,
                       "mode": engine.mode, "graphs": dataclasses.asdict(engine.graphs.report),
                       "noise_floor": floor, "stream_floor": stream_floor,
                       "quality": quality, "ratio": ratio,
                       "full_click_ratio": full_click, "alloc": alloc,
                       "stage_p50_ms": _stage_p50(engine, SKIP),
                       "gate": dataclasses.asdict(gate), "live": live_rep,
                       "events": live.events,
                       "checks": {k: bool(v) for k, v in checks.items()}})
    return 0 if all(checks.values()) else 1


# ----------------------------------------------------------------------------------- ab

# Step 2 of the live-quality work (docs/DEVIATIONS.md §7): one setting changed per variant,
# everything else as ``reference`` (the engine after the Step 1 fixes, at the flags given on
# the command line). Each: (shape, params, input conditioning) overrides, and what it is.
AB_VARIANTS: dict[str, tuple[dict, dict, dict, str]] = {
    "reference": ({}, {}, {}, "the defaults (coherent hops since 8.8)"),
    "classic": ({"decoder_context_ms": 0.0}, {"noise_roll": False, "sine_phase_carry": False},
                {}, "the pre-8.8 defaults: no decoder context, noise redrawn, sine phase reset"),
    "sola_blend3": ({"crossfade_ms": 30, "overlap_beyond_hop": True}, {}, {},
                    "pre-fix SOLA geometry: 30 ms overlap at the 20 ms hop (3-render blend)"),
    "overlap_10": ({"crossfade_ms": 10}, {}, {}, "10 ms crossfade (half the hop single-render)"),
    "raw_input": ({}, {}, {"enabled": False}, "no input conditioning (fix 4 off)"),
    "f0_interp": ({}, {"f0_unvoiced": "interp"}, {}, "unvoiced F0 interpolated (Q15)"),
    "envelope_0.3": ({}, {"volume_envelope": 0.3}, {}, "output follows the mic envelope (R4)"),
    "index_0.35": ({}, {"index_rate": 0.35}, {}, "paired index at 0.35"),
    "index_0.6": ({}, {"index_rate": 0.6}, {}, "paired index at 0.6"),
    "embed_pad": ({"embed_pad_ms": 15.0}, {}, {}, "ContentVec window replicate-padded 240 samples"),
    "f0_bridge": ({}, {"f0_bridge_ms": 60.0}, {}, "F0 dropouts <= 60 ms within 4 st filled"),
    "decctx_0": ({"decoder_context_ms": 0.0}, {}, {}, "no decoder left context (upstream)"),
    "noise_redraw": ({}, {"noise_roll": False}, {}, "decoder noise redrawn every hop (upstream)"),
    "trimtail_1": ({}, {"f0_trim_tail": 1}, {}, "RMVPE right edge as upstream: 1 frame"),
    "phase_reset": ({}, {"sine_phase_carry": False}, {}, "NSF sine phase reset every hop (upstream)"),
}


def cmd_ab(args) -> int:
    """Render the script once per variant (gated stream, the live path) to out/ab/<name>.wav
    and measure each the same way: R3 against its own full-file render, hop-seam click
    ratio, gate behaviour, voicing and HF/LF tilt, compute cost."""
    from vox.audio.resample import save_audio
    from vox.offline.convert import render_full
    from vox.offline.metrics import compare
    from vox.offline.simulate import StreamSimulator

    audio, registry, cache = _setup(args)
    names = list(AB_VARIANTS) if not args.variants else args.variants.split(",")
    out = ROOT / "out" / "ab"
    rows = []
    for name in names:
        shape_kw, param_kw, cond_kw, note = AB_VARIANTS[name]
        engine = _engine(args, registry, cache, shape_kw, cond_kw, **param_kw)
        plan, sr = engine.plan, engine.plan.device_sr
        start_s = _settle_s(plan)
        full = render_full(engine, audio, seed=0)
        floor = compare(full, render_full(engine, audio, seed=1), sr, start_s=start_s)
        sim = StreamSimulator(engine)
        open_run = sim.run(audio, seed=0)
        quality = _quality(full, open_run, sr, start_s)
        p99 = _stage_p50(engine, SKIP)  # host p50 per stage (keys: stages + total)
        from vox.profiling.report import stage_table

        total = stage_table(engine.timers, SKIP)["total"]
        gated = sim.run(audio, seed=0, gate=True)
        report = live_report(engine, audio, full, gated)
        save_audio(out / f"{args.prefix}{name}.wav", gated.aligned[:len(audio)], sr)
        cuts = report["mid_speech_cuts"]
        row = {"name": name, "note": note, "r3": quality["l1_speech"],
               "r3_ratio": quality["l1_speech"] / floor["l1_speech"], "bias": quality["bias"],
               "click": quality["click_ratio"], "toggles_8s": report["toggles_per_8s"],
               "toggles_in_phrases": cuts["toggles_in_phrases"],
               "speech_lost_ms": cuts["speech_lost_ms"],
               "voiced": report["stream"].get("voiced_ratio", float("nan")),
               "hf_lf": report["stream"]["hf_lf_p90_db"],
               "hf_lf_shared": report["stream_shared"]["hf_lf_p90_db"],
               "hf_lf_full_shared": report["full_shared"]["hf_lf_p90_db"],
               "active_dbfs": report["stream"]["active_rms_dbfs"],
               "hop_p50": total["host"]["p50"], "hop_p99": total["host"]["p99"]}
        rows.append(row)
        _record("ab", {"model": args.model, "audio": str(args.audio), "pitch": args.pitch,
                       "overrides": _overrides(args), "variant": name,
                       "shape": shape_kw, "params": param_kw, "conditioning": cond_kw,
                       "plan": dataclasses.asdict(plan), "noise_floor": floor,
                       "quality": quality, "live": report, "row": row, "p50": p99})
        print(f"[{name}] {note}: R3 {row['r3']:.3f} ({row['r3_ratio']:.2f}x floor) click "
              f"{row['click']:.2f} toggles {row['toggles_8s']:.1f}/8s lost "
              f"{row['speech_lost_ms']:.0f} ms voiced {row['voiced'] * 100:.1f}% HF/LF "
              f"{row['hf_lf']:+.1f} dB hop p99 {row['hop_p99']:.1f} ms", flush=True)
        engine.graphs.release()
        del engine
    print(f"\n{'variant':<14} {'R3':>6} {'xflr':>5} {'bias':>6} {'click':>6} {'tog/8s':>6} "
          f"{'in-ph':>5} {'lost':>6} {'voiced':>7} {'HF/LF':>6} {'shared':>7} {'full':>6} "
          f"{'dBFS':>6} {'p99':>5}")
    for r in rows:
        print(f"{r['name']:<14} {r['r3']:>6.3f} {r['r3_ratio']:>5.2f} {r['bias']:>+6.2f} "
              f"{r['click']:>6.2f} {r['toggles_8s']:>6.1f} {r['toggles_in_phrases']:>5d} "
              f"{r['speech_lost_ms']:>6.0f} {r['voiced'] * 100:>6.1f}% {r['hf_lf']:>+6.1f} "
              f"{r['hf_lf_shared']:>+7.1f} {r['hf_lf_full_shared']:>+6.1f} "
              f"{r['active_dbfs']:>6.1f} {r['hop_p99']:>5.1f}")
    print(f"\nwavs: {out}\\{args.prefix}<variant>.wav (gated stream, aligned to the input)")
    return 0


# -------------------------------------------------------------------------------- typed

TYPED_TEXT = ("This is text alone, not microphone input. It sounds robotic, as you can tell, "
              "but the voice conversion is clear, and it works.")
# Step 3 (typed speech only): (Kokoro voice, TypedPolish fields, what it is)
TYPED_VARIANTS: dict[str, tuple[str, dict, str]] = {
    "plain": ("af_heart", {}, "today's path"),
    "envelope": ("af_heart", {"envelope": 0.35, "rms_frame_ms": 40.0},
                 "volume_envelope 0.35 on 40 ms / 10 ms frames"),
    "contour": ("af_heart", {"contour_scale": 1.2}, "F0 contour x1.2 around the median"),
    "blend": ("af_heart*0.7+af_bella*0.3", {}, "Kokoro style 0.7 af_heart + 0.3 af_bella"),
    "all": ("af_heart*0.7+af_bella*0.3",
            {"envelope": 0.35, "rms_frame_ms": 40.0, "contour_scale": 1.2}, "all three"),
}


def cmd_typed(args) -> int:
    """Typed text through Kokoro and the model once per variant -> out/ab/typed_<name>.wav,
    with the prosody measures: F0 spread (semitones), loudness swing, voicing, HF/LF."""
    from vox.audio.resample import resample_array, save_audio
    from vox.config.schemas import RuntimeParams
    from vox.offline.convert import FullFileConverter
    from vox.offline.metrics import active_frames, frame_rms_dbfs
    from vox.offline.simulate import load_engine
    from vox.tts.backends import KokoroBackend
    from vox.tts.cli import TTS_DIR
    from vox.tts.speak import TypedPolish, convert_rendered, render_tts, split_pieces

    engine = load_engine(args.models_dir, args.model, None, RuntimeParams(), None,
                         warmup_hops=0, profile_dir=args.profile)
    converter = FullFileConverter.from_engine(engine)
    sr, device_sr = engine.loaded.model.tgt_sr, engine.plan.device_sr
    text = args.tts_text or TYPED_TEXT
    names = list(TYPED_VARIANTS) if not args.variants else args.variants.split(",")
    out = ROOT / "out" / "ab"
    print(f"text: {text!r}")
    print(f"{'variant':<10} {'F0 med':>7} {'F0 sd st':>8} {'loud sd':>8} {'voiced':>7} "
          f"{'HF/LF':>6} {'s':>5}  what")
    for name in names:
        voice, fields, note = TYPED_VARIANTS[name]
        backend = KokoroBackend(voice, args.models_dir / TTS_DIR)
        backend.load()
        pieces = split_pieces(text)
        said = convert_rendered(converter, engine.device, sr, render_tts(backend, pieces),
                                engine.params, None, seed=0, pieces=pieces,
                                polish=TypedPolish(**fields))
        save_audio(out / f"typed_{name}.wav", said.audio, sr)
        audio = resample_array(said.audio, sr, device_sr)
        f0 = _f0_100hz(engine, audio)
        mask = active_frames(audio, device_sr)
        n = min(len(f0), len(mask))
        voiced_f0 = f0[:n][mask[:n] & (f0[:n] > 0)]
        semis = 12 * np.log2(voiced_f0 / np.median(voiced_f0)) if voiced_f0.size else []
        level = frame_rms_dbfs(audio, device_sr)[:len(mask)][mask]
        metrics = _voice_metrics(engine, audio)
        row = {"variant": name, "voice": voice, "polish": fields,
               "f0_median_hz": float(np.median(voiced_f0)) if voiced_f0.size else 0.0,
               "f0_sd_semitones": float(np.std(semis)) if len(semis) else 0.0,
               "loudness_sd_db": float(np.std(level)), "seconds": len(said.audio) / sr,
               "pitch": said.pitch, **metrics}
        print(f"{name:<10} {row['f0_median_hz']:>7.1f} {row['f0_sd_semitones']:>8.2f} "
              f"{row['loudness_sd_db']:>8.2f} {metrics.get('voiced_ratio', 0) * 100:>6.1f}% "
              f"{metrics['hf_lf_p90_db']:>+6.1f} {row['seconds']:>5.2f}  {note}", flush=True)
        _record("typed", {"model": args.model, "text": text, **row})
    print(f"\nwavs: {out}\\typed_<variant>.wav")
    return 0


# --------------------------------------------------------------------------------- sing

# Singing A/Bs (docs/DEVIATIONS.md §8), one change per variant against the GUI's Sing
# preset for this model: index 0.3, protect 0.35, F0 ceiling 360 Hz, auto level capped at
# +10 dB with no trim. Run on a sustained-note recording (--audio, --start, --seconds).
SING_PRESET = {"index_rate": 0.3, "protect": 0.35, "f0_ceiling_hz": 360.0}
SING_CONDITIONING = {"max_gain_db": 10.0, "trim_db": 0.0}
SING_VARIANTS: dict[str, tuple[dict, dict, str]] = {
    "reference": ({}, {}, ("Sing preset on the defaults (phase carry + noise roll + 80 ms "
                           "decoder context, the A/B the user chose)")),
    "classic": ({"decoder_context_ms": 0.0}, {"sine_phase_carry": False, "noise_roll": False},
                "the pre-8.8 engine: sine phase reset, noise redrawn, no decoder context"),
    "phase_reset": ({}, {"sine_phase_carry": False}, "defaults but the sine phase reset per hop"),
    "noise_redraw": ({}, {"noise_roll": False}, "defaults but the decoder noise redrawn per hop"),
    "decctx_0": ({"decoder_context_ms": 0.0}, {}, "defaults but no decoder left context"),
    "trimtail_1": ({}, {"f0_trim_tail": 1}, "RMVPE right edge as upstream (1 frame)"),
    "ceiling_off": ({}, {"f0_ceiling_hz": 0.0}, "Sing preset without the F0 ceiling"),
}


def _sing_trace(engine, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open stream (every hop through the model, conditioned like live) with two per-hop
    probes: the SOLA offset the hop chose, and the pitch step at its seam, i.e. the cents
    between the F0 the previous hop rendered its saved tail with and the F0 this hop
    renders the same 10 ms frames with (voiced frames only)."""
    plan, pipe, sola, cond = engine.plan, engine.pipeline, engine.sola, engine.conditioner
    torch.manual_seed(0)  # before the reset: it draws the window's first noise
    pipe.reset()
    sola.reset()
    cond.reset()
    block, hops = plan.block, len(audio) // plan.block
    out = np.zeros(hops * block, dtype=np.float32)
    offsets, steps = np.zeros(hops, dtype=np.int64), []
    head, shift, fade = plan.skip_head, plan.shift_frames, plan.sola_buffer // plan.zc
    hist, prev = sola.offset_hist.clone(), None
    for n in range(hops):
        pipe.upload(cond.process(audio[n * block:(n + 1) * block]))
        stitched = sola.stitch(pipe.process_hop())
        out[n * block:(n + 1) * block] = stitched.cpu().numpy()
        now = sola.offset_hist.clone()
        offsets[n] = int(torch.argmax(now - hist))
        hist = now
        pitch = pipe.pitch_cache.pitchf_view[0].cpu().numpy().copy()
        if prev is not None:
            a, b = prev[head + shift:head + shift + fade], pitch[head:head + fade]
            voiced = (a > 0) & (b > 0)
            steps.append(np.abs(1200 * np.log2(a[voiced] / b[voiced])))
        prev = pitch
    return out, offsets, np.concatenate(steps) if steps else np.zeros(0)


def cmd_sing(args) -> int:
    from vox.audio.resample import save_audio
    from vox.offline.metrics import boundary_click_ratio

    audio, registry, cache = _setup(args)
    names = list(SING_VARIANTS) if not args.variants else args.variants.split(",")
    out_dir = ROOT / "out" / "ab"
    rows = []
    for name in names:
        shape_kw, param_kw, note = SING_VARIANTS[name]
        engine = _engine(args, registry, cache, shape_kw, SING_CONDITIONING,
                         **{**SING_PRESET, **param_kw})
        plan, sr = engine.plan, engine.plan.device_sr
        stream, offsets, steps = _sing_trace(engine, audio)
        settle = int(np.ceil(_settle_s(plan) * 1000 / plan.block_ms))
        offsets_ms = offsets[settle:] * 1000.0 / sr
        jumps = np.abs(np.diff(offsets_ms))
        aligned = stream[plan.stream_delay:][:len(audio)]
        f0 = _f0_100hz(engine, aligned)
        voiced = f0[f0 > 0]
        row = {"name": name, "note": note, "hop_ms": plan.block_ms,
               "latency_ms": plan.algorithmic_latency_ms,
               "offset_p50_ms": float(np.median(offsets_ms)),
               "offset_mode_share": float(np.mean(offsets_ms == np.median(offsets_ms))),
               "offset_zero_share": float(np.mean(offsets_ms == 0)),
               # hops that kept the previous offset: coherent renders line up where the
               # last one did (a jump by whole F0 periods is benign on a held note)
               "offset_held_share": float(np.mean(jumps == 0)),
               "offset_jump_mean_ms": float(jumps.mean()),
               "offset_jump_p95_ms": float(np.percentile(jumps, 95)),
               "seam_cents_p50": float(np.median(steps)) if steps.size else float("nan"),
               "seam_cents_p95": float(np.percentile(steps, 95)) if steps.size else float("nan"),
               "click": boundary_click_ratio(stream, plan.block),
               "f0_p95_hz": float(np.percentile(voiced, 95)) if voiced.size else float("nan"),
               "f0_max_hz": float(voiced.max()) if voiced.size else float("nan"),
               "histogram_ms": {f"{k * 1000.0 / sr:g}": int(v) for k, v in
                                zip(*np.unique(offsets[settle:], return_counts=True))}}
        rows.append(row)
        save_audio(out_dir / f"{args.prefix}sing_{name}.wav", aligned, sr)
        _record("sing", {"model": args.model, "audio": str(args.audio), "start": args.start,
                         "pitch": args.pitch, "variant": name, "shape": shape_kw,
                         "params": {**SING_PRESET, **param_kw}, "row": row})
        print(f"[{name}] {note}: offsets p50 {row['offset_p50_ms']:.2f} ms, at 0 "
              f"{100 * row['offset_zero_share']:.0f}%, held "
              f"{100 * row['offset_held_share']:.0f}%, jump mean {row['offset_jump_mean_ms']:.2f}"
              f" p95 {row['offset_jump_p95_ms']:.2f} ms; seam step p50 "
              f"{row['seam_cents_p50']:.1f} p95 {row['seam_cents_p95']:.1f} cents; click "
              f"{row['click']:.2f}; F0 p95 {row['f0_p95_hz']:.0f} max {row['f0_max_hz']:.0f} Hz",
              flush=True)
        engine.graphs.release()
        del engine
    print(f"\n{'variant':<30} {'lat':>4} {'off p50':>7} {'@0':>4} {'held':>4} {'jump':>5} "
          f"{'jump95':>6} {'seam50':>6} {'seam95':>6} {'click':>5} {'f0p95':>5}")
    for r in rows:
        print(f"{r['name']:<30} {r['latency_ms']:>4.0f} {r['offset_p50_ms']:>7.2f} "
              f"{100 * r['offset_zero_share']:>3.0f}% {100 * r['offset_held_share']:>3.0f}% "
              f"{r['offset_jump_mean_ms']:>5.2f} "
              f"{r['offset_jump_p95_ms']:>6.2f} {r['seam_cents_p50']:>6.1f} "
              f"{r['seam_cents_p95']:>6.1f} {r['click']:>5.2f} {r['f0_p95_hz']:>5.0f}")
    print(f"\nwavs: {out_dir}\\{args.prefix}sing_<variant>.wav")
    return 0


# -------------------------------------------------------------------------------- onset

# Phrase entrances through the gate (docs/DEVIATIONS.md §8): the pre-fix gate (VAD on the
# newest 512-sample chunk only, onset hop faded in over the whole overlap) against the fix.
ONSET_VARIANTS: dict[str, tuple[dict, dict, str]] = {
    "before": ({"onset_ramp_ms": 0.0}, {"vad_reduce": "last"},
               "pre-fix: VAD = newest chunk, onset hop fades in over the overlap"),
    "vad_max": ({"onset_ramp_ms": 0.0}, {}, "VAD max over the hop's chunks only"),
    "after": ({}, {}, "VAD max + 5 ms onset ramp (the defaults)"),
    "margin_3db": ({}, {"onset_margin_db": 3.0},
                   "A/B: the defaults + energy onset 3 dB over the threshold (not 6)"),
}


def _onset_times(audio: np.ndarray, sr: int, below_db: float = 30.0) -> list[float]:
    """Where each phrase of ``audio`` starts: the first 10 ms frame within ``below_db`` of
    the clip's loud frames, after at least 250 ms below it (seconds)."""
    from vox.offline.metrics import frame_rms_dbfs

    level = frame_rms_dbfs(audio, sr, win_ms=10.0)
    loud = level >= np.percentile(level, 95) - below_db
    starts, quiet = [], 25
    for i, on in enumerate(loud):
        if on and quiet >= 25:
            starts.append(i / 100.0)
        quiet = 0 if on else quiet + 1
    return starts


def cmd_onset(args) -> int:
    from vox.audio.resample import save_audio
    from vox.offline.metrics import frame_rms_dbfs
    from vox.offline.simulate import StreamSimulator

    audio, registry, cache = _setup(args)
    hops = [float(h) for h in args.hops.split(",")]
    out_dir = ROOT / "out" / "ab"
    sr = 48000
    entrances = _onset_times(audio, sr)
    print(f"input phrase entrances (s from --start): {', '.join(f'{t:.2f}' for t in entrances)}")
    rows = []
    for hop in hops:
        for name, (shape_kw, gate_kw, note) in ONSET_VARIANTS.items():
            engine = _engine(args, registry, cache, {**shape_kw, "block_ms": hop}, None,
                             gate_kw)
            result = StreamSimulator(engine).run(audio, gate=True, seed=0)
            out = result.aligned[:len(audio)]
            level_in = frame_rms_dbfs(audio, sr, win_ms=10.0)
            level_out = frame_rms_dbfs(out, sr, win_ms=10.0)
            thr_out = np.percentile(level_out, 95) - 30.0
            opens = [e["t"] for e in result.events if e["kind"] == "gate_open"]
            lags, kept = [], []
            for t in entrances:
                i = round(t * 100)
                # first output frame (from 100 ms before the entrance) that carries the voice
                window = level_out[max(0, i - 10):i + 60]
                on = np.flatnonzero(window >= thr_out)
                lag = (on[0] + max(0, i - 10) - i) * 10.0 if on.size else float("nan")
                lags.append(lag)
                # share of the entrance's first 150 ms of input energy the output carries
                seg = slice(i, i + 15)
                e_in = np.sum(10 ** (level_in[seg] / 10))
                e_out = np.sum(10 ** (level_out[seg] / 10))
                e_in_ref = np.sum(10 ** (level_in[i + 15:i + 60] / 10))
                e_out_ref = np.sum(10 ** (level_out[i + 15:i + 60] / 10))
                kept.append(float((e_out / e_out_ref) / (e_in / e_in_ref + 1e-12))
                            if e_out_ref > 0 else float("nan"))
            gate_lag = [min((o - t for o in opens if o >= t - 0.1), default=float("nan")) * 1000
                        for t in entrances]
            row = {"hop": hop, "name": name, "note": note, "lags_ms": lags, "kept": kept,
                   "gate_lag_ms": gate_lag}
            rows.append(row)
            stereo = np.stack([audio[:len(out)], out], axis=1)  # L: mic, R: converted
            save_audio(out_dir / f"{args.prefix}onset_{name}_hop{hop:g}.wav", stereo, sr)
            _record("onset", {"model": args.model, "audio": str(args.audio),
                              "start": args.start, **row})
            print(f"[hop {hop:g} {name}] {note}\n    output lag after entrance (ms): "
                  f"{', '.join(f'{x:.0f}' for x in lags)}\n    gate open after entrance (ms):"
                  f" {', '.join(f'{x:.0f}' for x in gate_lag)}\n    first-150-ms energy kept "
                  f"(1 = as loud as the rest of the phrase): "
                  f"{', '.join(f'{x:.2f}' for x in kept)}", flush=True)
            engine.graphs.release()
            del engine
    print(f"\nwavs (L mic, R output): {out_dir}\\{args.prefix}onset_<variant>_hop<ms>.wav")
    return 0


# ----------------------------------------------------------------------------- underrun

_BROADCAST_LOAD = """
import sys, time
import sounddevice as sd
dev = next(d["index"] for d in sd.query_devices()
           if "nvidia broadcast" in d["name"].lower() and d["max_input_channels"] > 0
           and sd.query_hostapis(d["hostapi"])["name"] == "Windows WASAPI")
with sd.InputStream(device=dev, channels=1, samplerate=48000, callback=lambda *a: None):
    time.sleep(float(sys.argv[1]))
"""


def cmd_underrun(args) -> int:
    """Paced runs (hop N available at N x hop, the output ring modelled as live) over hop x
    extra for one model: hop compute p50 / p99, the share of hops over budget and the
    underruns with their times. ``--broadcast-load`` holds the NVIDIA Broadcast virtual mic
    open in a side process meanwhile (its denoiser then runs on the same GPU, as when it is
    the input); the audio is discarded."""
    import subprocess

    from vox.offline.simulate import StreamSimulator
    from vox.profiling.report import stage_table

    audio, registry, cache = _setup(args)
    hops = [float(h) for h in args.hops.split(",")]
    extras = [float(e) for e in args.extras.split(",")]
    side = None
    if args.broadcast_load:
        side = subprocess.Popen([sys.executable, "-c", _BROADCAST_LOAD, "3600"])
        time.sleep(3.0)  # let Broadcast start its effect
    rows = []
    try:
        for extra in extras:
            for hop in hops:
                engine = _engine(args, registry, cache, {"block_ms": hop, "extra_ms": extra})
                result = StreamSimulator(engine).run(audio, realtime=True, seed=0)
                table = stage_table(engine.timers, SKIP)
                host = table["total"]["host"]
                gpu = {k: v.get("gpu", {}).get("p50") for k, v in table.items()
                       if k in ("f0", "embed", "synth")}
                times = [e["t"] for e in result.events if e["kind"] == "underrun"]
                row = {"model": args.model, "hop": hop, "extra": extra, "p50": host["p50"],
                       "p99": host["p99"], "max": host["max"],
                       "underruns": result.underruns, "underrun_t": times[:20],
                       "deadline_misses": result.deadline_misses, "gpu_p50": gpu,
                       "broadcast_load": bool(args.broadcast_load)}
                rows.append(row)
                _record("underrun", row)
                print(f"[{args.model} hop {hop:g} extra {extra:g}"
                      f"{' +Broadcast' if side else ''}] p50 {host['p50']:.1f} p99 "
                      f"{host['p99']:.1f} ms (budget {hop:g}); underruns {result.underruns}"
                      f"{' at ' + ', '.join(f'{t:.1f}s' for t in times[:8]) if times else ''};"
                      f" GPU p50 f0/embed/synth "
                      + "/".join(f"{v:.1f}" if v else "-" for v in gpu.values()), flush=True)
                engine.graphs.release()
                del engine
                torch.cuda.empty_cache()
    finally:
        if side is not None:
            side.terminate()
    return 0


# ----------------------------------------------------------------------------- suppress


def cmd_suppress(args) -> int:
    """RNNoise on the mic path, measured without the GPU: delay, CPU per hop, attenuation
    of a raw mic's pauses and speech (--audio), band loss on sustained sung vowels
    (--vowel), and the live toggle (step size around the switch)."""
    import time as _time

    from vox.audio import suppression
    from vox.audio.resample import load_audio, save_audio
    from vox.offline.metrics import frame_rms_dbfs

    reason = suppression.unavailable_reason()
    if reason:
        print(f"suppression unavailable: {reason}")
        return 1
    sr = suppression.SR
    raw, _ = load_audio(args.audio, sr)
    start = int(args.start * sr)
    raw = raw[start:start + int(args.seconds * sr)].astype(np.float64)

    def render(x: np.ndarray, block: int, mix: float) -> tuple[np.ndarray, list[float]]:
        sup = suppression.Suppressor(block, enabled=True, mix=mix)
        out, times = np.zeros(len(x) // block * block), []
        for i in range(0, len(out), block):
            t = _time.perf_counter()
            out[i:i + block] = sup.process(x[i:i + block])
            times.append((_time.perf_counter() - t) * 1000.0)
        return out, times

    # delay: cross-correlation of the fully wet output against the input
    wet, times20 = render(raw, 960, 1.0)
    _, times60 = render(raw, 2880, 1.0)
    lags = np.arange(0, 2 * suppression.DELAY)
    corr = [np.dot(raw[:len(wet) - lag], wet[lag:]) for lag in lags]
    delay = int(lags[int(np.argmax(corr))])
    print(f"delay: {delay} samples = {1000 * delay / sr:.1f} ms (RNNoise frame alignment)")
    for label, t in (("20 ms hop", times20), ("60 ms hop", times60)):
        t = np.asarray(t[10:])
        print(f"CPU per hop, {label}: p50 {np.percentile(t, 50):.2f} ms, p99 "
              f"{np.percentile(t, 99):.2f} ms, max {t.max():.2f} ms")

    # attenuation on the raw mic: pauses (room tone, fan, keys, bleed) vs speech frames
    level = frame_rms_dbfs(raw, sr)
    loud = np.percentile(level, 95)
    pause = level < loud - 30.0
    speech = level >= loud - 18.0
    print(f"raw mic: speech frames {speech.mean() * 100:.0f}%, pause frames "
          f"{pause.mean() * 100:.0f}% (input p95 {loud:.1f} dBFS)")
    rows = {}
    for mix in (1.0, 0.5):
        out, _ = render(raw, 960, mix)
        aligned = out[delay:]
        lvl = frame_rms_dbfs(aligned, sr)
        n = min(len(lvl), len(level))
        lvl, ref_level = lvl[:n], level[:n]
        p, s = pause[:n], speech[:n]
        # transients in the pauses: the loudest 2% of pause frames (clicks, keys, thumps)
        tr = p & (ref_level >= np.percentile(ref_level[p], 98))
        rows[mix] = {"pause_db": _mean_db(lvl, p) - _mean_db(ref_level, p),
                     "transient_db": _mean_db(lvl, tr) - _mean_db(ref_level, tr),
                     "speech_db": _mean_db(lvl, s) - _mean_db(ref_level, s)}
        save_audio(ROOT / "out" / "ab" / f"{args.prefix}suppress_mic_mix{mix:g}.wav",
                   aligned.astype(np.float32), sr)
        print(f"mix {mix:g}: pauses {rows[mix]['pause_db']:+.1f} dB, pause transients "
              f"{rows[mix]['transient_db']:+.1f} dB, speech {rows[mix]['speech_db']:+.1f} dB")

    if args.vowel is not None:
        sung, _ = load_audio(args.vowel, sr)
        sung = sung.astype(np.float64)
        f0 = _f0_track(sung, sr)
        steady = _sustained_frames(f0)
        print(f"sung vowels: {steady.sum() / 100:.1f} s of sustained notes (pitch within "
              f"+-50 cents over 300 ms) in {args.vowel.name}")
        for mix in (1.0, 0.5):
            out, _ = render(sung, 960, mix)
            aligned = out[delay:]
            loss = _band_loss_db(sung, aligned, sr, steady, (1500.0, 4000.0))
            low = _band_loss_db(sung, aligned, sr, steady, (100.0, 1000.0))
            print(f"mix {mix:g}: sustained-note band loss 1.5-4 kHz {loss:+.1f} dB "
                  f"(0.1-1 kHz {low:+.1f} dB)")
            save_audio(ROOT / "out" / "ab" / f"{args.prefix}suppress_sung_mix{mix:g}.wav",
                       aligned.astype(np.float32), sr)

    # the live toggle, switched inside speech: on in the longest phrase (the crossfade
    # starts after the warm-up), off in the second longest, 20 ms hops
    from vox.offline.metrics import phrases

    spans = sorted(phrases(raw.astype(np.float32), sr), key=lambda s: s[0] - s[1])[:2]
    warm = int(sr * suppression.WARMUP_MS / 1000)
    on_at = (spans[0][0] + spans[0][1]) // 2 // 960 * 960
    off_at = (spans[1][0] + spans[1][1]) // 2 // 960 * 960
    sup = suppression.Suppressor(960)
    out = np.zeros(len(raw) // 960 * 960)
    for i in range(0, len(out), 960):
        if i == on_at:
            sup.set(enabled=True)
        if i == off_at:
            sup.set(enabled=False)
        out[i:i + 960] = sup.process(raw[i:i + 960])
    steps = np.abs(np.diff(out))
    ref = np.percentile(np.abs(np.diff(raw[:len(out)])), 99.9)
    for label, at in (("on", on_at + warm), ("off", off_at)):
        near = steps[at - 480:at + 2 * 960]
        local = np.percentile(np.abs(np.diff(raw[at - 4800:at + 4800])), 99.9)
        print(f"toggle {label} at {at / sr:.2f} s (inside speech): largest step in the 20 ms "
              f"fade {near.max():.4f} vs the input's own p99.9 step there {local:.4f} "
              f"({near.max() / local:.2f}x; whole-file p99.9 {ref:.4f})")
    _record("suppress", {"audio": str(args.audio), "delay": delay, "rows": rows,
                         "cpu_p50_20": float(np.percentile(times20[10:], 50)),
                         "cpu_p99_20": float(np.percentile(times20[10:], 99))})
    return 0


def _mean_db(level_db: np.ndarray, mask: np.ndarray) -> float:
    """Mean power of the masked frames, in dB."""
    return float(10 * np.log10(np.mean(10 ** (level_db[mask] / 10)) + 1e-12))


def _f0_track(x: np.ndarray, sr: int) -> np.ndarray:
    """RMVPE over a whole device-rate signal (no engine needed), 100 Hz."""
    from vox.audio.resample import resample_array
    from vox.pitch import make_extractor

    extractor = make_extractor("rmvpe")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor.load(device, ROOT / "models")
    y = resample_array(x.astype(np.float32), sr, 16000)
    with torch.inference_mode():
        return extractor.extract(torch.from_numpy(y).to(device)).float().cpu().numpy()


def _sustained_frames(f0: np.ndarray, span: int = 30, cents: float = 50.0) -> np.ndarray:
    """10 ms frames inside a voiced run whose pitch stays within +-``cents`` of its median
    for ``span`` frames: held notes."""
    out = np.zeros(len(f0), dtype=bool)
    for i in range(len(f0) - span):
        seg = f0[i:i + span]
        if np.all(seg > 0):
            dev = 1200 * np.log2(seg / np.median(seg))
            if np.abs(dev).max() <= cents:
                out[i:i + span] = True
    return out


def _band_loss_db(ref: np.ndarray, test: np.ndarray, sr: int, mask: np.ndarray,
                  band: tuple[float, float]) -> float:
    """Energy change in ``band`` over the masked 10 ms frames (40 ms Hann windows)."""
    hop, win = sr // 100, sr // 25
    window = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    sel = (freqs >= band[0]) & (freqs < band[1])
    e_ref = e_test = 0.0
    for i in np.flatnonzero(mask):
        a = i * hop
        if a + win > min(len(ref), len(test)):
            break
        e_ref += np.sum(np.abs(np.fft.rfft(ref[a:a + win] * window))[sel] ** 2)
        e_test += np.sum(np.abs(np.fft.rfft(test[a:a + win] * window))[sel] ** 2)
    return float(10 * np.log10((e_test + 1e-12) / (e_ref + 1e-12)))


# --------------------------------------------------------------------------------- perf


def cmd_perf(args) -> int:
    import yaml

    from vox.offline.simulate import StreamSimulator
    from vox.profiling.report import stage_table
    from vox.profiling.stats import headroom

    grid = yaml.safe_load(Path(args.sweep).read_text())
    audio, registry, cache = _setup(args)
    rows = []
    # YAML reads bare on/off as booleans; accept both spellings.
    modes = [{True: "on", False: "off"}.get(m, str(m)) for m in grid.get("graphs", [args.graphs])]
    print(f"{'block':>5} {'extra':>5} {'f0':>6} {'mode':>6} | {'f0':>6} {'embed':>6} "
          f"{'synth':>6} {'total p50':>9} {'p95':>7} {'p99':>7} | {'RTF':>5} {'headroom':>8}"
          "  verdict")
    for mode in modes:
        args.graphs = mode
        for method in grid["f0_method"]:
            for extra in grid["extra_ms"]:
                for block in grid["block_ms"]:
                    engine = _engine(args, registry, cache,
                                     {"block_ms": block, "extra_ms": extra}, f0_method=method)
                    result = StreamSimulator(engine).run(audio)
                    table = stage_table(engine.timers, SKIP)
                    total = table["total"]["host"]
                    room = headroom(engine.plan.block_ms, total["p99"])
                    verdict = "pass" if room >= 1.5 else "marginal" if room >= 1.0 else "fail"
                    p50 = {k: v["host"]["p50"] for k, v in table.items()}
                    print(f"{block:>5} {extra:>5} {method:>6} {engine.mode:>6} | "
                          f"{p50['f0']:>6.2f} {p50['embed']:>6.2f} {p50['synth']:>6.2f} "
                          f"{total['p50']:>9.2f} {total['p95']:>7.2f} {total['p99']:>7.2f} | "
                          f"{result.rtf:>5.2f} {room:>7.2f}x  {verdict}", flush=True)
                    row = {"block_ms": block, "extra_ms": extra, "f0_method": method,
                           "stage_p50_ms": p50, "total": total, "rtf": result.rtf,
                           "headroom": room, "verdict": verdict, "mode": engine.mode,
                           "graphs_requested": mode,
                           "graphs": dataclasses.asdict(engine.graphs.report)}
                    rows.append(row)
                    _record("perf", {"model": args.model, **row})
                    engine.graphs.release()
                    del engine
    for mode in sorted({r["mode"] for r in rows}):
        stable = [r for r in rows if r["verdict"] == "pass" and r["mode"] == mode]
        best = min(stable, key=lambda r: r["block_ms"]) if stable else None
        print(f"\nsmallest hop with p99 headroom >= 1.5x ({mode}, throughput mode): "
              + (f"{best['block_ms']} ms (extra {best['extra_ms']} ms, {best['f0_method']})"
                 if best else "none in this grid"))
    return 0


# ------------------------------------------------------------------------- extra / sola


def _quality_sweep(args, name: str, points: list[dict], label) -> int:
    from vox.offline.convert import render_full
    from vox.offline.metrics import compare
    from vox.offline.simulate import StreamSimulator

    audio, registry, cache = _setup(args)
    reference_engine = _engine(args, registry, cache)
    sr = reference_engine.plan.device_sr
    full = render_full(reference_engine, audio, seed=0)
    engines = [_engine(args, registry, cache, pt.get("shape"), **pt.get("params", {}))
               for pt in points]
    # One common evaluation region: past the longest start-up among the points.
    start_s = max(_settle_s(e.plan) for e in engines)
    floor = compare(full, render_full(reference_engine, audio, seed=1), sr, start_s=start_s)
    print(f"evaluated from {start_s:.1f} s; noise floor (full vs full, new seed): "
          f"all {floor['l1']:.3f}, speech {floor['l1_speech']:.3f}")
    print(f"{'point':<26} {'L1 all':>7} {'speech':>7} {'/floor':>6} {'quiet':>6} {'bias':>6} "
          f"{'click':>6} {'offs p50':>8} {'p95':>5} {'hop p50 ms':>10}")
    for point, engine in zip(points, engines):
        result = StreamSimulator(engine).run(audio, seed=0, gate=args.gate == "on")
        quality = _quality(full, result, sr, start_s)
        offsets = result.sola_offsets_ms
        o50, o95 = np.percentile(offsets, [50, 95]) if offsets.size else (np.nan, np.nan)
        total = _stage_p50(engine, min(SKIP, result.hops // 2)).get("total", float("nan"))
        print(f"{label(point):<26} {quality['l1']:>7.3f} {quality['l1_speech']:>7.3f} "
              f"{quality['l1_speech'] / floor['l1_speech']:>6.2f} {quality['l1_quiet']:>6.3f} "
              f"{quality['bias']:>+6.2f} {quality['click_ratio']:>6.3f} {o50:>8.2f} "
              f"{o95:>5.2f} {total:>10.2f}")
        _record(name, {"model": args.model, **point, "noise_floor": floor, "start_s": start_s,
                       "gate": args.gate, "mode": engine.mode, **quality,
                       "sola_offset_p50_ms": float(o50), "sola_offset_p95_ms": float(o95),
                       "total_p50_ms": total})
    return 0


def cmd_extra(args) -> int:
    points = [{"shape": {"extra_ms": e}} for e in (100, 300, 500, 1000, 2500)]
    return _quality_sweep(args, "extra", points,
                          lambda p: f"extra {p['shape']['extra_ms']} ms")


def cmd_envelope(args) -> int:
    points = [{"shape": {"extra_ms": e}, "params": {"volume_envelope": v}}
              for e in (500, 2500) for v in (1.0, 0.5, 0.0)]
    return _quality_sweep(
        args, "envelope", points,
        lambda p: f"extra {p['shape']['extra_ms']} env {p['params']['volume_envelope']}")


def cmd_block(args) -> int:
    points = [{"shape": {"block_ms": b}} for b in (20, 50, 100, 250, 500)]
    return _quality_sweep(args, "block", points,
                          lambda p: f"block {p['shape']['block_ms']} ms")


def cmd_decctx(args) -> int:
    points = [{"shape": {"decoder_context_ms": c}} for c in (0, 40, 80, 120, 200)]
    return _quality_sweep(args, "decctx", points,
                          lambda p: f"dec context {p['shape']['decoder_context_ms']} ms")


def cmd_sola(args) -> int:
    points = [{"shape": {"crossfade_ms": o}, "params": {"sola_mode": m}}
              for o in (10, 20, 30, 40) for m in ("plain", "pv")]
    return _quality_sweep(
        args, "sola", points,
        lambda p: f"overlap {p['shape']['crossfade_ms']} ms {p['params']['sola_mode']}")


# --------------------------------------------------------------------------------- gate


def cmd_gate(args) -> int:
    """Step 2 acceptance: the silence gate over the LibriSpeech clip.

    Pause-region loudness (input below the gate threshold) of the gated stream vs the
    full-file render and the ungated stream; clicks at gate transitions; cost of a closed
    hop (the GPU-idle claim); speech-frame distance unchanged by gating.
    """
    from vox.audio.resample import save_audio
    from vox.offline.convert import render_full
    from vox.offline.metrics import (
        boundary_click_ratio,
        compare,
        frame_rms_dbfs,
        transition_click_ratio,
    )
    from vox.offline.simulate import StreamSimulator
    from vox.profiling.report import stage_table

    audio, registry, cache = _setup(args)
    engine = _engine(args, registry, cache)
    plan, sr = engine.plan, engine.plan.device_sr
    gate = engine.gate
    sim = StreamSimulator(engine)
    start_s = _settle_s(plan)
    full = render_full(engine, audio, seed=0)
    open_result = sim.run(audio, seed=0)
    gated = sim.run(audio, seed=0, gate=True)
    gated_table = stage_table(engine.timers, 0)

    n = len(audio)
    begin = int(start_s * sr)
    level_in = frame_rms_dbfs(audio[begin:n], sr)
    pause = level_in < gate.config.threshold_dbfs
    speech = level_in >= gate.config.threshold_dbfs + 10

    def region_dbfs(x: np.ndarray, mask: np.ndarray) -> float:
        frames = frame_rms_dbfs(x[begin:n], sr)
        m = mask[:len(frames)]
        power = np.mean(10 ** (frames[m] / 10.0)) if m.any() else 1e-12
        return float(10 * np.log10(power + 1e-12))

    rows = {"full-file": full, "stream (gate off)": open_result.aligned[:n],
            "stream (gate on)": gated.aligned[:n]}
    loud = {k: {"pause_dbfs": region_dbfs(v, pause), "speech_dbfs": region_dbfs(v, speech)}
            for k, v in rows.items()}
    q_open = compare(full, open_result.aligned[:n], sr, lag_search_ms=15.0, start_s=start_s)
    q_gated = compare(full, gated.aligned[:n], sr, lag_search_ms=15.0, start_s=start_s)

    toggles = np.flatnonzero(np.diff(gated.gate_open.astype(np.int8)) != 0) + 1
    # transitions in the emitted stream: a closed hop emits the drain (first) or zeros
    positions = toggles * plan.block
    click_transitions = transition_click_ratio(gated.audio, positions, plan.block)
    click_hops = boundary_click_ratio(gated.audio, plan.block)
    closed = gated.closed_hop_ms
    closed_share = 1.0 - gated.gate_open.mean()
    closed_p50 = float(np.median(closed)) if closed.size else float("nan")
    gpu_closed = gated_table.get("resample_in", {}).get("gpu", {}).get("p50", float("nan"))

    out = ROOT / "out"
    save_audio(out / "gate_full.wav", full, sr)
    save_audio(out / "gate_off.wav", open_result.aligned[:n], sr)
    save_audio(out / "gate_on.wav", gated.aligned[:n], sr)

    print(f"gate: threshold {gate.config.threshold_dbfs} dBFS, VAD {'on' if gate.vad else 'off'} "
          f"(thr {gate.config.vad_threshold}), hangover {gate.config.hangover_ms:.0f} ms, "
          f"extra {plan.extra_ms:.0f} ms, mode {engine.mode}")
    print(f"hops {gated.hops}: closed {closed_share * 100:.1f}%, transitions "
          f"{gated.gate_transitions}; closed-hop host p50 {closed_p50:.2f} ms "
          f"(ring advance GPU p50 {gpu_closed:.2f} ms)")
    print(f"{'render':<20} {'pause dBFS':>11} {'speech dBFS':>12}")
    for k, v in loud.items():
        print(f"{k:<20} {v['pause_dbfs']:>11.1f} {v['speech_dbfs']:>12.1f}")
    print(f"speech-frame log-mel L1 vs full: gate off {q_open['l1_speech']:.3f}, "
          f"gate on {q_gated['l1_speech']:.3f}; level bias gate off {q_open['bias']:+.3f}, "
          f"gate on {q_gated['bias']:+.3f} nepers")
    print(f"click ratio: gate transitions {click_transitions:.3f}, hop boundaries {click_hops:.3f}")
    pause_delta = loud["stream (gate on)"]["pause_dbfs"] - loud["full-file"]["pause_dbfs"]
    checks = {
        "gate closes and reopens on this clip (>= 2 transitions)": gated.gate_transitions >= 2,
        "no clicks at gate transitions (ratio <= 1.25)": click_transitions <= 1.25,
        "no clicks at hop boundaries (ratio <= 1.25)": click_hops <= 1.25,
        "pause-region loudness not above full-file + 1 dB": pause_delta <= 1.0,
        "closed hop costs < 2 ms host (GPU idle)": closed_p50 < 2.0,
        "speech frames unchanged by gating (<= 5%)":
            q_gated["l1_speech"] <= q_open["l1_speech"] * 1.05,
    }
    print()
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    _record("gate", {"model": args.model, "plan": dataclasses.asdict(plan), "mode": engine.mode,
                     "gate": dataclasses.asdict(gate.config), "loudness": loud,
                     "quality_open": q_open, "quality_gated": q_gated,
                     "click_transitions": click_transitions, "click_hops": click_hops,
                     "closed_share": closed_share, "closed_hop_p50_ms": closed_p50,
                     "transitions": gated.gate_transitions,
                     "checks": {k: bool(v) for k, v in checks.items()}})
    return 0 if all(checks.values()) else 1


# -------------------------------------------------------------------------- graph probe


def cmd_graph_probe(args) -> int:
    """Feasibility probe only (Phase 6 owns the real thing): capture one stage of the hop
    in a CUDA graph and compare eager with replay.

    Run one stage per process: a failed capture can poison the CUDA context. ``--stage hop``
    (the whole hop as a single graph) captures but its replay hung on the reference machine
    (driver 617.14, WDDM); see docs/DEVIATIONS.md before trying it without a timeout.
    """
    audio, registry, cache = _setup(args)
    engine = _engine(args, registry, cache, {"block_ms": args.block_ms})
    pipeline, sola, plan = engine.pipeline, engine.sola, engine.plan
    pipeline.timers.enabled = False
    for i in range(60):  # fill every buffer with real speech
        pipeline.upload(audio[(100 + i) * plan.block:(101 + i) * plan.block])
        chunk = pipeline.process_hop().clone()
        sola.stitch(chunk)
    pitch, pitchf = pipeline.pitch_cache.pitch_view, pipeline.pitch_cache.pitchf_view
    stages = {
        "sola": lambda: sola.stitch(chunk),
        "embed": lambda: pipeline.embedder.embed(pipeline.window_16k),
        "f0": lambda: pipeline.extractor.extract(pipeline.f0_tail),
        "synth": lambda: pipeline.model.infer(
            pipeline.feats, pipeline.p_len_t, pitch, pitchf, pipeline.sid_t, plan.skip_head,
            plan.return_length, plan.dec_context),
        "hop": lambda: sola.stitch(pipeline.process_hop()),
    }
    fn = stages[args.stage]

    def timed(call, n=200):
        times = []
        for _ in range(n):
            start = time.perf_counter()
            call()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
        return np.percentile(times[20:], [50, 95, 99])

    eager = timed(fn)
    print(f"{args.stage}: eager        p50 {eager[0]:6.2f}  p95 {eager[1]:6.2f}  "
          f"p99 {eager[2]:6.2f} ms", flush=True)
    graph = torch.cuda.CUDAGraph()
    try:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with torch.inference_mode(), torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
    except Exception as err:
        print(f"{args.stage}: capture FAILED: {type(err).__name__}: {err}")
        _record("graph_probe", {"model": args.model, "stage": args.stage, "captured": False,
                                "error": str(err)})
        return 1
    print(f"{args.stage}: captured, replaying...", flush=True)

    def replay():
        with torch.inference_mode():  # the graph's RNG state was registered in this mode
            graph.replay()

    replayed = timed(replay)
    print(f"{args.stage}: graph replay p50 {replayed[0]:6.2f}  p95 {replayed[1]:6.2f}  "
          f"p99 {replayed[2]:6.2f} ms   ({eager[0] / replayed[0]:.1f}x)")
    _record("graph_probe", {"model": args.model, "stage": args.stage, "captured": True,
                            "block_ms": plan.block_ms, "eager_ms": eager.tolist(),
                            "replay_ms": replayed.tolist()})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="vox bench", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("accept", "ab", "typed", "perf", "extra", "sola",
                                            "block", "decctx", "envelope", "gate",
                                            "graph-probe", "sing", "onset", "suppress",
                                            "underrun"))
    parser.add_argument("--model", default="base_f0_48k_v2")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--audio", type=Path,
                        default=ROOT / "benchmarks" / "corpus" / "libri_1272.wav")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--start", type=float, default=0.0,
                        help="seconds into --audio where the excerpt starts")
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--hubert-fp16", action="store_true")
    parser.add_argument("--hubert-autocast", choices=("auto", "fp16", "bf16", "off"),
                        default="auto", help="HuBERT autocast ladder (TARGET.md rule 2)")
    parser.add_argument("--profile", type=Path, default=ROOT / "profiles" / "default",
                        help="profile directory whose runtime.json stores warmup decisions")
    parser.add_argument("--graphs", choices=("on", "off", "auto"), default="auto",
                        help="CUDA graphs mode for the engine under test")
    parser.add_argument("--gate", choices=("on", "off"), default="off",
                        help="run the silence gate: quality sweeps (extra, block, ...) and "
                             "accept's live-path run")
    parser.add_argument("--gate-threshold", type=float, default=None,
                        help="gate energy threshold, dBFS (default: calibration.json)")
    parser.add_argument("--crossfade-ms", type=float, default=None)
    parser.add_argument("--conditioning", choices=("on", "off"), default=None,
                        help="the live input chain (high-pass, auto level, limiter); default on")
    parser.add_argument("--auto-level", choices=("on", "off"), default=None)
    parser.add_argument("--index-rate", type=float, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override any ShapeParams / RuntimeParams / GateConfig field, or "
                             "an input-conditioning one as cond_<field> (repeatable), e.g. "
                             "--set f0_unvoiced=interp")
    parser.add_argument("--tts-text", default=None,
                        help="accept: also render this text through Kokoro + the model (the "
                             "typed-path reference for voicing and HF/LF); typed: the text")
    parser.add_argument("--save-as", default=None,
                        help="accept: also save the live-path render as out/ab/NAME.wav")
    parser.add_argument("--variants", default=None,
                        help="ab: comma-separated subset of " + ", ".join(AB_VARIANTS))
    parser.add_argument("--prefix", default="",
                        help="ab: prefix for the out/ab/<prefix><variant>.wav files")
    parser.add_argument("--block-ms", type=float, default=20.0)
    parser.add_argument("--hops", default="60,20",
                        help="onset / underrun: comma-separated hop sizes (ms) to compare")
    parser.add_argument("--extras", default="1000,2200",
                        help="underrun: comma-separated context (extra) sizes in ms")
    parser.add_argument("--broadcast-load", action="store_true",
                        help="underrun: hold the NVIDIA Broadcast mic open meanwhile")
    parser.add_argument("--vowel", type=Path, default=None,
                        help="suppress: a sung recording; band loss is measured on its held "
                             "notes")
    parser.add_argument("--stage", choices=("f0", "embed", "synth", "sola", "hop"),
                        default="synth", help="graph-probe: which stage to capture")
    parser.add_argument("--sweep", type=Path,
                        default=ROOT / "benchmarks" / "sweeps" / "perf_grid.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("warning: no CUDA device; timings describe the CPU fallback", file=sys.stderr)
    handler = {"accept": cmd_accept, "ab": cmd_ab, "typed": cmd_typed, "perf": cmd_perf,
               "extra": cmd_extra, "sola": cmd_sola,
               "block": cmd_block, "decctx": cmd_decctx, "envelope": cmd_envelope,
               "gate": cmd_gate, "graph-probe": cmd_graph_probe, "sing": cmd_sing,
               "onset": cmd_onset, "suppress": cmd_suppress,
               "underrun": cmd_underrun}[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
