"""Benchmark harness (docs/research/10). Every number comes from the real engine, driven by
the stream simulator; there is no separate batch path.

    vox bench accept                  Phase 2 acceptance: streamed vs full-file, clicks,
                                      sync-free hop, allocation stability
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
    audio = audio[: int(args.seconds * 48000)]
    return audio, ModelRegistry(args.models_dir), {}


def _engine(args, registry, cache, shape_kw=None, **param_kw):
    from vox.config import defaults
    from vox.config.schemas import PrecisionParams, RuntimeParams
    from vox.offline.simulate import load_engine

    shape = dataclasses.replace(defaults.DEFAULT_SHAPE, **(shape_kw or {}))
    params = RuntimeParams(f0_up_key=args.pitch, **param_kw)
    precision = PrecisionParams(hubert_fp16=args.hubert_fp16, graphs=args.graphs,
                                hubert_autocast=args.hubert_autocast)
    return load_engine(args.models_dir, args.model, shape, params, precision,
                       registry=registry, cache=cache, profile_dir=args.profile)


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

    out = ROOT / "out"
    save_audio(out / "accept_full.wav", full_a, sr)
    save_audio(out / "accept_stream.wav", result.aligned[:len(audio)], sr)

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
    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    met = "met" if ratio <= 1.5 else "NOT met"
    print(f"  [info] strict equivalence target (<= 1.5x floor): {met} at {ratio:.2f}x; "
          "see docs/research/14 Q14")
    _record("accept", {"model": args.model, "plan": dataclasses.asdict(plan), "rtf": result.rtf,
                       "mode": engine.mode, "graphs": dataclasses.asdict(engine.graphs.report),
                       "noise_floor": floor, "stream_floor": stream_floor,
                       "quality": quality,
                       "full_click_ratio": full_click, "alloc": alloc,
                       "stage_p50_ms": _stage_p50(engine, SKIP),
                       "checks": {k: bool(v) for k, v in checks.items()}})
    return 0 if all(checks.values()) else 1


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
    parser.add_argument("command", choices=("accept", "perf", "extra", "sola", "block", "decctx",
                                            "envelope", "gate", "graph-probe"))
    parser.add_argument("--model", default="base_f0_48k_v2")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--audio", type=Path,
                        default=ROOT / "benchmarks" / "corpus" / "libri_1272.wav")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--hubert-fp16", action="store_true")
    parser.add_argument("--hubert-autocast", choices=("auto", "fp16", "bf16", "off"),
                        default="auto", help="HuBERT autocast ladder (TARGET.md rule 2)")
    parser.add_argument("--profile", type=Path, default=ROOT / "profiles" / "default",
                        help="profile directory whose runtime.json stores warmup decisions")
    parser.add_argument("--graphs", choices=("on", "off", "auto"), default="auto",
                        help="CUDA graphs mode for the engine under test")
    parser.add_argument("--gate", choices=("on", "off"), default="off",
                        help="quality sweeps (extra, block, ...): run the silence gate too")
    parser.add_argument("--block-ms", type=float, default=20.0)
    parser.add_argument("--stage", choices=("f0", "embed", "synth", "sola", "hop"),
                        default="synth", help="graph-probe: which stage to capture")
    parser.add_argument("--sweep", type=Path,
                        default=ROOT / "benchmarks" / "sweeps" / "perf_grid.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("warning: no CUDA device; timings describe the CPU fallback", file=sys.stderr)
    handler = {"accept": cmd_accept, "perf": cmd_perf, "extra": cmd_extra, "sola": cmd_sola,
               "block": cmd_block, "decctx": cmd_decctx, "envelope": cmd_envelope,
               "gate": cmd_gate, "graph-probe": cmd_graph_probe}[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
