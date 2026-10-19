"""``vox`` command line: run | devices | convert | models | bench | env (calibrate comes later).

Heavy imports happen inside the handlers so ``vox --help`` stays instant.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = REPO_ROOT / "models"
DEFAULT_PROFILE = REPO_ROOT / "profiles" / "default"
LATER = {"calibrate": "Phase 5 (calibration)"}

RUN_HELP = """\
live commands (type and press Enter):
  pitch N | index X | protect X | envelope X   hot parameters, applied before the next hop
  formant ST                                   formant shift, -4..+4 st (first use of a value: brief pause)
  ceiling HZ                                   soft F0 ceiling after transposition (0 = off)
  autopitch on|off | target HZ                 auto pitch: follow the register onto target
  gate on|off                                  silence gate
  level auto|off | level DB                    mic auto level on/off, or the input trim (dB)
  suppress on|off | suppress MIX               RNNoise on/off, or its wet/dry mix (0-1)
  hop MS | extra MS | crossfade MS | f0 rmvpe|fcpe   restart-class: brief gated pause
                                               (hops snap to multiples of 20 ms)
  stats                                        print one stats line now
  q                                            stop"""


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="model name under the models directory")
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--device", default=None,
                   help="compute device (default: the RTX 5060 Ti, asserted at startup; "
                        "cpu only with VOX_ALLOW_CPU=1, see TARGET.md)")
    p.add_argument("--pitch", type=float, default=None, dest="f0_up_key",
                   help="transpose in semitones")
    p.add_argument("--index-rate", type=float, default=None)
    p.add_argument("--protect", type=float, default=None)
    p.add_argument("--volume-envelope", type=float, default=None)
    p.add_argument("--f0-method", choices=("rmvpe", "fcpe"), default=None)
    p.add_argument("--f0-unvoiced", choices=("zero", "interp"), default=None)
    p.add_argument("--sola-mode", choices=("plain", "pv"), default=None)
    p.add_argument("--pitch-mode", choices=("fixed", "auto"), default=None,
                   help="auto: measure the register live and transpose it onto "
                        "--target-f0-hz (the intonation is kept); fixed: --pitch (default)")
    p.add_argument("--target-f0-hz", type=float, default=None, dest="target_f0_hz",
                   help="auto pitch target, Hz (default: the model's register, else 200)")
    p.add_argument("--formant", type=float, default=None, dest="formant_st",
                   help="formant shift in semitones, -4..+4 in 0.5 steps (pitch unchanged): "
                        "negative = deeper / heavier, positive = lighter; best within +-2")
    p.add_argument("--pitch-range", type=float, default=None,
                   help="auto pitch: 1 keeps the intonation, lower squeezes it")
    p.add_argument("--pitch-follow-s", type=float, default=None,
                   help="auto pitch: how fast the register estimate follows (default 2 s)")
    p.add_argument("--f0-ceiling-hz", type=float, default=None,
                   help="soft-knee F0 ceiling after transposition, Hz (0 = off; singing "
                        "safety: very high notes are bent flat)")
    p.add_argument("--f0-trim-tail", type=int, choices=(1, 2), default=None,
                   help="RMVPE frames dropped at the right edge (2 = guard, the default; "
                        "1 = upstream)")
    p.add_argument("--sine-phase-carry", choices=("on", "off"), default=None,
                   help="NSF sine phase continuous across hops (default on)")
    p.add_argument("--noise-roll", choices=("on", "off"), default=None,
                   help="decoder noise moves with the window (default on)")
    p.add_argument("--decoder-context-ms", type=float, default=None,
                   help="decoder left context in ms (default 80; 0 = upstream)")
    p.add_argument("--block-ms", type=float, default=None,
                   help="hop in ms, snapped to a multiple of 20 (whole HuBERT frames)")
    p.add_argument("--crossfade-ms", type=float, default=None)
    p.add_argument("--extra-ms", type=float, default=None)
    p.add_argument("--onset-ramp-ms", type=float, default=None,
                   help="fade-in of a gate-onset hop (default 5; 0 = the whole overlap)")
    p.add_argument("--hubert-fp16", action="store_true", help="same as --hubert-autocast fp16")
    p.add_argument("--hubert-autocast", choices=("auto", "fp16", "bf16", "off"), default=None,
                   help="HuBERT autocast: auto (default; first of fp16, bf16 that matches "
                        "the FP32 eager render at warmup, decision stored in the profile)")
    p.add_argument("--graphs", choices=("on", "off", "auto"), default=None,
                   help="CUDA graphs: on (fail if capture fails), off (eager), auto "
                        "(capture, verify against eager, else eager; the default)")
    p.add_argument("--gate", choices=("on", "off"), default=None,
                   help="silence gate (energy + Silero VAD, hangover from the profile)")
    p.add_argument("--gate-threshold", type=float, default=None,
                   help="gate energy threshold in dBFS (default: calibration.json)")
    p.add_argument("--auto-level", choices=("on", "off"), default=None,
                   help="microphone auto level (AGC toward -20 dBFS speech; default on)")
    p.add_argument("--input-gain-db", type=float, default=None,
                   help="input trim in dB on top of auto level (default: calibration.json)")
    p.add_argument("--conditioning", choices=("on", "off"), default=None,
                   help="the whole input chain (high-pass, level, limiter); off = raw mic")
    p.add_argument("--suppression-mix", type=float, default=None,
                   help="RNNoise wet/dry mix 0-1 when suppression is on (default 1)")


def conditioning_config(profile, args: argparse.Namespace):
    """``ConditioningConfig`` from calibration.json (noise floor, input gain) and flags."""
    from vox.audio.conditioning import ConditioningConfig

    cal = profile.calibration
    trim = getattr(args, "input_gain_db", None)
    mix = getattr(args, "suppression_mix", None)
    return ConditioningConfig(enabled=getattr(args, "conditioning", None) != "off",
                              auto_level=getattr(args, "auto_level", None) != "off",
                              trim_db=cal.input_gain_db if trim is None else trim,
                              noise_floor_dbfs=cal.noise_floor_dbfs,
                              suppression_mix=1.0 if mix is None else mix)


def gate_config(profile, args: argparse.Namespace):
    """``GateConfig`` from calibration.json, with the CLI on/off switch and threshold."""
    from vox.engine.gate import GateConfig

    cal = profile.calibration
    threshold = getattr(args, "gate_threshold", None)
    return GateConfig(enabled=args.gate != "off",
                      threshold_dbfs=cal.silence_gate_dbfs if threshold is None else threshold,
                      vad_threshold=cal.vad.threshold, hangover_ms=cal.vad.release_ms,
                      use_vad=cal.vad.engine == "silero")


def resolve(args: argparse.Namespace):
    """Profile + per-model config + CLI flags -> (ShapeParams, RuntimeParams, Precision)."""
    import dataclasses

    from vox.config.store import load_profile, resolve_runtime_params
    from vox.engine import chunker
    from vox.models.registry import ModelRegistry

    profile = load_profile(args.profile)
    args.gate_config = gate_config(profile, args)
    args.conditioning_config = conditioning_config(profile, args)
    config = ModelRegistry(args.models_dir).config(args.model)
    overrides_params = dict(vars(args))
    for flag in ("sine_phase_carry", "noise_roll"):
        if overrides_params.get(flag) is not None:
            overrides_params[flag] = overrides_params[flag] == "on"
    params = resolve_runtime_params(profile, config, overrides_params)
    if getattr(args, "target_f0_hz", None) is None:
        from vox.tts.speak import target_f0_hint

        params = dataclasses.replace(params, target_f0_hz=target_f0_hint(profile, config))
    shape = profile.runtime.shape()
    overrides = {k: getattr(args, k, None)
                 for k in ("block_ms", "crossfade_ms", "extra_ms", "onset_ramp_ms",
                           "decoder_context_ms")
                 if getattr(args, k, None) is not None}
    if "block_ms" in overrides:
        snapped = chunker.snap_block_ms(overrides["block_ms"])
        if snapped != overrides["block_ms"]:
            print(f"hop {overrides['block_ms']:g} ms -> {snapped:g} ms (whole 20 ms HuBERT "
                  "frames)")
        overrides["block_ms"] = snapped
    shape = dataclasses.replace(shape, **overrides)
    precision = profile.runtime.precision()
    if args.hubert_fp16:
        precision = dataclasses.replace(precision, hubert_fp16=True)
    if getattr(args, "hubert_autocast", None):
        precision = dataclasses.replace(precision, hubert_autocast=args.hubert_autocast)
    if getattr(args, "graphs", None):
        precision = dataclasses.replace(precision, graphs=args.graphs)
    args.profile_obj = profile
    return shape, params, precision


def cmd_convert(args: argparse.Namespace) -> int:
    from vox.offline.convert import convert_file
    from vox.profiling.report import format_report

    shape, params, precision = resolve(args)
    summary = convert_file(args.input, args.output, args.models_dir, args.model, args.mode,
                           shape, params, precision, args.device, args.out_sr,
                           gate=args.gate_config, profile_dir=args.profile,
                           conditioning=args.conditioning_config)
    print(f"wrote {summary['path']} ({summary['seconds']:.2f} s @ {summary['sr']} Hz, "
          f"mode={args.mode})")
    result = summary.get("result")
    if result is not None:
        engine = summary["engine"]
        plan = engine.plan
        print(format_report(
            engine.timers, hop_ms=plan.block_ms, skip=min(50, result.hops // 2),
            notes={"overlap": f"{plan.overlap_ms:.0f} ms", "search": f"{plan.search_ms:.0f} ms",
                   "extra": f"{plan.extra_ms:.0f} ms", "f0": params.f0_method},
            alloc=result.alloc, sola_offsets_ms=result.sola_offsets_ms))
        print(f"RTF {result.rtf:.3f}   mode {engine.mode}   algorithmic latency "
              f"{plan.algorithmic_latency_ms:.0f} ms (hop + overlap + search + lookahead; device and "
              f"ring terms come with live audio)")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    import dataclasses

    from vox.models import discovery
    from vox.models.metadata import MetadataError, inspect_pth
    from vox.models.registry import ModelRegistry

    try:
        if args.action == "list":
            registry = ModelRegistry(args.models_dir)
            for name in registry.names():
                c = registry.config(name)
                index = f"index {c.index.ntotal}x{c.index.dim}" if c.index.file else "no index"
                print(f"{name:<28} {c.version} {c.sr} Hz f0={int(c.f0)} spk={c.n_spk} {index}")
                for warning in c.warnings:
                    print(f"    warning: {warning}")
        elif args.action == "inspect":
            meta = dataclasses.asdict(inspect_pth(args.path))
            meta.pop("config")
            print(json.dumps(meta, indent=2))
        elif args.action == "import":
            target = discovery.import_model(args.path, args.models_dir, args.name, args.index)
            print(f"imported into {target}")
    except (MetadataError, FileExistsError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    from vox.profiling.report import environment_fingerprint

    print(json.dumps(environment_fingerprint(), indent=2))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    import runpy

    sys.argv = ["vox bench", *args.rest]
    runpy.run_path(str(REPO_ROOT / "benchmarks" / "bench.py"), run_name="__main__")
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    from vox.audio.devices import format_table, list_devices

    print(format_table(list_devices(None if args.all else args.hostapi)))
    print("\n* = host API default (input / output). Use the idx with 'vox run --input/--output'.")
    return 0


def _run_command(session, line: str) -> bool:
    """One typed command. Returns False to stop."""
    parts = line.strip().split()
    if not parts:
        return True
    cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else None)
    try:
        if cmd in ("q", "quit", "exit"):
            return False
        if cmd == "stats":
            print(session.stats_line())
        elif cmd == "pitch":
            print("params:", session.set_params(f0_up_key=float(arg)))
        elif cmd == "index":
            print("params:", session.set_params(index_rate=float(arg)))
        elif cmd == "formant":
            from vox.synthesis.formant import snap_formant

            print("params:", session.set_params(formant_st=snap_formant(float(arg))))
        elif cmd == "protect":
            print("params:", session.set_params(protect=float(arg)))
        elif cmd == "envelope":
            print("params:", session.set_params(volume_envelope=float(arg)))
        elif cmd == "unvoiced":
            print("params:", session.set_params(f0_unvoiced=arg))
        elif cmd == "autopitch":
            print("params:", session.set_params(pitch_mode="auto" if arg == "on" else "fixed"))
        elif cmd == "target":
            print("params:", session.set_params(target_f0_hz=float(arg)))
        elif cmd == "ceiling":
            print("params:", session.set_params(f0_ceiling_hz=float(arg)))
        elif cmd == "gate":
            print("gate:", session.set_gate(enabled=arg == "on"))
        elif cmd == "level":
            if arg in ("auto", "on", "off"):
                print("input:", session.set_conditioning(auto_level=arg != "off"))
            else:
                print("input:", session.set_conditioning(trim_db=float(arg)))
        elif cmd == "suppress":
            if arg in ("on", "off"):
                session.suppression = arg == "on"
                print("input:", session.set_conditioning(suppression=arg == "on"))
            else:
                print("input:", session.set_conditioning(suppression_mix=float(arg)))
        elif cmd == "hop":
            from vox.engine.chunker import snap_block_ms

            print("shape:", session.set_shape(block_ms=snap_block_ms(float(arg))))
        elif cmd == "extra":
            print("shape:", session.set_shape(extra_ms=float(arg)))
        elif cmd == "crossfade":
            print("shape:", session.set_shape(crossfade_ms=float(arg)))
        elif cmd == "f0":
            print("shape:", session.set_shape(f0_method=arg))
        elif cmd in ("help", "?"):
            print(RUN_HELP)
        else:
            print(f"unknown command {cmd!r}; type 'help'")
    except (TypeError, ValueError) as err:
        print(f"bad argument: {err}")
    return True


def _save_tap(path: Path, session) -> None:
    """The output tap as WAV + mask, and the whole event log, for cut attribution."""
    import numpy as np

    from vox.audio.resample import save_audio

    tap = session.audio.tap
    sr = tap.sr
    save_audio(path, tap.audio[:tap.pos], sr)
    np.savez_compressed(path.with_suffix(".npz"), from_ring=tap.from_ring[:tap.pos], sr=sr)
    events = session.engine.timers.events
    path.with_suffix(".events.json").write_text(json.dumps(
        {"counts": events.counts, "events": events.since(0),
         "output_started_s": session.started_at - events.t0}, indent=1))
    print(f"recorded {tap.pos / sr:.1f} s of output to {path} (+ .npz mask, .events.json)")


def cmd_run(args: argparse.Namespace) -> int:
    from vox.api.control import LiveConfig, LiveSession
    from vox.profiling.report import write_jsonl

    shape, params, precision = resolve(args)
    cfg = LiveConfig(model=args.model, models_dir=args.models_dir, shape=shape, params=params,
                     precision=precision, gate=args.gate_config,
                     conditioning=args.conditioning_config, input=args.input,
                     output=args.output, exclusive=args.exclusive,
                     exclusive_input=args.exclusive_input, callback_ms=args.callback_ms,
                     fill_target_ms=args.fill_ms, fill_rule=args.fill_rule,
                     fill_margin_ms=args.fill_margin_ms, input_file=args.input_file,
                     loop_file=args.loop_file, device=args.device, profile_dir=args.profile,
                     suppression={"on": True, "off": False}.get(args.suppression),
                     suppression_profile=args.profile_obj.calibration.rnnoise_enabled,
                     record_output_s=(60.0 * (args.minutes or 10.0) + 30.0
                                      if args.record_output else 0.0))
    session = LiveSession(cfg)
    session.start()
    print(RUN_HELP)
    stop = threading.Event()

    def stdin_loop() -> None:
        try:
            for line in sys.stdin:
                if not _run_command(session, line):
                    break
        except (EOFError, ValueError):
            pass
        else:
            stop.set()
            return
        stop.set()

    if not args.no_stdin:
        threading.Thread(target=stdin_loop, name="vox-stdin", daemon=True).start()
    deadline = time.perf_counter() + args.minutes * 60.0 if args.minutes else None
    snaps = []
    try:
        while not stop.is_set():
            time.sleep(1.0)
            snap = session.snapshot()
            snaps.append(snap)
            print(session.stats_line(snap), flush=True)
            if args.stats_log:
                write_jsonl(args.stats_log, snap)
            if snap["errors"]:
                print("errors:", snap["errors"], file=sys.stderr)
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
    if args.record_output and session.audio is not None and session.audio.tap is not None:
        _save_tap(args.record_output, session)
    if snaps:
        last = snaps[-1]
        totals = [s["total"]["p99"] for s in snaps if s["total"]["p99"] == s["total"]["p99"]]
        print(f"\nsession {last['t']:.0f} s: {last['hops']} hops ({last['gated_hops']} gated), "
              f"underruns {last['underruns']} (status flags {last['status_output_underflow']}),"
              f" refills {last['underrun_refills']}, drift trims {last['drift_trims']}, input "
              f"overruns {last['input_overruns']}, rebuilds {last['rebuilds']}; hop p99 over "
              f"windows: max {max(totals):.1f} ms; latency estimate {last['latency']['total']:.0f}"
              f" ms")
    return 0


def cmd_later(args: argparse.Namespace) -> int:
    print(f"'vox {args.command}' is not implemented yet: {LATER[args.command]}.",
          file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vox", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="convert a WAV file (stream engine or full-file)")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--mode", choices=("stream", "full"), default="stream",
                   help="stream: the realtime engine, hop by hop; full: canonical offline")
    p.add_argument("--out-sr", type=int, default=None)
    _add_engine_args(p)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("models", help="list, inspect or import RVC models")
    p.add_argument("action", choices=("list", "inspect", "import"))
    p.add_argument("path", type=Path, nargs="?")
    p.add_argument("--name", default=None)
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    p.set_defaults(func=cmd_models)

    from vox.tts.cli import add_say_parser

    add_say_parser(sub, DEFAULT_MODELS, DEFAULT_PROFILE)

    from vox.api.launch import add_gui_parser

    add_gui_parser(sub, REPO_ROOT, DEFAULT_MODELS, DEFAULT_PROFILE)

    p = sub.add_parser("bench", help="run the benchmark harness (benchmarks/bench.py)")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("env", help="print the environment fingerprint")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("devices", help="list audio devices (WASAPI by default)")
    p.add_argument("--hostapi", default="Windows WASAPI")
    p.add_argument("--all", action="store_true", help="every host API")
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser("run", help="live: microphone -> engine -> output device")
    _add_engine_args(p)
    p.add_argument("--input", default=None, help="input device index or name fragment")
    p.add_argument("--output", default=None, help="output device index or name fragment")
    p.add_argument("--suppression", choices=("on", "off"), default=None,
                   help="the engine's noise suppression (default: the profile's setting, "
                        "but off when the input is the NVIDIA Broadcast virtual mic)")
    p.add_argument("--exclusive", action="store_true",
                   help="WASAPI exclusive mode on the output stream (~5 ms instead of 22)")
    p.add_argument("--exclusive-input", action="store_true",
                   help="WASAPI exclusive on the input too; watch the INPUT RATE warning")
    p.add_argument("--callback-ms", type=float, default=10.0, help="device block (transport)")
    p.add_argument("--fill-ms", type=float, default=None,
                   help="output ring fill target (default: --fill-rule)")
    p.add_argument("--fill-rule", choices=("legacy", "hop_plus"), default="legacy",
                   help="fill target rule: legacy = 2 x hop + callback; hop_plus = hop + "
                        "--fill-margin-ms (docs/DEVIATIONS.md 13)")
    p.add_argument("--fill-margin-ms", type=float, default=20.0)
    p.add_argument("--minutes", type=float, default=None, help="stop after this long")
    p.add_argument("--input-file", type=Path, default=None,
                   help="software microphone: feed this WAV at real-time pace instead")
    p.add_argument("--loop-file", action="store_true")
    p.add_argument("--stats-log", type=Path, default=None, help="append 1 Hz snapshots (jsonl)")
    p.add_argument("--record-output", type=Path, default=None,
                   help="save what the output device was given (WAV) with an underrun mask "
                        "(.npz) and the event log (.events.json) beside it")
    p.add_argument("--no-stdin", action="store_true", help="do not read commands from stdin")
    p.set_defaults(func=cmd_run)

    for name, phase in LATER.items():
        p = sub.add_parser(name, help=f"(not yet: {phase})")
        p.set_defaults(func=cmd_later)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "models" and args.action != "list" and args.path is None:
        print("error: a path is required", file=sys.stderr)
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
