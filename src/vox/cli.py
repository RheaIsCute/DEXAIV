"""``vox`` command line: convert | models | bench | env (run / calibrate / devices come later).

Heavy imports happen inside the handlers so ``vox --help`` stays instant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = REPO_ROOT / "models"
DEFAULT_PROFILE = REPO_ROOT / "profiles" / "default"
LATER = {"run": "Phase 3 (live audio)", "devices": "Phase 3 (live audio)",
         "calibrate": "Phase 5 (calibration)"}


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="model name under the models directory")
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument("--pitch", type=float, default=None, dest="f0_up_key",
                   help="transpose in semitones")
    p.add_argument("--index-rate", type=float, default=None)
    p.add_argument("--protect", type=float, default=None)
    p.add_argument("--volume-envelope", type=float, default=None)
    p.add_argument("--f0-method", choices=("rmvpe", "fcpe"), default=None)
    p.add_argument("--f0-unvoiced", choices=("zero", "interp"), default=None)
    p.add_argument("--sola-mode", choices=("plain", "pv"), default=None)
    p.add_argument("--block-ms", type=float, default=None)
    p.add_argument("--crossfade-ms", type=float, default=None)
    p.add_argument("--extra-ms", type=float, default=None)
    p.add_argument("--hubert-fp16", action="store_true")


def resolve(args: argparse.Namespace):
    """Profile + per-model config + CLI flags -> (ShapeParams, RuntimeParams, Precision)."""
    import dataclasses

    from vox.config.store import load_profile, resolve_runtime_params
    from vox.models.registry import ModelRegistry

    profile = load_profile(args.profile)
    config = ModelRegistry(args.models_dir).config(args.model)
    params = resolve_runtime_params(profile, config, vars(args))
    shape = profile.runtime.shape()
    overrides = {k: getattr(args, k) for k in ("block_ms", "crossfade_ms", "extra_ms")
                 if getattr(args, k) is not None}
    shape = dataclasses.replace(shape, **overrides)
    precision = profile.runtime.precision()
    if args.hubert_fp16:
        precision = dataclasses.replace(precision, hubert_fp16=True)
    return shape, params, precision


def cmd_convert(args: argparse.Namespace) -> int:
    from vox.offline.convert import convert_file
    from vox.profiling.report import format_report

    shape, params, precision = resolve(args)
    summary = convert_file(args.input, args.output, args.models_dir, args.model, args.mode,
                           shape, params, precision, args.device, args.out_sr)
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
        print(f"RTF {result.rtf:.3f}   algorithmic latency {plan.algorithmic_latency_ms:.0f} ms"
              f" (hop + overlap + search; device and ring terms come with live audio)")
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

    p = sub.add_parser("bench", help="run the benchmark harness (benchmarks/bench.py)")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("env", help="print the environment fingerprint")
    p.set_defaults(func=cmd_env)

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
