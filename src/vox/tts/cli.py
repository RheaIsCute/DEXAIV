"""The ``vox say`` subcommand. Heavy imports stay inside ``run`` (``vox --help`` is instant)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TTS_DIR = "_tts"  # under the models directory; holds downloaded TTS weights
DEFAULT_OUT = Path("out") / "say.wav"

DESCRIPTION = """\
Speak typed text in the voice of an imported RVC model: a TTS engine renders the text,
then the full-file converter turns it into the model's voice.

Pitch: without --pitch, the transposition is measured from the rendered speech
(median F0 of the TTS voice vs. the model's target, 200 Hz unless the profile sets
target_f0_median_hz) and printed. Tip: a TTS voice already close to the target
(a female TTS voice for a female model, pitch near 0) usually sounds best."""


def add_say_parser(sub, default_models: Path, default_profile: Path) -> None:
    p = sub.add_parser("say", help="speak text in a model's voice (TTS -> RVC)",
                       description=DESCRIPTION,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="?", help="the text to speak (or use --file)")
    p.add_argument("--file", type=Path, default=None, help="read the text from a UTF-8 file")
    p.add_argument("--model", default=None, help="model name under the models directory")
    p.add_argument("--models-dir", type=Path, default=default_models)
    p.add_argument("--profile", type=Path, default=default_profile)
    p.add_argument("--backend", choices=("kokoro", "edge"), default=None,
                   help="TTS engine: kokoro (local, default) or edge (online); "
                        "default: kokoro, else edge when kokoro is not installed")
    p.add_argument("--voice", default=None, help="TTS voice name (see --list-voices)")
    p.add_argument("--list-voices", action="store_true",
                   help="print the backend's voice names and exit")
    p.add_argument("--pitch", type=float, default=None,
                   help="transpose in semitones (default: measured from the TTS audio)")
    p.add_argument("--index-rate", type=float, default=None)
    p.add_argument("--protect", type=float, default=None)
    p.add_argument("--f0-method", choices=("rmvpe", "fcpe"), default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None,
                   help=f"output WAV (default: {DEFAULT_OUT} unless --play is given)")
    p.add_argument("--play", action="store_true", help="play the result")
    p.add_argument("--device", default=None,
                   help="output audio device for --play: index or name (default: system default)")
    p.add_argument("--torch-device", default=None,
                   help="compute device (default: the RTX 5060 Ti, asserted; cpu only with "
                        "VOX_ALLOW_CPU=1, see TARGET.md)")
    p.set_defaults(func=run)


def _text(args: argparse.Namespace) -> str:
    if (args.text is None) == (args.file is None):
        raise ValueError("give the text either as an argument or with --file")
    return args.text if args.file is None else args.file.read_text(encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    from vox.tts.backends import pick_backend
    from vox.tts.base import TtsError

    try:
        backend = pick_backend(args.backend, args.voice, args.models_dir / TTS_DIR)
        if args.list_voices:
            print("\n".join(backend.voices()))
            return 0
        if args.model is None:
            raise ValueError("--model is required")
        text = _text(args)

        from vox.config.store import load_profile, resolve_runtime_params
        from vox.models.metadata import MetadataError
        from vox.models.registry import ModelRegistry
        from vox.tts import speak

        try:
            profile = load_profile(args.profile)
            config = ModelRegistry(args.models_dir).config(args.model)
        except MetadataError as err:
            raise ValueError(str(err)) from err
        params = resolve_runtime_params(profile, config, {
            "index_rate": args.index_rate, "protect": args.protect,
            "f0_method": args.f0_method})
        out = args.out if args.out or args.play else DEFAULT_OUT
        result = speak.speak(text, backend, args.models_dir, args.model, params, args.pitch,
                             speak.target_f0_hint(profile, config), profile.runtime.precision(),
                             args.torch_device, args.seed, out)
    except (TtsError, ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print(f"tts: {backend.name} voice {backend.voice}, {result.tts_seconds:.2f} s in "
          f"{result.pieces} piece(s)")
    if result.expressive:
        for n, piece in enumerate(result.hints):
            hints = []
            if piece.speed != 1.0:
                hints.append(f"speed {piece.speed:.2f}")
            if piece.gain_db:
                hints.append(f"gain {piece.gain_db:+.1f} dB")
            if piece.explicit and n < len(result.hints) - 1:
                hints.append(f"pause {piece.pause_after_s * 1000:.0f} ms")
            print(f"  piece {n + 1}: {piece.text!r}" + (f"  [{', '.join(hints)}]" if hints else ""))
    if result.auto_pitch:
        print(f"auto pitch: {result.pitch:+.0f} semitones (TTS median F0 "
              f"{result.tts_median_f0:.1f} Hz -> target {result.target_f0:.1f} Hz)")
    else:
        print(f"pitch: {result.pitch:+g} semitones")
    if result.path is not None:
        print(f"wrote {result.path} ({result.seconds:.2f} s @ {result.sr} Hz)")
    if args.play:
        speak.play(result.audio, result.sr, args.device)
        print("played on " + (args.device or "the default output device"))
    return 0
