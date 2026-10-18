"""The ``vox say`` subcommand. Heavy imports stay inside ``run`` (``vox --help`` is instant)."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import sys
from pathlib import Path

from vox.tts.normalize import LAUGHTER_MODES
from vox.tts.variation import CONTOUR_MAX, CONTOUR_TRY, DEFAULT_AMOUNT

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
    p.add_argument("--formant", type=float, default=None, dest="formant_st",
                   help="formant shift in semitones, -4..+4 (pitch unchanged)")
    p.add_argument("--f0-method", choices=("rmvpe", "fcpe"), default=None)
    p.add_argument("--variation", type=float, default=DEFAULT_AMOUNT,
                   help="0..1: how differently the same text is delivered each time (speed, "
                        "pitch, pauses, TTS style, RVC seed); 0 = the same every time "
                        f"(default {DEFAULT_AMOUNT})")
    p.add_argument("--seed", type=int, default=None,
                   help="reproduce one variation draw and RVC seed (default: a fresh draw; "
                        "with --variation 0 the seed is 0, as before)")
    p.add_argument("--f0-contour-scale", type=float, default=1.0,
                   help="widen (>1) the intonation around the utterance median in semitones; "
                        f"1 = off, {CONTOUR_TRY} is worth a listen")
    p.add_argument("--no-normalize", action="store_true",
                   help="read slang, numbers, emoji and casing literally")
    p.add_argument("--laughter", choices=LAUGHTER_MODES, default="haha",
                   help="lol / lmao / laughing emoji: say \"haha\" (default) or drop it")
    p.add_argument("--dry-run", action="store_true",
                   help="print raw -> normalized -> expressive pieces and exit (no model)")
    p.add_argument("--plain", action="store_true",
                   help="the model's own loudness instead of the TTS contour (the pre-8.8 path)")
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


def spoken_text(text: str, profile_dir: Path | None, normalize: bool = True,
                laughter: str = "haha") -> str:
    """``text`` as it will be spoken: normalised with the shipped slang list and the
    profile's ``slang.json`` (a broken file is reported and skipped, never fatal)."""
    if not normalize:
        return text
    from vox.tts import normalize as norm

    user = profile_dir / norm.USER_FILE if profile_dir is not None else None
    try:
        table = norm.table_for(user)
    except norm.SlangError as err:
        print(f"warning: {err}; using the shipped slang list only", file=sys.stderr)
        table = norm.table_for(None)
    return norm.normalize(text, table, laughter)


def piece_hints(piece, last: bool) -> str:
    """'speed 1.05, gain +1.5 dB, pause 250 ms' for a piece, or 'plain'."""
    hints = []
    if piece.speed != 1.0:
        hints.append(f"speed {piece.speed:.2f}")
    if piece.gain_db:
        hints.append(f"gain {piece.gain_db:+.1f} dB")
    if piece.explicit and not last:
        hints.append(f"pause {piece.pause_after_s * 1000:.0f} ms")
    return ", ".join(hints) or "plain"


def say_polish(speak, args: argparse.Namespace):
    """``--plain`` and ``--f0-contour-scale`` as the ``TypedPolish`` to speak with."""
    base = speak.TypedPolish() if args.plain else speak.DEFAULT_POLISH
    return dataclasses.replace(base, contour_scale=args.f0_contour_scale)


def _variation_amount(args: argparse.Namespace) -> float:
    if not 0.0 <= args.variation <= 1.0:
        raise ValueError("--variation must be between 0 and 1")
    if not 0.5 <= args.f0_contour_scale <= CONTOUR_MAX:
        raise ValueError(f"--f0-contour-scale must be between 0.5 and {CONTOUR_MAX}")
    return args.variation


def dry_run(args: argparse.Namespace) -> int:
    from vox.tts import variation
    from vox.tts.speak import split_pieces

    text = _text(args)
    amount = _variation_amount(args)
    spoken = spoken_text(text, args.profile, not args.no_normalize, args.laughter)
    pieces = split_pieces(spoken)
    print(f"raw:        {text!r}")
    print(f"normalized: {spoken!r}")
    print(f"pieces:     {len(pieces)}")
    for n, piece in enumerate(pieces):
        print(f"  {n + 1}. {piece.text!r}  [{piece_hints(piece, n == len(pieces) - 1)}]")
    if amount == 0:
        print("variation off: the same text renders the same way (RVC seed "
              f"{args.seed if args.seed is not None else 0})")
    else:
        print(variation.bounds(amount))
        if args.seed is not None:
            print(variation.draw(amount, args.seed).describe() + " (style blend is drawn "
                  "against the voice list when rendering)")
    if args.f0_contour_scale != 1.0:
        print(f"f0 contour x{args.f0_contour_scale:g} around the utterance median")
    return 0


def run(args: argparse.Namespace) -> int:
    from vox.tts.backends import pick_backend
    from vox.tts.base import TtsError

    for stream in (sys.stdout, sys.stderr):  # typed text may hold emoji; cp1252 consoles
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="backslashreplace")
    try:
        if args.dry_run:
            return dry_run(args)
        backend = pick_backend(args.backend, args.voice, args.models_dir / TTS_DIR)
        if args.list_voices:
            print("\n".join(backend.voices()))
            return 0
        if args.model is None:
            raise ValueError("--model is required")
        text = _text(args)
        amount = _variation_amount(args)

        from vox.config.store import load_profile, resolve_runtime_params
        from vox.models.metadata import MetadataError
        from vox.models.registry import ModelRegistry
        from vox.tts import speak, variation

        try:
            profile = load_profile(args.profile)
            config = ModelRegistry(args.models_dir).config(args.model)
        except MetadataError as err:
            raise ValueError(str(err)) from err
        params = resolve_runtime_params(profile, config, {
            "index_rate": args.index_rate, "protect": args.protect,
            "f0_method": args.f0_method, "formant_st": args.formant_st})
        spoken = spoken_text(text, args.profile, not args.no_normalize, args.laughter)
        seed = args.seed if args.seed is not None or amount > 0 else 0
        voices = backend.voices() if amount > 0 and getattr(backend, "blends", False) else ()
        var = variation.draw(amount, seed, backend.voice, voices)
        out = args.out if args.out or args.play else DEFAULT_OUT
        result = speak.speak(spoken, backend, args.models_dir, args.model, params, args.pitch,
                             speak.target_f0_hint(profile, config), profile.runtime.precision(),
                             args.torch_device, var.rvc_seed, out,
                             polish=say_polish(speak, args), variation=var)
    except (TtsError, ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    if spoken != text:
        print(f"spoken as: {spoken!r}")
    print(f"tts: {backend.name} voice {backend.voice}, {result.tts_seconds:.2f} s in "
          f"{result.pieces} piece(s)")
    if result.expressive:
        for n, piece in enumerate(result.hints):
            hints = piece_hints(piece, n == len(result.hints) - 1)
            print(f"  piece {n + 1}: {piece.text!r}" + (f"  [{hints}]" if hints != "plain" else ""))
    print(var.describe())
    offset = f" {result.pitch_offset:+.2f} variation" if result.pitch_offset else ""
    if result.auto_pitch:
        print(f"auto pitch: {result.pitch:+.0f} semitones{offset} (TTS median F0 "
              f"{result.tts_median_f0:.1f} Hz -> target {result.target_f0:.1f} Hz)")
    else:
        print(f"pitch: {result.pitch:+g} semitones{offset}")
    if result.path is not None:
        print(f"wrote {result.path} ({result.seconds:.2f} s @ {result.sr} Hz)")
    if args.play:
        speak.play(result.audio, result.sr, args.device)
        print("played on " + (args.device or "the default output device"))
    return 0
