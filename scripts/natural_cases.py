"""Natural typed speech acceptance: the dry-run table, the prep latency, and (unless
``--dry-run``) the wavs to listen to.

    .venv\\Scripts\\python scripts\\natural_cases.py --model MaeASMR
    .venv\\Scripts\\python scripts\\natural_cases.py --dry-run

Writes ``out/tts_natural/<case>.wav`` through the same steps ``vox say`` takes (normalise ->
expressive pieces -> one variation draw -> kokoro -> RVC), loading the model once. Prints a
markdown table of raw -> normalised -> pieces, the per-utterance prep time (normalise +
draw), and for the wavs their length, auto pitch, variation offsets and whether the "same"
renders are bit-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# (file stem, text, variation)
CASES = [
    ("idk_wbu", "idk wbu", None),
    ("tired_rn_ngl", "im so tired rn ngl", None),
    ("lol_ok_brb", "lol ok brb", None),
    ("omg_so_good", "omg that was SO good!!", None),
    ("hi_var_1", "hi", None),
    ("hi_var_2", "hi", None),
    ("hi_var_3", "hi", None),
    ("hi_fixed_1", "hi", 0.0),
    ("hi_fixed_2", "hi", 0.0),
    ("long_casual", ("ngl i thought the movie was gonna be mid but it was SO good lol, "
                    "we got there at 7:30 and it cost like $15 each... wbu, u still going tmrw?"),
     None),
]


def table(rows: list[tuple[str, str, list]]) -> str:
    from vox.tts.cli import piece_hints

    lines = ["| raw | normalized | pieces (hints) |", "|---|---|---|"]
    for raw, spoken, pieces in rows:
        cells = "<br>".join(f"`{p.text}` {piece_hints(p, n == len(pieces) - 1)}"
                            for n, p in enumerate(pieces))
        lines.append(f"| `{raw}` | `{spoken}` | {cells} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MaeASMR")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "tts_natural")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--variation", type=float, default=0.35)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from vox.tts import normalize, variation
    from vox.tts.speak import split_pieces

    table_ = normalize.table_for(None)
    normalize.normalize("warm up 3pm $5 lol", table_)
    rows, seen = [], set()
    for _, text, _ in CASES:
        if text in seen:
            continue
        seen.add(text)
        spoken = normalize.normalize(text, table_)
        rows.append((text, spoken, split_pieces(spoken)))
    print(table(rows))

    voices = ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
              "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"]
    times = []
    for _ in range(50):
        for _, text, _ in CASES:
            start = time.perf_counter()
            split_pieces(normalize.normalize(text, table_))
            variation.draw(args.variation, None, "af_heart", voices)
            times.append((time.perf_counter() - start) * 1000)
    times.sort()
    print(f"\nprep (normalize + split + draw) per utterance over {len(times)} runs: "
          f"median {times[len(times) // 2]:.3f} ms, p99 {times[int(len(times) * 0.99)]:.3f} ms, "
          f"max {times[-1]:.3f} ms")
    if args.dry_run:
        return 0

    from vox.audio.resample import save_audio
    from vox.offline.convert import FullFileConverter
    from vox.offline.simulate import load_engine
    from vox.tts import speak
    from vox.tts.backends import pick_backend
    from vox.tts.cli import TTS_DIR

    backend = pick_backend("kokoro", args.voice, args.models_dir / TTS_DIR)
    backend.load()
    engine = load_engine(args.models_dir, args.model, None, None, None, None, warmup_hops=0)
    converter = FullFileConverter.from_engine(engine)
    args.out.mkdir(parents=True, exist_ok=True)
    print("\n| file | text | variation draw | TTS median F0 | auto pitch | offset | length s |\n"
          "|---|---|---|---|---|---|---|")
    digests = {}
    for stem, text, amount in CASES:
        amount = args.variation if amount is None else amount
        spoken = normalize.normalize(text, table_)
        pieces = split_pieces(spoken)
        var = variation.draw(amount, 0 if amount == 0 else None, backend.voice,
                             backend.voices())
        rendered = speak.render_tts(backend, pieces, var)
        result = speak.convert_rendered(converter, engine.device, engine.loaded.model.tgt_sr,
                                        rendered, engine.params, None, 200.0, var.rvc_seed,
                                        pieces, pitch_offset=var.pitch_offset)
        path = args.out / f"{stem}.wav"
        save_audio(path, result.audio, result.sr)
        digests[stem] = hashlib.sha256(result.audio.tobytes()).hexdigest()[:12]
        mix = ", ".join(f"{w * 100:.1f}% {v}" for v, w in var.blend) or "no blend"
        draw = ("off" if var.neutral
                else f"speed x{var.speed:.3f}, pauses x{var.pause_scale:.2f}, {mix}")
        print(f"| {path.name} | `{text}` | {draw} | {result.tts_median_f0:.1f} Hz | "
              f"{result.pitch:+.0f} | {result.pitch_offset:+.3f} | {result.seconds:.2f} |")
    print("\nsha256 of the converted audio:", digests)
    print("hi_fixed_1 == hi_fixed_2:", digests["hi_fixed_1"] == digests["hi_fixed_2"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
