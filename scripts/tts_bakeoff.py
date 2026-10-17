"""TTS backend bake-off: the same eight casual lines from every engine, through RVC.

    .venv\\Scripts\\python.exe scripts\\tts_bakeoff.py --model Mom --input-file takes\\mic\\x.wav

Phase A, rendering, happens while a live session runs (the mic take looped through the
engine into ``--output``) so the live underruns each engine causes are counted. Every line
is normalised first (``vox.tts.normalize``, the layer that sits in front of any backend);
Kokoro takes its production path (expressive pieces, variation 0.35, seeded). The external
engines render each line twice: ``plain`` (normalised text, default controls) and
``expressive`` (the engine's own controls set by hand per line: Chatterbox exaggeration,
[laugh] / <laugh> tags, a Qwen instruction). Phase B converts every raw render with the
model (auto pitch, the shipped typed polish) once the live session is stopped.

Output: ``out/tts_bakeoff/<backend>/<variant>/NN_slug.wav`` (converted), ``.../raw/`` (the
TTS itself), ``results.json`` and ``results.md`` (render times, VRAM, underruns).
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (slug, raw text, mood)
LINES = [
    ("01_idk_wbu", "idk wbu", "neutral"),
    ("02_lol_ok_brb", "lol ok brb", "playful"),
    ("03_omg_so_good", "omg that was SO good!!", "excited"),
    ("04_question", "wait are you actually coming to the party tonight?", "question"),
    ("05_sad", "i really miss her... it just doesnt feel the same anymore", "sad"),
    ("06_excited", "WE WON!! i cant believe we actually won!!", "excited"),
    ("07_long", "ngl i thought the movie was gonna be mid but it was SO good lol, we got there "
                "at 7:30 and it cost like $15 each... wbu, u still going tmrw?", "playful"),
    ("08_hi", "hi", "neutral"),
]

QWEN_INSTRUCT = {
    "neutral": "Casual and relaxed, like talking to a friend.",
    "playful": "Playful and amused, with a light laugh in the voice.",
    "excited": "Very excited and happy, full of energy.",
    "question": "Curious and casual, asking a real question.",
    "sad": "Sad and quiet, a little choked up.",
}
CHATTERBOX = {  # (exaggeration, cfg_weight); Resemble: more exaggeration + lower cfg = livelier
    "neutral": (0.5, 0.5), "playful": (0.7, 0.4), "excited": (0.9, 0.3),
    "question": (0.55, 0.5), "sad": (0.4, 0.5),
}


def expressive(backend: str, text: str, mood: str) -> tuple[str, dict]:
    """The hand-set expressive render of one normalised line for ``backend``."""
    if backend == "chatterbox":
        ex, cfg = CHATTERBOX[mood]
        return text, {"exaggeration": ex, "cfg_weight": cfg}
    if backend in ("chatterbox-turbo", "orpheus"):
        laugh, sigh = ("[laugh]", "[sigh]") if backend == "chatterbox-turbo" else ("<laugh>",
                                                                                  "<sigh>")
        text = text.replace("Haha,", laugh).replace("haha,", laugh).replace("haha", laugh)
        if mood == "sad":
            text = f"{sigh} {text}"
        return text, {}
    if backend.startswith("qwen3"):
        return text, {"instruct": QWEN_INSTRUCT[mood]}
    return text, {}


EXPRESSIVE = {"chatterbox", "chatterbox-turbo", "orpheus", "qwen3", "qwen3-design"}


def gpu_used_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class Live:
    """A live session (file mic -> engine -> output device) whose underruns we count."""

    def __init__(self, args) -> None:
        from vox.api.controller import EngineController

        self.work = Path(tempfile.mkdtemp(prefix="vox_bakeoff_"))
        profile = self.work / "profile"
        shutil.copytree(ROOT / "profiles" / "default", profile)
        (profile / "gui_state.json").unlink(missing_ok=True)
        self.c = EngineController(args.models_dir, profile, model=args.model,
                                  output=int(args.output), input_file=args.input_file,
                                  loop_file=True, log=lambda m: None)
        self.c.start()
        end = time.time() + 180
        while self.c.state()["engine"]["status"] != "running":
            if time.time() > end or self.c.state()["engine"]["status"] == "error":
                raise RuntimeError(f"live session did not start: {self.c.state()['engine']}")
            time.sleep(0.2)
        time.sleep(2.0)

    @property
    def underruns(self) -> int:
        return self.c.session.out_ring.underruns

    def stop(self) -> None:
        self.c.stop()
        end = time.time() + 60
        while self.c.state()["engine"]["status"] != "stopped" and time.time() < end:
            time.sleep(0.2)
        self.c.shutdown()
        shutil.rmtree(self.work, ignore_errors=True)


def render_kokoro(lines, raw_dir: Path, seed: int) -> dict:
    """Kokoro's production path: pieces with render hints and one variation draw each."""
    from vox.audio.resample import save_audio
    from vox.offline.convert import SR
    from vox.tts import speak, variation
    from vox.tts.backends import KokoroBackend

    backend = KokoroBackend(None, ROOT / "models" / "_tts")
    start = time.perf_counter()
    backend.load()
    load_s = time.perf_counter() - start
    times, pieces_by = [], {}
    for n, (slug, text, _) in enumerate(lines):
        pieces = speak.split_pieces(text)
        var = variation.draw(0.35, seed + n, backend.voice, backend.voices())
        start = time.perf_counter()
        rendered = speak.render_tts(backend, pieces, var)
        times.append(time.perf_counter() - start)
        joined = speak.crossfade_concat(rendered, SR) if len(rendered) > 1 else rendered[0]
        save_audio(raw_dir / f"{slug}.wav", joined, SR)
        pieces_by[slug] = {"pieces": pieces, "rendered": rendered, "variation": var}
    return {"load_s": load_s, "times": times, "pieces": pieces_by, "voice": backend.voice}


def render_external(cls, lines, out: Path, live: Live | None) -> dict:
    backend = cls(None, ROOT / "models" / "_tts")
    if reason := backend.unavailable_reason():
        return {"error": reason}
    before = gpu_used_mb()
    under0 = live.underruns if live else 0
    start = time.perf_counter()
    info = backend.worker().start()
    load_s = time.perf_counter() - start
    rows = []
    variants = ("plain", "expressive") if cls.name in EXPRESSIVE else ("plain",)
    for variant in variants:
        (out / "raw" / variant).mkdir(parents=True, exist_ok=True)
        for slug, text, mood in lines:
            said, controls = (text, {}) if variant == "plain" else expressive(cls.name, text, mood)
            dst = out / "raw" / variant / f"{slug}.wav"
            u = live.underruns if live else 0
            start = time.perf_counter()
            backend.synthesize(said, dst, controls=controls)
            wall = time.perf_counter() - start
            r = backend.last_render
            rows.append({"variant": variant, "slug": slug, "text": said, "controls": controls,
                         "render_s": round(wall, 3), "audio_s": r["seconds"],
                         "rtf": round(wall / max(r["seconds"], 1e-3), 3),
                         "vram_peak_mb": r.get("vram_peak_mb"),
                         "underruns": (live.underruns - u) if live else None})
            print(f"  {cls.name:16s} {variant:10s} {slug:16s} {wall:6.2f} s for "
                  f"{r['seconds']:5.2f} s audio", flush=True)
    during = gpu_used_mb()
    backend.worker().stop()
    first = rows[0]["render_s"]
    warm = [r["render_s"] for r in rows[1:]]
    short = [r["render_s"] for r in rows[1:] if r["slug"] in ("01_idk_wbu", "08_hi",
                                                                "02_lol_ok_brb")]
    return {"voice": backend.voice, "load_s": round(load_s, 2), "first_render_s": first,
            "warm_median_s": round(statistics.median(warm), 3),
            "warm_short_median_s": round(statistics.median(short), 3) if short else None,
            "rtf_median": round(statistics.median(r["rtf"] for r in rows[1:]), 3),
            "vram_torch_peak_mb": max([r["vram_peak_mb"] or 0 for r in rows]
                                      + [info.get("vram_peak_mb") or 0]),
            "gpu_used_delta_mb": (during - before) if before is not None and during else None,
            "underruns": (live.underruns - under0) if live else None, "rows": rows}


def convert_all(results: dict, out_root: Path, args) -> None:
    """Phase B: every raw render through RVC (auto pitch, the shipped typed polish)."""
    from vox.audio.resample import load_audio, save_audio
    from vox.config.store import load_profile, resolve_runtime_params
    from vox.models.registry import ModelRegistry
    from vox.offline.convert import SR, FullFileConverter
    from vox.offline.simulate import load_engine
    from vox.tts import speak

    profile = load_profile(ROOT / "profiles" / "default")
    config = ModelRegistry(args.models_dir).config(args.model)
    params = resolve_runtime_params(profile, config, {})
    target = speak.target_f0_hint(profile, config)
    engine = load_engine(args.models_dir, args.model, None, params, profile.runtime.precision(),
                         None, warmup_hops=0)
    converter = FullFileConverter.from_engine(engine)
    sr = engine.loaded.model.tgt_sr
    for name, res in results.items():
        if "error" in res:
            continue
        base = out_root / name
        if name == "kokoro":
            for slug, item in res["pieces"].items():
                result = speak.convert_rendered(
                    converter, engine.device, sr, item["rendered"], engine.params, None, target,
                    item["variation"].rvc_seed, item["pieces"], speak.DEFAULT_POLISH,
                    pitch_offset=item["variation"].pitch_offset)
                save_audio(base / f"{slug}.wav", result.audio, result.sr)
                res.setdefault("pitch", {})[slug] = result.pitch
            continue
        for row in res["rows"]:
            audio, _ = load_audio(base / "raw" / row["variant"] / f"{row['slug']}.wav", SR)
            result = speak.convert_rendered(converter, engine.device, sr, [audio],
                                            engine.params, None, target, 0, None,
                                            speak.DEFAULT_POLISH)
            dst = base / row["variant"] / f"{row['slug']}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            save_audio(dst, result.audio, result.sr)
            row["auto_pitch"] = result.pitch
            row["tts_median_f0"] = round(result.tts_median_f0, 1)


def markdown(results: dict, args) -> str:
    lines = [f"# TTS bake-off ({time.strftime('%Y-%m-%d %H:%M')}, model {args.model})", "",
             "| backend | voice | load s | first render s | warm median s | warm short "
             "line s | RTF | VRAM torch peak MB | GPU used +MB | live underruns |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        if "error" in r:
            lines.append(f"| {name} | not run: {r['error']} |||||||||")
            continue
        lines.append(f"| {name} | {r['voice']} | {r['load_s']} | {r['first_render_s']} | "
                     f"{r['warm_median_s']} | {r.get('warm_short_median_s')} | "
                     f"{r.get('rtf_median')} | {r.get('vram_torch_peak_mb')} | "
                     f"{r.get('gpu_used_delta_mb')} | {r.get('underruns')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Mom")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--input-file", type=Path, default=None,
                    help="mic take to loop live while rendering (no live session without it)")
    ap.add_argument("--output", default="23", help="output device index for the live session")
    ap.add_argument("--backends", default="kokoro,chatterbox,chatterbox-turbo,orpheus,qwen3,"
                                          "qwen3-design,f5")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "tts_bakeoff")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from vox.tts import external
    from vox.tts.normalize import normalize, table_for

    table = table_for(None)
    lines = [(slug, normalize(raw, table), mood) for slug, raw, mood in LINES]
    for (slug, raw, _), (_, spoken, _) in zip(LINES, lines):
        print(f"{slug}: {raw!r} -> {spoken!r}")
    classes = {"chatterbox": external.ChatterboxBackend,
               "chatterbox-turbo": external.ChatterboxTurboBackend,
               "orpheus": external.OrpheusBackend, "qwen3": external.Qwen3Backend,
               "qwen3-design": external.Qwen3DesignBackend, "f5": external.F5Backend}
    wanted = [b.strip() for b in args.backends.split(",") if b.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    live = Live(args) if args.input_file else None
    try:
        for name in wanted:
            print(f"== {name}", flush=True)
            out = args.out / name
            if name == "kokoro":
                (out / "raw").mkdir(parents=True, exist_ok=True)
                under = live.underruns if live else 0
                k = render_kokoro(lines, out / "raw", args.seed)
                results[name] = {"voice": k["voice"], "load_s": round(k["load_s"], 2),
                                 "first_render_s": round(k["times"][0], 3),
                                 "warm_median_s": round(statistics.median(k["times"][1:]), 3),
                                 "warm_short_median_s": round(statistics.median(
                                     [k["times"][i] for i in (1, 7)]), 3),
                                 "vram_torch_peak_mb": 0, "gpu_used_delta_mb": 0,
                                 "underruns": (live.underruns - under) if live else None,
                                 "pieces": k["pieces"]}
            else:
                try:
                    results[name] = render_external(classes[name], lines, out, live)
                except Exception as err:  # noqa: BLE001 - one engine failing is a result
                    results[name] = {"error": f"{type(err).__name__}: {err}"}
                    external.ExternalBackend.shutdown_all()
            print(f"   {name}: { {k: v for k, v in results[name].items() if k not in ('rows', 'pieces')} }",
                  flush=True)
    finally:
        if live is not None:
            live.stop()
    convert_all(results, args.out, args)
    for r in results.values():
        r.pop("pieces", None)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (args.out / "results.md").write_text(markdown(results, args))
    print(markdown(results, args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
