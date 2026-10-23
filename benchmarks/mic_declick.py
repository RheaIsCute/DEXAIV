r"""De-clicker plosive check (docs/DEVIATIONS.md §15.1).

    .venv\Scripts\python.exe benchmarks\mic_declick.py [TAKE] [OUT_DIR]

Plosive check for the de-clicker on a clean take, input side only (no model): the
user's input chain with declick off / on (look-back applied as the engine does), every
clamp with the word Whisper puts there, and Whisper WER of both conditioned mics."""
import dataclasses, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "benchmarks"))
import paraling as pl
from vox.audio.conditioning import ConditioningConfig, InputConditioner
from vox.audio.resample import load_audio, resample_array, save_audio
from quality import load_script, wer

take = sys.argv[1] if len(sys.argv) > 1 else str(
    ROOT / "recordings/matrix_20261017_192237/shure_gateoff_agcoff_hop20_mic_raw.wav")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "out/ab/paraling" / Path(take).stem
out.mkdir(parents=True, exist_ok=True)
SR, BLOCK = 48000, 3840
audio, _ = load_audio(take, SR)


def condition(declick):
    c = InputConditioner(SR, BLOCK, ConditioningConfig(**pl.USER_COND, declick=declick))
    n = -(-len(audio) // BLOCK)
    x = np.zeros(n * BLOCK, np.float32); x[:len(audio)] = audio
    done = []
    for i in range(n):
        y = c.process(x[i * BLOCK:(i + 1) * BLOCK]).copy()
        if c.retro is not None:
            prev = np.concatenate(done) if done else np.zeros(0, np.float32)
            k = min(len(c.retro), len(prev))
            prev[len(prev) - k:] *= c.retro[len(c.retro) - k:]
            done = [prev]
        done.append(y)
    return np.concatenate(done)[:len(audio)], (c.declicker.clicks if c.declicker else [])


from faster_whisper import WhisperModel
model = WhisperModel("small.en", device="cpu", compute_type="int8")
script = load_script(ROOT / "benchmarks/corpus/speech_script.txt")
rows = {}
for name, on in (("off", False), ("on", True)):
    y, clicks = condition(on)
    save_audio(out / f"mic_declick_{name}.wav", y, SR)
    segs, _ = model.transcribe(resample_array(y, SR, 16000).astype(np.float32), language="en",
                               word_timestamps=True, beam_size=5)
    words = [w for s in segs for w in s.words]
    text = " ".join(w.word.strip() for w in words)
    # RNNoise delays the conditioned mic by 20 ms: a clamp at input time t is at t + 0.02
    near = []
    for t, g in clicks:
        tt = t + 0.02
        w = min(words, key=lambda w: 0 if w.start <= tt <= w.end else min(abs(w.start - tt), abs(w.end - tt)))
        inside = w.start - 0.05 <= tt <= w.end + 0.05
        near.append({"t": round(t, 3), "db": round(g, 1), "word": w.word.strip() if inside else None,
                     "pos_in_word": round((tt - w.start) / max(w.end - w.start, 1e-3), 2) if inside else None})
    rows[name] = {"wer": wer(script, text), "clamps": len(clicks), "clamp_detail": near, "text": text}
    print(name, "WER", round(rows[name]["wer"], 4), "clamps", len(clicks), flush=True)
json.dump(rows, open(out / "mic_declick.json", "w"), indent=1)
for r in rows["on"]["clamp_detail"]:
    print(r)
