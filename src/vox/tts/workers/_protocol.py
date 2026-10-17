"""The TTS worker side of ``vox.tts.external``: JSON lines on stdin, JSON lines back.

A worker runs in its own virtual environment (its own Python and torch), so this file and
the ``*_worker.py`` scripts next to it import nothing from ``vox``. Requests:

    {"op": "load"}                              -> {"ok": true, "sr": 24000, "load_s": 3.1}
    {"op": "voices"}                            -> {"ok": true, "voices": [...]}
    {"op": "synth", "text": "...", "out": "x.wav", "voice": "...", "speed": 1.0,
     "controls": {...}}                         -> {"ok": true, "sr": ..., "seconds": ...,
                                                    "render_s": ..., "vram_peak_mb": ...}
    {"op": "quit"}

Errors come back as {"ok": false, "error": "..."}; the worker keeps running. Anything a
library prints goes to stderr: the protocol owns the real stdout.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import wave


def _protocol_stream():
    """A private copy of stdout for replies; fd 1 itself is pointed at stderr so that
    stray prints (Python or C) can never corrupt the reply stream."""
    fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(fd, "w", encoding="utf-8", buffering=1)


def save_wav(path: str, audio, sr: int) -> float:
    """Mono float audio (-1..1, numpy) to 16-bit PCM WAV. Returns its length in seconds."""
    import numpy as np

    pcm = (np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
           * 32767.0).astype("<i2")
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sr))
        fh.writeframes(pcm.tobytes())
    return len(pcm) / float(sr)


def vram_peak_mb(reset: bool = True) -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return round(peak, 1)


def serve(engine) -> None:
    """Answer requests until "quit" or EOF. ``engine`` has ``load() -> sr``,
    ``voices() -> list`` and ``synth(text, voice, speed, controls) -> (audio, sr)``."""
    out = _protocol_stream()

    def reply(**msg):
        out.write(json.dumps(msg) + "\n")
        out.flush()

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "quit":
                reply(ok=True)
                return
            if op == "load":
                start = time.perf_counter()
                sr = engine.load()
                reply(ok=True, sr=sr, load_s=round(time.perf_counter() - start, 3),
                      vram_peak_mb=vram_peak_mb())
            elif op == "voices":
                reply(ok=True, voices=list(engine.voices()))
            elif op == "synth":
                start = time.perf_counter()
                audio, sr = engine.synth(req["text"], req.get("voice"),
                                         float(req.get("speed", 1.0)), req.get("controls") or {})
                render_s = time.perf_counter() - start
                seconds = save_wav(req["out"], audio, sr)
                reply(ok=True, sr=sr, seconds=round(seconds, 3), render_s=round(render_s, 3),
                      vram_peak_mb=vram_peak_mb())
            else:
                reply(ok=False, error=f"unknown op {op!r}")
        except Exception as err:  # noqa: BLE001 - reported, the worker stays up
            traceback.print_exc()
            reply(ok=False, error=f"{type(err).__name__}: {err}")
