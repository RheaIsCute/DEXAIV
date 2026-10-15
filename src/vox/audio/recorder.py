"""Session recording for diagnosis: the raw microphone, the microphone after the input chain,
and the converted output, captured hop by hop on the worker thread so the three tracks are
sample-aligned (the output trails the input by the SOLA delay, as it does live).

The worker calls ``add`` once per hop (three small copies); the control thread starts and
stops a take and writes it out. Float32 WAVs, so digital zeros and levels survive exactly.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

TRACKS = ("output", "mic_raw", "mic_conditioned")


class SessionRecorder:
    def __init__(self, sr: int, max_seconds: float = 1800.0) -> None:
        self.sr = sr
        self.max_samples = int(sr * max_seconds)
        self.samples = 0
        self.gate_open: list[bool] = []  # per hop: did the hop run through the model
        self._chunks: dict[str, list[np.ndarray]] = {t: [] for t in TRACKS}
        self._lock = threading.Lock()

    @property
    def seconds(self) -> float:
        return self.samples / self.sr

    @property
    def full(self) -> bool:
        return self.samples >= self.max_samples

    def add(self, mic_raw: np.ndarray, mic_conditioned: np.ndarray, output: np.ndarray,
            gate_open: bool) -> None:
        """One hop, on the worker thread. Stops growing at ``max_seconds``."""
        with self._lock:
            if self.samples >= self.max_samples:
                return
            self._chunks["output"].append(output.copy())
            self._chunks["mic_raw"].append(mic_raw.copy())
            self._chunks["mic_conditioned"].append(mic_conditioned.copy())
            self.gate_open.append(bool(gate_open))
            self.samples += len(output)

    def save(self, folder: Path, stem: str, meta: dict[str, Any]) -> dict[str, Path]:
        """Write ``<stem>_<track>.wav`` for each track plus ``<stem>.json`` (settings, gate
        per hop, events). Returns the paths by track (and ``"meta"``)."""
        import soundfile as sf

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with self._lock:
            tracks = {t: np.concatenate(c) if c else np.zeros(0, np.float32)
                      for t, c in self._chunks.items()}
            gate = list(self.gate_open)
        paths: dict[str, Path] = {}
        for name, audio in tracks.items():
            path = folder / f"{stem}_{name}.wav"
            sf.write(str(path), audio.astype(np.float32), self.sr, subtype="FLOAT")
            paths[name] = path
        meta = {**meta, "sr": self.sr, "seconds": self.seconds,
                "files": {k: p.name for k, p in paths.items()},
                "gate_open_per_hop": "".join("1" if g else "0" for g in gate)}
        paths["meta"] = folder / f"{stem}.json"
        paths["meta"].write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        return paths
