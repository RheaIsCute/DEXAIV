"""F5-TTS (SWivid; MIT code, CC-BY-NC-4.0 weights) as a TTS worker:
``python f5_worker.py <reference.wav>``.

Runs in ``.tts_envs/f5`` (Python 3.11, f5-tts 1.1.22, torch 2.7.1+cu128). F5 has no voices
or emotion knobs of its own: it continues the voice *and delivery* of a reference clip
(5-12 s), so it speaks the way the reference speaks. The voice is a reference WAV path
("default" = the one given on the command line); its transcript is made once by F5's
Whisper and kept next to it as ``<clip>.txt`` (edit that file to correct it). Native
``speed``. Controls: ``nfe_step`` (32), ``cfg_strength`` (2), ``seed``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _protocol import serve  # noqa: E402


class F5:
    def __init__(self, reference: str) -> None:
        self.reference = reference
        self.tts = None
        self._texts: dict[str, str] = {}

    def _ref_text(self, ref: str) -> str:
        if ref not in self._texts:
            sidecar = Path(ref).with_suffix(".txt")
            if sidecar.is_file():
                text = sidecar.read_text(encoding="utf-8").strip()
            else:
                text = self._transcribe(ref)
                sidecar.write_text(text + "\n", encoding="utf-8")
            self._texts[ref] = text
        return self._texts[ref]

    @staticmethod
    def _transcribe(ref: str) -> str:
        """F5's Whisper on the decoded clip (its own path wants ffmpeg for a file name)."""
        import soundfile as sf
        import torch
        import torchaudio.functional as AF
        from f5_tts.infer import utils_infer

        audio, sr = sf.read(ref, dtype="float32", always_2d=True)
        mono = torch.from_numpy(audio.mean(axis=1))
        raw = AF.resample(mono, sr, 16000).numpy()
        return utils_infer.transcribe({"raw": raw, "sampling_rate": 16000}, "english")

    def load(self) -> int:
        import torch
        from f5_tts.api import F5TTS

        torch.backends.cuda.matmul.allow_tf32 = True
        self.tts = F5TTS(device="cuda")
        self.synth("Hi there.", None, 1.0, {})
        return self.tts.target_sample_rate

    def voices(self) -> list[str]:
        return ["default"]

    def synth(self, text, voice, speed, controls):
        ref = self.reference if voice in (None, "", "default") else voice
        wav, sr, _ = self.tts.infer(
            ref, self._ref_text(ref), text, show_info=lambda *a, **k: None,
            speed=float(speed), nfe_step=int(controls.get("nfe_step", 32)),
            cfg_strength=float(controls.get("cfg_strength", 2.0)),
            seed=controls.get("seed"))
        return wav, sr


if __name__ == "__main__":
    serve(F5(sys.argv[1]))
