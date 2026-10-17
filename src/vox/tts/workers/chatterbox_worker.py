"""Chatterbox (Resemble AI, MIT) as a TTS worker: ``python chatterbox_worker.py [turbo]``.

Runs in ``.tts_envs/chatterbox`` (Python 3.11, chatterbox-tts 0.1.7 with torch overridden
to 2.7.1+cu128: the package pins torch 2.6.0, which has no Blackwell kernels).

Voices: "default" is the model's built-in voice (conds.pt); any other value is a path to a
reference clip (> 5 s for Turbo) the delivery is cloned from. Controls:
- original: ``exaggeration`` (0.25-2, default 0.5: emotion intensity), ``cfg_weight``
  (0-1, default 0.5: lower = slower, more deliberate), ``temperature`` (default 0.8).
- turbo: paralinguistic tags in the text ([laugh], [chuckle], [sigh], [cough], ...);
  ``temperature``. Turbo ignores exaggeration and cfg.
Both apply Resemble's Perth watermark to the output (imperceptible; part of the package).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _protocol import serve  # noqa: E402


class Chatterbox:
    def __init__(self, turbo: bool) -> None:
        self.turbo = turbo
        self.model = None
        self._prompt = None  # the reference clip the conditionals were made from

    def load(self) -> int:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        if self.turbo:
            from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
        else:
            from chatterbox.tts import ChatterboxTTS as Model
        self.model = Model.from_pretrained(device="cuda")
        self._default = self.model.conds
        self.synth("Hi there.", None, 1.0, {})  # the first generate initialises lazily (~25 s)
        return self.model.sr

    def voices(self) -> list[str]:
        return ["default"]

    def synth(self, text, voice, speed, controls):
        m = self.model
        prompt = None if voice in (None, "", "default") else voice
        if prompt != self._prompt:
            if prompt is None:
                m.conds = self._default
            else:
                kwargs = {"exaggeration": float(controls.get("exaggeration", 0.5))}
                m.prepare_conditionals(prompt, **kwargs)
            self._prompt = prompt
        if self.turbo:
            wav = m.generate(text, temperature=float(controls.get("temperature", 0.8)))
        else:
            wav = m.generate(text, exaggeration=float(controls.get("exaggeration", 0.5)),
                             cfg_weight=float(controls.get("cfg_weight", 0.5)),
                             temperature=float(controls.get("temperature", 0.8)))
        return wav.squeeze(0).cpu().numpy(), m.sr


if __name__ == "__main__":
    serve(Chatterbox(turbo="turbo" in sys.argv[1:]))
