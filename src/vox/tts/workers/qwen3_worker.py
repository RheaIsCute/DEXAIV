"""Qwen3-TTS (Alibaba Qwen, Apache-2.0) as a TTS worker: ``python qwen3_worker.py [repo]``.

Runs in ``.tts_envs/qwen3`` (Python 3.12, qwen-tts 0.1.1, torch 2.7.1+cu128). Only the
1.7B models take a natural-language ``instruct`` for emotion and style (the 0.6B ones
ignore it). Two kinds:

- CustomVoice (default, Qwen3-TTS-12Hz-1.7B-CustomVoice): nine preset speakers (none is a
  native-English woman: Serena / Vivian are Chinese, Ono_Anna Japanese, Sohee Korean).
  ``instruct`` is the delivery: "Excited, laughing a little".
- VoiceDesign (Qwen3-TTS-12Hz-1.7B-VoiceDesign): the voice is a description, passed as the
  ``voice``; the request's ``instruct`` is appended to it for the delivery.

Controls: ``instruct``, ``temperature``, ``language`` (default English).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _protocol import serve  # noqa: E402

DEFAULT_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DESIGN_VOICE = "A young American woman in her twenties with a warm, natural, casual voice."


class Qwen3:
    def __init__(self, repo: str) -> None:
        self.repo = repo
        self.model = None
        self.design = "VoiceDesign" in repo

    def load(self) -> int:
        import torch
        from qwen_tts import Qwen3TTSModel

        torch.backends.cuda.matmul.allow_tf32 = True
        self.model = Qwen3TTSModel.from_pretrained(self.repo, device_map="cuda:0",
                                                   dtype=torch.bfloat16,
                                                   attn_implementation="sdpa")
        _, sr = self.synth("Hi.", None, 1.0, {})
        return sr

    def voices(self) -> list[str]:
        if self.design:
            return [DESIGN_VOICE]
        return list(self.model.get_supported_speakers() or [])

    def synth(self, text, voice, speed, controls):
        kwargs = {"language": controls.get("language", "English")}
        if "temperature" in controls:
            kwargs["temperature"] = float(controls["temperature"])
        instruct = (controls.get("instruct") or "").strip()
        if self.design:
            described = voice if voice and voice != "default" else DESIGN_VOICE
            wavs, sr = self.model.generate_voice_design(
                text=text, instruct=f"{described} {instruct}".strip(), **kwargs)
        else:
            wavs, sr = self.model.generate_custom_voice(
                text=text, speaker=voice if voice and voice != "default" else "Serena",
                instruct=instruct or None, **kwargs)
        return wavs[0], sr


if __name__ == "__main__":
    serve(Qwen3(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO))
