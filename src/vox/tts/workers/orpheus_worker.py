"""Orpheus 3B (Canopy Labs) as a TTS worker: ``python orpheus_worker.py <llama-server> <gguf>``.

Runs in ``.tts_envs/orpheus`` (Python 3.12, torch 2.7.1+cu128, snac 1.2.1). The language
model runs in llama.cpp's ``llama-server`` (a CUDA build, ``.tts_envs/llama.cpp``) from a
quantised GGUF; this process prompts it with Canopy's token layout, collects the audio
codes it emits and decodes them with SNAC (hubertsiuzdak/snac_24khz, 24 kHz).

Voices: tara, leah, jess, leo, dan, mia, zac, zoe. Emotion tags go in the text:
<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>. Controls:
``temperature`` (0.6), ``top_p`` (0.9), ``repetition_penalty`` (1.1; Canopy says >= 1.1).
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _protocol import serve  # noqa: E402

VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
START_HUMAN, END_TEXT, END_HUMAN, START_AI, START_SPEECH = 128259, 128009, 128260, 128261, 128257
END_SPEECH = 128258
CODE_OFFSET = 128266  # custom_token_10: the first audio code
SR = 24000


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Orpheus:
    def __init__(self, server: str, gguf: str) -> None:
        self.server_exe, self.gguf = server, gguf
        self.server = None
        self.url = ""
        self.snac = None

    def _post(self, path: str, payload: dict, timeout: float = 300.0) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def load(self) -> int:
        import torch
        from snac import SNAC

        port = _free_port()
        self.url = f"http://127.0.0.1:{port}"
        self.server = subprocess.Popen(
            [self.server_exe, "-m", self.gguf, "--port", str(port), "--host", "127.0.0.1",
             "-ngl", "99", "-c", "4096", "--no-webui", "-np", "1",
             # end of speech ends the generation (else it runs on into a new text turn)
             "--override-kv", f"tokenizer.ggml.eos_token_id=int:{END_SPEECH}"],
            stdout=sys.stderr, stderr=sys.stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        end = time.time() + 300
        while True:
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as resp:
                    if resp.status == 200:
                        break
            except OSError:
                pass
            if self.server.poll() is not None or time.time() > end:
                raise RuntimeError("llama-server did not start")
            time.sleep(0.25)
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")
        self.synth("Hi.", "tara", 1.0, {})
        return SR

    def voices(self) -> list[str]:
        return list(VOICES)

    def _codes(self, text: str, voice: str, controls: dict) -> list[int]:
        ids = self._post("/tokenize", {"content": f"{voice}: {text}",
                                       "add_special": True})["tokens"]
        prompt = [START_HUMAN, *ids, END_TEXT, END_HUMAN, START_AI, START_SPEECH]
        reply = self._post("/completion", {
            "prompt": prompt, "n_predict": int(controls.get("max_tokens", 2400)),
            "temperature": float(controls.get("temperature", 0.6)),
            "top_p": float(controls.get("top_p", 0.9)),
            "repeat_penalty": float(controls.get("repetition_penalty", 1.1)),
            "cache_prompt": False, "return_tokens": True})
        tokens = reply.get("tokens") or []
        if END_SPEECH in tokens:  # the model may run on into a new text turn after it
            tokens = tokens[:tokens.index(END_SPEECH)]
        codes = []
        for token in tokens:
            code = token - CODE_OFFSET - (len(codes) % 7) * 4096
            if 0 <= code < 4096:
                codes.append(code)
        return codes[:len(codes) // 7 * 7]

    def synth(self, text, voice, speed, controls):
        import torch

        codes = self._codes(text, voice if voice in VOICES else "tara", controls)
        if not codes:
            raise RuntimeError("orpheus produced no audio codes")
        frames = torch.tensor(codes, dtype=torch.int32, device="cuda").view(-1, 7)
        layers = [frames[:, [0]].reshape(1, -1), frames[:, [1, 4]].reshape(1, -1),
                  frames[:, [2, 3, 5, 6]].reshape(1, -1)]
        with torch.inference_mode():
            audio = self.snac.decode(layers)
        return audio.reshape(-1).float().cpu().numpy(), SR


if __name__ == "__main__":
    engine = Orpheus(sys.argv[1], sys.argv[2])
    try:
        serve(engine)
    finally:
        if engine.server is not None:
            engine.server.terminate()
