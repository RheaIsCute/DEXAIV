"""TTS engines that run in their own Python environment, behind ``TtsBackend``.

Expressive TTS models pin their own torch / transformers versions (and some need another
Python), so they run as a small local worker process: ``vox/tts/workers/<x>_worker.py``
under ``.tts_envs/<env>/`` (one virtual environment per engine, never the main ``.venv``),
speaking JSON lines over stdin / stdout (``workers/_protocol.py``). The process starts on
first use and stays up, so only the first clip pays for loading the model.

``synthesize`` takes the usual arguments plus ``controls``: engine-specific delivery knobs
(Chatterbox ``exaggeration``, an instruction for Qwen3-TTS, ...), passed through as is.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from vox.tts.base import TtsBackend, TtsError

WORKERS = Path(__file__).resolve().parent / "workers"
# The editable install's checkout; VOX_TTS_ENVS moves the environments elsewhere.
ENVS = Path(os.environ.get("VOX_TTS_ENVS") or Path(__file__).resolve().parents[3] / ".tts_envs")
LOAD_TIMEOUT_S = 900.0  # the first load may download weights
SYNTH_TIMEOUT_S = 300.0


class WorkerProcess:
    """One worker process and its request / reply pipe (one request at a time)."""

    def __init__(self, python: Path, script: Path, args: list[str], env: dict[str, str],
                 log_path: Path | None = None) -> None:
        self.python, self.script, self.args, self.env = python, script, args, env
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.load_info: dict | None = None
        self._lock = threading.Lock()

    def start(self) -> dict:
        if self.proc is not None and self.proc.poll() is None:
            return self.load_info or {}
        stderr = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL  # noqa: SIM115
        self.proc = subprocess.Popen(
            [str(self.python), "-u", str(self.script), *self.args], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=stderr, env={**os.environ, **self.env},
            text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.load_info = self.request({"op": "load"}, LOAD_TIMEOUT_S)
        return self.load_info

    def request(self, msg: dict, timeout: float = SYNTH_TIMEOUT_S) -> dict:
        with self._lock:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                raise TtsError(f"{self.script.stem} is not running")
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
            reply: list[str] = []
            reader = threading.Thread(target=lambda: reply.append(proc.stdout.readline()),
                                      daemon=True)
            reader.start()
            reader.join(timeout)
            if reader.is_alive() or not reply or not reply[0]:
                self.stop()
                why = "timed out" if reader.is_alive() else "exited"
                raise TtsError(f"{self.script.stem} {why}"
                               + (f" (log: {self.log_path})" if self.log_path else ""))
        answer = json.loads(reply[0])
        if not answer.get("ok"):
            raise TtsError(f"{self.script.stem}: {answer.get('error')}")
        return answer

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=10)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            proc.kill()


class ExternalBackend(TtsBackend):
    """A ``TtsBackend`` whose engine runs in a worker process (subclasses name the
    environment, the script and its arguments)."""

    env_name: str
    script: str
    args: tuple[str, ...] = ()
    default_controls: dict = {}

    _workers: dict[tuple, WorkerProcess] = {}  # shared: one process per engine

    def unavailable_reason(self) -> str | None:
        if not self.python.is_file():
            return (f"its environment is not installed ({self.python.parent.parent}; "
                    "see docs/TTS_NOTES.md)")
        return None

    @property
    def python(self) -> Path:
        return ENVS / self.env_name / ("Scripts/python.exe" if sys.platform == "win32"
                                       else "bin/python")

    def _env(self) -> dict[str, str]:
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONNOUSERSITE": "1"}
        if self.cache_dir is not None:  # weights live with the other TTS weights
            env["HF_HOME"] = str(self.cache_dir / "hf")
        return env

    def worker_args(self) -> tuple[str, ...]:
        return self.args

    def worker(self) -> WorkerProcess:
        key = (self.env_name, self.script, self.worker_args())
        worker = self._workers.get(key)
        if worker is None:
            log = (self.cache_dir / f"{self.name}_worker.log") if self.cache_dir else None
            if log is not None:
                log.parent.mkdir(parents=True, exist_ok=True)
            worker = WorkerProcess(self.python, WORKERS / self.script, list(key[2]),
                                   self._env(), log)
            self._workers[key] = worker
        return worker

    def load(self) -> None:
        if self.unavailable_reason() is None:
            self.worker().start()

    def voices(self) -> list[str]:
        worker = self.worker()
        worker.start()
        return worker.request({"op": "voices"})["voices"]

    def synthesize(self, text: str, out_wav_path: Path, speed: float = 1.0,
                   blend: tuple[tuple[str, float], ...] = (), controls: dict | None = None
                   ) -> int:
        if reason := self.unavailable_reason():
            raise TtsError(f"{self.name}: {reason}")
        worker = self.worker()
        worker.start()
        reply = worker.request({"op": "synth", "text": text, "out": str(out_wav_path),
                                "voice": self.voice, "speed": float(speed),
                                "controls": {**self.default_controls, **(controls or {})}})
        self.last_render = reply
        return int(reply["sr"])

    @classmethod
    def shutdown_all(cls) -> None:
        for worker in list(cls._workers.values()):
            worker.stop()
        cls._workers.clear()


class ChatterboxBackend(ExternalBackend):
    """Chatterbox (0.5B, MIT): ``exaggeration`` controls emotion intensity."""

    name = "chatterbox"
    default_voice = "default"
    env_name = "chatterbox"
    script = "chatterbox_worker.py"


class ChatterboxTurboBackend(ExternalBackend):
    """Chatterbox Turbo (350M, MIT): paralinguistic tags such as [laugh] in the text."""

    name = "chatterbox-turbo"
    default_voice = "default"
    env_name = "chatterbox"
    script = "chatterbox_worker.py"
    args = ("turbo",)


class Qwen3Backend(ExternalBackend):
    """Qwen3-TTS 1.7B CustomVoice (Apache-2.0): preset speakers, ``instruct`` for the
    delivery ("excited, laughing")."""

    name = "qwen3"
    default_voice = "Serena"
    env_name = "qwen3"
    script = "qwen3_worker.py"


class Qwen3DesignBackend(ExternalBackend):
    """Qwen3-TTS 1.7B VoiceDesign (Apache-2.0): the voice is a description, ``instruct``
    the delivery."""

    name = "qwen3-design"
    default_voice = "default"
    env_name = "qwen3"
    script = "qwen3_worker.py"
    args = ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",)


class OrpheusBackend(ExternalBackend):
    """Orpheus 3B (Canopy Labs; Apache-2.0 code, Llama 3.2 weights) from a Q6_K GGUF in
    llama.cpp's server, SNAC-decoded: emotion tags such as <laugh> in the text."""

    name = "orpheus"
    default_voice = "tara"
    env_name = "orpheus"
    script = "orpheus_worker.py"
    gguf = "orpheus-3b-0.1-ft-Q6_K.gguf"

    @property
    def server(self) -> Path:
        return ENVS / "llama.cpp" / "llama-server.exe"

    @property
    def model_file(self) -> Path:
        return (self.cache_dir or Path()) / "orpheus" / self.gguf

    def worker_args(self) -> tuple[str, ...]:
        return (str(self.server), str(self.model_file))

    def unavailable_reason(self) -> str | None:
        if reason := super().unavailable_reason():
            return reason
        for path in (self.server, self.model_file):
            if not path.is_file():
                return f"{path} is missing (see docs/TTS_NOTES.md)"
        return None


class F5Backend(ExternalBackend):
    """F5-TTS v1 (MIT code, CC-BY-NC weights: personal use only): continues the voice and
    delivery of a reference clip, by default ``<tts cache>/f5/ref_user.wav``."""

    name = "f5"
    default_voice = "default"
    env_name = "f5"
    script = "f5_worker.py"

    @property
    def reference(self) -> Path:
        return (self.cache_dir or Path()) / "f5" / "ref_user.wav"

    def worker_args(self) -> tuple[str, ...]:
        return (str(self.reference),)

    def unavailable_reason(self) -> str | None:
        if reason := super().unavailable_reason():
            return reason
        if not self.reference.is_file():
            return f"no reference clip at {self.reference}"
        return None
