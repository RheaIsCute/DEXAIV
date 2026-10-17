"""Build the isolated environments of the expressive TTS engines (``vox.tts.external``).

    .venv\\Scripts\\python.exe scripts\\setup_tts_envs.py [chatterbox orpheus qwen3 f5]

Each engine gets its own virtual environment under ``.tts_envs/<name>/`` with its own Python
(downloaded by uv into ``.tts_envs/_python``; nothing is installed system-wide) and torch
2.7.1+cu128, the build with Blackwell kernels. The main ``.venv`` is never touched. Model
weights go to ``models/_tts/hf`` (Hugging Face cache) and ``models/_tts/orpheus``.
Pins and workarounds are the ones that were needed on 2026-10-17 (docs/TTS_NOTES.md).
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENVS = ROOT / ".tts_envs"
TTS = ROOT / "models" / "_tts"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH = ["torch==2.7.1", "torchaudio==2.7.1"]
LLAMA_BUILD = "b11327"  # llama.cpp release of 2026-10-01, CUDA 13.4 Windows build
ORPHEUS_GGUF = ("https://huggingface.co/unsloth/orpheus-3b-0.1-ft-GGUF/resolve/main/"
                "orpheus-3b-0.1-ft-Q6_K.gguf")

# engine -> (python, packages, extra steps after the install)
ENGINES = {
    # chatterbox-tts pins torch 2.6.0 (no sm_120); Perth's watermarker needs pkg_resources
    "chatterbox": ("3.11", ["chatterbox-tts==0.1.7", "setuptools<81"], []),
    "qwen3": ("3.12", ["qwen-tts==0.1.1"], []),
    "orpheus": ("3.12", ["snac==1.2.1", "numpy"], ["llama.cpp", "gguf"]),
    # datasets 2.x breaks on pyarrow >= 21; transformers 4.57 wants huggingface-hub < 1;
    # torchcodec (built for newer torch) breaks Whisper's preprocessing and F5 never uses it
    "f5": ("3.11", ["f5-tts==1.1.22", "datasets>=3.6", "huggingface-hub>=0.34,<1.0"],
           ["drop-torchcodec"]),
}


def run(*cmd: str, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})


def uv() -> str:
    exe = ENVS / "_tools" / "Scripts" / "uv.exe"
    if not exe.is_file():
        run(sys.executable, "-m", "venv", str(ENVS / "_tools"))
        run(str(ENVS / "_tools" / "Scripts" / "python.exe"), "-m", "pip", "install", "-q", "uv")
    return str(exe)


def fetch(url: str, dst: Path) -> None:
    if dst.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}", flush=True)
    tmp = dst.with_name(dst.name + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)


def llama_cpp() -> None:
    target = ENVS / "llama.cpp"
    if (target / "llama-server.exe").is_file():
        return
    base = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}"
    for name in (f"llama-{LLAMA_BUILD}-bin-win-cuda-13.4-x64.zip",
                 "cudart-llama-bin-win-cuda-13.4-x64.zip"):
        archive = ENVS / "_downloads" / name
        fetch(f"{base}/{name}", archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)


def build(name: str) -> None:
    python, packages, extras = ENGINES[name]
    env = {"UV_PYTHON_INSTALL_DIR": str(ENVS / "_python")}
    exe = ENVS / name / "Scripts" / "python.exe"
    tool = uv()
    if not exe.is_file():
        run(tool, "python", "install", python, env=env)
        run(tool, "venv", "--python", python, str(ENVS / name), env=env)
    overrides = ENVS / "torch_overrides.txt"
    overrides.write_text("\n".join(TORCH) + "\n")
    run(tool, "pip", "install", "--python", str(exe), "--index-strategy", "unsafe-best-match",
        "--extra-index-url", TORCH_INDEX, "--override", str(overrides), *TORCH, *packages,
        env=env)
    if "drop-torchcodec" in extras:
        run(tool, "pip", "uninstall", "--python", str(exe), "torchcodec")
    if "llama.cpp" in extras:
        llama_cpp()
    if "gguf" in extras:
        fetch(ORPHEUS_GGUF, TTS / "orpheus" / ORPHEUS_GGUF.rsplit("/", 1)[1])


def main() -> int:
    names = sys.argv[1:] or list(ENGINES)
    for name in names:
        if name not in ENGINES:
            print(f"unknown engine {name!r}; choose from {', '.join(ENGINES)}", file=sys.stderr)
            return 2
        build(name)
    print("done. The F5 engine also needs a reference clip: models/_tts/f5/ref_user.wav "
          "(5-12 s of the voice and delivery to continue).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
