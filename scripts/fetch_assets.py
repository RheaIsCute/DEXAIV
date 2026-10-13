"""Download and verify the shared model assets (docs/research/09 §6 layout).

    python scripts/fetch_assets.py              # ContentVec + RMVPE (required)
    python scripts/fetch_assets.py --test-model # + a public RVC v2 test voice (see below)
    python scripts/fetch_assets.py --test-audio # + LibriSpeech clips for the benchmark corpus
    python scripts/fetch_assets.py --all

Every file is checked against the SHA-256 pinned in scripts/assets.lock.json. A file with no
lock entry is checked against the hash Hugging Face publishes for it and then pinned.

The "test model" is the official RVC v2 pretrained base generator (f0G48k / f0G40k) repacked
as an inference checkpoint. It is a real multi-speaker RVC v2 synthesizer, good for
engineering tests; it is not a tuned target voice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
LOCK = Path(__file__).resolve().parent / "assets.lock.json"
HF = "https://huggingface.co/{repo}/resolve/main/{path}"

CORE = [
    ("lengyue233/content-vec-best", "config.json", "_embedders/contentvec/config.json"),
    ("lengyue233/content-vec-best", "pytorch_model.bin",
     "_embedders/contentvec/pytorch_model.bin"),
    ("lj1995/VoiceConversionWebUI", "rmvpe.pt", "_predictors/rmvpe.pt"),
]
PRETRAINED = [
    ("lj1995/VoiceConversionWebUI", "pretrained_v2/f0G48k.pth", "_pretrained/f0G48k.pth"),
    ("lj1995/VoiceConversionWebUI", "pretrained_v2/f0G40k.pth", "_pretrained/f0G40k.pth"),
]
# Constructor tuples for the official v2 base models (rvc-webui configs/v2/48k.json and
# configs/v1/40k.json): the 18 fields of docs/research/09 §2.
_COMMON = [192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]]]
TEST_MODELS = {
    "base_f0_48k_v2": ("_pretrained/f0G48k.pth",
                       [1025, 36, *_COMMON, [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256, 48000]),
    "base_f0_40k_v2": ("_pretrained/f0G40k.pth",
                       [1025, 32, *_COMMON, [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000]),
}
LIBRISPEECH = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
LIBRI_SPEAKERS = ("1272", "174", "84")  # dev-clean; one clip of >= CLIP_SECONDS each
CLIP_SECONDS = 30.0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def published_sha256(url: str) -> str | None:
    """Hugging Face exposes an LFS file's SHA-256 as X-Linked-ETag on the pre-redirect reply."""
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(url, method="HEAD")
    try:
        headers = opener.open(request, timeout=60).headers
    except urllib.error.HTTPError as err:
        headers = err.headers
    etag = (headers.get("X-Linked-ETag") or "").strip('"')
    return etag if len(etag) == 64 else None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done, shown = 0, -1
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            decile = done * 10 // total if total else 0
            if decile != shown:  # one line per 10 %, so redirected logs stay short
                shown = decile
                print(f"  {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB")
    tmp.replace(dest)


def fetch(repo: str, path: str, rel: str, lock: dict[str, str]) -> Path:
    dest = MODELS / rel
    url = HF.format(repo=repo, path=path)
    key = f"{repo}/{path}"
    if not dest.is_file():
        print(f"downloading {key}")
        download(url, dest)
    actual = sha256_of(dest)
    expected = lock.get(key)
    if expected is None:
        expected = published_sha256(url)
        if expected is None:  # small non-LFS file: pin what we got (trust on first use)
            expected = actual
            print(f"  pinned {key} on first use")
        lock[key] = expected
    if actual != expected:
        dest.rename(dest.with_name(dest.name + ".corrupt"))
        raise SystemExit(f"SHA-256 mismatch for {key}: got {actual}, expected {expected}")
    print(f"ok  {rel}  sha256={actual[:16]}...")
    return dest


def build_test_models() -> None:
    """Repack the training-format base generators as inference checkpoints."""
    import torch

    for name, (rel, config) in TEST_MODELS.items():
        out = MODELS / name / "model.pth"
        if out.is_file():
            print(f"ok  {name}/model.pth (exists)")
            continue
        ckpt = torch.load(MODELS / rel, map_location="cpu", weights_only=True)
        weight = {k: v.half() for k, v in ckpt["model"].items() if not k.startswith("enc_q.")}
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weight": weight, "config": config, "f0": 1, "version": "v2",
                    "sr": f"{config[-1] // 1000}k", "info": f"repacked from {Path(rel).name}"},
                   out)
        print(f"built {name}/model.pth ({len(weight)} tensors)")


def fetch_test_audio(corpus: Path) -> None:
    """Stream LibriSpeech dev-clean and cut one ~30 s 48 kHz clip per chosen speaker."""
    import io

    import numpy as np
    import soundfile as sf
    import soxr

    wanted = {s: corpus / f"libri_{s}.wav" for s in LIBRI_SPEAKERS}
    missing = {s for s, p in wanted.items() if not p.is_file()}
    if not missing:
        print("ok  benchmark corpus clips (exist)")
        return
    corpus.mkdir(parents=True, exist_ok=True)
    utterances: dict[str, dict[str, np.ndarray]] = {s: {} for s in missing}
    print(f"streaming {LIBRISPEECH} (338 MB) for speakers {sorted(missing)}")
    with urllib.request.urlopen(LIBRISPEECH, timeout=120) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                parts = Path(member.name).parts
                if not member.isfile() or not member.name.endswith(".flac") or len(parts) < 5:
                    continue
                speaker = parts[2]
                if speaker not in utterances:
                    continue
                data = tar.extractfile(member).read()
                audio, sr = sf.read(io.BytesIO(data), dtype="float32")
                assert sr == 16000, sr
                utterances[speaker][Path(member.name).stem] = audio
    for speaker in sorted(missing):
        chunks, seconds = [], 0.0
        for key in sorted(utterances[speaker]):
            chunks.append(utterances[speaker][key])
            seconds += len(chunks[-1]) / 16000
            if seconds >= CLIP_SECONDS:
                break
        if not chunks:
            raise SystemExit(f"speaker {speaker} not found in dev-clean")
        clip = soxr.resample(np.concatenate(chunks), 16000, 48000, quality="VHQ")
        sf.write(wanted[speaker], clip.astype(np.float32), 48000, subtype="PCM_16")
        print(f"wrote {wanted[speaker].relative_to(ROOT)} ({seconds:.1f} s)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-model", action="store_true")
    parser.add_argument("--test-audio", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    lock: dict[str, str] = json.loads(LOCK.read_text()) if LOCK.is_file() else {}
    try:
        for repo, path, rel in CORE:
            fetch(repo, path, rel, lock)
        if args.test_model or args.all:
            for repo, path, rel in PRETRAINED:
                fetch(repo, path, rel, lock)
            build_test_models()
    finally:
        LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    if args.test_audio or args.all:
        fetch_test_audio(ROOT / "benchmarks" / "corpus")
    return 0


if __name__ == "__main__":
    sys.exit(main())
