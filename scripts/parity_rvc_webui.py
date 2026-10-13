"""A/B our implementation against a read-only rvc-webui clone (Phase 1 acceptance).

    python scripts/parity_rvc_webui.py --ref _refs/rvc-webui

Three checks, from tightest to loosest:

1. synthesizer  - same checkpoint, same inputs, same seed on CPU: outputs must match to
                  float round-off, for both the full and the tail-only (skip_head) path.
2. rmvpe        - same audio: F0 tracks must agree.
3. offline A/B  - the whole canonical pipeline on a real clip: log-mel distance between the
                  reference render and ours, next to the noise floor between two of ours
                  (the synthesizer is stochastic, so exact equality is impossible).

Needs the extra reference deps: ``pip install librosa praat-parselmouth``.
It also writes both renders to ``out/`` for the manual listening A/B.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def check_synth(models_dir: Path, name: str) -> float:
    from infer.module.models import SynthesizerTrnMs768NSFsid  # reference

    from vox.synthesis.rvc_model import RvcModel

    path = models_dir / name / "model.pth"
    cpt = torch.load(path, map_location="cpu", weights_only=True)
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    ref = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
    del ref.enc_q
    ref.load_state_dict(cpt["weight"], strict=False)
    ref = ref.float().eval()
    ref.remove_weight_norm()
    ours = RvcModel.load(path, "cpu")

    gen = torch.Generator().manual_seed(1)
    frames = 120
    feats = torch.randn(1, frames, 768, generator=gen)
    pitchf = torch.rand(1, frames, generator=gen) * 300 + 80
    pitchf[:, 40:55] = 0  # an unvoiced stretch
    pitch = torch.randint(1, 255, (1, frames), generator=gen)
    lengths = torch.tensor([frames])
    sid = torch.tensor([0])
    worst = 0.0
    for skip_head, return_length in ((None, None), (70, 50), (10, 30)):
        with torch.no_grad():
            torch.manual_seed(7)
            a = ref.infer(feats, lengths, pitch, pitchf, sid, skip_head, return_length)[0][0, 0]
        torch.manual_seed(7)
        b = ours.infer(feats, lengths, pitch, pitchf, sid, skip_head or 0, return_length)
        diff = (a - b).abs().max().item()
        worst = max(worst, diff)
        print(f"  synth skip_head={skip_head} return_length={return_length}: "
              f"{tuple(b.shape)} max|diff| = {diff:.2e}")
    return worst


def check_rmvpe(models_dir: Path, audio_16k: np.ndarray) -> tuple[float, float]:
    from infer.rmvpe import RMVPE  # reference

    from vox.pitch.rmvpe import RmvpeExtractor

    ref = RMVPE(str(models_dir / "_predictors" / "rmvpe.pt"), is_half=False, device="cpu")
    f0_ref = ref.infer_from_audio(audio_16k, thred=0.03)
    ours = RmvpeExtractor()
    ours.load(torch.device("cpu"), models_dir)
    f0 = ours.extract(torch.from_numpy(audio_16k)).numpy()
    voiced = (f0_ref > 0) & (f0 > 0)
    agree = float(((f0_ref > 0) == (f0 > 0)).mean())
    cents = 1200 * np.abs(np.log2(f0[voiced] / f0_ref[voiced]))
    print(f"  rmvpe frames={len(f0)} voicing agreement={agree:.4f} "
          f"max deviation={cents.max():.3f} cents")
    return agree, float(cents.max())


def check_offline(models_dir: Path, name: str, audio_16k: np.ndarray, out_dir: Path,
                  f0_up_key: float) -> tuple[float, float]:
    from infer.vc.pipeline import Pipeline  # reference

    from vox.audio.resample import save_audio
    from vox.config.schemas import RuntimeParams
    from vox.features.contentvec import ContentVec
    from vox.offline.convert import FullFileConverter, SegmentPlan
    from vox.offline.metrics import log_mel_distance
    from vox.pitch.rmvpe import RmvpeExtractor
    from vox.synthesis.rvc_model import RvcModel

    device = torch.device("cpu")
    os.environ["rmvpe_root"] = str(models_dir / "_predictors")
    model = RvcModel.load(models_dir / name / "model.pth", device)
    embedder = ContentVec("v2")
    embedder.load(device, models_dir)
    extractor = RmvpeExtractor()
    extractor.load(device, models_dir)

    # Reference render. It reuses our loaded HF ContentVec instance and a reference-built
    # synthesizer, so the A/B covers F0, feature handling, padding and synthesis.
    from infer.module.models import SynthesizerTrnMs768NSFsid

    cpt = torch.load(models_dir / name / "model.pth", map_location="cpu", weights_only=True)
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float().eval()
    net_g.remove_weight_norm()
    config = SimpleNamespace(x_pad=1, x_query=6, x_center=38, x_max=41, is_half=False,
                             device=device)
    tgt_sr = cpt["config"][-1]
    pipe = Pipeline(tgt_sr, config)
    # Build the reference's lazily-created RMVPE now: constructing a network draws from the
    # global RNG, which would desynchronise the two renders' noise if it happened after
    # seeding.
    from infer.rmvpe import RMVPE

    pipe.model_rmvpe = RMVPE(str(models_dir / "_predictors" / "rmvpe.pt"), is_half=False,
                             device=device)
    audio = audio_16k.copy()
    peak = np.abs(audio).max() / 0.95
    if peak > 1:
        audio /= peak
    torch.manual_seed(0)
    ref_i16 = pipe.pipeline(embedder.model, net_g, 0, audio, [0, 0, 0], f0_up_key, "rmvpe",
                            "", 0, 1, tgt_sr, 0, 1, "v2", 0.33)
    ref = ref_i16.astype(np.float32) / 32768.0

    # Our renders: interp matches this rvc-webui revision's F0 handling.
    converter = FullFileConverter(model, embedder, extractor, device,
                                  segments=SegmentPlan(1, 6, 38, 41))
    params = RuntimeParams(f0_up_key=f0_up_key, f0_unvoiced="interp")
    torch.manual_seed(0)
    ours = converter.convert(audio_16k, params)
    torch.manual_seed(1)
    ours_again = converter.convert(audio_16k, params)

    save_audio(out_dir / "ab_reference_rvc_webui.wav", ref, tgt_sr)
    save_audio(out_dir / "ab_ours_full.wav", ours, tgt_sr)
    dist, _ = log_mel_distance(ref, ours, tgt_sr)
    floor, _ = log_mel_distance(ours, ours_again, tgt_sr)
    n = min(len(ref), len(ours))
    print(f"  offline lengths ref={len(ref)} ours={len(ours)} @ {tgt_sr} Hz   same-seed "
          f"sample max|diff| = {np.abs(ref[:n] - ours[:n]).max():.2e} "
          f"(int16 step = {1 / 32768:.1e})")
    print(f"  log-mel L1: ours vs rvc-webui = {dist:.4f}   noise floor (ours vs ours, "
          f"new seed) = {floor:.4f}")
    return dist, floor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", type=Path, default=ROOT / "_refs" / "rvc-webui")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--model", default="base_f0_48k_v2")
    parser.add_argument("--audio", type=Path,
                        default=ROOT / "benchmarks" / "corpus" / "libri_1272.wav")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.ref.resolve()))
    from vox.audio.resample import load_audio

    audio, _ = load_audio(args.audio, 16000)
    audio = audio[: int(args.seconds * 16000)]

    print("1. synthesizer parity (CPU, same seed)")
    synth = check_synth(args.models_dir, args.model)
    print("2. RMVPE parity")
    agree, cents = check_rmvpe(args.models_dir, audio)
    print("3. offline pipeline A/B")
    dist, floor = check_offline(args.models_dir, args.model, audio, ROOT / "out", args.pitch)

    ok = synth < 1e-4 and agree > 0.995 and cents < 1.0 and dist < max(1.5 * floor, 0.05)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
