"""Breath / click passthrough (engine/postchain.py ``Passthrough``, docs/DEVIATIONS.md §15)."""

import numpy as np
import torch

from vox.engine import chunker
from vox.engine.postchain import PostChain

SR = 48000


def setup():
    plan = chunker.plan(80, 30, 500, 10, SR, SR)
    post = PostChain(plan, "cpu")
    post.set_passthrough(True, 1.0)
    return plan, post


def run(post, plan, ref, pitchf):
    out = torch.zeros(plan.out_device)
    post.run(out, torch.from_numpy(ref.astype(np.float32)), torch.from_numpy(pitchf))
    return out.numpy(), post.passthrough.gain.numpy()


def frames_gain(gain, plan):
    return gain.reshape(plan.return_length, plan.zc).mean(1)


def lowpassed_noise(n, rng, cutoff=1500.0):
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[np.fft.rfftfreq(n, 1 / SR) > cutoff] = 0
    return np.fft.irfft(spec, n)


def highpassed_noise(n, rng, cutoff=5000.0):
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[np.fft.rfftfreq(n, 1 / SR) < cutoff] = 0
    return np.fft.irfft(spec, n)


def test_off_by_default():
    plan = chunker.plan(80, 30, 500, 10, SR, SR)
    post = PostChain(plan, "cpu")
    assert not post.passthrough_on
    out = torch.ones(plan.out_device)
    post.run(out, torch.ones(plan.out_device), torch.zeros(plan.p_len))
    assert torch.all(out == 1.0)


def test_a_breath_passes_and_a_sibilant_does_not():
    plan, post = setup()
    rng = np.random.default_rng(0)
    n = plan.out_device
    unvoiced = np.zeros(plan.p_len, np.float32)  # long unvoiced run: breath-eligible
    breath = 0.05 * lowpassed_noise(n, rng)
    out, gain = run(post, plan, breath, unvoiced)
    assert frames_gain(gain, plan).min() > 0.9
    assert np.sqrt(np.mean(out ** 2)) > 0.5 * np.sqrt(np.mean(breath ** 2))
    hiss = 0.05 * highpassed_noise(n, rng)  # sustained, centroid > 4 kHz
    out, gain = run(post, plan, hiss, unvoiced)
    assert np.abs(gain).max() == 0.0 and np.abs(out).max() == 0.0


def test_voiced_frames_never_pass_and_a_short_burst_does():
    plan, post = setup()
    rng = np.random.default_rng(1)
    n, zc = plan.out_device, plan.zc
    voiced = np.full(plan.p_len, 200.0, np.float32)
    out, gain = run(post, plan, 0.05 * lowpassed_noise(n, rng), voiced)
    assert np.abs(gain).max() == 0.0
    # a 10 ms smack in near-silence, inside a short unvoiced gap (not a breath)
    ref = 1e-4 * rng.standard_normal(n)
    k = plan.return_length // 2
    ref[k * zc:(k + 1) * zc] += 0.2 * rng.standard_normal(zc)
    pitch = voiced.copy()
    tail0 = plan.skip_head
    pitch[tail0 + k - 2:tail0 + k + 3] = 0.0
    out, gain = run(post, plan, ref, pitch[:plan.skip_head + plan.return_length])
    g = frames_gain(gain, plan)
    assert g[k] > 0.4 and g[0] == 0.0 and g[-1] == 0.0
