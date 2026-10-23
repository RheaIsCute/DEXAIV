"""Post-processing chain after synthesis (principle P7): RMS mix, then the breath / click
passthrough (a flag, off).

``volume_envelope`` (rms_mix_rate) 1.0 keeps the model's own loudness; lower values pull
the output envelope toward the input's. Envelopes use 40 ms frames at a 10 ms hop, like
the rvc-webui realtime path, but stay on the device.

``Passthrough`` (docs/DEVIATIONS.md §15, report 19 §3.2) mixes the band-passed conditioned
mic, sample-aligned under the decoded tail (``in_tail``), into the output on frames that
are unvoiced and either a short burst (a kiss, smack, tongue click: <= 30 ms) or a long
low-centroid breath; sustained frames whose centroid is above 4 kHz (sibilants: /s/ and
/sh/ carry the talker's sex) never pass.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from vox.engine.chunker import ChunkPlan


class RmsMix:
    graph_safe = True

    def __init__(self, plan: ChunkPlan, device: torch.device | str) -> None:
        self.hop = plan.zc
        self.frame = 4 * plan.zc
        self.size = plan.out_device
        self._exponent = torch.zeros((), device=device)

    def set_rate(self, rate: float) -> None:
        self._exponent.fill_(1.0 - rate)

    def _envelope(self, x: Tensor) -> Tensor:
        power = F.pad((x * x)[None, None], (self.frame // 2, self.frame // 2))
        rms = torch.sqrt(F.avg_pool1d(power, self.frame, self.hop))
        return F.interpolate(rms, size=self.size + 1, mode="linear",
                             align_corners=True)[0, 0, :-1]

    def run(self, audio: Tensor, reference: Tensor) -> None:
        """Scale ``audio`` in place by ``(rms_ref / rms_out) ** (1 - rate)``."""
        target = self._envelope(reference)
        current = self._envelope(audio).clamp(min=1e-3)
        audio.mul_(torch.pow(target / current, self._exponent))


# Passthrough classification (report 19 §3.2). Placeholders until the user's kiss / breath
# take: every one of these is a guess from the literature, not a measurement here.
PASS_BAND_HZ = (150.0, 10000.0)  # the band-pass on the mic
PASS_TAPS = 129
PASS_BURST_RATIO = 4.0  # burst: frame energy over both neighbours 30 ms away (+6 dB)
PASS_BURST_REACH = 3  # frames: a burst is at most this long (30 ms)
PASS_BREATH_CENTROID_HZ = 2500.0
PASS_SIBILANT_CENTROID_HZ = 4000.0
PASS_BREATH_MIN_FRAMES = 15  # unvoiced this long before a frame counts as breath (150 ms)
PASS_FLOOR_DBFS = -55.0  # frames quieter than this never pass (room tone)


class Passthrough:
    """Breath / click passthrough: ``audio += mix * g(t) * bandpass(reference)``.

    Per 10 ms frame of the reference (the conditioned mic under the decoded tail) and the
    decoded frames' pitch: power spectrum -> energy and centroid. A frame passes when it
    is unvoiced (``pitchf < 1``, the protect test), above ``PASS_FLOOR_DBFS``, and either a
    burst (``PASS_BURST_RATIO`` over both frames ``PASS_BURST_REACH`` away) or a breath
    (unvoiced for ``PASS_BREATH_MIN_FRAMES`` and centroid below 2.5 kHz); a non-burst frame
    with a centroid over 4 kHz never passes. The 0/1 frame decisions are smoothed
    [1/4, 1/2, 1/4] and interpolated to samples. Static shapes, no host sync (graph-safe);
    ``mix`` is hot (0-dim).
    """

    graph_safe = True

    def __init__(self, plan: ChunkPlan, device: torch.device | str) -> None:
        self.zc = plan.zc
        self.frames = plan.return_length
        self.size = plan.out_device
        self.mix = torch.zeros((), device=device)
        sr = plan.device_sr
        self.window = torch.hann_window(self.zc, device=device)
        self.freqs = torch.fft.rfftfreq(self.zc, 1.0 / sr).to(device)
        lo, hi = PASS_BAND_HZ
        n = torch.arange(PASS_TAPS, dtype=torch.float64) - (PASS_TAPS - 1) / 2
        sinc = lambda f: 2 * f / sr * torch.sinc(2 * f / sr * n)  # noqa: E731
        taps = (sinc(min(hi, 0.49 * sr)) - sinc(lo)) * torch.hamming_window(
            PASS_TAPS, periodic=False, dtype=torch.float64)
        self.taps = taps.to(torch.float32).to(device)[None, None]
        self.smooth = torch.tensor([0.25, 0.5, 0.25], device=device)[None, None]
        self.floor = PASS_FLOOR_DBFS * math.log(10.0) / 10.0  # natural log of the floor power
        self.gain = torch.zeros(self.size, device=device)  # the last g(t), for tests / stats
        # frames that passed / frames looked at, since the last reset (stats; no host sync:
        # read only after a run)
        self.frames_on = torch.zeros((), device=device)
        self.frames_seen = torch.zeros((), device=device)

    def reset_stats(self) -> None:
        self.frames_on.zero_()
        self.frames_seen.zero_()

    def set_mix(self, mix: float) -> None:
        self.mix.fill_(max(0.0, mix))

    def run(self, audio: Tensor, reference: Tensor, pitchf: Tensor) -> None:
        """``pitchf``: the decoded frames' pitch, ending with the tail's last frame (at
        least ``frames`` long; earlier frames give the breath its unvoiced run)."""
        n = self.frames
        frames = reference[:n * self.zc].view(n, self.zc) * self.window
        power = torch.fft.rfft(frames).abs().square()
        energy = power.mean(-1) + 1e-12
        centroid = (power * self.freqs).sum(-1) / power.sum(-1).clamp(min=1e-20)
        r = PASS_BURST_REACH
        padded = torch.nn.functional.pad(energy[None, None], (r, r), mode="replicate")[0, 0]
        burst = energy > PASS_BURST_RATIO * torch.maximum(padded[:n], padded[2 * r:])
        loud = torch.log(energy * (8.0 / 3.0) / self.zc) > self.floor  # Parseval, Hann: 3/8
        voiced = pitchf >= 1.0
        idx = torch.arange(pitchf.shape[-1], device=pitchf.device)
        last_voiced = torch.cummax(torch.where(voiced, idx, -10 ** 6), dim=-1).values
        quiet_run = (idx - last_voiced)[-n:]
        unvoiced = ~voiced[-n:]
        breath = (quiet_run >= PASS_BREATH_MIN_FRAMES) & (centroid < PASS_BREATH_CENTROID_HZ)
        sibilant = ~burst & (centroid > PASS_SIBILANT_CENTROID_HZ)
        take = (unvoiced & loud & (burst | breath) & ~sibilant).to(audio.dtype)
        self.frames_on.add_(take.sum())
        self.frames_seen.add_(float(n))
        g = torch.nn.functional.conv1d(
            torch.nn.functional.pad(take[None, None], (1, 1), mode="replicate"), self.smooth)
        g = torch.nn.functional.interpolate(g, size=self.size, mode="linear",
                                            align_corners=False)[0, 0]
        self.gain.copy_(g)
        band = torch.nn.functional.conv1d(reference[None, None], self.taps,
                                          padding=PASS_TAPS // 2)[0, 0]
        audio.add_(self.mix * g * band)


class PostChain:
    """Ordered, individually skippable stages. A disabled stage costs nothing."""

    def __init__(self, plan: ChunkPlan, device: torch.device | str) -> None:
        self.rms_mix = RmsMix(plan, device)
        self.rate = 1.0
        self.passthrough = Passthrough(plan, device)
        self.passthrough_on = False  # static: toggling changes the op sequence

    def set_volume_envelope(self, rate: float) -> None:
        self.rate = float(rate)
        self.rms_mix.set_rate(self.rate)

    def set_passthrough(self, on: bool, mix: float) -> None:
        self.passthrough_on = bool(on)
        self.passthrough.set_mix(mix)

    @property
    def needs_reference(self) -> bool:
        return self.rate < 1.0 or self.passthrough_on

    def run(self, audio: Tensor, reference: Tensor, pitchf: Tensor | None = None) -> None:
        if self.rate < 1.0:
            self.rms_mix.run(audio, reference)
        if self.passthrough_on and pitchf is not None:
            self.passthrough.run(audio, reference, pitchf)
