"""WAV in -> converted WAV out (docs/research/12 §3.11).

Two modes share the same loaded models:

* ``full`` - the canonical whole-file RVC inference (rvc-webui ``Pipeline.pipeline``):
  high-pass, reflect padding, segmentation at quiet points, full-context F0. This is the
  quality reference the streaming engine is measured against.
* ``stream`` - the realtime engine driven hop by hop by the stream simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from vox.audio.resample import load_audio, resample_array, save_audio
from vox.config.schemas import PrecisionParams, RuntimeParams, ShapeParams
from vox.engine.pipeline import protect_blend
from vox.features.base import Embedder
from vox.offline.simulate import Engine, StreamSimulator, load_engine
from vox.pitch.base import PitchExtractor, postprocess_f0, semitones_to_ratio
from vox.retrieval import gpu_knn
from vox.retrieval.gpu_knn import GpuIndex
from vox.synthesis.rvc_model import RvcModel

SR = 16000
WINDOW = 160


@dataclass(frozen=True)
class SegmentPlan:
    """Padding / segmentation of the canonical offline pipeline, in seconds."""

    x_pad: float = 1.0  # reflect padding each side of a segment
    x_query: float = 6.0  # search radius for a quiet cut point
    x_center: float = 38.0  # nominal segment length
    x_max: float = 41.0  # files shorter than this are converted in one piece


class FullFileConverter:
    def __init__(self, model: RvcModel, embedder: Embedder,
                 extractor: PitchExtractor | None, device: torch.device,
                 index: GpuIndex | None = None, segments: SegmentPlan = SegmentPlan()) -> None:
        self.model = model
        self.embedder = embedder
        self.extractor = extractor if model.has_f0 else None
        self.device = device
        self.index = index
        self.seg = segments

    @classmethod
    def from_engine(cls, engine: Engine, **kwargs) -> FullFileConverter:
        """Reuse the models an engine already loaded."""
        p = engine.pipeline
        return cls(p.model, p.embedder, p.extractor, engine.device, p.index, **kwargs)

    @torch.inference_mode()
    def convert(self, audio_16k: np.ndarray, params: RuntimeParams,
                rms_frame_ms: float | None = None, f0_contour_scale: float = 1.0) -> np.ndarray:
        """Mono 16 kHz in -> mono float32 at the model sample rate.

        Typed-speech options (docs/DEVIATIONS.md §7, Step 3; the defaults are upstream):
        ``rms_frame_ms`` sets the loudness-envelope frames of the ``volume_envelope`` mix
        (None: rvc-webui's 1 s frames at a 0.5 s hop; a value: that frame at a 10 ms hop).
        ``f0_contour_scale`` widens (> 1) the voiced F0 contour around the utterance's
        median in semitones before synthesis.
        """
        from scipy import signal

        seg = self.seg
        t_pad = int(SR * seg.x_pad)
        t_pad_tgt = int(self.model.tgt_sr * seg.x_pad)
        peak = float(np.abs(audio_16k).max()) / 0.95
        audio = (audio_16k / peak if peak > 1 else audio_16k).astype(np.float64)
        bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=SR)
        audio = signal.filtfilt(bh, ah, audio).astype(np.float32)

        cuts = self._cut_points(audio)
        padded = np.pad(audio, (t_pad, t_pad), mode="reflect")
        wave = torch.from_numpy(padded).to(self.device)
        pitch = pitchf = None
        if self.extractor is not None:
            p_len = len(padded) // WINDOW
            coarse, cont = postprocess_f0(
                self.extractor.extract(wave), semitones_to_ratio(params.f0_up_key),
                params.f0_min_hz, params.f0_max_hz, params.f0_unvoiced == "interp",
                round(params.f0_bridge_ms / 10.0), params.f0_ceiling_hz)
            if f0_contour_scale != 1.0:
                coarse, cont = scale_contour(cont, f0_contour_scale, t_pad // WINDOW)
            pitch, pitchf = coarse[:p_len].unsqueeze(0), cont[:p_len].unsqueeze(0)

        pieces: list[torch.Tensor] = []
        start = 0
        for cut in cuts:
            cut = cut // WINDOW * WINDOW
            end = cut + 2 * t_pad
            pieces.append(self._segment(wave[start:end + WINDOW], pitch, pitchf,
                                        start // WINDOW, end // WINDOW, params))
            start = cut
        pieces.append(self._segment(wave[start:], pitch, pitchf, start // WINDOW, None, params))
        out = torch.cat([piece[t_pad_tgt:-t_pad_tgt] for piece in pieces]).cpu().numpy()

        if params.volume_envelope != 1:
            out = _change_rms(audio, SR, out, self.model.tgt_sr, params.volume_envelope,
                              rms_frame_ms)
        peak = float(np.abs(out).max()) / 0.99
        return (out / peak if peak > 1 else out).astype(np.float32)

    def _cut_points(self, audio: np.ndarray) -> list[int]:
        """Cut long audio at the quietest sample near every ``x_center`` seconds."""
        seg = self.seg
        if len(audio) + WINDOW <= int(SR * seg.x_max):
            return []
        t_center, t_query = int(SR * seg.x_center), int(SR * seg.x_query)
        padded = np.abs(np.pad(audio, (WINDOW // 2, WINDOW // 2), mode="reflect"))
        csum = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
        energy = csum[WINDOW:WINDOW + len(audio)] - csum[:len(audio)]
        cuts = []
        for t in range(t_center, len(audio), t_center):
            lo = t - t_query
            cuts.append(lo + int(np.argmin(energy[lo:t + t_query])))
        return cuts

    def _segment(self, wave: torch.Tensor, pitch, pitchf, f_start: int, f_end: int | None,
                 params: RuntimeParams) -> torch.Tensor:
        feats = self.embedder.embed(wave.unsqueeze(0))
        if pitch is not None:
            pitch, pitchf = pitch[:, f_start:f_end], pitchf[:, f_start:f_end]
        use_index = self.index is not None and params.index_rate > 0
        use_protect = use_index and pitch is not None and params.protect < 0.5
        feats0 = feats.clone() if use_protect else None
        if use_index:
            gpu_knn.blend(feats, self.index, params.index_rate)

        def to_100hz(x: torch.Tensor) -> torch.Tensor:
            return F.interpolate(x.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        feats = to_100hz(feats)
        p_len = min(wave.shape[0] // WINDOW, feats.shape[1])
        feats = feats[:, :p_len]
        if pitch is not None:
            pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
        if use_protect:
            feats = protect_blend(feats, to_100hz(feats0)[:, :p_len], pitchf, params.protect)
        lengths = torch.tensor([p_len], dtype=torch.long, device=self.device)
        sid = torch.tensor([params.sid], dtype=torch.long, device=self.device)
        return self.model.infer(feats, lengths, pitch, pitchf, sid)


def _rms_frames(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    padded = np.pad(audio.astype(np.float64), (frame // 2, frame // 2))
    csum = np.concatenate(([0.0], np.cumsum(padded * padded)))
    starts = np.arange(0, len(padded) - frame + 1, hop)
    return np.sqrt((csum[starts + frame] - csum[starts]) / frame)


def _change_rms(source: np.ndarray, sr_in: int, out: np.ndarray, sr_out: int,
                rate: float, frame_ms: float | None = None) -> np.ndarray:
    """Offline loudness-envelope mix, ``rate`` = share of the output's own envelope.

    Frames: rvc-webui's 1 s at a 0.5 s hop by default (only the phrase-level loudness
    follows the source); with ``frame_ms`` that frame at a 10 ms hop, fine enough to carry
    syllable stress across (the typed-speech option)."""
    def envelope(x: np.ndarray, sr: int) -> np.ndarray:
        if frame_ms is None:
            frame, hop = sr // 2 * 2, sr // 2
        else:
            frame, hop = max(2, round(sr * frame_ms / 1000.0)), max(1, sr // 100)
        rms = _rms_frames(x, frame, hop)
        return np.interp(np.linspace(0, len(rms) - 1, len(out)), np.arange(len(rms)), rms)

    rms_in = envelope(source, sr_in)
    floor = 1e-6 if frame_ms is None else 1e-4  # short frames reach true silence
    rms_out = np.maximum(envelope(out, sr_out), floor)
    return out * np.power(rms_in, 1 - rate) * np.power(rms_out, rate - 1)


def scale_contour(pitchf: torch.Tensor, scale: float, edge: int = 0
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Widen (``scale`` > 1) or narrow the voiced F0 around its median in semitone space:
    ``f -> median * (f / median) ** scale``. The median is taken over voiced frames away
    from ``edge`` frames of padding at each end. Unvoiced frames stay 0. Returns
    ``(coarse, pitchf)`` (control path: the median syncs with the host)."""
    from vox.pitch.base import f0_to_coarse

    voiced = pitchf > 0
    core = voiced.clone()
    if edge:
        core[:edge] = False
        core[-edge:] = False
    if not bool(core.any()):
        return f0_to_coarse(pitchf), pitchf
    median = pitchf[core].median()
    scaled = torch.where(voiced, median * (pitchf.clamp(min=1.0) / median) ** scale, pitchf)
    return f0_to_coarse(scaled), scaled


def render_full(engine: Engine, audio: np.ndarray, seed: int | None = 0,
                params: RuntimeParams | None = None) -> np.ndarray:
    """Full-file reference render of device-rate ``audio``, returned at the device rate.

    The quality yardstick for the streaming engine: same models, same parameters, whole-file
    context. Output is trimmed/padded to the input length so the two share a timeline.
    """
    plan = engine.plan
    if seed is not None:
        torch.manual_seed(seed)
    out = FullFileConverter.from_engine(engine).convert(
        resample_array(audio, plan.device_sr, SR), params or engine.params)
    out = resample_array(out, engine.loaded.model.tgt_sr, plan.device_sr)
    fitted = np.zeros(len(audio), dtype=np.float32)
    fitted[:min(len(out), len(audio))] = out[:len(audio)]
    return fitted


def convert_file(src: Path, dst: Path, models_dir: Path, model_name: str, mode: str = "stream",
                 shape: ShapeParams | None = None, params: RuntimeParams | None = None,
                 precision: PrecisionParams | None = None, device: str | None = None,
                 out_sr: int | None = None, gate=None, profile_dir: Path | None = None,
                 conditioning=None) -> dict:
    """Convert one file. Returns a small summary dict (and the engine, for reporting).

    ``stream`` mode runs the live microphone path, input conditioning included
    (``conditioning``; default on); ``full`` mode never conditions."""
    engine = load_engine(models_dir, model_name, shape, params, precision, device,
                         warmup_hops=5 if mode == "stream" else 0, gate=gate,
                         profile_dir=profile_dir, conditioning=conditioning)
    summary: dict = {"mode": mode, "model": model_name, "engine": engine}
    if mode == "full":
        audio, _ = load_audio(src, SR)
        out = FullFileConverter.from_engine(engine).convert(audio, engine.params)
        sr = engine.loaded.model.tgt_sr
    elif mode == "stream":
        audio, _ = load_audio(src, engine.plan.device_sr)
        result = StreamSimulator(engine).run(audio, gate=gate is not None and gate.enabled)
        out, sr = result.aligned[:len(audio)], engine.plan.device_sr
        summary["result"] = result
    else:
        raise ValueError(f"mode must be 'full' or 'stream', not {mode!r}")
    if out_sr and out_sr != sr:
        out, sr = resample_array(out, sr, out_sr), out_sr
    save_audio(dst, out, sr)
    summary.update(seconds=len(out) / sr, sr=sr, path=Path(dst))
    return summary
