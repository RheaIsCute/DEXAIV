"""Profile and parameter schemas (docs/research/08 §5, 12 §3.10).

Plain dataclasses that round-trip through JSON. Two parameter classes make principle P10
explicit in types: ``RuntimeParams`` may change between hops, ``ShapeParams`` change buffer
shapes and therefore need a controlled stop -> realloc -> restart.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, TypeVar, get_type_hints

SCHEMA_VERSION = 1

T = TypeVar("T")


def to_dict(obj: Any) -> dict[str, Any]:
    return dataclasses.asdict(obj)


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build ``cls`` from a JSON dict. Unknown keys are ignored so newer files still load."""
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        value = data[f.name]
        target = hints[f.name]
        if dataclasses.is_dataclass(target) and isinstance(value, dict):
            value = from_dict(target, value)  # type: ignore[arg-type]
        kwargs[f.name] = value
    return cls(**kwargs)


# --------------------------------------------------------------------------- engine parameters


@dataclass(frozen=True)
class ShapeParams:
    """Restart-class parameters: every one of them changes a buffer shape."""

    block_ms: float = 20.0
    crossfade_ms: float = 20.0  # <= block_ms (the chunker clamps it)
    sola_search_ms: float = 10.0
    extra_ms: float = 1000.0  # 500 -> 1000: pause loudness bias (docs/DEVIATIONS.md R4)
    device_sr: int = 48000
    decoder_context_ms: float = 80.0  # decoder left context (0 = upstream); 8.8, Q17
    # A/B settings of the live-quality work (docs/DEVIATIONS.md §7), off by default:
    overlap_beyond_hop: bool = False  # pre-fix SOLA geometry: the crossfade may exceed the hop
    embed_pad_ms: float = 0.0  # replicate-pad the window's right edge before ContentVec
    # Gate onset (docs/DEVIATIONS.md §8): the first hop after the gate opens fades in over
    # this many ms instead of the whole overlap, so the decoded pre-onset consonant is kept.
    # 0 = the pre-fix fade (the full overlap from a silent SOLA tail).
    onset_ramp_ms: float = 5.0


@dataclass
class RuntimeParams:
    """Hot-reconfigurable parameters, applied between hops."""

    f0_up_key: float = 0.0
    index_rate: float = 0.0
    protect: float = 0.33
    volume_envelope: float = 1.0
    f0_method: str = "rmvpe"  # changing the extractor changes its tail size: restart-class
    f0_min_hz: float = 50.0
    f0_max_hz: float = 1100.0
    f0_unvoiced: str = "zero"  # "zero" (classic/Applio contract) | "interp" (current rvc-webui)
    sid: int = 0
    sola_mode: str = "plain"  # "plain" | "pv" (phase-vocoder crossfade, open question Q7)
    # A/B setting of the live-quality work (docs/DEVIATIONS.md §7), off by default:
    f0_bridge_ms: float = 0.0  # fill F0 = 0 runs this short between voiced runs (0 = off)
    noise_roll: bool = True  # decoder noise follows the window (on since 8.8; False = upstream)
    # Singing (docs/DEVIATIONS.md §8):
    f0_ceiling_hz: float = 0.0  # soft-knee F0 ceiling after transposition (0 = off); hot
    # Fresh RMVPE frames dropped at the right edge: 2 (default) halves the pitch step at
    # hop seams on held notes; 1 = upstream (docs/DEVIATIONS.md §8.4)
    f0_trim_tail: int = 2
    sine_phase_carry: bool = True  # NSF sine phase continuous across hops (8.8)


@dataclass(frozen=True)
class PrecisionParams:
    """Set once before warmup; never toggled while running (docs/research/06 §5)."""

    tf32: bool = True
    hubert_fp16: bool = False  # legacy: True forces hubert_autocast = "fp16"
    decoder_fp16: bool = False
    graphs: str = "auto"  # CUDA graphs: "on" | "off" | "auto" (capture, verify, else eager)
    # HuBERT autocast: "auto" tries fp16 then bf16 at warmup and keeps the first that
    # matches the FP32 eager render (TARGET.md rule 2); "fp16" | "bf16" force; "off".
    hubert_autocast: str = "auto"

    @property
    def hubert_candidates(self) -> tuple[str, ...]:
        """Autocast modes to try, in order; the first that passes the eager-match wins."""
        if self.hubert_fp16:
            return ("fp16",)
        return ("fp16", "bf16") if self.hubert_autocast == "auto" else (self.hubert_autocast,)


# ----------------------------------------------------------------------------- profile files


@dataclass
class SourceVoice:
    f0_median_hz: float = 0.0
    f0_p10_hz: float = 0.0
    f0_p90_hz: float = 0.0
    f0_p01_hz: float = 0.0
    f0_p99_hz: float = 0.0
    voiced_ratio: float = 0.0
    spectral_tilt_db_per_oct: float = 0.0


@dataclass
class F0ExtractorProfile:
    # None = not calibrated: the engine default (RuntimeParams) applies.
    method: str | None = None
    f0_min_hz: float | None = None
    f0_max_hz: float | None = None
    median_filter_frames: int = 0
    octave_flip_rate: dict[str, float] = field(default_factory=dict)


@dataclass
class VoiceProfile:
    schema: int = SCHEMA_VERSION
    created: str = ""
    source_voice: SourceVoice = field(default_factory=SourceVoice)
    f0_extractor: F0ExtractorProfile = field(default_factory=F0ExtractorProfile)


@dataclass
class VadProfile:
    engine: str = "silero"
    threshold: float = 0.5
    release_ms: int = 350  # hangover after the last speech (Applio: 200; 350 bridges mid-phrase dips)


@dataclass
class CalibrationProfile:
    schema: int = SCHEMA_VERSION
    mic_fingerprint: str = ""
    noise_floor_dbfs: float = -60.0
    speech_rms_dbfs: dict[str, float] = field(default_factory=dict)
    clipping_detected: bool = False
    input_gain_db: float = 0.0
    vad: VadProfile = field(default_factory=VadProfile)
    silence_gate_dbfs: float = -48.0
    rnnoise_enabled: bool = False
    raw_recordings: str = ""


@dataclass
class DeviceEndpoint:
    name: str = ""
    api: str = "WASAPI"
    exclusive: bool = False
    sr: int = 48000
    channels: int = 1
    enabled: bool = True


@dataclass
class AudioDeviceProfile:
    schema: int = SCHEMA_VERSION
    input: DeviceEndpoint = field(default_factory=DeviceEndpoint)
    output: DeviceEndpoint = field(default_factory=DeviceEndpoint)
    monitor: DeviceEndpoint = field(default_factory=lambda: DeviceEndpoint(enabled=False))


@dataclass
class RuntimeProfile:
    """Per-GPU scheduling choices, written by the benchmark (docs/research/10).

    Every shape and precision field is an override: None means "the engine default"
    (``config/defaults.py``) and is never written to disk. A file therefore holds only what
    someone chose, and a later change of an engine default is not shadowed by a stale copy
    of the old one (docs/DEVIATIONS.md §8).
    """

    schema: int = SCHEMA_VERSION
    gpu: str = ""
    block_ms: float | None = None
    crossfade_ms: float | None = None
    context_extra_ms: float | None = None
    sola_search_ms: float | None = None
    decoder_context_ms: float | None = None
    tf32: bool | None = None
    hubert_fp16: bool | None = None
    decoder_fp16: bool | None = None
    graphs: str | None = None
    hubert_autocast: str | None = None
    benchmark_ref: str = ""
    # Persisted warmup decisions keyed by the fixed shape (TARGET.md rule 4): the HuBERT
    # autocast that passed the eager-match and whether cuDNN autotune beat its heuristics,
    # so startup applies them instead of re-measuring. Written by the engine at warmup.
    tuned: dict[str, dict[str, Any]] = field(default_factory=dict)

    def shape(self, device_sr: int = 48000) -> ShapeParams:
        from vox.config.defaults import DEFAULT_SHAPE

        chosen = {"block_ms": self.block_ms, "crossfade_ms": self.crossfade_ms,
                  "sola_search_ms": self.sola_search_ms, "extra_ms": self.context_extra_ms,
                  "decoder_context_ms": self.decoder_context_ms}
        return dataclasses.replace(DEFAULT_SHAPE, device_sr=device_sr,
                                   **{k: v for k, v in chosen.items() if v is not None})

    def precision(self) -> PrecisionParams:
        from vox.config.defaults import DEFAULT_PRECISION

        chosen = {"tf32": self.tf32, "hubert_fp16": self.hubert_fp16,
                  "decoder_fp16": self.decoder_fp16, "graphs": self.graphs,
                  "hubert_autocast": self.hubert_autocast}
        return dataclasses.replace(DEFAULT_PRECISION,
                                   **{k: v for k, v in chosen.items() if v is not None})


@dataclass
class ModelOverrides:
    """Everything that depends on the (user, model) pair (docs/research/08 §7)."""

    schema: int = SCHEMA_VERSION
    default_model: str = ""
    models: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class IndexInfo:
    file: str = ""
    dim: int = 0
    ntotal: int = 0


@dataclass
class ModelConfig:
    """Per-model ``models/<Name>/config.json`` (docs/research/09 §6)."""

    schema: int = SCHEMA_VERSION
    name: str = ""
    source_file: str = ""
    sha256: str = ""
    version: str = "v2"
    sr: int = 48000
    f0: bool = True
    n_spk: int = 1
    sid: int = 0
    embedder: str = "contentvec"
    emb_dim: int = 768
    emb_output_layer: int = 12
    use_final_proj: bool = False
    vocoder: str = "HiFi-GAN"
    index: IndexInfo = field(default_factory=IndexInfo)
    defaults: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    imported_at: str = ""
    notes: str = ""


RUNTIME_PARAM_FIELDS = frozenset(f.name for f in dataclasses.fields(RuntimeParams))
