"""The PortAudio input/output streams. Callbacks do memcpy + metering only (P1).

Two *separate* streams (docs/research/03 §8): the microphone and the render device are
distinct clock domains, so a full-duplex stream would force one to the other's clock.
Each callback does exactly one thing with the audio: copy it into or out of a ring. The
input callback also raises ``hop_event`` when a whole hop is waiting. Status flags are
counted, never ignored (docs/research/04 §5.2).

``FileInput`` stands in for the microphone: a thread that feeds a file into the input ring
at the real-time rate in callback-sized blocks, stamping when each block entered. It is
how ``scripts/measure_rtt.py`` injects a known signal.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from vox.audio.devices import WASAPI, DeviceInfo
from vox.audio.rings import SPSCRing
from vox.audio.volume import RampedGain


@dataclass
class StreamConfig:
    input: DeviceInfo | None
    output: DeviceInfo
    sr: int = 48000
    callback_ms: float = 10.0  # device block: transport granularity, not the model hop
    exclusive: bool = False  # WASAPI exclusive mode for the OUTPUT stream
    exclusive_input: bool = False  # ...and for the input; see DEVIATIONS R9 before using it
    latency: str | float = "low"

    @property
    def callback_block(self) -> int:
        return int(round(self.sr * self.callback_ms / 1000.0))


@dataclass
class StreamHealth:
    input_overflows: int = 0  # PortAudio status flags
    output_underflows: int = 0
    input_callbacks: int = 0
    output_callbacks: int = 0
    input_latency_ms: float = 0.0  # reported by PortAudio once the streams run
    output_latency_ms: float = 0.0
    input_peak: float = 0.0  # meter, reset by whoever reads it
    output_peak: float = 0.0
    input_samples: int = 0  # for the delivered-rate check (DEVIATIONS R9)
    input_started: float = 0.0
    errors: list[str] = field(default_factory=list)


class OutputTap:
    """What the output callback handed the device, sample for sample, plus a mask of the
    samples that came out of the ring (0 = zero-filled on an underrun).

    Preallocated for ``seconds``; once full it stops recording. A zero run in ``audio``
    whose mask is 1 came from the worker (a gated hop); one whose mask is 0 is an underrun.
    """

    def __init__(self, sr: int, seconds: float) -> None:
        self.sr = sr
        self.audio = np.zeros(int(sr * seconds), dtype=np.float32)
        self.from_ring = np.zeros(len(self.audio), dtype=np.uint8)
        self.pos = 0

    def write(self, block: np.ndarray, real: int) -> None:
        n = min(len(block), len(self.audio) - self.pos)
        if n <= 0:
            return
        self.audio[self.pos:self.pos + n] = block[:n]
        self.from_ring[self.pos:self.pos + min(real, n)] = 1
        self.pos += n


class DuplexAudio:
    """Owns the two PortAudio streams; the rings and the hop event belong to the caller."""

    def __init__(self, cfg: StreamConfig, in_ring: SPSCRing, out_ring: SPSCRing,
                 hop_samples: int, hop_event: threading.Event,
                 tap: OutputTap | None = None) -> None:
        self.cfg = cfg
        self.in_ring = in_ring
        self.out_ring = out_ring
        self.hop = hop_samples
        self.hop_event = hop_event
        self.tap = tap
        self.health = StreamHealth()
        self._in_stream = None
        self._out_stream = None
        n = 4 * cfg.callback_block  # PortAudio may hand over larger blocks in shared mode
        self._mono_in = np.zeros(n, dtype=np.float32)
        self._mono_out = np.zeros(n, dtype=np.float32)
        # Output volume (``volume.target``, linear, set from the control path): the last
        # thing before the device, so the converted voice and typed clips move together and
        # the OUT meter and the output tap see what plays. Ramped, so a slider never clicks.
        self.volume = RampedGain.for_rate(cfg.sr)
        self.in_channels = 0
        self.out_channels = 0

    # ------------------------------------------------------------------ callbacks (RT)

    def _input_callback(self, indata, frames, time_info, status) -> None:
        if status:
            if status.input_overflow:
                self.health.input_overflows += 1
        mono = self._mono_in[:frames]
        if indata.shape[1] == 1:
            mono[:] = indata[:, 0]
        else:
            np.mean(indata, axis=1, out=mono)
        self.in_ring.write(mono)
        peak = float(np.abs(mono).max()) if frames else 0.0
        if peak > self.health.input_peak:
            self.health.input_peak = peak
        if self.health.input_callbacks == 0:
            self.health.input_started = time.perf_counter()
        self.health.input_callbacks += 1
        self.health.input_samples += frames
        if self.in_ring.available() >= self.hop:
            self.hop_event.set()

    def _output_callback(self, outdata, frames, time_info, status) -> None:
        if status:
            if status.output_underflow:
                self.health.output_underflows += 1
        mono = self._mono_out[:frames]
        real = self.out_ring.read(mono)
        volume = self.volume
        if not volume.unity:
            volume.apply(mono)
            if volume.now > 1.0 or volume.target > 1.0:
                np.clip(mono, -1.0, 1.0, out=mono)
        outdata[:] = mono[:, None]  # same signal on every channel
        if self.tap is not None:
            self.tap.write(mono, real)
        peak = float(np.abs(mono).max()) if frames else 0.0
        if peak > self.health.output_peak:
            self.health.output_peak = peak
        self.health.output_callbacks += 1

    # ---------------------------------------------------------------- control path

    def _extra(self, device: DeviceInfo, exclusive: bool):
        import sounddevice as sd

        if device.hostapi == WASAPI:
            return sd.WasapiSettings(exclusive=exclusive, auto_convert=not exclusive)
        return None

    def start_input(self) -> None:
        import sounddevice as sd

        cfg = self.cfg
        if cfg.input is None:
            return
        self.in_channels = min(2, cfg.input.max_input_channels)
        self._in_stream = sd.InputStream(
            device=cfg.input.index, samplerate=cfg.sr, blocksize=cfg.callback_block,
            channels=self.in_channels, dtype="float32", latency=cfg.latency,
            extra_settings=self._extra(cfg.input, cfg.exclusive_input),
            callback=self._input_callback)
        self._in_stream.start()
        self.health.input_latency_ms = 1000.0 * float(self._in_stream.latency)

    def start_output(self) -> None:
        """Started once the first hop is in the output ring, so the callback never starts
        on an empty ring (that would be a counted underrun before anything ran)."""
        import sounddevice as sd

        cfg = self.cfg
        self.out_channels = min(2, cfg.output.max_output_channels)
        self._out_stream = sd.OutputStream(
            device=cfg.output.index, samplerate=cfg.sr, blocksize=cfg.callback_block,
            channels=self.out_channels, dtype="float32", latency=cfg.latency,
            extra_settings=self._extra(cfg.output, cfg.exclusive),
            callback=self._output_callback)
        self._out_stream.start()
        self.health.output_latency_ms = 1000.0 * float(self._out_stream.latency)

    def start(self) -> None:
        self.start_input()
        self.start_output()

    def stop(self) -> None:
        for stream in (self._in_stream, self._out_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception as err:  # noqa: BLE001
                    self.health.errors.append(str(err))
        self._in_stream = self._out_stream = None

    @property
    def running(self) -> bool:
        return self._out_stream is not None and self._out_stream.active

    def input_rate_hz(self) -> float:
        """Samples per second the input actually delivers (nominal: ``cfg.sr``)."""
        h = self.health
        elapsed = time.perf_counter() - h.input_started
        return h.input_samples / elapsed if h.input_callbacks > 10 and elapsed > 0.5 else 0.0


class FileInput:
    """A software microphone: feeds ``audio`` into the input ring at real-time pace.

    Blocks of ``callback_block`` samples are written every ``callback_block / sr`` seconds
    on a thread; ``entered`` records ``perf_counter`` when each block went in, so a marker
    at a known sample position has a known entry time. Loops the audio when ``loop``.
    """

    def __init__(self, audio: np.ndarray, sr: int, callback_block: int, in_ring: SPSCRing,
                 hop_samples: int, hop_event: threading.Event, loop: bool = False) -> None:
        self.audio = np.ascontiguousarray(audio, dtype=np.float32)
        self.sr = sr
        self.block = callback_block
        self.in_ring = in_ring
        self.hop = hop_samples
        self.hop_event = hop_event
        self.loop = loop
        self.entered: list[tuple[int, float]] = []  # (sample position, perf_counter)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vox-file-input", daemon=True)
        self.position = 0
        self.health = StreamHealth()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        period = self.block / self.sr
        start = time.perf_counter()
        n = 0
        total = len(self.audio)
        zeros = np.zeros(self.block, dtype=np.float32)
        while not self._stop.is_set():
            due = start + (n + 1) * period
            while (remaining := due - time.perf_counter()) > 0:
                time.sleep(min(remaining, 0.002))
            pos = self.position
            if pos + self.block <= total:
                block = self.audio[pos:pos + self.block]
            elif self.loop and total >= self.block:
                self.position = 0
                pos = 0
                block = self.audio[:self.block]
            else:
                block = zeros
            self.entered.append((pos, time.perf_counter()))
            self.in_ring.write(block)
            self.position = pos + self.block if block is not zeros else pos
            self.health.input_callbacks += 1
            if self.in_ring.available() >= self.hop:
                self.hop_event.set()
            n += 1
