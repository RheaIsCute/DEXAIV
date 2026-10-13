"""Device enumeration and selection via sounddevice (docs/research/12 §3.1, 03 §8).

WASAPI is the host API of record on Windows: shared mode by default, exclusive behind a
flag. Devices are addressed by their PortAudio index (``vox devices``) or by a
case-insensitive name fragment, resolved within the chosen host API. Virtual cables
(VB-Cable, Virtual Audio Cable) are recognised by name so the rate check can warn when the
cable is not pinned at the engine rate.
"""

from __future__ import annotations

from dataclasses import dataclass

WASAPI = "Windows WASAPI"
CABLE_MARKERS = ("cable", "vb-audio", "virtual audio")


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    low_input_latency_s: float
    low_output_latency_s: float
    default_input: bool = False
    default_output: bool = False

    @property
    def is_cable(self) -> bool:
        return any(m in self.name.lower() for m in CABLE_MARKERS)


def _hostapi_name(index: int) -> str:
    import sounddevice as sd

    return sd.query_hostapis(index)["name"]


def list_devices(hostapi: str | None = WASAPI) -> list[DeviceInfo]:
    """All devices, or only those of one host API (default WASAPI)."""
    import sounddevice as sd

    out: list[DeviceInfo] = []
    defaults = {}
    for api in sd.query_hostapis():
        defaults[api["name"]] = (api["default_input_device"], api["default_output_device"])
    for i, d in enumerate(sd.query_devices()):
        api = _hostapi_name(d["hostapi"])
        if hostapi is not None and api != hostapi:
            continue
        din, dout = defaults.get(api, (-1, -1))
        out.append(DeviceInfo(
            index=i, name=d["name"], hostapi=api,
            max_input_channels=d["max_input_channels"],
            max_output_channels=d["max_output_channels"],
            default_samplerate=d["default_samplerate"],
            low_input_latency_s=d["default_low_input_latency"],
            low_output_latency_s=d["default_low_output_latency"],
            default_input=i == din, default_output=i == dout))
    return out


def resolve_device(spec: str | int | None, kind: str, hostapi: str | None = WASAPI
                   ) -> DeviceInfo:
    """``spec``: PortAudio index, name fragment, or None for the host API default.

    ``kind`` is ``"input"`` or ``"output"``; the device must have channels of that kind.
    """
    devices = list_devices(hostapi)
    want = "max_input_channels" if kind == "input" else "max_output_channels"
    usable = [d for d in devices if getattr(d, want) > 0]
    if spec is None or spec == "":
        for d in usable:
            if (d.default_input if kind == "input" else d.default_output):
                return d
        if usable:
            return usable[0]
        raise LookupError(f"no {kind} device under {hostapi}")
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        index = int(spec)
        for d in list_devices(None):
            if d.index == index:
                if getattr(d, want) <= 0:
                    raise LookupError(f"device {index} ({d.name}) has no {kind} channels")
                return d
        raise LookupError(f"no device with index {index}")
    needle = str(spec).lower()
    matches = [d for d in usable if needle in d.name.lower()]
    if not matches:
        raise LookupError(f"no {kind} device under {hostapi} matching {spec!r}")
    return matches[0]


def check_rate(device: DeviceInfo, sr: int, kind: str) -> str | None:
    """A warning string when the device cannot run at ``sr`` as asked, else None."""
    import sounddevice as sd

    try:
        if kind == "input":
            sd.check_input_settings(device=device.index, samplerate=sr, dtype="float32")
        else:
            sd.check_output_settings(device=device.index, samplerate=sr, dtype="float32")
    except Exception as err:  # noqa: BLE001 - PortAudio raises its own classes
        return f"{device.name}: {sr} Hz not accepted ({err}); shared mode will convert"
    if device.is_cable and int(device.default_samplerate) != sr:
        return (f"{device.name} reports {device.default_samplerate:.0f} Hz; pin the cable to "
                f"{sr} Hz in Windows sound settings to avoid a hidden resampler")
    return None


def format_table(devices: list[DeviceInfo]) -> str:
    lines = [f"{'idx':>4} {'in':>3} {'out':>3} {'rate':>6} {'lat in/out ms':>14}  name  [hostapi]"]
    for d in devices:
        flags = ("*" if d.default_input else " ") + ("*" if d.default_output else " ")
        lines.append(
            f"{d.index:>4} {d.max_input_channels:>3} {d.max_output_channels:>3} "
            f"{d.default_samplerate:>6.0f} {1000 * d.low_input_latency_s:>6.1f}/"
            f"{1000 * d.low_output_latency_s:<6.1f} {flags} {d.name}  [{d.hostapi}]"
            + ("  (virtual cable)" if d.is_cable else ""))
    return "\n".join(lines)
