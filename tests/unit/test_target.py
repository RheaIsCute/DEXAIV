"""TARGET.md rules at the control path: device assertion, cable-first output, Broadcast
suppression default, the autocast ladder's candidates and the persisted tuning record."""

from types import SimpleNamespace

import pytest
import torch

from vox.api.control import resolve_suppression
from vox.audio import devices
from vox.audio.devices import DeviceInfo
from vox.config import target
from vox.config.schemas import PrecisionParams, RuntimeProfile
from vox.config.store import load_as, load_tuned, save_tuned
from vox.config.target import TargetError

# ------------------------------------------------------------------- device assertion


def _props(name="NVIDIA GeForce RTX 5060 Ti", major=12, minor=0, gib=15.9):
    return SimpleNamespace(name=name, major=major, minor=minor, total_memory=int(gib * 2**30))


@pytest.fixture
def fake_cuda(monkeypatch):
    """A fake CUDA enumeration: ``set(props_list)`` installs the visible devices."""
    state = {"devices": [_props()], "cuda_version": "12.8"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(state["devices"]))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(state["devices"]))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: state["devices"][i])
    monkeypatch.setattr(torch.version, "cuda", state["cuda_version"])
    monkeypatch.delenv(target.CPU_ENV_FLAG, raising=False)
    return state


def test_the_target_gpu_is_selected_and_reported(fake_cuda):
    device, line = target.select_device(None)
    assert device == torch.device("cuda", 0)
    assert "hardware check" in line and "5060 Ti" in line and "OK" in line
    assert target.select_device("cuda")[0] == device
    assert target.select_device("cuda:0")[0] == device


def test_other_adapters_are_never_auto_picked(fake_cuda):
    fake_cuda["devices"] = [_props("AMD Radeon(TM) Graphics", 11, 0, 0.5), _props()]
    device, _ = target.select_device(None)
    assert device.index == 1  # the 5060 Ti, not the first enumerated device
    with pytest.raises(TargetError, match="not the supported GPU"):
        target.select_device("cuda:0")


@pytest.mark.parametrize("props, message", [
    (_props("NVIDIA GeForce RTX 4070"), "does not contain '5060 Ti'"),
    (_props(major=8, minor=9), "compute capability"),
    (_props(gib=8.0), "VRAM 8.0 GiB < 15.0"),
])
def test_wrong_hardware_is_a_hard_error(fake_cuda, props, message):
    fake_cuda["devices"] = [props]
    with pytest.raises(TargetError, match=message):
        target.select_device("cuda:0")


def test_wrong_torch_build_is_a_hard_error(fake_cuda, monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    with pytest.raises(TargetError, match="not 12.8"):
        target.select_device(None)


def test_no_cuda_is_a_hard_error_unless_cpu_is_allowed(fake_cuda, monkeypatch):
    fake_cuda["devices"] = []
    with pytest.raises(TargetError, match="no CUDA device"):
        target.select_device(None)
    monkeypatch.setenv(target.CPU_ENV_FLAG, "1")
    device, line = target.select_device(None)
    assert device.type == "cpu" and "unit-test" not in line and "cpu" in line


def test_cpu_is_for_unit_tests_only(fake_cuda, monkeypatch):
    with pytest.raises(TargetError, match="VOX_ALLOW_CPU=1"):
        target.select_device("cpu")
    monkeypatch.setenv(target.CPU_ENV_FLAG, "1")
    device, line = target.select_device("cpu")
    assert device.type == "cpu" and "unit-test path" in line


def test_index_cap_follows_the_budget():
    assert target.index_vector_cap(768) == (2 * 2**30) // (4 * 769)
    assert target.index_vector_cap(256) > target.index_vector_cap(768)
    assert target.index_vector_cap(768, budget_bytes=1) == 1


# -------------------------------------------------------------- audio routing defaults


def _dev(index, name, ins, outs, default_in=False, default_out=False):
    return DeviceInfo(index, name, devices.WASAPI, ins, outs, 48000.0, 0.003, 0.003,
                      default_in, default_out)


@pytest.fixture
def fake_devices(monkeypatch):
    listed = [
        _dev(18, "Speakers (NVIDIA Broadcast)", 0, 2),
        _dev(19, "Line 1 (Virtual Audio Cable)", 0, 2),
        _dev(20, "Realtek HD Audio 2nd output (Realtek(R) Audio)", 0, 2, default_out=True),
        _dev(22, "Microphone (Realtek(R) Audio)", 2, 0, default_in=True),
        _dev(23, "Microphone (NVIDIA Broadcast)", 2, 0),
        _dev(24, "Line 1 (Virtual Audio Cable)", 2, 0),
    ]
    monkeypatch.setattr(devices, "list_devices", lambda hostapi=devices.WASAPI: listed)
    return listed


def test_default_output_is_the_cable_and_default_input_the_microphone(fake_devices):
    assert devices.resolve_device(None, "output").index == 19
    assert devices.resolve_device(None, "input").index == 22
    assert devices.resolve_device("broadcast", "input").index == 23
    assert devices.resolve_device("Realtek", "output").index == 20


def test_default_output_falls_back_to_the_host_default_without_a_cable(fake_devices,
                                                                       monkeypatch):
    without = [d for d in fake_devices if not d.is_cable]
    monkeypatch.setattr(devices, "list_devices", lambda hostapi=devices.WASAPI: without)
    assert devices.resolve_device(None, "output").index == 20


@pytest.mark.parametrize("requested, profile, name, expected, why", [
    (None, True, "Microphone (Realtek(R) Audio)", True, "profile"),
    (None, False, "Microphone (Realtek(R) Audio)", False, "profile"),
    (None, True, "Microphone (NVIDIA Broadcast)", False, "Broadcast"),
    (True, True, "Microphone (NVIDIA Broadcast)", True, "explicit"),
    (False, False, "Microphone (Realtek(R) Audio)", False, "explicit"),
    (None, True, None, True, "profile"),  # file-backed input: no device name
])
def test_suppression_defaults_off_for_the_broadcast_mic(requested, profile, name, expected,
                                                        why):
    value, reason = resolve_suppression(requested, profile, name)
    assert value is expected and why in reason


# ------------------------------------------------------------ precision and persistence


def test_hubert_autocast_candidates():
    assert PrecisionParams().hubert_candidates == ("fp16", "bf16")
    assert PrecisionParams(hubert_autocast="bf16").hubert_candidates == ("bf16",)
    assert PrecisionParams(hubert_autocast="off").hubert_candidates == ("off",)
    assert PrecisionParams(hubert_fp16=True).hubert_candidates == ("fp16",)
    assert RuntimeProfile(hubert_autocast="fp16").precision().hubert_candidates == ("fp16",)


def test_tuned_record_roundtrips_through_runtime_json(tmp_path):
    key = "Voice|hop20|extra1000|xf30|search10|sr48000|rmvpe|dec32|torch2.7.1+cu128"
    assert load_tuned(tmp_path, key) is None
    assert load_tuned(None, key) is None
    record = {"hubert_autocast": "fp16", "cudnn_autotune": False,
              "hubert_match": {"fp16": {"log_mel_l1": 0.01, "passed": True}}}
    save_tuned(tmp_path, key, record)
    save_tuned(None, key, record)  # no profile: nothing written, no error
    assert load_tuned(tmp_path, key) == record
    runtime = load_as(RuntimeProfile, tmp_path / "runtime.json")
    assert runtime.block_ms == 20.0 and runtime.tuned == {key: record}  # defaults kept
    save_tuned(tmp_path, "other", {"cudnn_autotune": True})
    assert set(load_as(RuntimeProfile, tmp_path / "runtime.json").tuned) == {key, "other"}
