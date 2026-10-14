# rvc-vox

Realtime RVC voice conversion engine (Windows, PyTorch, CUDA graphs) plus `vox say`
(typed text -> TTS -> the model's voice). Layout and dependency rules: `12_PROJECT_STRUCTURE.md`,
enforced by `tests/unit/test_import_rules.py`. Measurements and departures from the research
docs: `docs/DEVIATIONS.md`. TTS findings: `docs/TTS_NOTES.md`.

## Target hardware

This project supports exactly one machine, described in [TARGET.md](TARGET.md): RTX 5060 Ti
(16 GiB, compute capability 12.0, torch 2.7.1+cu128), Ryzen 9 7900X, Windows 11, Virtual
Audio Cable (not VB-Cable) and NVIDIA Broadcast present. Assume it in every session:

- The engine asserts the GPU at startup (`vox.config.target.select_device`); the CPU path
  exists only for unit tests behind `VOX_ALLOW_CPU=1` (`tests/conftest.py` sets it).
- Do not add compatibility branches, device-capability switches or conservative defaults
  for other hardware. Blackwell defaults are the defaults: TF32 on, HuBERT autocast
  candidates on by default behind the eager-match check, memory guards sized to 16 GiB.
- Audio defaults: input = the user's microphone, output = the VAC "Line 1" input line,
  48 kHz pinned on the cable. When the input is the NVIDIA Broadcast virtual mic, the
  engine's own Suppression defaults off.
- Every benchmark number in `docs/` was or must be measured on this machine, one GPU job at
  a time (`.venv\Scripts\python.exe`; the bare `python` has no torch).
