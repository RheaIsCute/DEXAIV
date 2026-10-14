# rvc-vox

Low-latency real-time RVC voice changer for one user, one mic, one GPU (Windows + NVIDIA).
Headless engine first; the research and design live in [docs/research/](docs/research/).
It targets exactly one machine, described in [TARGET.md](TARGET.md) (RTX 5060 Ti, Virtual
Audio Cable, NVIDIA Broadcast); the engine asserts that GPU at startup.

**Status: Phases 0-2 plus the reordered Phase 3 of the
[roadmap](docs/research/13_IMPLEMENTATION_ROADMAP.md) are built.** The conversion engine
runs offline and live through the same objects: per-stage CUDA graphs are the default fast
path (`graphs: auto`), the silence gate (energy + Silero VAD) idles the GPU in pauses, and
`vox run` streams microphone -> engine -> any output device with hot parameter changes and
a 1 Hz profiler line. Measured results, deviations from the research docs and newly found
open questions are in [docs/DEVIATIONS.md](docs/DEVIATIONS.md) and
[docs/research/14_OPEN_QUESTIONS.md](docs/research/14_OPEN_QUESTIONS.md).

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1   # venv + torch 2.7.1 cu128
.venv\Scripts\python scripts\fetch_assets.py --all               # ContentVec, RMVPE, test model, test audio
.venv\Scripts\python -m pytest                                   # unit + engine tests
```

`fetch_assets.py` verifies every download against `scripts/assets.lock.json`. `--all` also
builds two public test voices (`base_f0_48k_v2`, `base_f0_40k_v2`, the official RVC v2 base
generators repacked as inference checkpoints) and three LibriSpeech clips under
`benchmarks/corpus/`.

## Use

```powershell
vox models import path\to\voice.pth          # detects v1/v2, sample rate, f0, pairs an index
vox models list
vox convert in.wav out.wav --model MyVoice --pitch 7                 # the streaming engine
vox convert in.wav out.wav --model MyVoice --pitch 7 --mode full     # canonical whole-file
vox bench accept                             # Phase 2 acceptance + profiler report
vox bench perf                               # block x extra x f0 sweep (--sweep perf_graphs.yaml: graphs vs eager)
vox bench gate                               # silence-gate acceptance
vox devices                                  # WASAPI devices with indices
vox run --model MyVoice --pitch 7 --input 22 --output 20   # live; type 'help' for commands
python scripts/measure_rtt.py --cable "Virtual Audio Cable"   # loopback round-trip latency
vox env                                      # environment fingerprint
```

`vox calibrate` is a placeholder for Phase 5.

## Layout

The structure is the blueprint of
[docs/research/12_PROJECT_STRUCTURE.md](docs/research/12_PROJECT_STRUCTURE.md).

| Path | Role |
|---|---|
| `src/vox/engine/` | `chunker.py` (the chunking law), `pipeline.py` (the per-hop function as capturable stages), `graphs.py` (per-stage CUDA graphs), `gate.py` (silence gate), `worker.py` (inference thread), `sola.py`, `postchain.py`, `warmup.py` |
| `src/vox/pitch/`, `features/`, `retrieval/`, `synthesis/` | stage packages: RMVPE / FCPE, ContentVec, GPU k-NN index, RVC synthesizer |
| `src/vox/audio/` | `resample.py` (incremental 48 k -> 16 k and model -> device), `rings.py` (SPSC rings), `streams.py` (WASAPI in/out streams, file-backed input), `devices.py` |
| `src/vox/api/control.py` | `LiveSession`: engine + rings + worker + streams, hot and restart-class changes, stats snapshots |
| `src/vox/profiling/` | stage timers (host + CUDA events read one hop late), statistics, reports |
| `src/vox/models/`, `config/` | checkpoint detection, model registry, JSON profiles |
| `src/vox/offline/` | `simulate.py` (stream simulator), `convert.py` (full-file + CLI), `metrics.py` |
| `benchmarks/bench.py` | acceptance run and sweeps, all through the stream simulator |
| `scripts/parity_rvc_webui.py` | A/B against a read-only rvc-webui clone |
| `scripts/measure_rtt.py`, `scripts/probe_whole_hop_graph.py` | loopback latency measurement; the Q16 whole-hop graph probe |

Later-phase modules (`calibration/`, `api/events.py`) are stubs that state their contract.

## Reference code

`scripts/parity_rvc_webui.py` compares this implementation with a local clone of
[RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
(MIT). The synthesizer and RMVPE networks under `src/vox/synthesis/nets/` and
`src/vox/pitch/_rmvpe_net.py` are inference-only reimplementations that keep upstream
parameter names so published checkpoints load unchanged.

## Text to speech (`vox say`)

Speak typed text in a model's voice: a TTS engine renders the text, then the full-file
converter turns it into the model's voice. Needs `pip install -r requirements/tts.txt`.

```powershell
vox say "Hello there." --model MyVoice --out out\hello.wav     # kokoro (local) by default
vox say "Hello there." --model MyVoice --play --device Speakers
vox say --file script.txt --model MyVoice --backend edge --voice en-US-AriaNeural
vox say --list-voices --backend kokoro
```

Without `--pitch` the transposition is measured from the rendered speech and printed.
Picking a TTS voice already close to the target (a female TTS voice for a female model,
pitch near 0) usually sounds best. Details are in [docs/TTS_NOTES.md](docs/TTS_NOTES.md).

Informal text sounds informal (`src/vox/tts/expressive.py`): `...` and dashes become
pauses that grow with the dot count, "heyyy" / "nooooo" render slower, a word in CAPS or
`!!` / `??` gets emphasis (a little faster and louder). Plain text is untouched.

## The window (`vox gui`)

Four screens over the same engine `vox run` drives: Live (start/stop, latency, levels,
pitch, type-to-speak), Models (import by drag and drop, index per model), Performance
(per-stage timings, 60 s hop-time chart) and Settings. Needs
`pip install -r requirements/gui.txt` for its own window; without it the pages open in the
browser.

```powershell
vox gui                                  # opens the window; press START to go live
vox gui --output "Line 1" --start        # pick the output device and go live at once
vox gui --no-window                      # only the local control API; prints its URL
```

Typing in the bar on the Live screen speaks the text in the model's voice over the same
output as the live voice; the microphone is muted while the clip is rendered and played.
Suppression, headphone monitoring, hop auto-tuning and calibration are shown but not built
yet. Design, deviations and measured acceptance: [docs/gui/GUI_SPEC.md](docs/gui/GUI_SPEC.md)
and section 6 of [docs/DEVIATIONS.md](docs/DEVIATIONS.md).
