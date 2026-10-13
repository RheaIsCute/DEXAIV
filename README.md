# rvc-vox

Low-latency real-time RVC voice changer for one user, one mic, one GPU (Windows + NVIDIA).
Headless engine first; the research and design live in [docs/research/](docs/research/).

**Status: Phases 0-2 of the [roadmap](docs/research/13_IMPLEMENTATION_ROADMAP.md) are built.**
The conversion engine runs offline, hop by hop, through the same objects live audio will
use. Live audio (Phase 3) is not started. Measured results, deviations from the research
docs and newly found open questions are in [docs/DEVIATIONS.md](docs/DEVIATIONS.md) and
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
vox bench perf                               # block x extra x f0 sweep
vox env                                      # environment fingerprint
```

`vox run`, `vox devices` and `vox calibrate` exist as placeholders for Phases 3 and 5.

## Layout

The structure is the blueprint of
[docs/research/12_PROJECT_STRUCTURE.md](docs/research/12_PROJECT_STRUCTURE.md).

| Path | Role |
|---|---|
| `src/vox/engine/` | `chunker.py` (the chunking law), `pipeline.py` (the per-hop function), `sola.py`, `postchain.py`, `warmup.py` |
| `src/vox/pitch/`, `features/`, `retrieval/`, `synthesis/` | stage packages: RMVPE / FCPE, ContentVec, GPU k-NN index, RVC synthesizer |
| `src/vox/audio/resample.py` | incremental 48 k -> 16 k and model -> device resamplers |
| `src/vox/profiling/` | stage timers (host + CUDA events read one hop late), statistics, reports |
| `src/vox/models/`, `config/` | checkpoint detection, model registry, JSON profiles |
| `src/vox/offline/` | `simulate.py` (stream simulator), `convert.py` (full-file + CLI), `metrics.py` |
| `benchmarks/bench.py` | acceptance run and sweeps, all through the stream simulator |
| `scripts/parity_rvc_webui.py` | A/B against a read-only rvc-webui clone |

Later-phase modules (`audio/streams.py`, `engine/worker.py`, `engine/gate.py`,
`engine/graphs.py`, `calibration/`, `api/`) are stubs that state their contract.

## Reference code

`scripts/parity_rvc_webui.py` compares this implementation with a local clone of
[RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
(MIT). The synthesizer and RMVPE networks under `src/vox/synthesis/nets/` and
`src/vox/pitch/_rmvpe_net.py` are inference-only reimplementations that keep upstream
parameter names so published checkpoints load unchanged.
