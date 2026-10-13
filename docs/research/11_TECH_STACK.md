# 11 — Tech Stack Recommendations

Status: research complete · Target: Windows 10/11, NVIDIA CUDA, Python backend, headless engine
Grounding: `Vonovox/requirements.txt` + README ("Python 3.12.8, PyTorch 2.7.0, CUDA 12.8"),
`Applio/requirements-engine.txt`, `voice-changer` server stack.

Guiding rule: **boring, pinned, single-process**. No Docker, no databases, no services — this is one
Python process on one gaming PC feeding a virtual audio cable.

---

## 1. Python version

**Recommendation: Python 3.12 (pin a micro, e.g. 3.12.x).**

- Vonovox ships exactly **3.12.8** (compiled `cp312` pyds); it is the proven realtime-RVC config.
- Wheel coverage: PyTorch CUDA wheels, faiss-cpu, onnxruntime-gpu, sounddevice, llvmlite/numba
  (librosa dep) all publish cp312 wheels. 3.13 coverage is still patchy for the numba/faiss tail;
  3.11 works but buys nothing. Avoid free-threaded builds (3.13t) — the audio stack is untested there.
- Applio's engine requirements branch on `python_version >= "3.12"` (scipy pins) — they target 3.12 too.

## 2. PyTorch + CUDA wheel strategy

**Recommendation: pin `torch==2.7.x + cu128` wheels from `download.pytorch.org/whl/cu128`,
with torchaudio matched; revisit upward only via the benchmark regression gate (doc 10 §7).**

- What the references pin: **Vonovox: PyTorch 2.7.0 / CUDA 12.8** (README; requirements list bare
  `torch` because setup.bat installs the cu wheel). **Applio: `torch==2.11.0` + `torchaudio==2.11.0`**
  with explicit `nvidia-*-cu12` wheels on Windows (`requirements-engine.txt`) — i.e. both ride the
  cu12 line; Applio tracks newer torch aggressively, Vonovox freezes what it validated.
- Why pin hard: CUDA graphs, cuDNN heuristics, and fp16 kernels change timing behavior between torch
  minors; an unpinned torch silently invalidates every benchmark baseline. Install via
  `--index-url https://download.pytorch.org/whl/cu128` (self-contained wheels, no system CUDA
  toolkit needed; only the NVIDIA driver matters).
- cu128 covers Blackwell (sm_120) through Maxwell-era cards Vonovox supports; if the user's GPU is
  RTX 30/40/50, cu126/cu128 are both fine — pick cu128 for headroom.
- Record the exact `torch.version.cuda` + driver in every benchmark result (doc 10).

## 3. Audio I/O

**Recommendation: `sounddevice` (PortAddio) with WASAPI, shared mode default.**

| Option | Verdict | Why |
|---|---|---|
| **sounddevice==0.5.x** | **pick** | PortAudio callbacks + WASAPI (incl. exclusive via `WasapiSettings`), numpy-native, used by both Vonovox and Applio realtime (`sounddevice` in both requirements) — community-validated for exactly this workload |
| pyaudio | no | older PortAudio bindings, worse maintenance, no advantage |
| python-sounddevice raw / CFFI custom | no | sounddevice already is that |
| JACK/ASIO-first stacks | no | ASIO optional via PortAudio if an interface exists; Vonovox README: "Use WASAPI unless you have an ASIO interface" |
| sd exclusive mode | optional flag | shaves a few ms, but device-hogging and fragile; default off (matches Vonovox guidance) |

Fixed engine rate 48 kHz mono to match VAC lines; resample internally to model rates (16 k for
ContentVec/F0, model SR out).

## 4. Core numerics & retrieval

| Component | Pick | Why |
|---|---|---|
| **numpy** | 2.x pinned (Applio pins 2.4.6) unless a dep forces 1.26 (Vonovox pins 1.26.4 for older numba/librosa) — resolve at lock time, then freeze | ABI churn between 1.x/2.x broke many audio pkgs; pin whatever the full lock solves to |
| **faiss-cpu** | **faiss-cpu** (Applio 1.14.3 / Vonovox 1.9.0.post1 — take current 1.x) | index search is k=8 over a ~100–500 MB IVF index for ~50–100 queries/block: sub-ms on CPU and it keeps VRAM + GPU streams free for the decoder. faiss-gpu has no official Windows wheels and would fight the inference stream — not worth it (both reference apps agree: faiss-cpu) |
| **resampling** | `torchaudio.functional.resample` on-GPU in the hot path; **soxr** (Applio pins `soxr==1.1.0`) for offline/calibration quality | hot path must stay on-device; soxr is best-in-class quality for file work |
| **F0** | torchfcpe==0.0.4 (Applio pin) + vendored RMVPE weights/impl (every repo vendors RMVPE; there is no canonical package) | matches doc 08 extractor strategy; vendoring RMVPE is the ecosystem norm |
| **VAD** | `silero-vad` (Vonovox pins 5.1.2) | tiny, torch-native, proven in Vonovox with 400 ms release; webrtcvad (Applio) is cheaper but much cruder |
| **denoise (optional)** | `pyrnnoise` (Vonovox 0.2.7), default off pending calibration noise-floor verdict | doc 08 §4.2 |
| **onnxruntime-gpu** | optional, behind a flag (Applio pins 1.22.0 on win32) | useful later for RMVPE-ONNX or decoder export; adds a second CUDA runtime to manage — not in v1 critical path |
| speaker embedding (tuning only) | resemblyzer or speechbrain-ECAPA, in a **separate tools extra**, not the runtime env | keeps runtime deps slim |
| ASR (tuning only) | faster-whisper (ctranslate2) | WER gate, offline only |

## 5. Config format

**Pick: JSON** (stdlib `json`, schema-versioned files as in doc 08 §5).

- The profiles are **machine-written** (calibration + tuning loop write them); TOML's human-editing
  ergonomics don't apply, and round-tripping TOML with comments needs `tomlkit` (another dep,
  formatting landmines). stdlib `tomllib` is read-only.
- JSON diff/merge tooling is universal; numeric fidelity fine for our fields.
- Human-facing niceties (comments) are replaced by `"schema"` + generated `tuning_report` provenance.
- One exception allowed later: a tiny hand-edited `settings.toml` for GUI preferences — decide when a
  GUI exists.

## 6. Logging & profiling

- **Logging**: stdlib `logging` + `rich` handler for the console (Vonovox already ships `rich`).
  JSON-lines file log for engine events (underruns, device changes, profile loads). **Never log per
  block in the hot path** — aggregate counters flushed once per second.
- **Profiling (shipped)**: CUDA events per stage + perf_counter, NVML sampling — i.e. the doc 10
  instrumentation is compiled in and toggleable, so field issues are diagnosable.
- **Profiling (dev)**: `py-spy` (sampling, attach to live process, finds GIL/CPU stalls),
  Nsight Systems for stream/overlap analysis, `torch.profiler` for op-level. None are runtime deps.

## 7. Packaging (later)

**Recommendation: portable embedded-Python distribution (the reference-project pattern), not PyInstaller.**

- Vonovox (`setup.bat` + zips/), w-okada, Applio all ship a folder with an embedded/portable Python
  + pinned wheels + `start.bat`. It survives CUDA-wheel size (multi-GB), keeps `torch` importable
  normally, and updates = replace folder.
- PyInstaller with torch/CUDA/faiss is a known tarpit (hidden imports, DLL resolution, 6 GB onefile
  unpack times). Not worth it for a single-user tool.
- Defer until the engine stabilizes; dev runs from a `uv`/`pip-tools`-locked venv.

## 8. GUI (later — backend stays headless)

Engine = headless process with a local control surface (stdin/JSON-RPC or a localhost websocket,
pattern proven by w-okada's server/client split). GUI candidates, **noted only, no decision now**:
CustomTkinter (Vonovox — adequate, pure-Python), PySide6 (richest, heavier), or a tiny web UI over
the websocket (w-okada style). Nothing in the engine may import GUI code.

## 9. Requirements sketch

```text
# runtime.txt — engine (install torch via --index-url https://download.pytorch.org/whl/cu128)
torch==2.7.1+cu128
torchaudio==2.7.1+cu128
numpy==<lock-resolved>          # 1.26.4 or 2.x per dependency solve, then frozen
sounddevice==0.5.1
faiss-cpu==1.9.0.post1          # or current 1.x at lock time
torchfcpe==0.0.4
silero-vad==5.1.2
soxr==1.1.0                     # offline resampling (calibration/bench)
librosa==0.10.2.post1           # analysis only (calibration), not hot path
soundfile==0.13.*
pyrnnoise==0.2.7                # optional extra [denoise]
rich==14.*
psutil==7.*
pynvml==12.*                    # NVML metrics (doc 10)

# tools.txt — offline tuning/benchmark only (separate venv or extras)
resemblyzer / speechbrain       # speaker-cosine (pick one at lock time)
faster-whisper                  # WER gate
praat-parselmouth               # calibration analysis
matplotlib                      # bench reports

# dev.txt
py-spy, pytest, ruff
```

(Lock with `uv pip compile` / pip-tools; the sketch shows intent, the lockfile is the truth.)

## 10. Decision table (summary)

| Decision | Choice | Primary reason | Reference precedent |
|---|---|---|---|
| Python | 3.12.x | full cp312 wheel coverage; Vonovox-proven | Vonovox 3.12.8 |
| Torch/CUDA | 2.7.x + cu128, hard pin | benchmark stability; driver-only install | Vonovox 2.7.0/cu128; Applio cu12 line |
| Audio I/O | sounddevice/WASAPI shared | callback model, numpy-native, both repos use it | Vonovox, Applio |
| Index | faiss-cpu | sub-ms at our query sizes; no Windows GPU wheels; frees VRAM | both repos |
| Resample | torchaudio (hot) + soxr (offline) | stay on GPU in hot path | Applio pins soxr |
| F0 | torchfcpe pkg + vendored RMVPE | ecosystem norm | Applio, Vonovox |
| VAD | silero-vad | quality + torch-native | Vonovox |
| Config | JSON, schema-versioned | machine-written profiles | — |
| Logging | stdlib + rich, JSONL events | zero hot-path cost | Vonovox ships rich |
| Packaging | portable embedded Python (later) | CUDA wheels vs PyInstaller tarpit | Vonovox/w-okada/Applio pattern |
| GUI | headless engine + IPC; GUI TBD | testability, benchmark = product | w-okada split |
| Docker/DB/services | **none** | single user, single box | — |

## Key takeaways

1. **Python 3.12 + torch 2.7.x/cu128, everything hard-pinned** — Vonovox proves this exact combo in
   production realtime RVC; unpinned torch invalidates benchmark baselines.
2. **sounddevice/WASAPI** is the uncontested audio layer (both reference apps), exclusive mode an
   expert flag only.
3. **faiss-cpu, never faiss-gpu on Windows**: k=8 retrieval is CPU-trivial, GPU wheels don't exist,
   and VRAM belongs to the decoder.
4. **JSON profiles** (machine-written by calibration/tuning) beat TOML here; stdlib logging + rich;
   CUDA-event profiling compiled in.
5. Package later as a **portable embedded-Python folder** like every reference project; keep the
   engine headless with GUI candidates (CustomTkinter/PySide6/web) explicitly deferred. No Docker,
   no databases, no services.
