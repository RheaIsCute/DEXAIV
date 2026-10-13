# 10 — Benchmark & Profiling System

Status: research complete · Goal: repeatable, single-machine measurement of latency, stability and
quality for the realtime RVC engine, gating every code change.
Code references: `Applio/rvc/realtime/` (block/SOLA structure), `voice-changer/server/voice_changer/RVC/RVCSettings.py`, doc 08 (profiles, tuning loop).

---

## 1. Principles

1. **One engine, two feeders.** The benchmark feeds WAV files through the *identical* realtime code
   path (same ring buffers, SOLA crossfade, context windows, CUDA graphs) via a stream simulator —
   never through a separate batch pipeline. Any divergence between "benchmark engine" and "live
   engine" invalidates every number.
2. **Fixed corpus, fixed seed.** The user's calibration recordings (doc 08 §4) are the quality
   corpus; deterministic chunking; `torch.manual_seed` pinned; cuDNN benchmark mode allowed but
   warmed up before timing.
3. **Perf and quality are separate sweeps.** Perf first (block/context/precision on silence+speech),
   quality second (pitch/index/protect on the locked perf winners). Mixing them explodes the grid.
4. **Honest numbers.** Every metric table states whether values are measured or hypothetical;
   objective quality metrics ship with their caveats attached.

## 2. Stream simulation (offline mode)

```
WAV (48 kHz mono) ──chunker(block_ms)──▶ engine.process_block() ──▶ output accumulator ──▶ metrics
                      │                                                  │
                      └── two clock modes ───────────────────────────────┘
```

- **Throughput mode** (`realtime=False`): feed blocks back-to-back as fast as the engine accepts
  them. Measures per-stage compute cost and RTF without scheduler noise. Used for sweeps.
- **Realtime emulation mode** (`realtime=True`): a paced feeder delivers block N at t = N·block_ms
  (perf-counter clock); if the engine hasn't returned block N−1 by its deadline, count an
  **underrun** and (configurably) drop or queue — mirroring the live callback contract. Used for
  stability/jitter verdicts. Optionally add synthetic feeder jitter (±1–2 ms) to emulate WASAPI
  callback variance; true device/driver latency (buffering, DAC) is *not* captured — measured once
  separately with a loopback cable and reported as a constant offset.
- Output is written to WAV per run, named by the full parameter hash → every quality metric and
  every human A/B replays from the same artifact.

## 3. Performance metrics & collection

### 3.1 Per-stage timing

Stages instrumented individually: input resample → (RNNoise) → VAD → F0 extraction → ContentVec →
FAISS/index blend → decoder (net_g) → SOLA/crossfade → RMS/post-FX → output resample.

| Concern | Tool | Notes |
|---|---|---|
| GPU stage times | **CUDA events** (`torch.cuda.Event(enable_timing=True)` pairs per stage) | the only correct way; `time.perf_counter()` around async CUDA launches measures nothing. Events recorded every block, elapsed read lazily after the run (reading forces sync). |
| CPU stages & end-to-end block wall time | `time.perf_counter_ns()` | end-to-end includes one deliberate sync at block end (the live engine syncs there anyway to hand audio to the callback) |
| GPU/CPU utilization, VRAM | `pynvml` (NVML) sampled at 10 Hz on a side thread; `torch.cuda.max_memory_allocated()` per run; `psutil` for CPU% | NVML util is coarse (1 s window on some drivers) — treat as trend, not truth |
| Deep dives (dev only) | Nsight Systems (`nsys profile`), `py-spy dump` for CPU stalls, `torch.profiler` for op-level | not part of CI numbers |

### 3.2 Reported statistics per run

- Per-stage and end-to-end **p50 / p95 / p99** latency (ms), after discarding a 50-block warmup.
- **Jitter** = p99 − p50 of end-to-end block time; also max.
- **RTF** = total compute time / audio duration (throughput mode).
- **Headroom** = block_ms − p99 end-to-end (realtime viability requires > 0 with margin).
- **Underruns / dropped blocks** per minute (realtime mode), plus time-to-first-underrun.
- **VRAM peak**, mean GPU util, mean CPU util, clocks/thermals at start vs end (NVML) to catch
  thermal throttling over a 10-minute soak run.

### 3.3 Why percentiles, not means

A 15 ms block budget with mean 9 ms and p99 16 ms glitches audibly several times a minute. The gate
metric is **p99 (and max) vs block budget**, never the mean. GC pauses, cudaMalloc on cache miss,
and FAISS CPU spikes all live in the tail.

## 4. Quality corpus spec

Fixed recordings of the one user (superset of the calibration script), captured once through the
production input chain at 48 kHz, stored immutable (`corpus/v1/*.wav` + transcript JSON):

| Clip | Content / script suggestion | ~Len | Probes |
|---|---|---|---|
| `read_normal` | Rainbow Passage excerpt, conversational effort | 25 s | baseline WER, F0 stats, speaker-sim |
| `read_quiet` | same text, near-whisper | 15 s | VAD threshold, noise gate, low-SNR F0 |
| `read_loud` | same text, projected | 15 s | clipping headroom, RMS matching |
| `pitch_low` | sentence deliberately at bottom of range ending in fry: "and that was the end of that, I suppose…" | 10 s | octave errors, fry handling |
| `pitch_high` | excited high sentence + rising questions | 10 s | F0 clamp ceiling, falsetto artifacts |
| `fast` | tongue-twister run: "Peter Piper…" at speed | 10 s | context-size smearing, ASR WER sensitivity |
| `slow_vowels` | long sustained /a/ /i/ /u/ glides | 10 s | pitch stability, block-boundary phase (SOLA) artifacts, wobble |
| `sibilants` | "She sells seashells…; sixty-six sisters" + isolated /s/ /ʃ/ holds | 10 s | protect/index settings, sibilant centroid |
| `plosives` | "put the cup back; big bad bug" close-mic | 8 s | transient handling, protect, pop sensitivity |
| `laughter` | natural laugh, chuckle | 6 s | VAD false-negatives, nonverbal rendering |
| `breaths` | sentence with audible inhales between phrases | 10 s | breath pass-through vs gating |
| `silence_room` | room tone | 10 s | noise floor, idle GPU load, gate behavior |

~2.5 min total → one full-corpus run is cheap enough to attach to every sweep point.

## 5. Parameter sweep harness

CLI: `bench sweep --config sweeps/perf_grid.yaml --profile profiles/main`

### 5.1 Perf grid (throughput + realtime modes, `read_normal` + `silence_room`)

| Axis | Values |
|---|---|
| block size | 5, 10, 15, 20, 30, 50 ms |
| extra context (lookahead/history) | 50, 100, 200 ms |
| F0 method | rmvpe, fcpe |
| precision | fp16, fp32 (and TF32 on) |

### 5.2 Quality grid (locked perf winners; full corpus)

| Axis | Values |
|---|---|
| f0_up_key | estimate −2…+2 |
| index_rate | 0, 0.25, 0.5, 0.75 |
| protect | 0.2, 0.33, 0.5 |
| block/context | top-2 perf configs |

Output: one JSON per run (`results/<hash>.json`: params, perf stats, quality metrics, output-WAV
path) + an aggregated HTML/Markdown report with the three correlation views: **latency vs block
size**, **quality-proxy vs block size** (quality degrades below some block floor — find it), and
**stability (underruns) vs GPU headroom**.

### 5.3 Example result tables — HYPOTHETICAL NUMBERS (illustrative format only)

Perf sweep (RTX 4070, fp16, rmvpe, 100 ms context — *invented values*):

| block_ms | p50 ms | p95 ms | p99 ms | headroom p99 | underruns/min | RTF | verdict |
|---|---|---|---|---|---|---|---|
| 5 | 6.1 | 7.9 | 9.4 | −4.4 | 41 | 1.22 | fail |
| 10 | 6.8 | 8.3 | 9.9 | +0.1 | 3.1 | 0.68 | marginal |
| 15 | 7.4 | 9.0 | 10.8 | +4.2 | 0.0 | 0.49 | **pass** |
| 20 | 8.2 | 9.8 | 11.5 | +8.5 | 0.0 | 0.41 | pass |
| 50 | 12.9 | 15.0 | 17.2 | +32.8 | 0.0 | 0.26 | pass (latency waste) |

Quality sweep excerpt (block 15 ms — *invented values*):

| f0_up_key | index_rate | protect | out F0 median (target 196) | spk-cos | WER % (src 4.1) | flips/min | note |
|---|---|---|---|---|---|---|---|
| +6 | 0.25 | 0.33 | 183 | 0.81 | 4.9 | 0.3 | slightly low |
| +7 | 0.25 | 0.33 | 194 | 0.83 | 5.0 | 0.3 | **candidate A** |
| +7 | 0.75 | 0.33 | 195 | 0.85 | 8.7 | 0.4 | WER gate fail |
| +7 | 0.25 | 0.5 | 195 | 0.84 | 5.6 | 0.3 | candidate B (duller /s/?) → A/B |
| +8 | 0.0 | 0.5 | 206 | 0.82 | 4.6 | 0.9 | candidate C |

## 6. Objective quality metrics — with honest caveats

| Metric | Implementation | Tag | Caveats |
|---|---|---|---|
| Output F0 stats (median, P10/P90, flip-rate, % frames > 350 Hz) | RMVPE on output | PROVEN | validates pitch mapping & stability only; blind to timbre |
| Speaker similarity | resemblyzer GE2E (or ECAPA/WavLM-sv) cosine vs target preview renders | LIKELY | great at separating broken from plausible; top candidates differ by less than embedding noise (±0.02); embeddings partly encode channel/recording conditions |
| Intelligibility | faster-whisper (small, int8) WER vs known scripts | PROVEN mech. | Whisper's robustness hides mild smearing; use as gate (≤ source WER + 3 pts), not ranker |
| PESQ / POLQA | pesq pkg, 16 kHz | EXPERIMENTAL (misuse) | **intrusive metrics need the same utterance as reference** — the "reference" here is a different voice by design; PESQ scores will be low and meaningless. Only valid use: engine-vs-engine on identical settings to catch new DSP artifacts (diff mode) |
| UTMOS / NISQA (no-reference MOS) | pretrained nets | EXPERIMENTAL | trained on TTS/telephony distortions; correlation with VC gender-credibility unverified; report, never gate |
| Sibilant centroid / spectral tilt deltas | numpy on segmented /s/ holds | EXPERIMENTAL | segmentation fragility; useful trend line for protect sweeps |
| Chunk-consistency | spectral/F0 discontinuity at block boundaries vs a full-file offline render of the same settings | LIKELY | directly measures realtime-vs-offline degradation; our best "stability of quality" proxy |

Human A/B remains the final arbiter (doc 08 §6.3); objective metrics shrink the candidate set and
catch regressions.

## 7. Regression gating between code changes

- `bench regress` runs a **pinned reference config** (the shipped profile) on the full corpus:
  ~5 min wall time.
- **Perf gates**: p99 end-to-end ≤ baseline × 1.10; zero underruns in realtime mode at shipped
  block size; VRAM peak ≤ baseline + 256 MB.
- **Quality gates**: output F0 median within ±2 Hz of baseline; flip-rate ≤ baseline + 0.2/min;
  WER ≤ baseline + 1 pt; speaker-cos ≥ baseline − 0.02; **bit/spectral diff**: log-mel L1 distance
  of output vs baseline output — if > ε, the change altered audio and requires a listen before
  merge (metrics can't bless an audible change, only flag it).
- Baselines stored as `baselines/<git-sha>.json` + the reference WAVs; updating a baseline is an
  explicit, reviewed action with a mandatory listening note.
- Environment fingerprint recorded in every result (GPU, driver, CUDA, torch, cuDNN versions,
  power/thermal state) — cross-machine or cross-driver comparisons are flagged, not silently merged.

## Key takeaways

1. Benchmark = the **real engine behind a stream simulator** (throughput mode for cost, paced
   realtime mode for underruns/jitter); a separate batch path would measure a different product.
2. Gate on **p99 vs block budget and underruns**, measured with **CUDA events** for GPU stages;
   means and CPU timers around async CUDA are lies.
3. Quality lives on a **fixed ~2.5 min corpus of the user's own voice** covering dynamics, pitch
   extremes, fry, sibilants, plosives, laughter and breaths — the same clips feed calibration,
   tuning, and regression.
4. Sweep perf first (block/context/F0 method/precision), lock winners, then sweep
   pitch/index/protect; report latency-vs-quality-vs-stability correlations from one JSON-per-run
   store. Example tables above are **hypothetical**.
5. Objective metrics are triage, not truth: F0 stats and WER gates PROVEN, speaker-cosine and
   chunk-consistency LIKELY, PESQ is structurally invalid for VC (reference is a different voice),
   UTMOS EXPERIMENTAL. Any audible (log-mel) diff from baseline forces a human listen before merge.
