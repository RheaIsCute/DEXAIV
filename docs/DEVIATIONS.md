# Deviations from the research docs, and what reality showed

Recorded during Phases 0-2 (2026-10-13). Reference machine: RTX 5060 Ti 16 GB, driver
617.14, Ryzen 9 7900X, Windows 11, Python 3.13.15, torch 2.7.1+cu128. Measurements use the
public test voice `base_f0_48k_v2` and LibriSpeech clip `libri_1272.wav`; they describe the
engine, not a tuned target voice. Unresolved items are tracked in
[research/14_OPEN_QUESTIONS.md](research/14_OPEN_QUESTIONS.md).

## 1. Where the implementation departs from the docs

| # | Doc says | What was built | Why |
|---|---|---|---|
| D1 | Python 3.12 (doc 11 §1) | Python 3.13.15 | The only interpreter installed. Every pinned wheel has a cp313 build, including torch 2.7.1+cu128. `requirements/runtime.txt` is frozen from the working environment. |
| D2 | Version sketch in doc 11 §9 | Newer pins: numpy 2.3.5, faiss-cpu 1.15.1, transformers 5.19.0, sounddevice 0.5.6, silero-vad 6.2.3; `nvidia-ml-py` instead of `pynvml` | "The lockfile is the truth" (doc 11). `transformers` is added because doc 12 specifies the HF ContentVec loader. `librosa` and `praat-parselmouth` are needed only by `scripts/parity_rvc_webui.py`. |
| D3 | Docs under `docs/research/` | Copied there from `docs_research/docs/research/` | That is where the zip was extracted. The originals are untouched and gitignored. |
| D4 | Kickoff prompt: `offline/stream_sim.py` | `offline/simulate.py` | Doc 12 is the blueprint and names it `simulate.py`. |
| D5 | Doc 12 file list | Three additions: `synthesis/nets/` and `pitch/_rmvpe_net.py` (the vendored networks doc 12 assigns to `rvc_model.py` / `rmvpe.py`, split out to keep modules small) and `offline/metrics.py` (log-mel comparisons shared by tests, bench and later tuning) | Module size; no boundary changes. |
| D6 | Defaults of 32 ms crossfade, 16 ms stretch hop (docs 04, 12) | All sizes snap to 10 ms units: 32 -> 30 ms, 16 -> 20 ms | Doc 12 §3.6 also says "all rounded to `zc = sr // 100` units", and the pitch cache can only shift by whole 10 ms frames. The two statements conflict; the integer-frame rule is the hard constraint. |
| D7 | `skip_head = extra // 160`, `sola_buffer = min(crossfade, 40 ms)` | Same values, but the overlap used everywhere is the capped `sola_buffer`, and `skip_head = convert_frames - return_length` | With crossfade > 40 ms the doc formulas would decode audio SOLA never uses. Identical whenever crossfade <= 40 ms. |
| D8 | "Unvoiced gaps are linearly interpolated" (doc 02 §3.2) vs "unvoiced = f0 0 -> bucket 1" (doc 05 §1) | Both, selectable: `f0_unvoiced = "zero"` (default) or `"interp"` | Current rvc-webui interpolates in training and inference; Applio and older RVC do not. The right value depends on what trained the model: see Q15. |
| D9 | Protect mask "gate on `pitchf > 0`" (doc 02 §3.6 summary) | `pitchf < 1 -> protect` | That is what the upstream code does; a unit test compares against it. |
| D10 | "faiss-cpu is fine in this phase" | faiss reads the file; search is already the torch brute-force k-NN of doc 07 | It is the final design and shorter than a CPU round-trip path. Verified against exact FAISS search. |
| D11 | "zero Python-level tensor allocation" in the steady-state hop | Verified instead: no driver mallocs, no reserved-memory growth, no live-memory drift, every persistent buffer pointer-stable, and no host sync (torch sync-debug mode set to `error`) | Eager PyTorch modules create ~1,700 pool-served temporaries per hop by construction. They vanish under graph replay, not before. |
| D12 | HuBERT FP16 autocast on by default (doc 12 `defaults.py`) | Implemented, default off | Doc 13 schedules the precision ladder for Phase 6, after correctness. |
| D13 | FCPE "RealtimeMel + on-device decoder" | Stock `torchfcpe` wrapper, `graph_safe = False` | Making it sync-free belongs with graph capture (Phase 6). |
| D14 | CUDA events "read one hop late" | Same, plus a non-blocking `Event.query()` check before each read | Without it a read could hit a still-pending event and fail (`device not ready`), seen once. |
| D15 | No decoder context in any reference | Optional `decoder_context_ms` (frames decoded ahead of the tail and discarded), default 0 = upstream behaviour | Added on a theory that a 60 ms tail is rendered entirely against zero padding. It showed no measurable effect on the log-mel metric (Q17), so it is off. |

## 2. What reality showed that the docs did not predict

### R1. Eager per-hop compute is 2-4x the docs' estimate, and it is launch-bound

Doc 04 §3.6 budgets 10-20 ms of compute per hop. Measured, eager, idle GPU:
**~38 ms p50 at every hop size and EXTRA** (RMVPE ~14 ms, ContentVec ~6 ms, synth ~16 ms),
p99 around 80 ms; ~27-31 ms p50 with FCPE in place of RMVPE.
The hop runs ~1,700 small operations and each costs 30-70 us to launch on this machine
(a tiny conv1d: ~50 us; the same code pattern on Linux is typically several times cheaper).
Compute barely matters: a six-frame decode costs as much as a much longer one, and cuDNN
settings move little.

Consequence: **in eager mode only the 100 ms hop reaches 1.5x headroom, and not at every
setting** (perf table under Q1). Phase 3's acceptance test (zero underruns at a hop
>= 32 ms, eager) cannot pass on this machine as planned.

### R2. CUDA graphs help ~2-3x per stage, not an order of magnitude; one-graph replay hangs

`vox bench graph-probe --stage S`, replay p50 (p99): RMVPE 7.3 (8.1) ms, ContentVec 2.6
(3.1) ms, synth 4.5 (5.1) ms, SOLA 0.13 (0.40) ms, against eager p50 of roughly 13, 6-10, 16
and 0.2 ms. Per-stage graphs would put the hop near **15 ms** (about 17 ms summing the
p99s), against ~38 ms eager, and without the eager tail. Capturing the **whole hop as one graph succeeds but its
replay never returns** (process spins; killed after 4 minutes, twice). Phase 6 needs either
per-stage graphs or a root cause for that hang.

The GPU reports a low power state (P3, ~1860 MHz, 24 W) during these runs: the load is a
stream of tiny kernels and never pushes the clocks up. Forcing maximum-performance power
management may reduce per-kernel time; untested (Q16).

### R3. A streamed render is not log-mel-equivalent to a full-file render

Speech-frame log-mel L1 after start-up, default plan: **0.56 streamed vs full-file, against
a 0.30 noise floor** (two full-file renders with different seeds): 1.9x. Ruled out as the
cause, one experiment each: hop size (20-250 ms: 0.555-0.570), EXTRA beyond 500 ms
(1000 ms: 0.543, 2500 ms: 0.547), F0 (an oracle serving full-file F0: 0.561), a longer RMVPE
tail, decoder left context, lookahead of 20-250 ms (0.535-0.593), and latent noise (with the
noise switched off the gap is still 0.51 while a stream differs from itself by 0.06).

What does explain it: this generator is extremely sensitive to imperceptible input changes.
With the same seed, swapping the 16 kHz resampler moves a full-file render by 0.125, adding
-80 dB noise by 0.115, a one-sample delay by 0.103 and a 10 ms delay by 0.207. Every hop
renders the same audio from a window shifted by one hop, so consecutive renders differ, and
at a 20 ms hop with 30 ms overlap every output sample is a crossfade of two of them. At
rvc-webui's own defaults (250 ms block, 2.5 s extra) the deterministic gap is 0.32.

So the metric measures a real property of small-hop SOLA streaming, and it cannot say
whether the result is audibly worse. That needs ears: `out/accept_full.wav` vs
`out/accept_stream.wav` (Q14).

### R4. Short context inflates loudness in pauses

At 500 ms extra the stream is ~2.8 dB louder overall than the full-file render, almost all
of it in quiet frames (quiet-frame L1 0.98 vs 0.34 floor). At 2500 ms the bias is gone.
ContentVec normalises over the window, so a mostly-silent window has its noise floor
amplified into speech-like features. `volume_envelope = 0` matches the level but not the
spectrum of those frames. This is the practical argument for the Phase 4 gate, and for
considering 1000 ms extra (bias +0.16 instead of +0.40).

## 3. Phase 1 A/B against rvc-webui (clone at commit 81eed5e)

`python scripts/parity_rvc_webui.py`:

- Synthesizer, same weights, inputs and seed on CPU: **max |diff| = 0.0** (bit-exact) on the
  full path and on two tail-only (`skip_head` / `return_length`) paths.
- RMVPE on 12 s of speech: voicing agreement 100%, largest deviation 0.001 cents.
- Whole offline pipeline (`Pipeline.pipeline` vs `FullFileConverter`, `f0_unvoiced=interp`,
  same seed): identical length, sample max |diff| 2.6e-3, log-mel L1 **0.0054** against a
  0.327 new-seed noise floor.

Both renders are written to `out/ab_reference_rvc_webui.wav` and `out/ab_ours_full.wav` for
the manual listening check.

## 4. Phase 3 reorder (2026-10-13): CUDA graphs first, silence gate second, then live audio

R1/R2 made the roadmap order untenable: eager compute (~40 ms per hop, launch-bound) cannot
meet any small hop, and the per-stage graph probes said graphs would. So the Phase 6
capture work moved ahead of live audio, and the Phase 4 gate came with it (it fixes R4 and
idles the GPU in pauses). Same machine and environment as above.

### 4.1 Deviations

| # | Doc says | What was built | Why |
|---|---|---|---|
| D16 | CUDA graphs in Phase 6; capture the *whole* hop as one graph (doc 06 §2.4, doc 12 `graphs.py`) | Phase 3, **per-stage graphs**: resample_in, F0, ContentVec, index, synth, post and SOLA are seven `torch.cuda.CUDAGraph`s replayed in sequence; `graphs: on / off / auto` (auto = capture, verify ten replayed hops against eager on a synthetic voiced signal, else eager with the reason logged) | Whole-hop replay hung on this machine (R2); per-stage replay is what the probes measured. Timers and CUDA events stay outside the graphs (events cannot be captured), so the profiler still reports per stage. |
| D17 | Hot parameters may skip stages (`index_rate == 0` skips retrieval, `protect >= 0.5` skips the protect blend, `volume_envelope == 1` skips the RMS mix) | The op sequence of a stage depends only on the static configuration (index loaded, extractor present, `f0_unvoiced`). A disabled feature is a numerically exact no-op: `lerp` with weight 0, protect weight 1, RMS exponent 0 | A captured graph is a fixed op sequence. This way every hot parameter is a value in a device scalar and never invalidates a graph; only `f0_unvoiced`, index load/unload and shape changes recapture. |
| D18 | Decoder noise drawn inside the synthesizer (`torch.randn_like`, upstream) | Pre-filled per hop into two pointer-stable buffers (`noise_z`, `noise_src`) outside any graph; the synthesizer accepts them as optional arguments and draws itself only when they are absent (the full-file path, so the Phase 1 parity holds) | Nothing random may run inside a captured region (the prime suspect for the R2 hang). Statistically identical output. |
| D19 | `extra = 500 ms` default (Phase 2 provisional) | **1000 ms** | R4: quiet-frame level bias +0.16 vs +0.40 nepers. Under graphs the cost is flat (R5), so the latency/compute argument against it is gone. |
| D20 | Gate protocol: on close zero the context rings and refill them at onset (doc 03 §5, Applio) | On close: zero the pitch cache and feature buffers, fade the SOLA tail out exactly once (`Sola.drain_step`), emit silence. The 16 kHz context ring is **advanced every closed hop** (resample stage only, ~0.3 ms) instead of being zeroed and refilled | A ring that is always the true recent input can never replay stale speech and is already full at onset, so there is no refill stall and the model sees the same context the full-file render sees. Hangover 200 ms (the profile's 400 ms release became 200, Applio's value). |
| D21 | Pinned upload with `non_blocking=True` (Phase 2) | Same, but the staging buffer waits on a CUDA event before it is rewritten | Latent race: under graph replay the host launches a hop in ~1 ms, so the next block could overwrite the pinned buffer while the previous DMA was still queued. Found by `test_pitch_cache_matches_full_window_f0` the moment graphs went on. |
| D22 | `RuntimeProfile.use_cuda_graphs: bool` | `graphs: "auto"` string in the profile and in `PrecisionParams` | Three modes, not two. |
| D23 | Silero VAD on the hop's 16 kHz window (GPU) | Silero on the **CPU**, on a host-side FIR decimation of the device-rate block, in 512-sample (32 ms) chunks | ~0.3 ms per chunk on this CPU and no GPU sync in the gate; the VAD can lag the hop by at most one hop. |
| D24 | `api/control.py` minimal start/stop in Phase 3; runtime changes via the Phase 8 API | `LiveSession` with hot params, gate switch and restart-class changes (hop / extra / crossfade / F0 method rebuild in place: drain, replan, recapture, resume); `vox run` reads those commands from stdin | Needed now to exercise the rebuild path; the socket API can wrap the same object later. |

### 4.2 R5. Per-stage graphs: the hop is now GPU-bound at ~16 ms, with no eager tail

`vox bench perf --sweep benchmarks/sweeps/perf_graphs.yaml` (throughput mode, 20 s clip,
first 50 hops discarded, RMVPE, same process for all eight points):

| hop | extra | mode | total p50 | p95 | p99 | RTF | headroom (hop / p99) |
|---|---|---|---|---|---|---|---|
| 20 ms | 500 | eager | 46.2 | 104.9 | 118.4 | 2.82 | 0.17x |
| 30 ms | 500 | eager | 48.8 | 103.6 | 120.3 | 1.98 | 0.25x |
| 20 ms | 1000 | eager | 41.7 | 95.3 | 114.0 | 2.53 | 0.18x |
| 30 ms | 1000 | eager | 42.4 | 91.0 | 97.6 | 1.60 | 0.31x |
| 20 ms | 500 | **graphs** | **16.9** | 18.3 | **19.1** | 0.85 | 1.04x |
| 30 ms | 500 | **graphs** | **16.2** | 19.0 | **20.8** | 0.55 | 1.44x |
| 20 ms | 1000 | **graphs** | **15.3** | 17.5 | **20.7** | 0.78 | 0.96x |
| 30 ms | 1000 | **graphs** | **15.3** | 17.7 | **20.1** | 0.52 | 1.49x |

- Acceptance for Step 1 (p50 <= 18 ms, p99 <= 25 ms at hop 20-30 ms, extra 1000, RMVPE)
  is met at every graphed point. The eager tail is gone: p99 - p50 is 2-5 ms instead of
  50-70 ms.
- Where the time goes under replay (CUDA events, `vox bench accept --graphs on`, hop 20 ms):
  RMVPE 7.9 ms, ContentVec 3.0 ms, synth 4.8 ms, SOLA 0.2 ms, resample/index/post < 0.1 ms
  each: **~16 ms of GPU time per hop**. The host now launches the hop in ~0.6 ms and waits
  in `download` for the rest. The hop is GPU-bound; the next lever is per-stage GPU time
  (RMVPE first), not launch count.
- `pool allocs/hop = 0` under replay (eager: ~1,700). Driver mallocs, reserved growth and
  live drift stay 0.
- Replay vs eager from identical state and noise: log-mel L1 0.0006-0.0011 over ten hops,
  relative RMS difference ~2e-4 (TF32 kernel rounding). The `auto` tolerance is 0.05.
- Capture costs ~1.2 s (3 warmups + capture for seven stages + the 10-hop verification).
- Eager in this run is ~20% slower than the Phase 2 table at every stage (46 vs 38 ms p50);
  the ratio between stages is unchanged, so it is host-side launch cost varying between
  sessions, not the engine. Eager is the fallback and is not tuned further.
- At a 20 ms hop the graphed p99 sits at the hop (headroom ~1.0x). Live audio therefore
  needs the output ring to absorb ~1 hop of jitter at 20 ms, or a 30 ms hop (1.45x).
- GPU clocks are not the lever (Q16 power-state theory): with the SM clock locked at
  2000-3090 MHz (`nvidia-smi -lgc`, P1 under load) the graphed hop is 16.3 / 19.1 ms
  (p50 / p99) against 16.1 / 20.5 unlocked; eager 41.2 / 90.5 against 40.9 / 90.4.

### 4.3 R6. The gate: pauses go quiet, closed hops cost 0.4 ms, transitions are seamless

`vox bench gate` (20 s LibriSpeech clip, hop 20 ms, extra 1000 ms, graphs, gate threshold
-48 dBFS + Silero 0.5, hangover 200 ms):

| render | pause-region level (input < -48 dBFS) | speech-region level |
|---|---|---|
| full-file | -43.1 dBFS | -21.0 dBFS |
| stream, gate off | -41.6 dBFS (+1.5 dB) | -21.0 dBFS |
| stream, gate on | -44.9 dBFS (-1.8 dB) | -21.1 dBFS |

- 11.6% of hops closed on this clip, 7 transitions; closed hops cost **0.41 ms host p50**
  (the 16 kHz ring advance, 0.04 ms of GPU time): the GPU is idle in gated silence.
- Click ratio at gate transitions 0.00 (edges are smoother than mid-hop audio: the drain
  fades the SOLA tail out, the onset fades in through the crossfade); hop boundaries 1.01.
- Speech-frame log-mel distance to the full-file render is identical with and without the
  gate (0.545). Gating changes nothing while speech is on.
- The "within +-1 dB of the full-file render" target reads differently on each side of the
  gate. **Ungated pauses** (within the 200 ms hangover) at extra 1000 ms sit +1.5 dB above
  the full-file render (R4 measured +2.8 dB at 500 ms). **Gated pauses** are digital
  silence, so the pause region as a whole lands 1.8 dB *below* the reference. The engine
  never renders louder than the reference in pauses, which was R4's complaint; the exact
  1 dB figure is set by how much of a pause the hangover leaves ungated.
- R4 re-measured with the new engine (`vox bench extra`, hop 20 ms, graphs), level bias in
  nepers over all frames after 3.1 s: gate off 100/300/500/1000/2500 ms = +1.05 / +0.63 /
  +0.38 / **+0.13** / -0.12 (Phase 2: +1.06 / +0.65 / +0.40 / +0.16 / -0.09, so graphs and
  the always-on post chain changed nothing); gate on = +0.32 / +0.01 / -0.15 / **-0.32** /
  -0.50. Quiet-frame L1 at 1000 ms: 0.731 gate off, 1.342 gate on (zeros vs the reference's
  noise floor: the metric penalises silence, the ear does not). Speech-frame L1 is 0.540-0.545
  either way. Hop p50 is 16-17 ms at every extra, as R5 predicts.
- The default is now extra = 1000 ms with the gate on (D19, D20).

### 4.4 R7. Live audio: 10 minutes, 30,047 hops, zero underruns; the latency budget is in the periphery

`vox run --model base_f0_48k_v2 --input 22 --output 19 --gate off --minutes 10` (Realtek
microphone -> engine -> Virtual Audio Cable, WASAPI shared, callback 10 ms, hop 20 ms,
overlap 30, search 10, extra 1000, graphs, fill target 50 ms; gate off so every hop runs
the model). 600 one-second windows:

| | p50 | p95 | p99 |
|---|---|---|---|
| hop compute, median window | 15.7 ms | 16.7 ms | 17.1 ms |
| hop compute, worst window | 18.4 ms | 22.5 ms | **23.7 ms** |

- **0 output underruns** (ring) and **0 PortAudio status flags** (output underflow, input
  overflow), 0 refills, 0 drift trims, 0 input overruns, over 30,047 hops. The output ring
  fill stayed between 20 and 60 ms around its 50 ms target; the two devices did not drift
  measurably against each other in 10 minutes.
- Host time per hop is ~0.6 ms of launches plus the wait for the GPU in `download`: the
  worker thread is idle most of the hop, so the GIL is not a factor at this hop size.
- The microphone sat at silence during the soak (nobody was speaking; input peak 0.00),
  which exercises the compute path but not the gate. The gated path is measured in the RTT
  runs below and in `tests/engine/test_worker.py`.
- **End-to-end estimate 143-159 ms**, and that is the real finding: hop 20 + compute 17 +
  output ring 22-39 + SOLA 40 (overlap 30 + search 10) + **devices 44 (PortAudio reports
  22 ms for each shared-mode WASAPI stream at "low" latency)**. Compute is 12% of it.

### 4.5 R8. Loopback round trip confirms the estimate; exclusive mode and a 20 ms crossfade reach ~90 ms

`scripts/measure_rtt.py` plays eight speech bursts from a file-backed input (perf_counter
stamped as they enter the input ring), renders to the Virtual Audio Cable and records the
cable's capture end; round trip = recorded onset - injected onset - recorder input latency.

| configuration | measured median (p95) | engine estimate | delta |
|---|---|---|---|
| shared, overlap 30, fill 50 (the soak setup) | **110 ms** (125) | 122 ms | -13 |
| shared, overlap 20, fill 40 | **88 ms** (114) | 102 ms | -14 |
| **exclusive**, overlap 20, fill 40 | **72 ms** (96) | 90 ms | -18 |

- The estimate is consistently ~13-18 ms conservative: the output-ring term is the fill
  *before* the hop is written (the wait of the hop's first sample), and PortAudio's device
  latency figure is an upper bound. Within the +-15 ms acceptance for the shared-mode runs,
  2.6 ms outside for exclusive mode.
- Where the 90 ms target lives: exclusive WASAPI takes the two device buffers from 22 ms
  each to ~5 ms each (PortAudio reports 10.7 ms for the pair), and overlap 20 instead of 30
  saves 10 ms at ~1% on the Q7 metric. Compute (16-17 ms) and the hop (20 ms) are now the
  minority of the budget. The remaining lever is the output fill target (40 ms here, of
  which ~29 ms is waited on average): it must cover compute p99 (24 ms worst window) plus
  one callback block; 30 ms is the floor at this hop and GPU.
- A click burst is not a usable probe through an RVC model with the gate off: the model
  renders digital silence as a -30 dBFS hiss (R4) and the onset detector cannot find the
  converted burst reliably. The speech bursts with the gate on are the realistic path and
  include the gate's own onset lag (one VAD chunk, <= 32 ms, inside the measured numbers).
- Not measured: a physical microphone's own capture buffer (the file-backed input replaces
  it). PortAudio reports it as 22 ms shared / ~5 ms exclusive; `vox run` adds it to `dev`.

### 4.6 R9. WASAPI exclusive mode: fine on the cable output, broken on the Realtek microphone

A 2-minute gated soak with `--exclusive` on both streams produced 8,630 hops in 120 s (50/s
expected) and 1,630 "drift trims". Measured directly with a bare `sounddevice.InputStream`:
the Realtek microphone opened in exclusive mode at a nominal 48,000 Hz **delivers 72,045
samples/s** (750 callbacks of 480 frames in 5 s; shared mode: 48,016). The output side of
the Virtual Audio Cable is correct in both modes (48,138-48,181 samples/s; latency 22.0 ms
shared, 10.7 ms exclusive) and the microphone's exclusive latency reads 13.0 ms.

| D25 | `--exclusive` meant both streams | `--exclusive` is the **output** stream only; `--exclusive-input` is separate, and the stats line prints `INPUT RATE ... != 48000` whenever the delivered input rate is off by more than 2% | A 1.5x input rate is inaudible in the counters (no flags, no overruns: everything is "working", just at the wrong speed) and only shows as pitch-shifted, time-compressed output. The check is cheap and permanent. |

The RTT table in R8 used a file-backed input, so its exclusive row is unaffected (the
output path was exclusive). Re-run with the output exclusive and the microphone shared
(`--exclusive --crossfade-ms 20 --fill-ms 40 --gate on`, 2 minutes): 6,009 hops in 120 s,
0 underruns, 0 flags, 0 trims; device latency 22.0 in + 10.7 out; estimate 108 ms with the
gate closed on a silent microphone (the output-ring term reads the full fill target while
gated). For the microphone the shared-mode 22 ms stands until the
driver's exclusive format is understood (PortAudio may be opening a different native rate
and not converting); a USB microphone or ASIO would be the way around it.

## 5. Single-target hardening (2026-10-13): RTX 5060 Ti only, and the expressive TTS layer

`TARGET.md` (repo root) now defines the one supported machine; `CLAUDE.md` points every
session at it. The rules it implies were implemented on `feature/expressive-tts` together
with the expressive typed-speech layer (`docs/TTS_NOTES.md`, "Expressive typed speech").

### 5.1 Deviations

| # | Doc says | What was built | Why |
|---|---|---|---|
| D26 | Device = `cuda` if available else CPU (doc 12, Phase 0-3 code) | `config/target.py::select_device`: the engine resolves to the CUDA device whose name contains "5060 Ti" and asserts cc (12, 0), VRAM >= 15 GiB and the cu128 torch build; any other GPU, a `cuda:N` that is not it, or no CUDA is a `TargetError`. CPU only with `VOX_ALLOW_CPU=1` (set by `tests/conftest.py`). Every `load_engine` logs the check line | TARGET.md rule 1. The iGPU and the virtual display adapter are not CUDA devices, so CUDA enumeration could never pick them; the assertion guards against a second NVIDIA card, a wrong torch wheel, or running on another machine |
| D27 | HuBERT FP16 autocast "default off until Phase 6 validates it" (D12) | `hubert_autocast: "auto"` (default): at warmup the ladder tries FP16 then BF16, renders 30 eager hops of the synthetic warmup signal with seeded decoder noise, and keeps the first candidate whose log-mel L1 against the FP32 eager render is <= 0.05 (the bar the graph replay has to clear). The choice is stored with the profile under the shape key and applied on later startups without measuring | TARGET.md rule 2. The gate is the eager-match check the doc 06 ladder asks for; the numbers are in R10 |
| D28 | cuDNN `benchmark=False` always (doc 06 §5, D-less) | Still False everywhere a stage could run eagerly. During a graph capture's warmup runs `graphs._capture` may turn it on (`StageGraphs.autotune`), so cuDNN times its conv algorithms for the fixed shapes and the captured graph records the winners. Warmup captures once each way, times 60 replays of each, keeps autotune only if it is > 5% faster, and persists the decision under the shape key (`runtime.json: tuned`) | TARGET.md rule 4. PyTorch has no API to export cuDNN's algorithm cache, so the persisted thing is the decision plus the measurements; a later startup captures once (as before) and gets the same kernels because the shapes are the same. Live rebuilds (`worker._rebuild`) reuse the current decisions and never measure |
| D29 | Index cap 200k vectors (doc 07, Phase 2) | `target.index_vector_cap(emb_dim)`: 2 GiB of the 12 GiB engine budget, ~700k vectors at 768-d; the startup log prints reserved/allocated memory against the budget | TARGET.md rule 3 |
| D30 | Output device = host API default (doc 03 §8, Phase 3) | Default output = the first device whose name marks it as a virtual cable (VAC "Line 1"); default input unchanged (the host default microphone); `check_rate` keeps warning when the cable is not at 48 kHz | TARGET.md rule 6 |
| D31 | No suppression setting at runtime (RNNoise is a calibration flag, Phase 5) | `LiveSession` resolves an effective `suppression` at start: explicit `--suppression on/off`, else the profile's `rnnoise_enabled`, except that an input named "NVIDIA Broadcast" defaults it off; the decision and its reason are logged and shown in the session line | TARGET.md: the Broadcast virtual mic is already denoised and double denoising hurts. The RNNoise stage itself is still Phase 5 work; this is the setting it will read |
| D32 | `vox say` renders every piece at speed 1.0 and joins pieces with a constant 200 ms gap | Per-piece speed (kokoro `speed`, edge-tts `rate`), per-boundary pause and per-piece post-RVC gain from `tts/expressive.py`; the plain path is byte-identical (verified against the pre-change tree with a fixed TTS wav) | Feature prompt Part 1; see `docs/TTS_NOTES.md` |

### 5.2 R10. What the hardening changed in the numbers (same machine, GPU otherwise idle)

`vox bench perf --sweep benchmarks/sweeps/perf_graphs.yaml`, same clip, same model, same
settings as the Phase 3 table in R5. New defaults: HuBERT autocast ladder (picked **FP16**
at every point: eager-match log-mel L1 0.028-0.045 against the 0.05 bar, seeded noise,
30 hops), cuDNN autotune measured and **rejected** at every point (persisted as
`cudnn_autotune: false`), TF32 on as before. Logs: `out/perf_graphs_hardening.log`,
`out/graph_probe_hardening_*.log`, `out/perf_locked_hardening.log`.

| hop | extra | mode | Phase 3 p50 / p99 | hardened p50 / p99 | delta p50 / p99 |
|---|---|---|---|---|---|
| 20 ms | 500 | eager | 46.2 / 118.4 | 53.3 / 120.1 | +7.1 / +1.7 |
| 30 ms | 500 | eager | 48.8 / 120.3 | 43.8 / 89.4 | -5.0 / -30.9 |
| 20 ms | 1000 | eager | 41.7 / 114.0 | 40.3 / 113.3 | -1.4 / -0.7 |
| 30 ms | 1000 | eager | 42.4 / 97.6 | 50.7 / 120.5 | +8.3 / +22.9 |
| 20 ms | 500 | **graphs** | 16.9 / 19.1 | **15.8 / 18.4** | -1.1 / -0.7 |
| 30 ms | 500 | **graphs** | 16.2 / 20.8 | **15.1 / 17.2** | -1.1 / -3.6 |
| 20 ms | 1000 | **graphs** | 15.3 / 20.7 | **16.2 / 18.3** | +0.9 / -2.4 |
| 30 ms | 1000 | **graphs** | 15.3 / 20.1 | **16.0 / 19.0** | +0.7 / -1.1 |

- **Graphed p99 is 1-4 ms lower at every point; p50 is flat (within +-1 ms).** The eager
  rows swing +-8 ms either way, which is the session-to-session host launch variance R5
  already noted; eager is the fallback and is not tuned.
- **HuBERT FP16 autocast buys nothing measurable per hop.** Graph probe, embed stage:
  replay p50 3.25 ms (p99 3.85) with FP16 autocast against 2.6-3.0 ms FP32 in R2/R5; f0
  7.36 ms and synth 4.77 ms are unchanged. At a 1.0-1.5 s window the HuBERT forward is
  launch/memory-bound, so halving the matmul precision moves nothing and the autocast
  casts add a few kernels. It passes the eager-match check (L1 0.02-0.045, against a
  same-seed floor of ~0.001 for TF32 rounding and a new-seed floor of 0.30), so it stays
  the default per TARGET.md rule 2; `--hubert-autocast off` is one flag away and the
  decision is in `runtime.json`. **BF16** was never selected because FP16 passed first; in
  the one engine build where FP16 failed before the noise was seeded (L1 0.076, a different
  noise draw), BF16 scored 0.249: the lower mantissa shows, as doc 06 §5 predicted.
- **cuDNN autotune vs heuristics: 14.5-16.2 ms either way, differences of -4% to +2%,
  i.e. noise.** The 5% margin keeps the heuristics; a 2% margin flipped the decision from
  run to run. The autotune path costs one extra capture (~1.2 s) and 2x60 timed replays
  once per shape; subsequent startups read the record and capture once. The thing the
  prompt asked to persist (the chosen algorithm configuration) cannot be exported from
  PyTorch; the persisted decision gives the same kernels because the shapes are fixed.
- **Q16 (power state), re-tested with the hardened engine:** `nvidia-smi -lgc 2000,3090`
  (no admin prompt on this machine) and the hop 20 / extra 1000 graphed point: locked
  15.7 / 18.7 ms against unlocked 16.2 / 18.3. Same result as R5: clocks are not the lever;
  the GPU already sits at P1 / 2.8 GHz under the graphed load. "Prefer maximum performance"
  in the NVIDIA control panel is a per-process driver profile that cannot be toggled from
  the CLI; the clock lock is the stronger version of the same intervention and shows no
  delta, so the control-panel setting is not worth a manual A/B. Q16 is closed.
- **Startup log** now prints, in order: the hardware check line, `hubert autocast: ...`,
  `cudnn autotune: ...` (or `(from profile)` for both), the graphs capture line, and a
  memory line (`reserved 1.2-2.8 GiB of a 12 GiB engine budget`).
- Nothing regressed in the engine tests (`tests/engine`, run after the sweep); the TTS
  path is byte-identical for formal text (TTS_NOTES).
- Caveat on measurement hygiene: a second checkout (`DEXAIV-tts`) was running GPU work
  from another session during the first attempt (`vox run`, `pytest tests/engine`); those
  numbers were discarded and the sweep re-run when `nvidia-smi` showed the GPU idle.

## 6. Phase 8 (2026-10-13): the control API and the DEXAIV window

Built from `docs/gui/GUI_SPEC.md` and the four mockups in `docs/gui/mockups/`, on top of
the Phase 3 live engine and the `vox say` TTS path. Measured on the reference machine with
a looping speech file as the microphone (`--input-file`) and the Virtual Audio Cable as
the output, unless a row says otherwise.

### 6.1 Deviations

| # | Spec said | What was built | Why |
|---|---|---|---|
| G1 | JSON-lines stdio and/or a localhost websocket (doc 12 §3.12, doc 13 Phase 8) | Local **HTTP + server-sent events**, standard library only (`api/server.py`): `POST /api/command`, `GET /api/events`, `POST /api/upload`, bound to 127.0.0.1 on an ephemeral port behind a random token | The UI is HTML, so a browser-native transport needs no client library and no websocket dependency. Commands are request/response and events are one-way, which is exactly SSE. The token keeps other local pages from driving the engine. |
| G2 | "pywebview or PySide6: decide and justify" | **pywebview** shell (`ui/app.py`) over static pages (`ui/web/`), started by `vox gui` as a **separate process** that is given only the URL | The mockups are plain HTML/CSS, so the screens are the mockups' own markup with the inline styles lifted into theme variables; Qt would have meant re-drawing them. The window process imports nothing from `vox` (WebView2 renders out of process as well), so the GUI cannot take the engine's GIL. Without pywebview, or with `--browser`, the same pages open in the default browser. |
| G3 | `EngineController` in `api/control.py` | `api/controller.py` holds `EngineController`; `api/control.py` keeps `LiveSession`, which the controller wraps and `vox run` still uses directly | One object per file; the CLI path is untouched. Slow commands (start, model switch, clip render, import) run in order on one control thread and report through the state document; sliders apply at once. |
| G4 | "Pitch card: value from the profile" | What the user changes is stored in `profiles/<name>/gui_state.json` (gitignored) and layered over the profile; the calibrated files are never rewritten from the GUI | The "auto · calibrated" badge needs the calibrated value to survive an edit. Until Phase 5 writes a calibration the badge reads "profile default". |
| G5 | (nothing) | **Every PortAudio call is made on the control thread** (device listing, stream open/close, clip playback), and the state document serves a cached device list | Found the hard way: PortAudio's WASAPI backend sets COM up on the thread that first touches it. With the device list first read on an HTTP thread, opening the output stream from the control thread failed with `PaErrorCode -9999 Unanticipated host error` (the WDM-KS text in the message is stale). It reproduced whenever the state was requested before the first start and never otherwise. |
| G6 | "Converted mic is muted for the clip's duration" | Muted from the moment the clip's **GPU render starts**: `worker.hold` closes the live voice the way the gate does (SOLA tail faded once, context ring kept current), then `engine/override.py` plays the clip into the same output ring and fades live back in | At hop 20 ms the engine uses ~16.5 of every 20 ms of GPU (R5), so rendering a clip beside an open gate made hops late: **10 underruns over 9 clips** before the hold. The extra mute is the render time, 160-480 ms measured, plus 4 hops for the fade-out to reach the ring. |
| G7 | (nothing) | While held, the worker queues 0.5 s of extra silence in the output ring and takes it back out when the hold ends; it is excluded from the latency estimate | A held hop still waits on the GPU once (the upload event, D21), so it can run late behind the render's kernels. Only silence is queued then, so a cushion of silence costs nothing audible. With it: **0 underruns over 18 clips** (0.9-7.6 s each), latency back to 120-150 ms afterwards. |
| G8 | Index loaded lazily, only when `index_rate > 0` (P6) | When the selected model has an index the GUI starts the engine with it loaded even at rate 0 | Otherwise the Index-rate slider would need an engine restart to leave zero. Cost: the k-NN stage runs every hop, 0.4 ms p50 / 0.9 ms p99 for 44k vectors. |
| G9 | Toggles for Suppression (RNNoise), Monitor in headphones, Auto-tune hop; "Recalibrate my voice" | Shown **disabled** with a "not built into the engine yet" tooltip; the commands answer with that message; Recalibrate says calibration is Phase 5 | None of the four processes audio in the engine. Section 5 added the suppression *setting* (resolved at start, off by default on the NVIDIA Broadcast mic) but no RNNoise stage runs yet, so the switch stays disabled. A switch that does nothing would be worse than one that says so. |
| G10 | "A dropped .pth imports via `vox models import`" | Dropped files are streamed to `POST /api/upload`, then `import_model` calls the same `discovery.import_model`; a `.pth` dropped with exactly one `.index` is staged alone in a folder so the pair is matched | A web page never sees a dropped file's path. Same code as the CLI import from there on. "Browse…" in the Index menu uploads an `.index` into the model's folder the same way. |
| G11 | Profiler stats as in `LiveSession.snapshot` | Fixed a window bug there (stages the gate skips were measured against the total hop count and dropped out of the snapshot once any hop had been gated) and added per-stage CUDA-event times | Under graphs the host time of a stage is the ~0.05 ms launch of its replay; the Performance bars show device time. |
| G12 | Mockup copy | "In use" (disabled) instead of a second "Use" on the active card; tags come from real data (in use, test voice, vector count, speakers); all four screens show the nav icons the Live mockup has; headroom is `1 - p99 / hop`; dips in the hop-time chart are windows of gated hops | The mockups show placeholder data. Where a label had no source in the engine it was derived from what the engine does report. |

The mockups reference a `support.js` that was not in the package, so they do not render on
their own. For the side-by-side comparison they were rendered with a 50-line stand-in
(template substitution only, values from each mockup's own script).

### 6.2 Acceptance

| # | Check | Result |
|---|---|---|
| 1 | `vox gui` opens; Live shows real meters, latency, model | Pass. Meters at 14-15 events/s, profiler at 2/s; latency estimate 121 ms shared mode (hop 20 + ring 27 + compute 16 + SOLA 40 + device 22), 111 ms with the output exclusive. |
| 2 | Start/stop; pitch, gate threshold, gate toggle, index rate, protect live without dropouts | Pass: 12 changes 0.4 s apart with the gate open, underruns 0 -> 0, no rebuild. Suppression is not built (G9). |
| 3 | Typed clip over the cable while live | Pass: 18 clips, 0 underruns, no step at either seam above what the speech itself contains in the cable recording; live resumes. The mic is muted 0.25-0.55 s longer than the clip (G6). |
| 4 | `.pth` + `.index` dropped together | Pass through the upload API with a real pair (54 MB + 132 MB): one card, index paired with 43,857 vectors. The drag gesture itself in the window was not exercised by a human. |
| 5 | Pitch tracker or hop change re-captures in place | Pass: hop 20 -> 30 -> 20 and RMVPE -> FCPE -> RMVPE, 1.8-2.4 s each, session never restarted. CUDA graphs off/on and exclusive on/off reopen the session (3-5 s). |
| 6 | CLI unaffected | Pass: after closing the backend, `vox run` on the same model: 451 hops, 0 underruns, p99 16.7 ms. |
| 7 | Screens match the mockups | Side-by-side captures at 1200x780 under `out/shots/` (`mock-*.png`, `final-*.png`). Layout, spacing, colours and type match; content differs where the mockup shows placeholders (G12). |

GIL check (doc 13): hop p50 / p99 medians over 30 half-second windows were 16.3 / 17.7 ms
with no UI attached and 16.5 / 18.7 ms with a page attached and rendering (one run each,
so within the 20% session-to-session swing already noted).

Another session was running GPU jobs on this machine during part of these runs, so the
timing figures above were not all taken on an otherwise idle GPU.

Tests: `tests/api/` (controller against a fake session, HTTP transport),
`tests/unit/test_override.py`, and two new import rules (`ui/` reaches `vox` only through
`vox.api`; the engine never imports the transport or `vox.tts`).

After merging the section 5 branch (`feature/expressive-tts`) the engine tests (20), the
other tests (314) and the whole live acceptance were run again on an idle GPU: all pass,
9 more typed clips with 0 underruns, hop p50 / p99 15.6 / 16.4 ms. Typed clips in the
window go through the expressive layer (speed, pauses and gain per piece) like `vox say`;
the latency readout stays at ~130 ms while a clip renders (the cushion is not counted).

Pre-push check of the merged branch found two faults in the hold (G6/G7), both fixed and
now covered by `tests/engine/test_hold.py` and `tests/unit/test_rings.py`:

- The worker read `hold` twice in one hop. When the control thread raised it in between,
  the hop was a normal speech hop but the 0.5 s cushion was queued *in front of it*: the
  live voice cut straight into silence and one hop of speech played half a second later
  (3 of 14 clips). The flag is now read once per hop. The cushion also went in ahead of
  the last 10 ms of the fade-out; it now waits for a hop that is pure silence.
- The cushion was taken back out with `SPSCRing.drop` from the worker thread, which races
  with the output callback's own position update; a lost update left the cushion in as
  0.5 s of latency until the next drift trim cut it out of live speech (one underrun and a
  cut 0.8 s after a clip). The worker now uses `SPSCRing.request_drop`, applied by the
  reader on its next read.

After the fixes: 24 clips in a row with no cut and no underrun; then "nooooo waaaay",
"hey... hey!!" and "WAIT... what?! no way, that is SO cool!!" typed while live: the clips
last what `vox say` renders (1.16 s, 1.67 s, 4.19 s), the pauses are there (0.41-0.53 s at
-66 to -72 dB on the cable while live speech sat at about -13 dB before the clip, so the
microphone does not leak into them), the largest step at a clip seam is 0.11 against 0.22
inside ordinary speech, and a gained clip peaks at 0.985 with the same RMS as a plain one.

### 6.3 Not verified

- A real microphone through the GUI: every live run used the file-backed input.
- Listening: clicks and gaps were judged from the cable recording, not by ear.
- The window itself under long use; headless Chrome rendered the captures.
- Kokoro's first use after a fresh install (weights download) from inside the GUI.

## 7. Live-mic quality (2026-10-14): the external audio report, checked against the code

A code-aware report on a real session (live mic raspy, whispery and chopped; typed speech
clean) listed nine "bugs". Each was re-checked at 5fa21d2, the confirmed ones fixed, the
rest turned into A/B variants. Same machine; NVIDIA Broadcast, OBS and browsers were
resident on the GPU (~4.5 GB, 8-13% utilisation) for every run, so "idle GPU" means no
other compute job.

**Test material.** The report's recording used **MaeASMR** (32 kHz, no index); the work
used **Mom** (32 kHz, 43,857-vector index), the voice the GUI had selected. The raw
microphone is the OBS capture of 2026-10-13 22:55 (default mic at unity, no filters): a male
voice, F0 median ~100 Hz (p10 67 Hz), speech at **-42 dBFS** active RMS over a -60..-98 dBFS
floor, with quiet bleed (another voice, ~200 Hz, 20 dB below) in the pauses. Its two speech
passages make `benchmarks/corpus/mic_script.wav` (25.75 s, not committed). Settings: Mom,
pitch +12 (the saved GUI value), hop 20, overlap 30 requested, extra 1000, index rate 0,
`f0_unvoiced` zero, envelope 1.0.

**Gate threshold.** The prompt said -31 dB; the GUI state saved at the end of the reported
session says -57 dBFS. At this mic level a -31 gate never opens (0.1% of 20 ms blocks reach
it; measured: 100% of hops closed), so the baseline is at -57. After input conditioning the
gate measures the lifted level and -40 is the setting that works (7.3).

**Measures** (`vox bench accept`, `offline/metrics.py`): gate transitions per 8 s of script;
*phrases* = input frames within 18 dB of its loud frames, gaps < 250 ms bridged (11 spans,
13.2 s: the talker, not the bleed); *speech lost* = digital-zero runs >= 10 ms inside
phrases, each attributed to gated hops or an underrun; voicing = RMVPE on the output over
active frames; HF/LF = 90th percentile of the 3-12 kHz / 0.1-1 kHz energy ratio; R3 =
speech-frame log-mel L1 of the gate-off stream against the full-file render, over the
new-seed floor. "Typed" = the report's two sentences through Kokoro `af_heart` + Mom.

### 7.1 Deviations

| # | Prompt / report said | What was built | Why |
|---|---|---|---|
| L1 | SOLA: with overlap > block the saved tail is double-windowed; copy it before the crossfade **and** clamp, default 20 ms | Clamp (overlap <= hop) and the 20 ms default; the **save order kept** (tail saved after the crossfade); `overlap_beyond_hop` keeps the old geometry as an A/B setting | The old order is not a seam bug: the tail it saves starts on the mixed sample the emitted block ended on, a three-render blend whose weights sum to one (envelope step / crossfade slope 1.00, `test_sola.py`). Copying first, at overlap > block, steps at every seam (229x). Under the clamp the order is immaterial. The 20 ms default costs seam smoothness (7.3) |
| L2 | Strong onset (>= threshold + 6 dB) opens before the VAD | Also needs a 12 dB rise over the quietest block of the last 200 ms; held 100 ms unless the VAD confirms | After auto level the room tone of this mic sits above threshold + 6 on its own: without the rise rule the gate reopened every ~120 ms through pauses |
| L3 | Hysteresis: close at threshold - 6 dB | Same, plus VAD hysteresis (stay open down to 0.5 - 0.15) | Silero's own neg_threshold rule |
| L4 | Hangover 300-350 ms | 350 ms (`calibration.json` `release_ms`, `VadProfile` default, `GateConfig`) | The profile's 200 overrode the gate's default |
| L5 | AGC: target -20 dBFS active RMS, max +18 dB, frozen below the noise floor, soft limit near -3 dBFS; HPF 70-80 Hz | As specified (`audio/conditioning.py`): level = 500 ms mean power of blocks above floor + 6 dB, gain up 400 ms / down 60 ms ramped per block, gain held in pauses, tanh knee from -6 to -3 dBFS, 2nd-order HPF at **70 Hz** | A symmetric power average is the active RMS; a fast-up dB tracker followed the syllable peaks and landed 3.6 dB low. 70 Hz: this talker's F0 reaches 67 Hz |
| L6 | The gate measures the post-conditioning level | So does the IN meter (the gate's level) | A threshold set on the raw mic (-57) then passes lifted room tone and bleed to the VAD: 12.4 toggles / 8 s. The setting moves to about -40 dBFS |
| L7 | Typed speech must not pass the input chain | By construction: clips go to the output ring | |
| L8 | (none) | Step 0 tooling: `StageTimers.events` (gate open/close, underrun; hop + time), `SimResult.events` and a modelled output ring in paced runs, `OutputTap` + `vox run --record-output` + `scripts/analyze_tap.py`, `vox bench ab` / `typed` | Every cut needs a cause |
| L9 | GUI input-gain slider as a trim | Added (there was none): "Input gain" -12..+24 dB and "Auto level", stored in `gui_state.json`, hot | |

This supersedes part of D7 (the overlap in use is now `min(crossfade, 40 ms, hop)`) and of
D20 (on close the pitch cache advances with the ring instead of being zeroed, and the
features are no longer zeroed; the SOLA drain and the output buffers are as before).

### 7.2 Step 0: attribution, and the baseline (5fa21d2)

Gate closures and underruns are logged apart with hop index and time; `vox bench accept`
prints both and the cut attribution; the live tap marks every sample the output callback
zero-filled. Baseline, gate -57 dBFS raw, hangover 200 ms:

| | baseline |
|---|---|
| gate transitions / 8 s (closures in 25.7 s) | 8.7 (14) |
| speech lost to cuts | 860 ms in 3 cuts, all gate (the report's measure, which counts the bleed as speech: 7 cuts, 1,260 ms) |
| underruns | 0 (paced simulation); live, 90 s looped through the VAC: 0, no zero-filled sample, 54 zero runs all from gated hops, 21 of them mid-speech |
| voiced: stream (its active frames; on the frames shared with the full-file render) / full-file (shared frames) / typed | 74.3% (76.5%) / 75.0% / 79.3% |
| HF/LF p90, same columns | +11.8 (+12.0) / +14.5 / +2.8 dB |
| R3 (Mom) | 0.671 vs floor 0.266 = **2.52x** (test voice: 1.82x); seam click ratio 1.02 |
| latency | algorithmic 60 ms; live estimate median 124 ms; tap trails input 90 ms |

So on this machine every cut was the gate; none was an underrun. The HF/LF excess is not
a streaming artefact: the full-file render of the same input is brighter still.

### 7.3 Step 1: the five fixes, measured cumulatively

Same script and settings; gate -57 raw until the input chain is in.

| after | trans / 8 s | closures | lost ms | click | R3 | voiced | HF/LF |
|---|---|---|---|---|---|---|---|
| baseline | 8.7 | 14 | 860 | 1.02 | 2.52x | 74.3% | +11.8 |
| 1 SOLA clamp, overlap 20 | 8.7 | 14 | 880 | **1.33** | 2.56x | 71.1% | +11.9 |
| 2 output resampler, 3 pitch cache | 8.7 | 14 | 880 | 1.34 | 2.56x | 71.4% | +12.1 |
| 5 gate (raw input) | 6.8 | 11 | 50 | 1.34 | 2.56x | 70.3% | +13.0 |
| 4 input chain, gate -57 | 12.4 | 20 | 40 | 1.38 | 2.73x | 67.2% | +12.6 |
| 4 input chain, **gate -40** | **5.6** | **9** | **150** | 1.39 | 2.73x | 68.4% | +12.6 |
| 4 input chain, gate -31 | 5.6 | 9 | 181 | 1.43 | 2.73x | 70.4% | +12.6 |

(voiced and HF/LF over the gated stream's own active frames; click ratio of the gate-off
stream, +-0.03 between identical runs. The gate row "5" predates the onset rise rule of
L2, which only matters once the input chain lifts the room tone.)

1. **SOLA.** Latency -10 ms (algorithmic 60 -> 50). The seam ratio rises to 1.33-1.4 on Mom,
   1.34 on the test voice, 1.41 on the 40 k test voice: with overlap == hop every sample but
   the seams is a 50/50 mix of two partly uncorrelated renders, the seams are single
   renders. The engine's own seam guard (< 1.25) fails; `tests/engine/test_engine.py`
   records it as two strict xfails pending the listening A/B, and checks that the old
   geometry passes. Noise rolling does not cure it (7.4).
2. **Output resampler.** Now carries 10 ms of the previous tail; a 32 k / 40 k streamed
   sine matches the whole-signal resample to < 1e-4 (stateless: > 1e-2 at the left edge).
   No measurable effect on the stream: SOLA already weights both edges ~0.
3. **Pitch cache.** Closed hops shift it like open ones (`advance_silent`), aligned with the
   ring; features are no longer zeroed. Within the metrics' noise: after a reopen RMVPE
   rewrites the newest 280 ms, the decoded tail reads only the newest 50 ms.
4. **Input chain.** Lifts this mic by the full +18 dB (speech -42.7 -> -24.7 dBFS). With the
   gate at -40 it ends the soft-word cuts; at -57 the lifted room tone and bleed reach the
   VAD (gate open 6.5 s outside phrases). R3 rises 2.56 -> 2.73x: the stream's input is now
   high-passed and lifted, the full-file reference's is not.
5. **Gate.** The VAD sees every block; hysteresis, 350 ms hangover, onset rule (L2). At -40
   after the input chain: one open and one close per phrase (5.6 per 8 s is exactly 2 x 9
   phrases), none inside a phrase, 150 ms of speech lost against 860.

### 7.4 Step 2: A/B variants (on the Step 1 engine, gate -40; defaults unchanged)

`vox bench ab --model Mom --audio mic_script.wav --pitch 12 --gate-threshold -40`; each wav
is the gated stream in `out/ab/<variant>.wav`.

| variant | R3 | click | voiced | HF/LF | level | reading |
|---|---|---|---|---|---|---|
| reference (Step 1) | 2.73x | 1.39 | 68.3% | +12.6 | -13.9 | |
| sola_blend3 (30 ms at the 20 ms hop) | **2.65x** | **1.00** | 71.8% | +13.3 | -13.9 | better seams and R3, +10 ms |
| overlap_10 | 2.89x | 0.86 | 69.4% | +10.9 | -14.6 | smooth seams, worse R3 |
| raw_input (no input chain) | 2.56x | 1.33 | 80.2% | +10.7 | -13.6 | 2.6 s lost at -40 (gate tuned for the lifted level) |
| **f0_interp** | 2.73x | 1.33 | **76.8%** | **+8.3** | -13.8 | 4.3 dB less hiss, 8.5 points more voicing |
| envelope_0.3 | 6.64x | 1.28 | 64.7% | +11.6 | -21.1 | follows the mic envelope: 7 dB quieter |
| index_0.35 | 2.64x | 1.36 | 68.1% | +12.9 | -14.4 | |
| index_0.6 | 2.62x | 1.33 | 68.2% | +13.2 | -15.0 | |
| embed_pad (240 samples) | 2.87x | 1.35 | 68.8% | +13.8 | -14.6 | worse on both |
| f0_bridge (<= 60 ms, 4 st) | 2.73x | 1.39 | 69.9% | +12.8 | -13.9 | no measurable change |
| decctx_80 | 2.73x | 1.36 | 69.6% | +12.7 | -13.7 | no measurable change (as Q17) |
| noise_roll | 2.74x | 1.44 | 70.7% | +13.4 | -13.8 | no measurable change |

Gate behaviour is identical across variants (5.6 / 8 s, 0 inside phrases, 140-160 ms lost)
except raw_input. Hop p99 17.6-18.8 ms for all.

### 7.5 Step 3: typed speech (A/B, defaults unchanged)

`vox bench typed --model Mom`: the report's two sentences, Kokoro, auto pitch;
`out/ab/typed_<variant>.wav`. Options: `TypedPolish(envelope, rms_frame_ms,
contour_scale)` in `convert_rendered`, voice blends as `"af_heart*0.7+af_bella*0.3"`.

| variant | F0 sd | loudness sd | voiced | HF/LF |
|---|---|---|---|---|
| plain | 3.01 st | 8.93 dB | 77.9% | +3.0 |
| envelope 0.35, 40 ms frames | 3.11 | 8.17 | 78.2% | +3.3 |
| contour x1.2 | **3.43** | 9.03 | 79.1% | +2.6 |
| blend 0.7 af_heart + 0.3 af_bella | 2.74 | 8.92 | 80.0% | +1.6 |
| all three | 3.30 | 8.60 | 78.9% | +0.9 |

The report's "flat wall of sound" (`_change_rms` skipped) does not show: the plain render
already varies 8.9 dB, and the 40 ms envelope mix narrows it. The contour scale does what
it says (+14% F0 spread, median unchanged).

### 7.6 Acceptance

| check | before | after (gate -40) | target | |
|---|---|---|---|---|
| Step 1 fixes in, with tests | | 368 CPU tests pass; engine 25 pass + 2 strict xfail (L1) | suite green | met except the 20 ms seam guard |
| gate toggles / 8 s | 8.7 | 5.6, 0 inside phrases | <= 4 | not reachable on this script without bridging real pauses: 9 phrases with gaps of 0.5-1.4 s need 5.6 |
| mid-speech zeros attributed | 100% gate | 100% gate (sim and live tap) | all | met |
| underruns, idle GPU | 0 | 0 (sim; live 90 s: 0, no zero-filled sample) | 0 | met |
| voiced ratio vs typed (79.1%) | 74.3% | 68.4% (f0_interp: 76.8%) | within 8 points | met only with f0_interp |
| HF/LF p90 vs typed (+2.5) | +11.8 | +12.6 (f0_interp +8.3) | within 4 dB | not met; the full-file render of the same input is +18.9 |
| R3 for Mom | 2.52x | 2.73x | recorded | |
| latency | 60 / 124 / 90 ms | 50 / 119 / 80 ms | ~-20 ms | -5 to -10 ms (algorithmic / live estimate / tap lag) |

### 7.7 Not verified

- Nothing was listened to: the A/B wavs are for the user's ears, and no Step 2/3 default
  was changed.
- The engine's input level as heard in the GUI session: taken equal to the OBS capture.
- MaeASMR, the model of the report's own recording.
- Live runs used the file-backed input; the Shure MV6 was not connected.

## 8. Singing, onsets, noise suppression (2026-10-15)

The user now sings as well as speaks (sung F0 180-280 Hz against ~100-130 Hz speech). Two
takes recorded with the Record button were analysed by a second external report:
`take_20261015_004247` (Mom, 7 s) and `take_20261015_004329` (GoatMommy, 113 s), both
through the NVIDIA Broadcast virtual mic with hops of 60 ms. Its findings were triaged by
the feature prompt (v3); where the two differ, the prompt was followed. Same machine; NVIDIA
Broadcast, Discord, browsers and OBS were resident on the GPU throughout (no other compute
job), so "idle GPU" below means that. No A/B of section 7 had been listened to when this
work started (the user had not found the files), so none of them became a default.

### 8.1 Deviations

| # | Prompt / report said | What was built | Why |
|---|---|---|---|
| S1 | Persisted runtime.json / gui_state.json store only what the user changed; new engine fields fall through | `RuntimeProfile`'s shape and precision fields and the voice profile's F0 method / range are `None` = "engine default" and never written (`store.save_as` drops unset values); `profiles/default/runtime.json` and `voice_profile.json` lost the ten copies of defaults they held; `gui_state.json` is saved compacted (no nulls, no empty groups, `exclusive` only once switched) | `save_tuned` rewrote the whole `RuntimeProfile` on every warmup, so runtime.json held a full copy of the defaults of the day: any later default (e.g. `decoder_context_ms`) would have been overridden by a stale `0.0`. The report's claim that the A/B fields were *disabled by* the profile is half right: they were off because they are off by default (no A/B approved); the clobbering was latent |
| S2 | Shape / plan desync (take 1: shape 60, plan 30) | `LiveSession.describe()` and the take JSON read the worker's plan and a new `worker.shape` (set from the plan the chunker built at each rebuild); `engine.plan` is only the plan at start | Take 1 ran at 60 ms (119 hops in 7.14 s, fill target 140 = 70 x 60/30). The session had started at 30 ms and been rebuilt; the take reported the start plan |
| S3 | Hop snapped to multiples of 20 ms in the controller and CLI; GUI slider steps by 20 | `chunker.snap_block_ms` (halves round up, minimum 20) in `EngineController.set_shape` and `_shape` (an old saved 30 becomes 40), `vox run --block-ms` / `hop MS`, `vox convert`; slider 20-100 step 20 | `hubert_phase_stable` always holds. The chunker still plans any 10 ms multiple: `vox bench` measures 30 ms hops on purpose |
| S4 | Gate VAD: max probability over the hop's 512-sample chunks | `GateConfig.vad_reduce = "max"` (default); `"last"` keeps the pre-fix rule; a hop that completes no chunk keeps the previous probability | Stated in the prompt |
| S5 | Onset: short (~5 ms) ramp instead of the overlap fade from a silent SOLA tail | `ShapeParams.onset_ramp_ms = 5` (0 = pre-fix); `Sola.mark_onset()` blends to the short sin^2 ramp through a 0-dim device weight inside the captured SOLA graph, for one hop; the worker and the simulator call it (and `pipeline.on_onset()`) on a gate onset when no drain is pending | The head of the onset hop is decoded audio from just before the block (the overlap + search decoded ahead of it), i.e. the consonant the gate opened on. It can only keep what was decoded: `overlap + search` = 30 ms before the block. `pv` SOLA mode is unchanged |
| S6 | Speak / Sing preset switch, per model, each with its own pitch, index rate, protect, ceiling; Sing pitch = Speak - 5..6 st (configurable), index 0.3, protect 0.35 | Controller-level presets: the model entry of `gui_state.json` is Speak (as before), `models.<name>.sing` holds only the Sing values the user moved; Sing defaults: Speak pitch - 5 (`sing_pitch_offset` overrides), index 0.3, protect 0.35, ceiling 360 Hz; switching is hot (four `set_params` fields + the input chain), no recapture. Live screen: Speak / Sing next to Pitch; Settings: a Pitch ceiling slider (per preset) | The engine knows nothing of presets: they are four hot parameters and two input-chain bounds |
| S7 | Soft F0 ceiling, semitone soft knee after transposition; off in Speak, ~360 Hz in Sing | `RuntimeParams.f0_ceiling_hz` (hot, 0 = off): `pitch.base.soft_ceiling`, unchanged up to 4 st below the ceiling, then `knee * 2^(4 tanh(st/4) / 12)`; a device scalar (inf = exact no-op), applied to the continuous F0 before the coarse buckets, live and in the full-file converter | It bends very high notes off-key and is a safety, not the fix: with the ceiling at 360 Hz a transposed 300 Hz note is untouched (0.01 st), 330 Hz comes out 0.28 st flat, 400 Hz 2.2 st flat, and 600 Hz is pinned near 360 Hz (8.9 st flat) (`tests/unit/test_pitch.py`) |
| S8 | RMVPE right-edge guard: `TRIM_TAIL` 1 -> 2, A/B the pitch step at hop seams | `RuntimeParams.f0_trim_tail`, a recapture-class flag like `f0_bridge_ms`; **default 2** (the measurement in 8.4 agreed); `--f0-trim-tail 1` / `vox bench ... --set f0_trim_tail=1` is upstream | Like upstream, the trimmed estimate is written flush with the end of the cache, so the guard also puts F0 10 ms later against the features: it buys the seam frames 10 ms more right context that way. `test_pitch_cache_matches_full_window_f0` now states the alignment for either value |
| S9 | Experimental `sine_phase_carry`: SineGen phase continuous across hops; persistent 0-dim buffer advanced by `shift_frames` each hop; reset on gate onset and recapture; graph-safe | As specified, but the buffer is the pipeline's (`HopPipeline.sine_phase`, `[1, 1, 1]`) and is passed down through `RvcModel.infer` -> `SineGen.forward(phase, phase_frames)`, which reads it and writes the phase at frame `shift_frames` in place. **On by default** since the listening in 8.8, together with `noise_roll` and 80 ms of decoder context | The loaded model is shared with typed-clip renders on a second CUDA stream: a phase buffer stored on the model would be advanced by the clip render and raced by it |
| S10 | RNNoise (48 kHz, 10 ms frames) after the high-pass, before AGC and the gate; live toggle with ~20 ms crossfade; `suppression_mix` | `audio/suppression.py`: the `rnnoise.dll` of the **pyrnnoise 0.4.5** wheel (xiph RNNoise, installed `--no-deps`, `requirements/suppression.txt`) through ctypes on preallocated buffers, one state carried across hops; mix and toggle as specified, plus a 100 ms unheard warm-up after an idle period; the user's on/off is remembered per microphone (`gui_state.json`), so the Broadcast mic keeps its own default (off, TARGET.md) | pyrnnoise is maintained and ships a Windows DLL (the machine has no compiler); its Python layer allocates per frame and imports audiolab and matplotlib, so only the DLL is used. Measured: **20 ms delay** (960 samples, correlation 0.99) and **0.39 ms CPU per 10 ms frame** (8.5) |
| S11 | Sing preset: AGC max +10 dB, trim 0 dB; one-time toast when Broadcast is the input while singing | As specified; the input-gain slider is locked at 0 in Sing; the toast shows once per app run, on switching to Sing, choosing the Broadcast mic, or starting with both | |

Not followed from the report (hard rules or not approved): lowering SineGen's unvoiced
noise (`sine_amp / 3` is what the models were trained with); `f0_unvoiced = "interp"` as a
default (it disables protect, which keys on `pitchf < 1`); `embed_pad_ms = 20`,
`f0_bridge_ms = 50`, `volume_envelope = 0.4`, protect 0.35 / index 0.3 as engine defaults
(the user heard no difference between index 0.35 and 0.6, 8.8; the Sing preset carries index
and protect only for singing); `decoder_context_ms` and `noise_roll` did become defaults, at
80 ms and together with the phase carry, because the user chose that combination by ear
(8.8), not on the report's reasoning; a 40 ms default hop; an onset margin of 3 dB (measured as an
A/B, 8.3); a protect floor above 300 Hz. Two of the report's factual claims did not hold:
the gate no longer wipes the pitch cache on close (section 7, fix 3), and "Promise you" was
not cut by 170 ms (8.3).

### 8.2 Step 1: hop size and the underruns

The take-1 JSON counted 222 underruns *in the session*, none in the take. The user's
account: "20 worked fine on a model but was stuttering on another, but 10 was worst on
both" (the old hop slider went down to 10 ms). Measured with `vox bench underrun` (paced:
hop N is available at N x hop, the output ring modelled as live with the default fill of
two hops + 10 ms; 20 s of `mic_script.wav`, pitch +6, input chain on, gate off so every hop
runs the model) and two 60 s live runs (`vox run --input-file ... --loop-file --output
"Line 1" --exclusive --gate off`, the takes' output mode). "+Broadcast" = the NVIDIA
Broadcast mic held open by a side process (audio discarded) so its denoiser runs on the
same GPU, as when it is the input.

| model | hop | extra | hop p50 / p99 (ms) | underruns |
|---|---|---|---|---|
| GoatMommy (48 kHz) | 20 | 1000 | 14.8 / 16.5 | 0 |
| GoatMommy | 40 / 60 | 1000 | 14.9 / 16.6, 15.1 / 18.2 | 0 / 0 |
| GoatMommy | 20 | **2200** (the takes) | 15.9 / 17.7 | 0 |
| GoatMommy | 40 / 60 | 2200 | 15.9 / 17.4, 16.3 / 17.9 | 0 / 0 |
| GoatMommy +Broadcast | 20 | 1000 / 2200 | 14.8 / 16.5, 16.1 / 18.0 | 0 / 0 |
| GoatMommy +Broadcast | **10** | 2200 | 15.9 / 17.6 | **361 in 20 s** |
| Mom (32 kHz) | 20 / 40 / 60 | 1000 | 14.8 / 16.6, 14.9 / 16.6, 15.5 / 19.0 | 0 / 0 / 0 |
| Mom | 20 / 40 / 60 | 2200 | 15.9 / 17.6, 15.5 / 17.7, 15.8 / 17.7 | 0 / 0 / 0 |
| Mom +Broadcast | 20 | 1000 / 2200 | 15.0 / 16.2, 16.5 / 18.5 | 0 / 0 |
| live, GoatMommy, exclusive | 20 | 2200 | window p99 max 19.1 (60 s, 3,007 hops) | 0, no PortAudio flag |
| live, GoatMommy +Broadcast | 20 | 2200 | window p99 max 18.9 (60 s) | 0, no flag |

- **Hop 10 cannot work on this GPU**: the hop costs ~16 ms (R5), over a 10 ms budget every
  time. 361 underruns in 20 s; a few seconds there account for the session's 222. The hop
  is now snapped to multiples of 20 ms everywhere a user picks it, so 10 is no longer
  offered.
- **Hop 20 does not underrun** on an idle GPU at either context, with either model, with or
  without Broadcast's denoiser running (it costs ~0.3-0.9 ms of p99 at 2200 ms context).
  The "stutter on one model at 20 ms" is not reproduced as an underrun. What does change
  between 20 and 60 ms on sung notes is the seam rate (50 vs 17 per second) with SOLA
  offsets jumping ~3 ms at most seams (8.4), which is audible as flutter on held notes
  rather than as gaps. The Performance screen's underrun counter tells the two apart on the
  user's side: flutter with 0 underruns is the former.
- Not hidden by a larger hop: the default stays 20 ms. Take 1's "fill target 140 ms" is a
  70 ms target (a 30 ms start) scaled by the rebuild to 60 ms.
- The 2-minute gated soaks of 8.5 show p99 windows of 21-24 ms at the same script position
  every loop (the first hops after a gated pause), with or without suppression: a pre-existing
  effect, absorbed by the 50 ms fill target except once with suppression on.

### 8.3 Step 2: phrase entrances

`vox bench onset` on take 2's raw mic track from 21.0 s through 75 s (so the auto level has
its session state at both entrances), GoatMommy +6, gate -45 dBFS, trim -3 dB (the take's
settings), hop 60 (the takes) and 20. "Lag" = the first output frame within 30 dB of the
output's loud frames, minus the input entrance (10 ms frames).

| entrance | hop | pre-fix | VAD max | + 5 ms ramp (default) | A/B: onset margin 3 dB |
|---|---|---|---|---|---|
| "Promise you" (22.0 s) | 60 | gate +20 ms, lag 0 | same | same; first 30 ms +2.7 dB | same as default |
| | 20 | gate +60 ms, lag 40 | same | same; first 30 ms -0.9 dB | same as default |
| "In you and me" (72.83 s) | 60 | gate **+190 ms**, lag 160 | same | same; first 30 ms +1.9 dB | gate **+10 ms**, lag 0 |
| | 20 | gate **+170 ms**, lag 140 | same | same | gate **+30 ms**, lag 0 |

- **"Promise you" was not amputated.** On the Broadcast mic the pause before it is digital
  silence (Broadcast gates its output) and the "Pr" is -75 to -60 dBFS in the raw track:
  Broadcast removed the plosive before the engine saw it. The energy-onset rule (a rise of
  >12 dB to >6 dB over the threshold) opens the gate on the first loud hop. The report's
  21.85 s start does not match the track, which is digital silence until 21.96 s.
- **"In you and me" was clipped by 140-190 ms, and the two prompt fixes do not change it.**
  The sung entrance is soft: about -41 dBFS after conditioning, 4 dB over the -45 gate and
  2 dB under the +6 dB energy margin, so the gate waits for Silero, which reaches 0.5 only
  ~140 ms in (0.09, 0.36, 0.39, 0.25, 0.66 over the first 32 ms chunks). Taking the max over
  a hop's chunks cannot help when every chunk is under 0.5, and the ramp can only keep what
  is decoded ahead of the block (overlap + search = 30 ms).
- **An energy-onset margin of 3 dB opens it 10-30 ms after the entrance** (the A/B column;
  `GateConfig.onset_margin_db`, unchanged default 6). Not made a default: it was not asked
  for, and on a raw mic whose room tone the auto level lifts it may reopen the gate in
  pauses more often (section 7 needed the rise rule for exactly that; not measured here).
- The short ramp does what it should where the onset hop has decoded audio before the block:
  the first 30 ms of output gain ~2-3 dB at hop 60; at hop 20 the onset hop's head is the
  previous hop's region and gains little. `tests/unit/test_sola.py` and
  `tests/engine/test_singing.py` cover it (a decoded burst at the head of the onset hop keeps
  >65% of its energy against <15% with the overlap fade).
- Listening clips (L = raw mic, R = output): `out/ab/onset_{promise,inyou}_{before,vad_max,
  after,margin_3db}_hop{20,60}.wav`.

### 8.4 Step 3: singing

`vox bench sing` on take 2's first 13.5 s (continuous singing, 7.8 s of held notes), Sing
preset (index 0.3, protect 0.35, ceiling 360 Hz, input chain +10 dB max / trim 0), pitch +6,
hop 20 unless stated, every hop through the model. Offsets and jumps after the first second;
"held" = hops that kept the previous hop's SOLA offset; "seam" = |cents| between the F0 a hop
rendered its saved tail with and the F0 the next hop renders the same frames with (voiced
frames). Click = seam vs mid-hop roughness (+-0.07 between seeds, 8.6). These ran before the
trim-tail default changed, so every row but `trimtail 2` has `f0_trim_tail = 1`.

| model, hop | variant | offset p50 | held | jump mean / p95 (ms) | seam p50 / p95 (cents) | click | output F0 p95 / max (Hz) |
|---|---|---|---|---|---|---|---|
| GoatMommy, 20 | reference (trim 1) | 5.88 ms | - | 2.96 / 7.35 | 1.6 / **21.1** | 1.35 | 325 / 341 |
| | **trim tail 2** | 6.33 | - | 2.95 / 7.57 | 1.1 / **9.9** | 1.41 | 325 / 341 |
| | phase carry | 5.58 | - | 2.56 / 7.27 | 1.6 / 21.1 | 1.48 | 326 / 341 |
| | noise roll | 5.81 | 7% | 2.53 / 7.25 | 1.6 / 21.1 | 1.44 | 325 |
| | carry + roll | 5.00 | 9% | 2.03 / 7.26 | 1.6 / 21.1 | 1.37 | 325 |
| | carry + roll + 80 ms decoder context | 4.02 | 19% | 1.15 / 5.07 | 1.6 / 21.1 | 1.40 | 326 |
| | no ceiling | 5.77 | - | 3.04 / 7.73 | 1.7 / 21.1 | 1.33 | **331 / 357** |
| GoatMommy, 60 | reference / trim 2 | 3.99 / 4.41 | - | 2.90 / 3.44 | 1.8 / 22.2 -> 1.3 / **17.3** | 0.86 / 0.80 | 324 |
| | phase carry | 4.08 | - | 2.71 / 6.32 p95 | 1.8 / 22.2 | 0.92 | 323 |
| Mom, 20 | reference (trim 1) | 5.71 | 1% | 3.05 / 7.53 | 1.6 / 21.1 | 1.24 | 332 |
| | trim tail 2 | 5.83 | 1% | 2.95 / 7.70 | 1.1 / **9.9** | 1.33 | 332 |
| | phase carry | 6.19 | 2% | 2.59 / 8.01 | 1.6 / 21.1 | 1.39 | 331 |
| | carry + roll + 80 ms decoder context | 4.52 | 11% | 1.54 / 6.06 | 1.6 / 21.1 | 1.43 | 330 |
| | no ceiling | 5.85 | 1% | 3.05 / 7.71 | 1.7 / 21.1 | 1.31 | 336 |

On speech (`vox bench ab`, Mom +12, `mic_script.wav`, gate -40, the section 7 setup):

| variant | R3 | click | voiced | HF/LF p90 |
|---|---|---|---|---|
| reference (trim 1) | 0.725 (2.73x) | 1.31 | 67.5% | +13.0 dB |
| trim tail 2 | 0.727 (2.74x) | 1.38 | 69.4% | +12.5 dB |
| phase carry | 0.728 (2.74x) | 1.23 | 69.8% | +13.0 dB |

- **Trim tail 2 halves the pitch step at the seams of held notes** (p95 21 -> 10 cents at
  hop 20 on both voices, 22 -> 17 at hop 60) and is neutral on speech R3. It costs a
  slightly rougher seam by the click metric (+0.06-0.09, at the edge of the seed spread)
  and puts F0 10 ms later against the features (S8; the +1.9 points of voicing on speech
  are likely voicing reaching 10 ms further into consonants). Made the default, as the
  prompt states; `f0_trim_tail = 1` is the A/B the other way.
- **The ceiling** at 360 Hz pulled GoatMommy +6's sung peaks from 357 to 341 Hz (F0 p95
  331 -> 325); on Mom +6, p95 336 -> 332. On these takes it rarely engages; it is there for
  +11-type transpositions of a sung line.
- **Phase carry does not make the offsets collapse toward 0** (at 0: 1-2% of hops in every
  variant). Alone it trims the mean offset jump by 15% and roughens the seams (click +0.13);
  combined with noise roll and 80 ms of decoder context the renders become more coherent
  (19% of hops keep the previous offset, mean jump 1.15 ms) but the click ratio still ends
  up worse than the reference. The sine phase is not what decorrelates consecutive hops: the
  latent noise is redrawn per hop (noise roll fixes that), and the head of each decoded slice
  sits on the decoder's zero-padded left edge (decoder context fixes that). On held notes
  SOLA also jumps between whole F0 periods (2.5-4 ms at 255-395 Hz), which is benign. Not
  clearly helpful: `sine_phase_carry` stays off, and the 5 ms search was not tried (the
  chunker plans search in whole 10 ms frames; 5 ms would need sub-frame planning, and the
  premise did not hold).
- Wavs: `out/ab/sing_{phasecarry_off,phasecarry_on,noiseroll,phasecarry_noiseroll,
  phasecarry_noiseroll_decctx80,trimtail_2,ceiling_off}.wav` (GoatMommy, hop 20),
  `hop60_sing_*` (GoatMommy, hop 60), `mom_sing_*` (Mom, hop 20), `s8_{reference,
  trimtail_2,phase_carry}.wav` (speech).

### 8.5 Step 4: noise suppression

`vox bench suppress` (CPU; RMVPE only to find held notes) on the raw default-mic OBS capture
of 2026-10-13 (49 s, unity gain, no filters) and take 2's mic track; live numbers from two
2-minute soaks (GoatMommy +6 with the Sing preset's values, hop 20, extra 1000, gate -40,
`mic_script.wav` looped, VAC exclusive).

| measure | result |
|---|---|
| added latency | **20.0 ms** (960 samples; cross-correlation 0.99). Blocks are whole 10 ms frames, so there is no further alignment buffer. The latency estimate and the take JSON (`input_chain_delay_ms`) include it while it runs |
| CPU per hop (worker thread) | 20 ms hop: p50 0.78 ms, p99 1.02 ms; 60 ms hop: 2.28 / 2.52 ms (0.39 ms per 10 ms frame) |
| raw-mic pauses (room tone, fan, bleed) | mix 1.0: **-11.7 dB**; mix 0.5: -5.1 dB |
| loudest 2% of pause frames (clicks, keys, thumps) | mix 1.0: -12.2 dB; mix 0.5: -5.2 dB |
| speech frames | -0.1 dB at either mix |
| held sung notes, 1.5-4 kHz | mix 1.0: -0.1 dB; mix 0.5: 0.0 dB (0.1-1 kHz: 0.0) |
| live toggle inside speech | largest step in the 20 ms fade: 0.30x (on) and 1.07x (off) the speech's own p99.9 step there: no click |
| live soak, suppression on vs off | 1 underrun (10 ms, at 11.2 s) vs 0; window p99 median 17.9 vs 17.2 ms, max 24.2 vs 23.4 ms |

- The sung-note figure is on take 2's mic track, which NVIDIA Broadcast had already
  denoised (and, per the report, notched): it shows RNNoise does not carve those notes
  further, not what it does to a raw sustained note. That needs the user's raw-Shure
  re-record.
- RNNoise's ~0.8 ms per 20 ms hop is on the worker's critical path (it has to precede the
  upload and the gate). It pushed one of the existing 21-24 ms post-pause spikes into a
  10 ms underrun in 2 minutes. Moving it off that path (into the input callback, whose
  10 ms blocks are exactly one RNNoise frame) is the follow-up if the user's soak shows
  underruns with suppression on; it would move the high-pass with it.
- The Broadcast rule stands: on the "NVIDIA Broadcast" mic suppression defaults off (and
  the state says why); switching it on there is remembered for that mic only.

### 8.6 What changed by default, and the tests

Defaults: hop choices snap to 20 ms; the gate takes the max VAD probability over a hop's
chunks; the onset hop fades in over 5 ms; `f0_trim_tail = 2`; shape and precision come from
the engine defaults unless a profile file sets them; after the listening in 8.8,
`sine_phase_carry` and `noise_roll` on with `decoder_context_ms = 80`, and typed speech
(`vox say`, the GUI) rendered with `TypedPolish(envelope=0.35, rms_frame_ms=40)`. New and
opt-in: the Sing preset (Speak is the old behaviour), the soft ceiling (off in Speak),
suppression (off unless switched on or `rnnoise_enabled` in calibration.json; off on the
Broadcast mic).

Tests: 400 CPU tests (unit, api, tts; 368 before) and the GPU engine tests (28 pass, one
seam xfail, one seam xpass) pass. New: profile fall-through
(`test_schemas`, `test_target`, `test_controller`), hop snapping (`test_controller`), VAD
max-prob (`test_gate`), onset ramp (`test_sola`, `test_singing`), suppression state,
alignment, crossfade and allocation (`test_suppression`), presets and persistence
(`test_controller`), soft ceiling and trim tail (`test_pitch`), phase carry
(`test_sine_phase`, `test_singing`), take report (`test_live_session`). The two seam-click
xfails of section 7 are no longer strict: the ratio moves 1.18-1.45 with the decoder-noise
seed alone on the test clip, and trim tail 2 with seed 0 landed under the 1.25 guard.

### 8.7 Not verified

- Nothing was listened to; every new wav is for the user's ears.
- No raw (non-Broadcast) singing: the acceptance re-record (raw Shure, Sing preset) is the
  user's. The entrance and suppression conclusions above rest on Broadcast-processed takes.
- The GUI controls were driven through the controller API and its tests, and rendered once
  headless (Live and Settings, in Speak and in Sing, on a copy of the profile); a person did
  not click them in the window. The Broadcast toast was checked in the tests only.
- Live runs used the file-backed input; the microphone path through the GUI was not run.
- The 3 dB onset margin was not measured for false openings on a raw mic in pauses.

### 8.8 The user's listening (2026-10-15, later)

The user listened to every file in `out/ab/` (sections 7 and 8). Their verdicts, in their
words where it matters:

- Every mic render of the speech script "stutters BADLY", regardless of the section 7
  variant (gate, SOLA geometry, interp, envelope, index, bridge); `overlap_10` "sounds
  horrible", `raw_input` "stutters much less but still rough", `fix1_sola` "slightly
  better". Index 0.35 vs 0.6: "no difference". Typed renders never stutter.
- Singing (hop 20, GoatMommy): `sing_phasecarry_noiseroll_decctx80` is "the closest to
  perfection"; `sing_phasecarry_noiseroll` "kinda goated, a small small stutter";
  `sing_ceiling_off` "more human" but stutters; phase carry alone, trim tail 2 alone and the
  reference are "meh". The hop 60 and Mom sets: "a bit better on singing, stutters there,
  sounds a bit AI".
- Typed: `typed_envelope` "may be the best one", `typed_blend` and `typed_contour` "really
  good", `typed_all` "much more AI", `typed_plain` "not bad".

What was made of it:

| decision | measured consequence |
|---|---|
| `sine_phase_carry = True`, `noise_roll = True`, `decoder_context_ms = 80` (engine defaults) | GoatMommy hop 20: p99 19.5-19.7 ms (was 17.7-18.0), 0 underruns paced at extra 1000 / 2200; Mom 17.5-19.0. Live 2-minute soak (Sing values, gate -40, VAC exclusive): 1 underrun, window p99 median 18.9 ms, max 23.2 (18.3 / 23.4 before). A live hop change (rebuild + recapture) now mutes 2.9-3.0 s instead of 1.8-2.4 (the synth graph decodes 80 ms more). On the speech script (Mom +12 / GoatMommy +6, `out/ab/speech_{reference,coherent}.wav`, `speech_goat_*`) R3, voicing and HF/LF are within noise of the old defaults; the click ratio moves +0.03 / -0.12. The metrics did not predict the user's preference either way (8.4) |
| `DEFAULT_POLISH = TypedPolish(envelope=0.35, rms_frame_ms=40)` for `vox say` and the GUI's typed clips; `vox say --plain` is the old path | Section 7.5: loudness sd 8.9 -> 8.2 dB, voicing and F0 spread unchanged. The blend and contour variants stay options (`vox say --voice "af_heart*0.7+af_bella*0.3"`; `TypedPolish(contour_scale=1.2)`) |
| Index rate: nothing | |
| `HopPipeline.reset()` draws fresh decoder noise; the simulator and the sing trace seed before resetting | With `noise_roll` a new stream otherwise started on the previous stream's draws, and a seeded run was not reproducible (found by `test_gate_onset_hop_keeps_its_head`, whose two runs compared different noise). The gated-clip seam ratio of `test_gate_streams...` then read 1.31-1.39 over seeds with either default set: it was the Q7 guard passing on a leaked draw, and is now left to the `OVERLAP_EQUALS_HOP` tests |

The "stutter" in the mic renders is the open item. With the coherent defaults the hop-to-hop
render differences are what the user's chosen sing file had; whether the speech script now
sounds the same to them is the next listening (`speech_coherent.wav` vs `speech_reference.wav`).
If it still stutters, the remaining candidates are the gate (150 ms of speech lost in 9
cuts at -40; `raw_input` lost more yet sounded smoother, which points away from the gate
and toward the input chain's gain ramps or the 20 ms hop's seam density: a 40 ms hop costs
no extra compute, 8.2) and the two-render blend at overlap == hop (Q7).

## 9. Auto pitch (2026-10-15, later)

The user: "keep a specific pitch steady even if I speak low or high, cause my voice isn't
always the same and it can break the pitch or the way the RVC model sounds." A fixed
transposition (+6 on GoatMommy) lands their speech (~100 Hz) at ~150 Hz and their singing
(180-280 Hz) at 255-395 Hz; the model was trained on one register.

| # | Asked | Built | Why |
|---|---|---|---|
| A1 | Hold the output pitch steady whatever register the talker is in | `RuntimeParams.pitch_mode = "auto"`: `pitch.base.RegisterTracker`, a running mean of log2 F0 over each hop's fresh frames (time constant `pitch_follow_s`, default 2 s, counted in voiced frames so pauses move nothing), turned into a per-frame ratio that puts the register on `target_f0_hz`; the fixed `f0_up_key` applies until 50 voiced frames (0.5 s) were heard. `pitch_range` (default 1.0) scales the intonation around the target. State: two 0-dim device tensors updated inside the captured F0 stage (snapshotted with the pipeline, reset with the stream); the applied offset is read on the control thread for the GUI | Register normalisation, not pitch flattening: the user's intonation is what makes it theirs; `pitch_range` < 1 is there for sung lines a model cannot span. Graph replay verified against eager (`tests/engine/test_singing.py`); a 110 Hz LibriSpeech voice lands at 170-240 Hz median under replay |
| A2 | Target | `target_f0_hz`, defaulting to the model's register as `vox say` already derives it (`target_f0_hint`: calibrated median, model config, else 200 Hz); per model and per preset in the GUI (`gui_state.json`: `auto_pitch`, `target_hz`, `pitch_range`) | One notion of "the model's pitch" for typed and live speech |
| A3 | GUI | "Auto pitch" toggle under the Pitch offset slider with the live applied offset ("now +7"); Auto pitch target and range sliders on Settings (greyed while off); `vox run --pitch-mode auto --target-f0-hz --pitch-range --pitch-follow-s`, live commands `autopitch on|off`, `target HZ` | |

Measured (`out/ab/autopitch_{speech,sing}_{fixed6,auto,auto_range70}.wav`, GoatMommy,
target 200 Hz; the speech script and the first 14 s of the sung take, gate on, index 0.3,
protect 0.35, no ceiling; output F0 by RMVPE over voiced frames):

| input | input median (p10-p90) | fixed +6: output median (p10-p90) | auto: output (applied) | auto, range 0.7 |
|---|---|---|---|---|
| speech script | 101 Hz (80-117) | 143 Hz (118-169) | **193 Hz** (148-215), +12.1 st | 195 Hz (143-215) |
| sung take | 159 Hz (139-210) | 225 Hz (196-299) | **194 Hz** (168-257), +4.2 st | 196 Hz (177-243) |

One toggle puts both registers on the model's pitch; the intonation spread is the input's
(p10-p90 ~0.6 octave for speech either way). Under graph replay the F0 stage costs the
same (two extra reductions over the 32-frame tail). The applied offset is shown live in
the GUI; the CLI's `stats` line does not carry it.

A first live run (take 2026-10-15 11:40) died on a hop change with auto pitch on: the
GUI's 2 Hz snapshot read the tracker's offset straight off the device while the worker
was capturing graphs, and the sync inside a capture poisoned the CUDA context (every
restart then failed with "Offset increment outside graph capture" until the app was
closed). The worker now mirrors the two scalars to the host after the hop's own sync
(`RegisterTracker.publish`), `offset_st` reads only that mirror, and
`test_auto_pitch_readout_never_syncs_during_a_live_rebuild` polls it at 50 Hz through a
live rebuild. Rule restated: nothing off the hop thread touches a device tensor.

Not verified: listening (`autopitch_*` are for the user); a voice that drifts *within* a
phrase faster than the 2 s follow; what the tracker does on a sung line that spans more
than the model's range (the `pitch_range` slider is the lever there).

## 10. Volume controls (2026-10-17)

Live screen, next to Input gain and the suppression amount; all hot (applied between hops
or inside the output callback), all kept in `gui_state.json` under `volume`:

| control | range | where it acts | how |
|---|---|---|---|
| Output volume | -24..+6 dB | `DuplexAudio` output callback: the last step before the device, so the converted voice and typed clips move together, and the OUT meter and the Record output tap see what plays | `audio.volume.RampedGain`: a new value ramps in linearly over 10 ms (480 samples) whatever the callback block, no allocation; unity is a no-op; clipped at full scale only while boosting |
| Typed speech volume | -12..+12 dB | the clip player (`engine/override.py`), on the clip only | same ramp, so moving it mid-clip is click-free; clips played while not live get it (and the output volume) applied before `sd.play` |
| Match typed speech to live level | on by default | controller, before the typed volume | the worker follows the converted voice's level over its speech hops (`SpeechLevel`: energy average, 3 s time constant, hops under -60 dBFS skipped); a clip's speech level (`active_level_dbfs`: 20 ms frames within 30 dB of its loudest, pauses excluded) is moved onto it, by at most -24 / +18 dB, then lowered if needed so that with the typed volume the clip peaks at -0.1 dBFS. Before the voice has spoken in a session, the last session's level is used (saved on stop), else -20 dBFS. The Live screen says which |
| Monitor volume | — | not built | headphone monitoring does not exist in the engine yet (the Settings toggle is shown disabled), so there is nothing to set |

Measured (`scripts/volume_check.py`: the mic script looped through Mom into VAC Line 1,
gate off, OUT meter p75 of the GUI's own meter events per 5 s phase; three runs):
output 0 / -12 / -24 dB read -7.0..-7.9 / -18.6..-19.0 / -31.1..-31.8 dB (steps of
-11.2 and -24.1 dB on average), +6 dB pins the meter at 0 dB (the clip), and values came
back after a controller restart. Typed clips at typed 0 / -12 / +12 dB read about -9 /
-21 / -3.6 dB on the meter (+12 is held back by the peak guard). Underruns: none on any
slider move. Around typed clips there were 3 underruns in 13 clips with this code and 3
in 7 clips on e3ba942 with no volume code at all (same script, `--typed 0,...`), so they
belong to the typed-over-live hold, not to the volume path; still open.

## 11. Modernization, Phase 0 (2026-10-17): the ear-and-metric loop

The modernization report (`docs/research/15_MODERNIZATION_REPORT.md`, built from a code
audit of main@2b2cf93 and ~100 sources) found the project strong on performance and blind on
quality: no WER, speaker-similarity, MOS or blind-A/B number anywhere, defaults chosen by
ear against metrics that disagreed with the ear, and one dominant complaint (the speech
"stutter") never heard on a raw microphone. Phase 0 builds the measurements; it changes no
engine default. Phase 1 (inference-time fixes) waits for the user's raw-mic matrix (11.4).
Same machine; NVIDIA Broadcast, OBS, Discord and browsers resident on the GPU (~4.4 GB).

### 11.1 Deviations

| # | Prompt / report said | What was built | Why |
|---|---|---|---|
| M1 | New `benchmarks/quality.py`, reusable by every bench subcommand | `benchmarks/quality.py` (pure functions + CPU scorers + `QualityKit`), wired into `vox bench ab --quality --pairs`, a standalone `vox bench quality --files`, and `vox bench take --analyze`; `benchmarks/target_f0.py` and `benchmarks/matrix.py` beside it | The scorers (faster-whisper small.en int8, speechbrain ECAPA-TDNN or WavLM-SV, UTMOS22 strong via torch.hub) run on the CPU so a GPU render is never disturbed; weights land under `models/_tools`, `_embedders`, `_predictors/hub` (gitignored). `requirements/tools.txt` pins faster-whisper 1.2.1 and speechbrain 1.1.2 |
| M2 | WER on a fixed script with the known transcript | `benchmarks/corpus/speech_script.txt` (150 words, no digits, sibilant and plosive runs, one near-whisper line, one shout); `--transcript FILE`, `--transcript mic` (the un-converted mic's own Whisper transcript is the reference: for recordings of no known script, such as `mic_script_2026-10-13.wav`) | The existing mic script was never read from a text; `mic` mode makes the mic's WER 0 and an output's WER "what the conversion lost" |
| M3 | Speaker similarity against held-out target recordings; `vox say` renders as a labelled proxy if unavailable | `takes/target/<Model>/*.wav` (none exist yet); the proxy is two seeds of the script through the model, labelled "proxy: vox say" in every table and record; the targets' own pairwise cosine is printed as the ceiling (0.986 for the proxy: two seeds of one TTS render, so it is not a real ceiling) | The user must supply the clips; until then every similarity number measures distance to the model's typed voice |
| M4 | Seed floor: two renders, different seeds | In `ab --quality` the gated stream is rendered twice (seed 0 and 1) and their cosine is the floor; in `quality --files` and `take` there is one render per file, so the floor column is empty | A second live take cannot be re-seeded |
| M5 | UTMOS as a naturalness proxy, report but never rank on it | Scored in 15 s chunks averaged by duration (the SSL front end on a 60 s clip on the CPU); the table footer repeats the rule | |
| M6 | F0 stats: median, IQR, octave flips/min, voiced ratio | RMVPE track (the engine's own) at 100 Hz; IQR in semitones; flips = steps >= 8 st between consecutive voiced frames per voiced minute | A glide never moves 8 st in 10 ms; an octave error is 12 |
| M7 | No MCD, F0-RMSE, PESQ, NISQA-PSTN | Not implemented; `test_quality_never_offers_the_metrics_the_report_rules_out` keeps them out | SVCC 2023/2025 |
| M8 | `--pairs`: blind A/B pairs, randomised X/Y, answer key | `write_pairs`: `pairNN_X.wav` / `pairNN_Y.wav`, `pairs.md` (what to listen for, a blank per pair, no answers), `answer_key.json`; one fixed seed (2026) so a sheet can be regenerated | |
| M9 | 0.2: render a 60 s script through each model with `vox say` (Kokoro, pitch auto off), write the output's median F0 as `target_f0_median_hz`, auto pitch must use it | `vox bench target-f0` does exactly that (pitch 0, `TypedPolish()` plain) and writes `defaults.target_f0_median_hz` into `models/<Name>/config.json` **and** `profiles/default/model_overrides.json` (the committed copy; `target_f0_hint` reads the override first). Beside it, `target_f0_calibration` keeps two other estimators: a -8..+8 st sweep (cents error, voicing kept, flips, UTMOS per shift) and the coarse-pitch embedding's movement against the pretrained base (f0G32k.pth fetched for the 32 kHz models) | **The prompt's estimator is circular**: RVC copies the source F0 contour, so at pitch 0 the output median is the TTS voice's (193 Hz) for every model (11.2). The written value is kept as specified; the embedding estimator is the one that reads the training register (Q13) |
| M10 | 0.3: `vox bench take --matrix`: input {Shure, Broadcast} x gate x AGC x hop {20, 40}; the user talks, the command sequences, labels, scores, pairs | As specified; one `LiveSession` per input, hop changes as in-place rebuilds, gate and auto level hot; the Enter key starts and ends each cell (`--take-seconds` for a fixed length); `--plan`, `--only`, `--analyze DIR`. Gate threshold per cell: -40 dBFS with auto level, -55 without (`--gate-threshold-agc/raw`) | The gate measures the conditioned level: the same threshold cannot serve both AGC states (7.3). Inputs are named `Microphone (Shure MV6)` / `Microphone (NVIDIA Broadcast)` and a resolved cable is refused: the user renamed the cable's capture end "(Shure MV6x)", so "Shure MV6" alone recorded the engine's own output in the smoke test |
| M11 | DEVIATIONS §10 | This is §11: §10 is the volume controls of 2026-10-17 | |

### 11.2 Per-model target register (`vox bench target-f0 --utmos`)

Kokoro `af_heart`, the 150-word script (56 s of TTS), full-file conversion, RMVPE on the
output. "Rendered" is the written value.

| model | rendered (pitch 0) | TTS voice | sweep best | embedding register (p10-p90, peak/median) | UTMOS by shift -8/-4/0/+4/+8 |
|---|---|---|---|---|---|
| Mom (32k) | 194 Hz | 193 Hz | -4 / -8 st (two runs) | **265 Hz** (94-677, x5.0) | 2.77 / 3.16 / **3.47** / 3.45 / 3.10 |
| GoatMommy (48k) | 194 Hz | 193 Hz | +8 / +4 st | 304 Hz (85-540, x2.2) | 3.29 / 3.89 / 4.06 / **4.13** / 3.61 |
| MaeASMR (32k) | 193 Hz | 193 Hz | -8 / -4 st | 545 Hz (120-958, x2.2) | 3.59 / 4.10 / **4.17** / 3.83 / 3.66 |
| base_f0_48k_v2 | 192 Hz | 193 Hz | -4 st | (is the base) | 4.37 / 4.39 / 4.39 / 4.33 / 3.84 |

Reading: the rendered column is the TTS register to within 2 Hz on every model, as predicted
from the pipeline (the written value therefore changes auto pitch by -0.5 st against the old
200 Hz fallback and nothing more). The sweep's cents error is 1-25 cents at every shift
(the decoder follows the sine wherever it is asked), so its "best" rides on the octave-flip
count (6-74/min across shifts) and moved between two runs of the same command: not an
estimator. The embedding estimator has a clear profile only for Mom (peak 5x the median;
265 Hz with a wide p10-p90) and a weak one for the other two. UTMOS peaks at 0 or +4 st for
the three voices and is flat for the base, which trained on every register. What auto pitch
should aim at is now Phase 1.4's A/B (194 vs 265 Hz on Mom), not a measurement.

### 11.3 The §7.4 variants under the new metrics (`vox bench ab --quality --pairs`)

Mom +12, the mic script (25.75 s), gate -40 after the input chain, coherent-hop defaults,
`--transcript mic` (the microphone's own Whisper transcript: "Like, it, trust me, it's
working. It's working, but not too well. ..."), speaker targets = the `vox say` proxy.

| variant | WER | spk cos (proxy) | seed floor | UTMOS | F0 med | IQR st | flips/min | voiced | click | R3 x | HF/LF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 25.7% | 0.577 | 0.959 | 1.54 | 197 | 2.5 | 57.3 | 69% | 1.27 | 2.75 | +13.3 |
| f0_interp | **20.0%** | 0.585 | 0.964 | 1.44 | 198 | 2.7 | 55.9 | 76% | 1.32 | 2.77 | **+9.3** |
| sola_blend3 | 25.7% | **0.604** | 0.963 | 1.50 | 198 | 2.5 | 56.2 | 71% | **1.04** | 2.73 | +12.2 |
| index_0.35 | 28.6% | 0.566 | 0.952 | 1.48 | 197 | 2.4 | 65.5 | 69% | 1.34 | 2.66 | +13.5 |

Reading: a quarter of the words the microphone carried do not survive the conversion at
this gate setting (part of that is the 150 ms of gated speech and Whisper's tolerance of a
quiet, bled-into mic recording); interp recovers a fifth of the loss. Speaker cosine moves
by less than the seed floor's distance from 1 (0.04) for every variant, so none of them
changes identity measurably against the proxy. UTMOS sits at 1.4-1.5 for every mic render
against 3.5 for the same model on typed speech (11.2): the naturalness proxy sees the
microphone path as the problem, which is what the user hears. Blind pairs for these three
comparisons: `out/ab/p0_pairs/` (`pairs.md`, then `answer_key.json`).

### 11.4 The matrix (`vox bench take --matrix`): built, smoke-tested, not recorded

A two-cell smoke run on the quiet room (hop 20 then 40, gate and auto level on, 5 s each)
exercised the whole path: session start on the named microphone, the in-place hop rebuild
(2.2 s), start/stop recording, file labelling, the JSON (cell, script, SOLA delay), the
analysis table and one blind pair. It also found M10's cable-name trap (the first attempt
recorded the cable's capture end at -99 dBFS). The 16-cell matrix needs the user: about 16
minutes of reading, `vox bench take --matrix` (or `--hops 20` for 8 cells), then
`out/ab/matrix_<stamp>_pairs/pairs.md` for the verdicts. Phase 1.1 (the gate redesign) is
conditional on those verdicts and nothing in Phase 1 was started.

### 11.5 Acceptance

| check | result |
|---|---|
| Phase 0.1 metrics in `vox bench`, with tests | `tests/unit/test_quality.py` (12), `tests/unit/test_matrix_and_target_f0.py` (10): pure functions, pairs, kit plumbing with stubbed scorers, take analysis on a synthetic take |
| 0.2 per-model target F0 written and used | `target_f0_median_hz` in every model's config.json and `model_overrides.json`; `target_f0_hint` (vox say, `vox run`, the GUI's auto pitch) reads it: the 200 Hz fallback no longer applies to Mom, GoatMommy, MaeASMR, base_f0_48k_v2 |
| 0.3 matrix protocol | built and smoke-tested; recording waits for the user |
| suite | CPU suites (unit, api, tts) green; engine (GPU) see below |
| defaults changed | none |

### 11.6 Not verified

- Nothing was listened to. `out/ab/p0_pairs/` and the matrix pairs are for the user.
- Speaker similarity against the real target voices: no held-out clips exist
  (`takes/target/<Model>/`).
- Whether the matrix's gate thresholds (-40 / -55 dBFS) suit the Shure at the user's
  distance and the Broadcast mic's level; the meter line during each cell shows the level.
- UTMOS on 48 kHz RVC output at these levels: a proxy, calibrated on TTS at 16 kHz.

## 12. Formant control, and whether the index does anything (2026-10-17/18)

Branch `feature/formant` (off `feature/modernize-p1`). Same machine. The DEXAIV GUI was
running on the GPU during the formant renders and the worker tests (~13-16% load), so the
timings in 12.2 are on a shared GPU; render quality is unaffected.

### 12.1 Deviations

| # | Prompt said | What was built | Why |
|---|---|---|---|
| F1 | rtrvc's math: decode `ceil(return_length * f)` frames, resample by `floor(f * upp) / upp` | One rational factor `f_e = round(f * upp) / upp` drives both the stretch and the resample (`synthesis/formant.py`) | rtrvc's two ratios differ; at our 5-frame tail that is up to a quarter semitone of pitch and timing error. `f_e` is within 0.03 st of `f` |
| F2 | (not in the prompt) | The stretch is anchored to absolute time: one extra frame is decoded and each hop picks its phase `r_n = (n * shift * upp_res) mod upp` from a precomputed table, copied into static buffers before the synth stage (outside the graph, like the noise) | The first version drifted sharp when streamed (+28 cents at -2 st on a steady tone; per-hop chunks and the full-file render were exact). The window moves `shift * f_e` stretched frames per hop, not a whole number, so a window-anchored stretch put the decoder's grid elsewhere every hop |
| F3 | (not in the prompt) | Positions past the decoded window read **zero**, not the edge frame | Holding the last frame rendered every hop's right edge (the region the next hop's SOLA aligns to) differently from the interior: -6.6 samples of hop-to-hop lag at -2 st, which SOLA turned into a steady pitch rise. Zero, like the convolutions' own padding at an unstretched edge: -0.35 samples. (At formant 0 the engine already shows +3.1 samples of edge lag: the stitched stream reads ~11 cents flat of the requested pitch on a steady tone. Unchanged here; worth a look under Q7) |
| F4 | "Chunker owns the scaled decode length (`decoder_out` grows)" | `ChunkPlan.decode_frames` (tail + decoder context) is the chunker's; the stretched count and the phase table live in `synthesis/formant.py`; `decoder_out` is unchanged because the audio is resampled back inside the synth stage | Formant is a hot parameter, not a shape: the plan, buffers and every other graph stay as they are |
| F5 | Graph cache per 0.5 st step, LRU ~4, a cached step switches with no mute | As specified (`StageGraphs.sync_formant`, `FORMANT_GRAPHS_KEPT = 4`); only the synth graph is per step. A first visit builds the step's phase table (20-105 ms) and captures its synth graph (~250 ms) with the voice faded out around both, as a hop change does | |
| F6 | Pitch pre-compensation: synthesize with F0 / factor (`f0_up_key - formant_st`) | The excitation keeps the requested pitch (slowed by the stretch, restored by the resample); only the coarse pitch the text encoder sees is `pitch / f_e` (a hot device scalar in the f0 stage) | rtrvc does exactly this once its `pitchf * return_length2 / return_length` is unfolded; transposing both by `-formant_st` would put the output pitch off by the factor |
| F7 | GUI slider sends live | The value is shown while dragging and sent on release (and on the double-click reset) | Every 0.5 st the slider passes is its own engine shape: sending each would capture up to 16 graphs in one drag |
| F8 | F1/F2 via Parselmouth or LPC within 15% | LPC F1/F2 on the 48 kHz test model (male, ~114 Hz) within 15%: passes. On Mom +12 (~208 Hz) LPC picks harmonics (its "F1" is the fundamental), so the bench uses the cepstral long-term envelope (`benchmarks/formant_check.py`) | Parselmouth is not installed; LPC formant tracking is unreliable on high voices |
| F9 | Pitch within +-10 cents of formant 0, by output median F0 | By the per-frame median over frames voiced in both renders | A whole-file median also moves when different frames happen to be voiced (it read 10.02 cents on a render whose per-frame offset is 2-5) |
| F10 | v2 `formant_mode="envelope"` | Not built | The prompt gates it on v1 shipping and a blind A/B |

### 12.2 Formant measurements (`vox bench formant`, `benchmarks/formant_check.py`)

Live = the raw-Shure matrix take (`shure_gateoff_agcoff_hop20_mic_raw`, first 45 s) through
Mom +12 with the input chain (stream simulator, graphs); typed = two lines of the script
through Kokoro and the full-file converter at Mom's auto pitch.

| step | factor | live envelope shift | live pitch vs 0 | typed envelope shift | typed pitch vs 0 | synth GPU p50 | hop p99 (shared GPU) |
|---|---|---|---|---|---|---|---|
| -2 | 0.891 | 0.912 | +2.1 c | 0.911 | -3.1 c | 5.37 ms | 24.3 ms |
| -1 | 0.944 | 0.951 | +4.0 c | 0.958 | -2.7 c | 6.43 | 24.4 |
| 0 | 1 | 1 | 0 | 1 | 0 | 5.44 | 24.5 |
| +1 | 1.059 | 1.058 | +3.8 c | 1.057 | +0.6 c | 5.93 | 24.1 |
| +2 | 1.122 | 1.101 | +5.0 c | 1.087 | +1.7 c | 5.54 | 27.8 |
| +4 | 1.259 | | | | | 5.94 | 25.7 |

Envelope shift = the factor that best aligns the cepstrally smoothed long-term spectra; it
reads 80-100% of the nominal factor because breath, sibilants and room noise are in the
average too. Pitch = per-frame median cents against step 0. The synth stage costs ~0.5 ms
more at +4 st than at 0 (the decoder renders up to 26% more samples); the hop p99 of
24-28 ms was measured with the GUI on the GPU and is not comparable with the 17-20 ms of
§8. Seam click ratio 1.32 at 0, 1.33-1.38 at the other steps. Live switching (worker test):
six switches between cached steps, no underrun; a first visit captured once and the voice
came back.

Wavs: `out/ab/formant/live_{-2.0,-1.0,+0.0,+1.0,+2.0}.wav` and `typed_{...}.wav`.

### 12.3 Does the index do anything? (`vox bench index`)

The live session line and the take JSON now name the index in use (file, vectors, dim,
rate) or say it is not loaded. Mom's index: `model.index`, 43,857 x 768, loaded; protect
0.33. The same raw-Shure take (45 s), Mom +12, one engine with the index loaded throughout,
open stream, seed 0; a second seed at rate 0 is the noise floor.

| index_rate | feature cosine distance (decoded frames) | log-mel L1 vs rate 0 | WER | speaker cos (proxy) | sibilant centroid | active level |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 34.7% | 0.798 | 5428 Hz | -14.7 dBFS |
| 0.3 | 0.0027 | 0.172 | 33.3% | 0.805 | 5417 | -15.0 |
| 0.6 | 0.0136 | 0.275 | 33.3% | 0.808 | 5437 | -15.6 |
| 1.0 | 0.0471 | 0.520 | 40.7% | 0.765 | 5411 | -17.9 |
| 0, seed 1 | | **0.336** (the floor) | 33.3% | 0.798 | 5412 | -14.7 |

WER is against the whole script while the take is its first 45 s, so every row carries the
same missing tail (the microphone itself reads 24.7%): compare rows, not absolutes.

**Verdict: inert for Mom.** At 0.6 the index changes the output less than a different random
seed does (0.275 < 0.336), and the features the decoder sees move by 1.4% in cosine: this
voice's ContentVec frames already sit close to their nearest training vectors, and protect
pulls the unvoiced ones back. Only at 1.0 does it pass the floor, and then it is 3 dB
quieter, loses words and moves away from the proxy. That matches the user's "no difference
at any setting". Applied per the rule: the verdict is stored in `models/Mom/config.json`
and `profiles/default/model_overrides.json` (`index_effect`, tied to the index file
measured). For an inert model the GUI does not load the index (the index stage then only
upsamples), keeps the rate at 0, refuses a new rate and greys the slider with "No audible
effect measured for this model". Another index file for the same model is not covered. No
ABX was written: the rule calls for one only when the output changes. Wavs:
`out/ab/index/index_{0,0.3,0.6,1}.wav` and `index_0_seed1.wav`.

### 12.4 Tests

`tests/unit/test_formant_math.py` (factor, decode length, stretch+compress identity,
hop-to-hop coherence of excitation and latent grid), `tests/unit/test_formant_index_bench.py`
(verdict, ABX writer, envelope estimator), `tests/engine/test_formant.py` (streamed and
full-file pitch within 10 cents, F1/F2 moved within 15%, seams, graph cache and LRU,
replay vs eager), `tests/engine/test_worker.py` (live switching), `tests/api` (presets,
persistence, inert index). Under the GUI's GPU load four pre-existing worker tests also
failed on the pre-formant commit (underruns): not a regression; they need a quiet GPU.

### 12.5 Not verified

- Listening: none of the wavs above was heard. Whether +-2 st sounds natural on Mom, and
  whether the typed and live formants match by ear, is the user's call.
- The index verdict on GoatMommy (124,683 vectors) and on other takes: only Mom was measured.
- Hop timings on a quiet GPU.

## 13. Streaming engine, phase A (2026-10-19): the no-retraining fixes of report 18

Branch `feature/streaming-a` (off `feature/formant`). Report 18 (`docs/research/18_STREAMING_
ENGINE_REPORT.md`) answered brief 17: the stutter is structural (every emitted sample is a
crossfade of two renders that disagree at their right edge, re-timed by SOLA fifty times a
second). This section builds its §F items 1-8 and 10, every one as a flag, no default
changed. Same machine; the GUI was closed; the GPU had only the one job unless stated.

The take throughout: the raw-Shure matrix take (`shure_gateoff_agcoff_hop20_mic_raw`, first
45 s) levelled once by the GUI's input chain (`out/ab/model_vs_engine/0_your_mic.wav`), Mom
+12, index 0, formant 0, conditioning off, seed 0 (the §12 / brief 17 setup). "L1" = speech-
frame log-mel L1 against the whole-file render of the same take (`out/ab/right/wholefile.wav`),
whose reseed floor is 0.254. "click" = the seam roughness ratio. "held" = share of hops that
kept the previous SOLA offset; "jump" = mean hop-to-hop |offset change|.

### 13.1 Deviations

| # | Prompt said | What was built | Why |
|---|---|---|---|
| S1 | Commit per item | A1-A3 share one commit, A4/A6/A7 another | The items touch the same lines of `simulate.py` and `worker.py`; splitting the hunks would have left intermediate commits that do not import |
| S2 | A1: "the decoder's `dec_context` is unaffected" | The decoder also decodes the `right` frames (`dec_right`) and drops them | Otherwise the kept tail would still be decoded against zero padding on its right and the lookahead would only reach the encoder, flow and F0 tracker. Cost ~+0.1 ms of synth per 20 ms |
| S3 | A2: "NCC within eps of the maximum" | The score is the correlation normalised by both sides (in [-1, 1]) | So `eps` is scale-free; the plain NCC is scaled by the tail's level and a fixed `eps` would mean different things in a loud and a quiet hop |
| S4 | A4: "peak over the neighbouring modulation floor, in dB" | Also `seam_depth_pct`, the fundamental read as an AM depth; and only bands at least 2.5 x the hop rate wide take part | On speech the floor-relative dB compress (a 20 % imposed dip reads +3 dB, 5 % +0.3 dB): the AM copies the talker's own envelope spectrum into sidebands and lifts the floor with it. The depth tracks the imposed depth ~linearly on top of the talker's own 50 Hz content (13.5). Bands narrower than ~2.5 x the hop rate filter the sidebands out (at 50 Hz: everything below ~500 Hz), so they are left out |
| S5 | A6: "the F0 it had when it was N ms old" | Reported twice: nominally paired and with the `f0_trim_tail` lag taken out | The cache's frame *a* holds the tracker's frame *a + 1 - trim* (§8.4: 10 ms later per trimmed frame beyond 1); the nominal pairing reads that lag as 75-89 cents of RMS error at every age, the lagged pairing shows the real age dependence |
| S6 | A7: statistics "accumulated over the first N seconds of speech" | Measured once over the whole take and stored (`vox bench drift --store-norm`) | A running accumulation inside the hot loop would change the embed stage's op sequence at the moment it froze (a recapture mid-stream); the stored statistics are what a live mode would load |
| S7 | A1: "underruns in a 60 s live run with the matrix settings" | As asked, plus the A3 soaks at 120 s | |
| S8 | Hop p50/p99 "quiet GPU" | The per-render timings were measured while CPU jobs (unit tests, the seam calibration) ran beside them; the quiet numbers come from one dedicated 20 s pass per variant (13.7) | Rendering took ~3 min per variant; waiting for silence between every render would have cost the session |

### 13.2 A1: right context (`extra_right_ms`)

`out/ab/right/live_right_{0,20,40,80,100}.wav`, `live_hop40_right40.wav`, `wholefile.wav`.

| variant | algorithmic latency | L1 (floor 0.254) | click | held | synth GPU p50 |
|---|---|---|---|---|---|
| hop 20, right 0 (today) | 50 ms | 0.586 | 1.28 | 20 % | 4.66 ms |
| right 20 | 70 | 0.568 | 1.37 | | 5.16 |
| right 40 | 90 | 0.545 | 1.29 | | 5.27 |
| right 80 | 130 | 0.513 | 1.28 | | 5.04 |
| right 100 | 150 | 0.509 | 1.32 | | 5.16 |
| hop 40 + overlap 40 + right 40 | 130 | 0.523 | 1.34 | | 5.12 |
| hop 200 (brief 17, for scale) | 230 | 0.51 | | | |

Right context moves the metric as far as a 200 ms hop did (0.509 at 100 ms vs 0.51), at
hop 20 and 150 ms of algorithmic latency instead of 230. The click ratio does not move: the
seams are still two-render mixes, now of renders that agree better. Live latency and
underruns: 13.6.

### 13.3 A2: SOLA offset damping (`sola_hold_eps`)

`out/ab/sola_hold/live_eps_{0,0.02,0.05}.wav`, `live_right40_eps_0.05.wav`. Offsets and jumps
from the engine's histograms (`Sola.offset_hist`, the new `Sola.jump_hist`), all hops.

| variant | held | jump mean / p50 / p95 | offset p50 / p95 | L1 | click |
|---|---|---|---|---|---|
| eps 0 (today) | 20 % | 1.23 / 0.15 / 5.6 ms | 4.5 / 9.2 ms | 0.586 | 1.34 |
| eps 0.02 | 49 % | 1.07 / 0.02 / 5.8 | 4.3 / 9.4 | 0.589 | 1.25 |
| eps 0.05 | 62 % | 0.85 / 0.00 / 5.2 | 4.2 / 9.3 | 0.588 | 1.28 |
| right 40 + eps 0.05 | **82 %** | **0.30 / 0.00 / 2.0** | 4.4 / 8.9 | 0.548 | **1.11** |

Damping alone halves the small jumps and leaves the large ones (p95 still 5 ms: when the
renders disagree, no offset within eps of the best is the old one). With 40 ms of right
context the renders agree, the hold takes, and the jitter the report called the second
percept is mostly gone (mean jump 0.3 ms, p95 2 ms) with the lowest click ratio measured on
this take. The hold is a `torch.where` on device tensors inside the captured stitch; the
first version indexed the score with a tensor (`score[prev]`), which is a host read-back and
is refused inside a capture (`tests/unit/test_sola.py::test_hold_never_synchronises_with_the_host`
now covers eps > 0 on the GPU).

### 13.4 A5: decoder left context 80 -> 120 ms, and the phase vocoder at 30 / 40 ms

| variant | L1 | click | held | synth GPU p50 | algorithmic latency |
|---|---|---|---|---|---|
| decoder context 80 (today) | 0.583 | 1.33 | 20 % | 4.72 ms | 50 ms |
| decoder context 120 | 0.589 | 1.37 | 19 % | 5.86 ms | 50 |
| plain, overlap 30 (`overlap_beyond_hop`) | 0.597 | 1.05 | 25 % | 4.54 | 60 |
| **pv**, overlap 30 | 0.596 | 1.03 | 25 % | 4.46 | 60 |
| plain, overlap 40 | 0.566 | 0.92 | 27 % | 5.48 | 70 |
| **pv**, overlap 40 | 0.566 | 0.90 | 27 % | 6.54 | 70 |

The full receptive field on the left (report 18 §0.2) changes nothing the metrics see and
costs 1.1 ms of synth per hop: `out/ab/decctx/live_decctx{80,120}.wav` is in the pack for the
ear, the default stays 80. The phase vocoder is within 0.02 of plain on every number at both
overlaps (Q7 again); the overlaps themselves lower the click ratio below 1 at 40 ms (the
three-render blend). `out/ab/pv/live_{plain,pv}_ov{30,40}.wav`.

### 13.5 A4: the seam-rate modulation prominence (`seam_prominence`)

Per-1/3-octave Hilbert envelopes at 1 kHz, 400 ms Hann blocks at half overlap, the envelope
spectrum's peak at the hop rate (+-1.5 bins) over the median of +-0.4 x hop rate around it,
energy-weighted over bands, averaged over speech blocks; also as an AM depth. Calibration on
the whole-file render with `impose_am` (`out/ab/seam_calibration/`):

| imposed AM | depth | hop-20 metric (50 Hz): dB / depth % | hop-40 (25 Hz) | hop-60 (16.7 Hz) |
|---|---|---|---|---|
| none | 0 | 2.29 / 21.2 | 1.56 / 32.5 | -1.29 / 32.8 |
| 50 Hz | 2 % | 2.34 / 21.3 | 1.55 | -1.28 |
| 50 Hz | 5 % | 2.55 / 21.8 | 1.53 | -1.28 |
| 50 Hz | 10 % | 3.33 / 23.6 | 1.52 | -1.27 |
| 50 Hz | 20 % | 5.24 / 29.6 | 1.49 | -1.26 |
| 25 Hz | 2 / 5 / 10 / 20 % | 2.28 / 2.26 / 2.23 / 2.16 | 1.60 / 1.72 / 2.02 / 3.00 | -1.28 / -1.27 / -1.30 / -1.48 |
| 16.7 Hz | 2 / 5 / 10 / 20 % | 2.29 / 2.28 / 2.28 / 2.25 | 1.56 / 1.55 / 1.53 / 1.39 | -1.20 / -1.00 / -0.49 / +0.56 |

Monotonic in depth at every rate and specific to its rate (the other columns stay flat or
fall), as the acceptance asked; the synthetic test (`tests/unit/test_seam_metric.py`) spans
> 15 dB at 50 Hz. On real speech the span is small: the talker's own envelope has content at
50 Hz (depth 21 % with nothing imposed), so a seam train of a few per cent is a sub-dB,
sub-per-cent excess over the whole-file render of the same take. That is how it is reported
in 13.7: stream minus whole-file. It is a measurement of the AM mechanism, not of what the
user hears; the pack decides whether it tracks the ear.

### 13.6 A3: output-ring fill rule, and the live runs

`chunker.fill_target_ms`: `legacy` = 2 x hop + callback (today), `hop_plus` = hop + 20 ms
(`LiveConfig.fill_rule`, `vox run --fill-rule`, `vox bench underrun --fill-rule`; a live hop
change re-derives the target from the rule instead of scaling the old one). Paced bench
(`vox bench underrun --model Mom --pitch 12 --audio <take> --seconds 45 --hops 20,40,60
--extras 1000 --broadcast-load --auto-level off`) and 120 s live soaks (the take looped as a
software microphone, VAC exclusive, gate off, auto level off, Broadcast's denoiser held open
on the GPU):

| hop | rule | fill target | paced: hop p50 / p99, underruns (45 s) | live 120 s: p50 / p99 / window p99 max, underruns | live latency estimate (ring part) |
|---|---|---|---|---|---|
| 20 | legacy | 50 ms | 17.1 / 21.8, 0 | 19.4 / 20.9 / 25.5, **1** (one 10 ms refill) | 193 ms (113)* |
| 20 | hop_plus | 40 | 17.8 / 21.1, **3** | 17.4 / 18.6 / 24.3, **1** | 158 (80)* |
| 40 | legacy | 90 | 18.0 / 22.3, 0 | 18.4 / 19.7 / 28.7, 0 | 197 (78) |
| 40 | hop_plus | 60 | 18.1 / 20.4, 0 | 18.3 / 19.3 / 26.1, 0 | **167 (48)** |
| 60 | legacy | 130 | 18.2 / 21.3, 0 | 19.5 / 23.1 / 25.7, 0 | 257 (116) |
| 60 | hop_plus | 80 | 19.2 / 22.2, 0 | 19.9 / 23.0 / 24.4, 0 | **206 (66)** |

\* After its one underrun the worker tops the ring back up (§8.2's refill), so the hop 20
estimates were read with the ring over target (120 / 90 ms); before it they sat at 50 / 40.

Both rules underrun once in two minutes at hop 20 with Broadcast on (hop p99 ~21 ms against
a 20 ms budget: the same first-hops-after-a-stall effect §8.2 and §8.8 saw); hop 40 and 60
never do under either rule. As the prompt's rule says: `legacy` stays for hop 20 and
`hop_plus` is the measured choice from hop 40 up, where it takes 30-50 ms off the live
estimate for nothing. No default changed; the GUI does not expose the rule yet.

**A1 live, 60 s, matrix settings** (same soak, Broadcast off, which is how the take was
recorded):

| variant | hop p50 / p99 / window max | underruns | live latency estimate |
|---|---|---|---|
| hop 20, right 0 | 17.0 / 18.0 / 24.3 | 0 | 114 ms |
| right 20 | 18.1 / 19.8 / 22.3 | 0 | 139 |
| right 40 | 18.8 / 21.8 / 22.2 | 0 | 157 |
| right 80 | 17.5 / 18.2 / 25.2 | 1 | 260 (ring refilled to 100 after it; 180 before) |
| right 100 | see below | | |
| hop 40 + overlap 40 + right 40 | 18.5 / 21.6 / 21.8 | 0 | 235 (legacy fill; ~205 with hop_plus) |
| eps 0.05 | 17.1 / 17.8 / 23.7 | 0 | 119 |
| right 40 + eps 0.05 | 17.8 / 18.9 / 21.4 | 0 | 157 |
| decoder context 120 | 18.0 / 19.2 / 23.3 | 0 | 116 |

The first right-100 run collapsed (1,195 underruns, hop p50 45 ms, 499 refills in 60 s); a
20 s rerun minutes later ran clean (p50 18.1, p99 19.6, 0 underruns, estimate 221 ms) and the
60 s rerun is in 13.8. Nothing in the engine distinguishes right 100 from right 80 (two more
decoded frames); the first timing pass of this session also read 3-4 ms high on every
variant until something outside the engine let go of the GPU. Treated as external load,
not reproduced.

### 13.7 A6 and A7: how far the window's features and F0 are from whole-file, and the frozen norm

`vox bench drift --model Mom --pitch 12 --audio <take> --seconds 45 [--right 40] [--norm frozen]`
(`out/ab/a567_metrics.json`). Ages count from the window's end; with `extra_right_ms` = R the
kept region's newest frame has age R. 1,769 active frames per age after 1.5 s. The whole-file
reference is the embedder and RMVPE over the entire take on the window's 50 Hz grid parity.

| age | feature cosine distance mean / p95 | F0 cents RMSE / p95, nominal pairing | the same, trim lag taken out | octave flips | voicing agreement |
|---|---|---|---|---|---|
| 0 ms | 0.818 / 1.046 | 89.2 / 102.1 | **46.2 / 38.4** | 0.46 % | 0.920 |
| 20 | 0.827 / 1.044 | 74.6 / 85.3 | 39.9 / 26.7 | 0.36 % | 0.940 |
| 40 | 0.459 / 0.891 | 74.9 / 86.6 | **12.3 / 20.0** | 0.26 % | 0.945 |
| 80 | 0.267 / 0.485 | 83.5 / 86.8 | 12.6 / 19.3 | 0.35 % | 0.946 |
| 120 | 0.320 / 1.035 | 73.9 / 86.4 | 14.0 / 18.2 | 0.35 % | 0.945 |

Scale, measured directly on one window at 4.0 s (`drift_debug`): adjacent whole-file 50 Hz
frames are 0.16 apart in cosine; the same frames seen from a 1.1 s window are 0.22 from their
whole-file values on average (0.1 deep inside, 0.4-0.5 in the last 40 ms, 0.96 for the
window's first frame); two windows one hop apart agree to 0.025 on their shared frames. So
the sliding window's features are not "noisy": they carry a steady offset from whole-file
of the order of a neighbouring frame (GroupNorm + global attention, report 18 §C7) plus a
large edge zone, the last 20-40 ms, where they are nearly unrelated to the frame's
whole-file vector. The cached F0 is 46 cents RMS off at the edge and settles to 12 cents by
40 ms; its 0.3-0.5 % octave flips are RMVPE's. Both say the same thing for the commit-once
design (report 18 §F9): committing at the edge freezes the worst version of every frame;
40 ms of lookahead buys most of the settling for F0 and about half of it for the features,
80 ms most of it. The right-40 run gives the identical table (ages are from the window's
end), as it should.

**A7, frozen GroupNorm.** Statistics over the 45 s take (`models/_embedders/contentvec/
frozen_norm.pt`), `contentvec_norm="frozen"`: the drift table is the same to three digits
(0.817 / 0.826 / 0.461 / 0.267 / 0.322), the whole-file render moves by 0.015 L1 (seed floor
0.254), the stream reads L1 0.578 vs 0.586, click 1.34 vs 1.28. Conv layer 0's over-time
statistics barely vary across 1.1 s windows of this talker, so freezing them changes nothing
measurable; the window dependence is in the attention. `out/ab/frozen_norm/live_frozen.wav`
is in the pack regardless. The R4 pause inflation at short context is left untested here
(extra 1000 throughout).

### 13.8 The one table, and the pack

Same take, same settings. Hop p50 / p99 from one quiet pass of 20 s per variant
(`out/ab/timing.json`); underruns and the live estimate from the 60 s runs of 13.6
(Broadcast off); "seam" = the stream's seam-rate prominence minus the whole-file render's,
in dB and in AM depth points (13.5: the whole-file reads 2.29 dB / 21.2 % at 50 Hz on its own,
1.57 / 32.3 at 25 Hz); click from the render (the hop grid); R3 = L1 vs whole-file over a
0.254 floor; drift = the kept region's newest frame (age = the lookahead) from 13.7.

| variant | algorithmic | live estimate | hop p50 / p99 | underruns / 60 s | seam excess dB / depth pts | click | R3 (x floor) | feature drift at the kept edge | F0 cents at the kept edge (lag out) |
|---|---|---|---|---|---|---|---|---|---|
| whole-file render | - | - | - | - | 0 / 0 | - | 0 | 0 | 0 |
| hop 20 (today) | 50 ms | 114 ms | 15.7 / 19.9 | 0 | +1.04 / +7.6 | 1.28 | 0.586 (2.31) | 0.82 | 46 |
| right 20 | 70 | 139 | 16.6 / 19.0 | 0 | +0.69 / +4.9 | 1.37 | 0.568 (2.24) | 0.83 | 40 |
| right 40 | 90 | 157 | 16.3 / 19.4 | 0 | +0.26 / +3.4 | 1.29 | 0.545 (2.15) | 0.46 | 12 |
| right 80 | 130 | 180 (260 after its refill) | 16.4 / 19.2 | 1 | -0.03 / +1.6 | 1.28 | 0.513 (2.02) | 0.27 | 13 |
| right 100 | 150 | 221 | 17.1 / 19.2 | 1 (rerun; 0 in 20 s) | -0.33 / +1.4 | 1.32 | 0.509 (2.00) | 0.32 | 14 |
| hop 40 + overlap 40 + right 40 | 130 | 235 (legacy fill) | 16.4 / 17.9 | 0 | +0.02 / +2.7 | 1.34 | 0.523 (2.06) | 0.46 | 12 |
| eps 0.05 | 50 | 119 | 15.8 / 17.5 | 0 | +1.16 / +7.4 | 1.28 | 0.588 (2.31) | 0.82 | 46 |
| **right 40 + eps 0.05** | 90 | 157 | 16.5 / 19.4 | 0 | +0.29 / +3.2 | **1.11** | 0.548 (2.16) | 0.46 | 12 |
| decoder context 120 | 50 | 116 | 16.6 / 19.5 | 0 | +1.07 / +8.1 | 1.37 | 0.589 (2.32) | 0.82 | 46 |
| frozen norm | 50 | - | 15.4 / 16.7 | - | +1.25 / +7.9 | 1.34 | 0.578 (2.28) | 0.82 | 46 |
| plain, overlap 30 (`blend30`) | 60 | - | 17.2 / 19.6 | - | +0.04 / +3.7 | 1.05 | 0.597 (2.35) | 0.82 | 46 |
| pv, overlap 30 | 60 | - | 15.6 / 17.3 | - | +0.03 / +3.2 | 1.03 | 0.596 (2.35) | 0.82 | 46 |
| plain, overlap 40 | 70 | - | - | - | +0.17 / +2.8 | 0.92 | 0.566 (2.23) | 0.82 | 46 |
| pv, overlap 40 | 70 | - | 16.8 / 19.0 | - | +0.32 / +3.1 | 0.90 | 0.566 (2.23) | 0.82 | 46 |
| hop 40, overlap 20 (brief 17) | 70 | - | 16.0 / 18.7 | - | +1.40 / +14.3 | - | 0.586 (2.31) | 0.82 | 46 |
| hop 40, overlap 40 (brief 17) | 90 | - | - | - | +1.12 / +6.1 | - | 0.556 (2.19) | 0.82 | 46 |

What the numbers say, before the ear: the seam-rate excess is the one metric that ranks the
variants the way report 18 predicted. It falls with right context (7.6 -> 4.9 -> 3.4 -> 1.6
-> 1.4 depth points from 0 to 100 ms), and at 80-100 ms the stream's 50 Hz content is the
whole-file's own. It does not move with offset damping (eps 0.05: 7.4), because the hold
changes the time base, not the amplitude of the seam; the click ratio and the jump
statistics are its measures, and those move only once right context lets the hold take
(right 40 + eps 0.05: click 1.11, jump 0.3 ms). Longer overlaps lower it too (3-4 points at
30-40 ms) by spreading each seam, at the same latency as right 20-40 ms which lowers it by
agreeing instead. Hop 40 at a 20 ms overlap is the worst row (14 points at 25 Hz): fewer
seams, each twice as deep; its 40 ms overlap halves that; right 40 on top brings it to 2.7.
Decoder context 120 and the frozen norm do nothing. Hop timings: right context costs
0.6-1.4 ms of p50 (two to ten more decoded frames), nothing underruns at hop 20 on a quiet
GPU, hop 20 keeps ~0.5 ms of headroom at p99 in every row.

**The pack.** `out/ab/streaming_a_pairs/` (31 blind pairs, `pairs.md` to fill in,
`answer_key.json` after): every variant above (level-matched to -14 dBFS active, copies in
`out/ab/streaming_a/`) against today's hop 20 and against the whole-file render, plus the
unheard brief-17 files (`blend30`, `hop40`, `hop40_xf40`), plain vs pv at 30 and 40 ms, and
hop 20 vs whole-file itself. `out/ab/streaming_a/scores.json` holds the table's seam and L1
numbers per file.

**If the user hears what the report predicts** (right context at 40-100 ms sounds like hop
200 did, and the hold removes the warble): the default becomes hop 20, `extra_right_ms` 40,
`sola_hold_eps` 0.05 (90 ms algorithmic, ~157 ms live), with `fill_rule="hop_plus"` from hop
40 up; 80-100 ms of right context stays the "quality" preset until item 9 (commit once,
stream the decoder) replaces the crossfade.

**Verdict (2026-10-20).** The user played pair11 blind and called X "much worse"; X was
`hop20_ref`, Y `right40_eps0.05`. `ShapeParams.extra_right_ms` is now 40 and
`RuntimeParams.sola_hold_eps` 0.05 by default (algorithmic latency +40 ms); `fill_rule` stays
legacy. The other 30 pairs are unheard. With OBS, Broadcast, browsers and a media player
on the GPU (~25 % busy), `tests/engine/test_worker.py` underran in 2 tests on the old
defaults and in 6 on the new ones (6-11 underruns per test); the gate test's closed-hop
median read 2.47 ms once and passed on rerun. The new defaults have less headroom under
load; a quiet-GPU rerun of the worker tests is still owed.

### 13.9 Tests

`tests/unit/test_chunker.py` (chunking law with `extra_right_ms`: sizes, latency, p_len,
hop 40, fill rules), `tests/unit/test_pipeline_math.py` (`dec_right` keeps the tail's length
and position and changes its rendering), `tests/unit/test_sola.py` (hold within eps, jump
beyond, eps 0 = argmax, histograms, reset, GPU sync-free at eps > 0),
`tests/unit/test_seam_metric.py` (monotonic on imposed AM at 50 and 16.7 Hz, rate-specific,
depth reads the imposed depth, silence is nan), `tests/engine/test_drift.py` (drift ordered
by age on the test model, CPU or GPU; frozen norm reproduces GroupNorm on its own statistics,
changes the static key, streams clean through `load_engine`), `tests/engine/test_engine.py`
(right 40, and hop 40 + right 100 + decoder context 120 run clean).

### 13.10 Not verified

- Listening: none of the 31 pairs has been heard. Every number above is a mechanism
  measurement; whether the seam-rate excess tracks "stutter" is exactly what the pack tests.
- The suite: 414 unit and engine tests pass; the five `tests/engine/test_worker.py` live
  tests failed once when run right after the pack's CPU scoring (underruns) and pass on a
  quiet machine, as §12.4 recorded for the same tests.
- The right-100 live collapse (13.6) was seen once and not reproduced.
- The frozen-norm result (nothing changes) was measured with statistics from this take only
  and at extra 1000; the R4 pause bias at short context was not re-measured with it.
- `hop_plus` live at hop 20 with Broadcast underruns once in two minutes, as `legacy` does;
  the paced bench put 3 underruns in 45 s on `hop_plus` at hop 20 and 0 on `legacy`. The
  rule's split (legacy at 20, hop_plus from 40) follows the prompt; it is not exposed in the
  GUI.
- The seam metric's floor-relative dB compresses on speech (13.5); its depth form is the
  number used, and both were calibrated on this one take.
- GoatMommy: nothing in this section was rendered with it.

## 14. Streaming engine, phase B (2026-10-22): polish of the new default

The user heard phase A live (13.8) and called it "a billion times better"; Mom is now the
main model. Four complaints, each measured before anything was built. Branch
`feature/streaming-b` off `feature/streaming-a`.

**Step 0, what the user actually runs.** The code defaults are hop 20, `extra_right_ms`
40, `sola_hold_eps` 0.05, `fill_rule` legacy (c7121fd, chosen by ear from blind pair11 of
13.8: X = today's hop 20, Y = right 40 + eps 0.05, "X is much worse"; the other 30 pairs
are unheard). The GUI state (`profiles/default/gui_state.json`) does not run the code
defaults: **hop 80 ms, extra 2400, gate threshold -28 dBFS, RNNoise on for the Shure,
auto level on with +6 dB trim, Mom +12, protect 0.35, ceiling 330 Hz, formant -0.5,
index 0.56 (inert, 12.3)**. Live latency estimate at that shape (legacy fill 170 ms):
hop 80 + ring ~80-230 + compute ~7-18 + sola 30 + lookahead 40 + devices 11 + RNNoise
20 = **~300-370 ms** (the 2-minute soaks of 14.1 read the ring at 230 ms after a refill).
Every number below is at the user's shape unless stated, on the raw-Shure matrix take
(`recordings/matrix_20261017_192237/shure_gateoff_agcoff_hop20_mic_raw.wav`, 59 s, 18
phrases) and the sung take 2 (`takes/live/take_20261015_004329_mic_raw.wav`, first 30 s).

### 14.0 Deviations

| # | Prompt said | What was built | Why |
|---|---|---|---|
| T1 | B1: verify the lookahead loss, then flush on close | Verified: the loss is 0 ms. No flush. Instead a guard (hangover never under the stream delay), the drift-trim fix, and a hangover A/B | 14.1: the 350 ms hangover already covers the 70 ms lookahead at every close; what cuts the sentence is the gate itself (closes inside trailing voice) and, live, the drift trim |
| T2 | Commit per item | B1 and B4 share one commit; `schemas.py` carries the B3 fields in it | gate.py, worker.py, sola.py and the schema are touched by both; splitting would leave commits that do not import |
| T3 | B2 (a): record 20 s of the user laughing | Not possible from here: the harness (`b2_laugh.py`) is built and was run on a typed "hahaha!" proxy; the real take is the user's | the Record button needs the user |
| T4 | B3: the note detector's reference is the run's first frame | An EMA of the run (`NOTE_REF_ALPHA` 0.1 / frame) | vibrato and a slow drift stay inside the +-1 st; a first-frame reference broke the sung take's held notes into pieces (only 2 runs >= 400 ms in 30 s) |
| T5 | B3: "switch the ceiling knee and the pitch_range for that note" | Also freezes the auto-pitch register once the run is `note_freeze_ms` (100 ms) old | a running mean slides a held note toward the target: measured 14.3 |
| T6 | B4: a 60 s take with deliberate creaks, bumps and taps | A synthetic proxy (`out/ab/phase_b/b4/proxy_take.wav`: the take's phrases with 1.5 s pauses and one creak / bump / tap per pause at 0 / +4 / +8 dB over the voice's p95) | no such take exists; the real one is the user's to record, the runner (`b4_leaks.py`) takes any take plus its event list |
| T7 | B4: "suppression strength ... a stronger max setting if the current one is capped" | `suppression_strength = "max"`: RNNoise's per-frame speech probability gates its residual (-20 dB under 0.5, 50 ms one-pole) | RNNoise has no strength knob; the user's mix is already 1.0 |
| T8 | B4: onset probation "for 2 chunks" | `onset_persist_chunks` counts completed 32 ms VAD chunks, hop-independent | at hop 80 a hop holds 2.5 chunks, at hop 20 it holds 0.6; a hop count would mean different things |
| T9 | Live soaks on the virtual cable | On the Realtek second output (device 24) after the first two: device 23 ("Shure MV6x") is what the user monitors and they heard the looped take | VAC "Line 1" is no longer in the device list; nothing in the soak depends on the device |

### 14.1 B1: "my sentence cuts off slightly before it actually ends"

**The lookahead hypothesis, measured** (`b1_lost.py`: simulator, gate on, the user's chain;
per gate close, the speech frames of the conditioned input between the last emitted
converted input sample `close * hop - stream_delay` and the first one the reopen renders;
"speech" = Silero >= 0.35 on 32 ms chunks / RMVPE voiced / energy >= threshold - 6):

| shape | closes | lost in the lookahead (voiced) | lost voiced, total / max per close | lost VAD, total | closes with any voiced loss |
|---|---|---|---|---|---|
| hop 80, right 40 (the user) | 11 | **0 ms** | 100 / 100 ms | 40 | 1 |
| hop 80, right 0 | 11 | 0 | 100 / 100 | 70 | 1 |
| hop 20, right 40 | 16 | 0 | 100 / 100 | 40 | 1 |
| hop 20, right 0 | 16 | 0 | 100 / 100 | 40 | 1 |
| hop 80, right 40, references from the chain without RNNoise | 11 | 0 | 280 / 200 | 220 | 2 |
| same, RNNoise off in the chain | 8 | 0 | 280 / 200 | 130 | 2 |

Nothing is lost inside the lookahead at any close: the hangover (350 ms) is five times the
stream delay (70 ms), so the drain comes long after the lookahead has been rendered. The
hypothesis is wrong for this engine; the flush was not built. A guard went in instead
(`SilenceGate._set_hops`: the hangover is never shorter than the stream delay, so no
future setting can make it true; `test_hangover_never_undercuts_the_stream_delay`).

**What does cut the sentence.** Two things, both at the user's -28 dBFS threshold:

1. *The gate closes inside trailing voice.* At 2 of 11 closes RMVPE still finds 80 and
   200 ms of voiced input after the last rendered sample (the second is the take's last
   words, spoken softly); by Silero, 4 closes lose 10-110 ms. RNNoise is not the cause:
   over the last 150 ms of every phrase it changes the level by -0.06 dB (p50; worst
   -0.3), the same as mid-phrase (-0.15). The take's soft endings fall under -34 dBFS
   (threshold - hysteresis) with the VAD under 0.35, and 350 ms later the gate drains.
2. *The gate flickers.* 7 of the 11 closes last 80-240 ms: a pause inside a sentence of
   ~350 ms closes the gate, the drain plays, and the reopen fades in 5 ms later. That is
   heard as a bite out of the sentence, not as a pause.

Hangover A/B (same take, same chain; `out/ab/phase_b/b1/`):

| hangover | closes | flickers (closed <= 240 ms) | lost voiced total | lost VAD total | gated hops / 743 |
|---|---|---|---|---|---|
| 350 (today) | 11 | 7 | 100 ms | 40 | 84 |
| 500 | 6 | 1 | 100 | 40 | 64 |
| 600 | 5 | 0 | 100 | 40 | 58 |

500 ms removes six of the seven flickers and costs 20 more open hops (1.6 s of GPU per
minute); 600 removes the last one. The 100 ms at the take's end is the soft last word
under the threshold at every hangover; only a lower threshold reaches it (the gate's
threshold is the user's slider). Wavs: `base_hop80_right40.wav` (350) vs
`hang500_hop80_right40.wav` vs `hang600_hop80_right40.wav`. `vox run --hangover-ms`
exposes it; the GUI still reads `calibration.json`'s `release_ms`.

**The live-only cut: the drift trim.** `InferenceWorker._hop` trimmed the output ring back
to its target on the *first* closed hop whenever the ring sat more than one hop over
target (after an underrun refill it always does, 8.2). `SPSCRing.drop` discards the oldest
unread samples, and on that hop the oldest queued output is still the hangover's render:
the end of the sentence when the gate closed inside it. Fixed: the trim waits until every
queued sample was written by a silent hop (`_silent_run * block >= fill + block`;
`InferenceWorker.trim_waits_for_silence`, the pre-fix rule kept for the A/B soak). Live
A/B (120 s, the take looped, gate -28, the ring pushed two hops over target after the
first hop, `drop` instrumented to peak-read what it discards):

| rule | trims | trimmed samples | peak of what was discarded | silent hops before the trim |
|---|---|---|---|---|
| pre-fix | 1 | 4320 (90 ms) | 0.0004 (-68 dBFS: the hangover's render of room tone) | 0 |
| fixed | 1 | 4320 | 0.0 (digital silence) | 10 |

On this take the discarded 90 ms was the converted room tone of the hangover, not voice
(-68 dBFS); with a hangover shorter than the ring (80 ms, `test_drift_trim_waits_until_
the_queued_output_is_silence`) it is the sentence, which the test checks sample for sample
against the simulator. A bug fix, not a flag.

**2-minute live soaks** (hop 80 / right 40, gate -28, RNNoise, Broadcast's denoiser on the
GPU, the take looped, output on device 24):

| hangover / onset rule | hops | closes in 120 s | gated hops | underruns | hop p50 / p99 / window-p99 max (ms, closed hops included) | ring (ms) | live estimate |
|---|---|---|---|---|---|---|---|
| 350, persist 1 (today) | 1503 | 22 | 181 | **0** | 7.2 / 31.5 / 36.6 | 177 | 376 ms |
| 500, persist 1 | 1503 | 13 | 142 | **0** | 6.9 / 26.8 / 36.2 | 170 | 369 |
| 500, persist 2 | 1503 | 13 | 142 | **0** | 7.4 / 30.8 / 35.6 | 170 | 369 |

(hop p50 is low because a closed hop costs ~1 ms; the open-hop cost is the 17-19 ms of
13.6. The estimate is hop 80 + ring 170 + compute + sola 30 + lookahead 40 + devices 22 +
RNNoise 20: the hop-80 shape the user chose costs ~210 ms more than the code default.)

Acceptance: lost speech per close <= 10 ms by the lookahead definition (it is 0); the
final-consonant test (`test_gate_renders_the_final_consonant_of_a_sentence`: speech, then
120 ms of noise 20 dB down, then silence: the noise is rendered within 30 dB of the
speech and the close comes after it); no underruns in the 2-minute soaks (table above).

### 14.2 B2: "I'd love a way to laugh"

(a) Live. The harness `b2_laugh.py` runs a take through the user's chain in Speak and in
Natural and reports, per laugh-like segment (>= 3 voiced pulses of 40-300 ms inside 1.2 s,
or segments given on the command line): gate closes and opens inside it, the share of
unvoiced frames RMVPE gives the pulses, the pulses' F0 p10/p90, and how far the auto-pitch
register moves across it. No recording of the user laughing exists; on the typed
"hahaha!" proxy (not a laugh: three TTS syllables) it finds one segment of 2 pulses,
28 % unvoiced, F0 192-268 Hz, 0 gate closes, register move +0.2 st. The real
measurement waits for a Record-button take of the user laughing and talking (20 s).

(b) Typed. Neither backend has a laugh tag (Kokoro has no expressive markup; edge-tts
strips SSML), so the insert is the spelling. `[laugh]`, `[laughs]`, `[laughing]`,
`*laughs*`, `(laughs)` now become the same spoken "haha" as `lol` (set off by commas,
dropped with `--laughter drop`; `test_an_explicit_laugh_insert_is_said_like_lol`).
Spellings rendered through Mom for the ear, `out/ab/phase_b/b2/`: `typed_haha.wav`
("haha", the insert's rendering), `typed_ha_ha_ha.wav`, `typed_hahaha_bang.wav`
("hahaha!"), `typed_heh_heh.wav`, `typed_laugh_tag.wav` ("[laugh]", identical in text to
the first), `typed_laugh_dropped.wav` (the same sentence with the laugh dropped). If none
convinces, the answer is "the backend cannot", and the insert stays a comma-set "haha".

### 14.3 B3: "the singing option doesn't seem to do much"

**What Sing changes on Mom** (`b3_presets.py`: open stream at the user's shape, Speak =
pitch 12 / protect 0.35 / ceiling 330 / formant -0.5 / index 0 (inert) / trim +6; Sing =
the user's Sing entry: pitch 12 (they set it), protect 0.5 (= off), ceiling 360, index 0.3
(inert), trim 0 / max gain +10):

| take | preset | output F0 p50 / p95 / max (Hz) | voiced frames above the ceiling's knee | click | level |
|---|---|---|---|---|---|
| sung (30 s) | Speak | 310 / 330 / 346 | **96 %** (knee 262) | 1.11 | -16.1 |
| sung | Sing | 321 / 358 / 366 | 87 % (knee 286) | 1.02 | -16.9 |
| speech (30 s) | Speak | 209 / 266 / 320 | 6.4 % | 0.99 | -14.2 |
| speech | Sing | 208 / 265 / 337 | 2.3 % | 1.21 | -14.2 |

Sing differs from Speak by 30 Hz of ceiling and protect off (the user set both pitches to
+12; the index is inert on Mom); on speech that moves 4 % of voiced frames, which is why
it "doesn't do much". The sung take's real problem is the transposition: +12 st puts a
sung line at 310-360 Hz, above Mom's register (194.5 Hz median), and the soft ceiling
is compressing 87-96 % of every voiced frame. (The seam metric is undefined at hop 80's
12.5 Hz seam rate with 400 ms blocks and is left out.)

**Natural** (`PRESETS` gains "natural"; `NoteDetector` in `pitch/base.py`;
`RuntimeParams.note_detect / note_min_ms 400 / note_tolerance_st 1 / note_ceiling_hz /
note_pitch_range / note_freeze_ms 100`): Speak's sliders and input chain, auto pitch on
with `target_f0_hz` = the model's own (194.5), the ceiling always armed (Speak's, else the
Sing ceiling), protect at the Speak value; a voiced run of 400 ms within +-1 st of its own
running pitch is a held note, and while it lasts the Sing ceiling (360) and
`NATURAL_NOTE_RANGE` (1.0) replace the speech values and the register stops following
(from 100 ms into the run). All hot (device scalars and a `torch.where` in the F0 stage;
`test_note.py`, the controller test). Measured on the same takes:

| take | preset | output F0 p50 / p95 / max (Hz) | above the knee | held notes (>= 400 ms, +-1 st) | glide over a note, cents |p50| / |p95| | click | wav (`out/ab/phase_b/b3/`) |
|---|---|---|---|---|---|---|---|
| sung | Speak (+12 fixed) | 310 / 330 / 346 | 96 % | 2 (930 ms) | 27 / 52 (the singer's own) | 1.11 | `sung_speak.wav` |
| sung | Sing | 321 / 358 / 366 | 87 % | 2 | 59 / 71 | 1.02 | `sung_sing.wav` |
| sung | **Natural** | **197 / 276 / 297** | **11 %** | 2 | 43 / 77 | 1.20 | `sung_natural.wav` |
| sung | Natural, note range 0.8 | 196 / 276 / 297 | 11 % | 2 | 42 / 70 | 1.15 | `sung_natural_range80.wav` |
| sung | Natural without the register freeze | 193 / 267 / 291 | 7 % | 2 | 47 / 74 | 1.38 | `sung_natural_nofreeze.wav` |
| sung | auto pitch alone (no note switch) | 193 / 267 / 291 | 7 % | 2 | 47 / 74 | 1.38 | `sung_natural_nonote.wav` |
| speech | Speak | 209 / 266 / 320 | 6.4 % | 0 | - | 0.99 | `speech_speak.wav` |
| speech | **Natural** | 189 / 247 / 291 | 2 % | 0 | - | 0.99 | `speech_natural.wav` |

Auto pitch is what changes the sung take: the register lands on Mom's 194.5 Hz instead of
+12 st's 310, and the ceiling stops working on nearly every frame (96 % -> 7-11 %). The
note switch is small on top of it on this take (only two runs of >= 400 ms within +-1 st in
30 s of this singer's vibrato): the freeze keeps the register 4 cents further from the
note (43 vs 47 cents of glide; the other 27 are the singer's), and the note ceiling
(360) lets the held notes sit 4 % more often above the 330 knee. Speech under Natural is
Speak's output 1 st lower (auto pitch puts the user's ~105 Hz median on 194.5 instead of
+12 = 210) with the same click ratio.

The glide column is the output's F0 drift over each held note (400 ms in to the end,
cents, |p50| / |p95|); in Speak (fixed pitch) it is the singer's own 27 / 52 cents.

Natural is the third button in the Live screen; Speak and Sing are unchanged; no default
changed. The register's follow (2 s) is still the wrong constant for a sung line that
changes key mid-phrase (Q-open since §9); the note freeze does not touch it.

### 14.4 B4: "moving the mic or my chair sometimes leaks a tiny bit"

Four options, all flags (`GateConfig.onset_persist_chunks`, `ConditioningConfig
.suppression_strength`, `.highpass_hz`, `ShapeParams.onset_ramp_energy_ms`; `vox run
--onset-persist-chunks / --suppression-strength / --onset-ramp-energy-ms`), measured on
the proxy take (T6) at the user's gate (-28, Silero, hangover 350) and chain. "Leaked" =
the gate opened during a transient (or within 150 ms after it) with no phrase within
300 ms; "lost speech" = speech frames (Silero >= 0.35 and energy >= -34 on the dry chain)
never rendered; onset lag = first rendered input frame after each phrase start.

*Transients at -8 to -2 dB under the voice's p95 (first proxy):* no variant leaked any of
the 18 (baseline included); lost speech 340 ms / 78 s baseline, 580 with "max"
suppression, 470 with the 90 Hz high-pass; onset lag p50 0 ms in every variant (the
lookahead renders the audio before the open); WER 6.0 % baseline (Whisper small.en
against the script), 6.0-8.0 across variants; UTMOS 2.69-2.74 (the variation of one
render); sibilant centroid 6541-6549 Hz; conditioned speech level -15.30 to -15.34 dBFS.
At that level a creak is 6 dB under the energy-only onset (threshold + 6) and Silero
calls it non-speech, so today's gate already holds.

*Transients at 0 / +4 / +8 dB over the voice's p95 (second proxy):*

| variant | leaked / 18 | lost speech, ms per 60 s | onset lag p50 / p95 | WER | UTMOS |
|---|---|---|---|---|---|
| baseline | 0 | 270 | 0 / 5 | 9.3 % | 2.72 |
| persist 2 | 0 | 270 | 0 / 5 | 7.3 | 2.72 |
| suppression max | 0 | 416 | 0 / 33 | 8.7 | 2.74 |
| high-pass 90 | 0 | 293 | 0 / 0 | 7.3 | 2.75 |
| energy-onset ramp 10 ms | 0 | 270 | 0 / 5 | 6.0 | 2.72 |
| all four | 0 | 362 | 0 / 61 | 9.3 | 2.74 |

Still nothing leaks, and the reason is in front of the gate: **RNNoise removes the
synthetic transients before the gate sees them.** Peak 10 ms level of each event after
the chain, dry vs with RNNoise (dBFS): bumps -6..-11 -> -71..-81, taps -10..-14 -> -76..-81,
creaks -5..-9 -> -54..-81 with one exception at -22.8 (the energy-only onset needs -22).
So on this proxy the gate rules are untested by construction, and the leaks the user
hears are transients RNNoise lets through, i.e. speech-like ones (a harmonic chair
creak, a bump in the voice band). "Max" suppression costs 150 ms of speech per minute
(RNNoise's own VAD calls soft onsets non-speech first) and the 90 Hz high-pass 20 ms, both
without a gain here; the ramp and the persistence cost nothing because no energy-only
onset ever fired (every phrase opened with the VAD, lag 0-5 ms thanks to the lookahead).
WER moves 6-9 % between renders of the same take (Whisper on a 78 s proxy; the seed's
noise), so none of the variants is distinguishable by it; UTMOS 2.72-2.75 likewise.

*RNNoise off in the chain (the transients reach the gate; what the rules do to a leak):*

| variant | leaked / 18 (creak / bump / tap) | gate opens | lost speech, ms per 60 s | onset lag p50 / p95 |
|---|---|---|---|---|
| baseline | 18 (6 / 6 / 6) | 36 | 200 | 0 / 48 |
| **persist 2** | **9 (6 / 1 / 2)** | 27 | 200 | 0 / 48 |
| high-pass 90 | 18 (6 / 6 / 6) | 36 | 46 | 0 / 0 |
| energy-onset ramp 10 ms | 18 | 36 | 200 | 0 / 48 |
| persist 2 + ramp 10 | 9 (6 / 1 / 2) | 27 | 200 | 0 / 48 |

Without RNNoise every transient opens the gate on energy alone. The persistence rule
stops the 25 ms taps and 60 ms bumps (one chunk each) and none of the 150 ms creaks (two
chunks loud), at no cost in speech or onset lag. The 90 Hz high-pass stops nothing but
changes what the auto level sees in pauses (46 vs 200 ms of "lost" speech is the AGC
settling differently, not the gate). The ramp changes only how an energy-only onset
fades in (the wav, not the count).

### 14.5 Tests

`tests/unit/test_gate.py` (+2: the hangover guard, onset persistence with the energy-only
flag), `tests/unit/test_note.py` (new, 5: the detector's timing, vibrato vs jumps,
unvoiced resets and the disabled state, the register freeze and range override, the
ceiling switch), `tests/unit/test_suppression.py` (+1: "max" strength),
`tests/tts/test_normalize.py` (+1 x 7 spellings), `tests/api/test_controller.py` (+1 the
Natural preset; the Sing expectation grew the four mode fields `set_preset` now sends),
`tests/engine/test_worker.py` (+1 drift trim), `tests/engine/test_engine.py` (+1 final
consonant). Suite: 842 passed, 1 skipped, 2 xfail; `test_worker_rebuilds_shape_live_and_keeps_running` failed once in the full run (graphs fell back to eager after the hop change, with the GUI and the desktop apps on the GPU) and passes with `test_worker.py` alone (7 passed), as 12.4 / 13.10 recorded for the same file.

### 14.6 Not verified

- Listening: none of the phase-B wavs has been heard. The B1 hangover A/B, the Natural
  renders and the typed-laugh spellings are the user's to decide.
- B2 (a) and B4 on real material: both need a Record-button take (laughing; creaks and
  bumps). The proxy transients are synthetic; a real creak's spectrum and the VAD's answer
  to it may differ.
- The drift-trim fix was shown to discard digital silence instead of the hangover's
  -68 dBFS render; a case where the pre-fix rule discards voice needs a hangover shorter
  than the ring (the test) or an underrun during a sentence end (not caught in a soak).
- The Natural preset's register follow on key changes mid-line (§9's open question).
- `NATURAL_NOTE_RANGE` is 1.0 (intervals kept); 0.8 was rendered for the ear only.

### 14.7 What the user heard (2026-10-22, the files above, not blind)

- B1: `base_hop20_right0` "ass"; `base_hop20_right40` "a little better but unhuman";
  `base_hop80_right0` "stutters"; `base_hop80_right40` "nothing actually cut off";
  **`hang500_hop80_right40` "sounds really good and smooth ... as for pronunciation and
  cut offs, its great"**; `hang600` "small stutters but no cutoffs". Verdict: hangover
  500. The hangover default (`VadConfig.release_ms`, and the profile's calibration.json)
  is now 500 ms.
- B3: `sung_speak` "sounds so bad"; `sung_natural` "sounds decent actually, but voice did
  drop in pitch"; `sung_natural_range80` "decent but it dropped the high pitch voice to
  SUPER low"; `speech_speak` "goat"; `speech_natural` "sounds okay yea". Natural is
  better than Speak on singing and not worse on speech, but the sung register landing on
  Mom's *spoken* median (194.5 Hz) is too low for singing: the next step is a sung
  target above the spoken one (a `note_target_offset_st` on held notes, or a higher
  `target_f0_hz` in Natural), heard before it becomes a default. Natural stays a preset.
- B2: `typed_hahaha_bang` "would be great with some improvements"; `typed_haha` "too
  forced"; `ha_ha_ha` "needs serious improvement"; `heh_heh` no; the dropped one is no
  laugh. Verdict: the backend's laugh is not convincing yet; "hahaha!" is the best
  spelling but still "needs improvements", so `LAUGH_SPOKEN` stays "haha" and the laugh
  (spelling, speed, pitch contour, breath) is open expressive work.
- B4: `all` (the proxy with its deliberate 1.5 s pauses) "sounds really good ... when the
  louder part came in it kinda bugged out just a tiny bit"; `persist2_hp90` "sounds great
  too, almost like narrating a story slowly". No leak to judge on the proxy (14.4); the
  "louder again" glitch is the auto level's fast-down gain on the shout (8.2), not B4.

**Defaults, one line each, if the user hears what the numbers say:** B1: hangover 500 ms (done, 14.7). B2: no default change (the best spelling is still unconvincing, 14.7); nothing live until the take exists. B3: Natural as the default preset once its sung register is raised (14.7); until then a preset. B4: nothing until a real creak/bump take; `onset_persist_chunks = 2` is the candidate (halves the leaks RNNoise does not catch, costs nothing measurable; the user liked `persist2_hp90` by ear), the high-pass stays 70, "max" suppression stays off (150 ms/min of speech)..

## 15. Paralinguistics and transients: the parts that need no new take (2026-10-23)

Report 19 (`docs/research/19_PARALINGUISTICS_AND_TRANSIENTS.md`) ranks its work behind
three takes the user has not recorded yet (typing, laughing, whispering and kissing). This
section builds the parts that do not need them, **each behind a flag that is off by
default**: the in-speech de-clicker (19 §1.4), context-based probation (19 §1.3), the laugh
guard and ceiling lift (19 §3.3), the breath / click passthrough (19 §3.2), and a bench
command that runs any take through all of them. **No default changed.** Branch
`feature/paraling` off `main` @ 439a314.

Every number is at the user's live shape (14, step 0: hop 80, extra 2400, right 40, gate
-28 dBFS / hangover 500, RNNoise on, auto level +6 dB, Mom +12, protect 0.35, ceiling 330,
formant -0.5) on the raw-Shure matrix take
(`recordings/matrix_20261017_192237/shure_gateoff_agcoff_hop20_mic_raw.wav`, 59 s), or on a
keystroke proxy built from it (15.5), unless stated. Outputs:
`out/ab/paraling/shure_gateoff_agcoff_hop20_mic_raw/` (`report_run*.json` per run).

### 15.0 Deviations

| # | Report / prompt said | What was built | Why |
|---|---|---|---|
| P1 | 19 §1.4: compare the fast envelope with E_slow; "fell back within 6 dB of E_slow" | Candidate: fast vs the slow envelope **2 ms earlier**. Fell back: the signal's **RMS** over [12, 20] ms after the candidate within 6 dB of the **RMS of the 6..2 ms before it** | Measured on the test click and the proxy: the slow envelope rises under the click and hides it; through a stop closure it still remembers the vowel, so a weak aspiration "fell back" (a /p/ was clamped); in a quiet room (-91 dBFS) the 1.5 ms fast envelope's own decay is still 30 dB up at 20 ms, so the keystroke's push was rejected and only its second sub-impulse was clamped (-6 dB instead of -40) |
| P2 | 19 §1.4: the clamp spans "candidate to where it fell back" | To where the fast envelope is back **for good** within the 12 ms | a keystroke push is two sub-impulses (touch, hit, 19 §1.1); the dip between them is not the end |
| P3 | 19 §1.4: apply the attenuation retroactively to the ring and `in_tail` | `Declicker` returns a look-back gain; `InputConditioner` splits it across RNNoise's 20 ms delay (the part landing in the current output is applied there) and hands the rest to `HopPipeline.apply_retro`, which scales `in_tail`, the 16 kHz ring and the input resampler's carried window | the clamp is decided on the high-passed signal, before RNNoise; RNNoise moves everything after it 20 ms later. The resampler's window must be scaled too or the next hop's recomputed 10 ms would bring the click back |
| P4 | 19 §3.3: flicker guard at 3 flips/s | `laugh_flip_rate` **10** flips/s over a **1 s** mean | speech on the matrix take flips 3 flips/s (p50), 5 (p90) over 1 s windows, up to 8.3 in its busiest stretches (15.3): at 3 the register would freeze in ordinary talk. A laugh is ~8.8 flips/s (19 §3.3), inside the speech range: the flicker flag cannot separate them on this data |
| P5 | 19 §3.3: jump flag on every hop | Only once the register has heard `REGISTER_WARM_FRAMES` | before that the level is the first hop's pitch and every hop "jumps" (weight 0.001 at 3.8 s on the take) |
| P6 | 19 §1.3: onset flag on "today's rule" sub-frame rise | Sub-frame level >= threshold + 6 dB **and** >= 12 dB over the quietest sub-frame of the last 200 ms; the rise rule is kept but does nothing with RNNoise on | RNNoise puts the room at about -84 dBFS: every logged onset "rose" 61-68 dB |
| P7 | Prompt: b2_laugh.py | Moved into the repo as `benchmarks/b2_laugh.py` (was a scratch script) | the user's laugh take must find it next session |

### 15.1 In-speech de-clicker (`audio/conditioning.py` `Declicker`, `ConditioningConfig.declick`)

After the high-pass, before RNNoise and the gain. Fast (1.5 ms) and slow (25 ms) one-pole
power envelopes; a candidate is a rising edge of fast >= 4 x slow-2-ms-earlier (+12 dB) above
-80 dBFS; 20 ms later (the lookahead the engine already buys, 19 §1.2) it is a click when the
RMS over [12, 20] ms is within 6 dB of the RMS just before it, and its span is scaled to that
level + 6 dB with 2 ms cosine ramps (a soft clamp). Numpy, host, per block; zero added
latency (the look-back samples are inside the 40 ms right context; the engine test
`test_declick_lookback_rewrites_every_buffer_holding_the_samples` checks all three buffers).

**Clean take (the plosive check).** 25 clamps in 59 s, all false positives by definition (no
clicks were added). Mapped to Whisper's word timings (input side, `benchmarks/mic_declick.py` -> `mic_declick.json`):
11 outside words (mouth noises, breath onsets, the take's first and last sample), 14 inside
or at the edge of words: "and" (-3, -7 dB), "The" (-16 dB, 0.27 of a word *before* it: a lip
smack), "pickled" /p/ (-4), "big" (0), "the" (-8, -1), "to" (-5, -4), "can" /k/ (-2), "kept"
final /pt/ (-3), "Thank" (-9, -4). Whisper on the conditioned mic: WER **2.0 % off, 2.0 %
on**; the earlier (P1-less) version moved it 2.0 -> 4.7 % by punctuation and a hallucinated
"Thank you for watching", no word changed. On the rendered output: WER 5.3 % both, UTMOS
2.63 -> 2.57, sibilant centroid 5977 -> 5959 Hz. **Whether the -3..-16 dB clamps on bursts are
audible is the ear's call** (`mic_declick_off.wav` vs `mic_declick_on.wav`, then
`base.wav` vs `declick.wav`).

**Keystroke proxy.** With RNNoise on (the user's chain) the synthetic keystrokes do not leak
at all, de-clicker or not: in pauses RNNoise removes them before the gate (as in 14.4), and
inside speech the log-mel change they cause is at the measurement floor (median 0.25 vs
floor p90 0.27-0.30). The de-clicker caught 3 of 18 in-speech clicks at -6 dB re the voice's
p95 and every click in a pause or the hangover, plus each key's release impulse 80-100 ms
later. On the input side (no model) with RNNoise off, the median click in the hangover
drops from **+56 dB to +18 dB** over the click-free input (pause: +49 -> +11); with RNNoise
on, the clicks were already gone and the de-clicker leaves RNNoise less disturbed (-7.5 ->
-1.5 dB). On renders with RNNoise off: gate opens on pause clicks **12 -> 3** of 18, hangover
log-mel change 0.98 -> 0.72 (floor 0.35-0.37), still a leak by the 1.5 x floor rule in 15 of
18 (the clamp leaves the click 6 dB over the room, and the release impulse is clamped less:
the slow envelope still holds the push).

### 15.2 Context-based probation (`engine/gate.py`, `GateConfig.onset_confirm_ms`)

`onset_confirm_ms` > 0 replaces the energy-only `strong` rule: a 10 ms sub-frame flag, the
gate held closed for the next 30 ms (they sit in the lookahead: with the 40 ms right context
nothing is lost, the open hop still renders the flagged sub-frame), then open if the level
has not decayed 8 dB from its peak within 20 ms ("held") or Silero's probability rose 0.25
("vad"); a rejected flag mutes new ones for 120 ms (a key's release). Every flag goes to
`SilenceGate.onset_log` with its crest factor, HF ratio (energy share above 2 kHz) and
verdict; the legacy rule logs its energy-only onsets the same way (verdict "legacy").

- Clean take: the same 3 energy-only onsets the legacy rule opened, all "held"; open share
  91.4 % both; WER 5.3 % and UTMOS 2.63 both. No speech lost.
- Proxy, RNNoise off (the only chain where clicks reach the gate): gate opens on pause
  clicks **12 -> 0** of 18; opens 30 -> 18 (the 18 phrases); 18 flags rejected, 9 held, 1 by
  the VAD rule. Hangover clicks are untouched (17 of 18 leak either way: the gate is already
  open there, 19 §0 path 1, which is the de-clicker's job).
- Logged distributions, proxy RNNoise off: rejected clicks crest 5.7 / 8.7 / 12.3 dB
  (p10/p50/p90), HF ratio 0.44 / 0.54 / 0.70; confirmed speech onsets crest 9.7 / 10.6 /
  13.9 dB, HF 0.22 / 0.84 / 1.0. **Neither separates clicks from speech** (19 §1.3 said not
  to threshold on them; the data agrees). With RNNoise on, the rise is 61-68 dB on every
  onset (P6).

### 15.3 Laugh guard and ceiling lift (`pitch/base.py` `LaughGuard`, `RuntimeParams.laugh_*`)

In the f0 stage, before the note detector: per hop, jump = sigmoid((|mean voiced log2 F0 -
register| - 0.75 oct) / 1 st) once the register is warm, flicker = sigmoid((flips/s over 1 s
- 10) / 1); the register's update is multiplied by (1 - jump)(1 - flicker); a lift (0..1,
~100 ms glide) toward max(jump upward, flicker) raises the soft ceiling log-linearly toward
`laugh_ceiling_hz` (600; an absent ceiling stays absent, the lift never lowers one). Hot
0-dim tensors, the same op sequence on or off (off = weight exactly 1, lift exactly 0:
`test_off_is_an_exact_no_op`), so toggling needs no recapture
(`test_new_flags_are_off_exact_and_only_passthrough_changes_the_op_sequence`).

- Unit test (`test_a_laugh_an_octave_up_barely_moves_the_register`): 1 s of 272 Hz calls on
  a 110 Hz register moves it > 4 st unguarded (19 §3.3 predicted ~6), < 1 st guarded.
- Speech, matrix take (replay of the engine's own classes, `b2_laugh.py --no-render`): the
  guard still dips on 7 of the 20 bursty stretches the laugh finder flags in plain speech
  (weight min 0.06 at 3.8 s, just after warm-up; 0.40 at 27.5 s where RMVPE reads 67 Hz
  creak; >= 0.63 elsewhere), lift <= 0.22 (ceiling up to ~375 Hz briefly), ceiling
  engagements 20 -> 15 frames. Rendered: WER 5.3 -> 4.7 %, UTMOS 2.63 -> 2.61.
- Typed "hahaha!" clip (`out/ab/phase_b/b2/typed_hahaha_bang.wav`, a TTS voice, 3 s: only a
  code-path run; auto pitch never warms in 3 s): **every voiced laugh frame is over the
  330 Hz ceiling's knee** (31 of 31, output p90 536 Hz vs knee 262 Hz), 19 §3.3 prediction 2.
  The real effect of the guard on a laugh needs the user's take.

### 15.4 Breath / click passthrough (`engine/postchain.py` `Passthrough`, `RuntimeParams.passthrough*`)

After the RMS mix, in the post stage: per 10 ms frame of `in_tail_ref` (the conditioned mic
under the decoded tail) and the decoded frames' pitch: energy, centroid; a frame passes when
unvoiced (`pitchf < 1`), above -55 dBFS, and either a burst (4 x both frames 30 ms away) or a
breath (unvoiced for >= 150 ms, centroid < 2.5 kHz); never a non-burst frame with a
centroid > 4 kHz (sibilants: 19 §3.2). Smoothed [1/4, 1/2, 1/4], interpolated to samples,
band-passed 150 Hz-10 kHz (129-tap FIR), mixed at `passthrough_mix` (0.25). Graph-safe;
switching it on changes the post stage's op sequence (`set_params` reports it, the worker
recaptures; the mix is hot). A device counter gives the share of decoded frames it fired on.

- Clean take: fired on **1.9 %** of decoded frames (3.0 % with RNNoise off); sibilant
  centroid 5977 -> 5974 Hz, active level unchanged, WER 5.3 % both, UTMOS 2.63 -> 2.62.
- **It cannot tell a kiss from a key.** With RNNoise off, `all` (de-clicker + passthrough)
  leaks *more* in the hangover than the de-clicker alone (log-mel change 1.19 vs 0.72,
  excess +3.8 dB): the passthrough mixes the clamped click back in as a "burst". 19 §3.2
  asked for a "mouth" marker (close source, no release impulse 90 ms later) decided before
  the de-clicker; it is not built: it needs the kiss + typing take to calibrate. Until
  then, passthrough and de-clicker should not be on together without RNNoise.

### 15.5 `vox bench paraling` and `benchmarks/b2_laugh.py`

`vox bench paraling --audio TAKE [--variants base,declick,confirm,laugh,passthrough,all]
[--preset speak|natural] [--proxy-clicks] [--quality] [--set key=value ...]` renders any take
through the user's chain (`benchmarks/paraling.py` `USER_*`) once per variant into
`out/ab/paraling/<take>/<variant>.wav` with `report.json` (gate opens / onset log, de-clicker
clamps, passthrough share, sibilant centroid and level, WER / UTMOS). `--proxy-clicks` builds
a keystroke proxy from the take (its phrases spaced by 1.2 s of its own room tone; per phrase
one two-impulse keystroke inside it, per pause one at 0.2 s (inside the hangover) and one at
0.85 s (gate closed); push 10 ms RMS at -6 / 0 / +6 dB re the take's speech p95, cycling per
place) and scores every click against the same render without clicks: output energy excess,
and the log-mel change (a waveform difference is useless: one click nudges the SOLA
alignment and every later hop shifts by a few samples, so the two renders differ by as much
as the signal everywhere; the log-mel change on click-free speech windows is the floor).

`benchmarks/b2_laugh.py --take T [--segments a-b,...]` renders Speak / Natural with and
without the guard, then replays the engine's tracker, guard and ceiling on the conditioned
input's F0 and reports per laugh segment: register move and peak (st), **recovery time**
(s until the register is back within 1 st), **ceiling engagements** (voiced frames over the
knee, share, output F0 p90 vs the knee), **voicing flips/s** (and the speech baseline outside
every segment), guard weight and lift, gate closes inside.

### 15.6 Placeholders waiting for the user's takes

| Threshold | Value | Waits for |
|---|---|---|
| `declick_ratio` (fast/slow) | 4 (+12 dB) | typing take: in-speech clacks quieter than that are not caught (3 of 18 at -6 dB) |
| `DECLICK_SETTLE_MS` / `DECIDE_MS` / `BACK_DB` / `PRE_MS` / `LAG_MS` | 12 / 20 / 6 dB / 6 / 2 ms | typing take (real clack decay and room level) and the ear on the plosive clamps (15.1) |
| `onset_confirm_ms`, `onset_decay_db`, `onset_vad_rise`, `onset_pair_ms` | 30 ms, 8 dB, 0.25, 120 ms | typing take (real clacks in pauses, key release spacing); note RNNoise removes the synthetic ones |
| crest factor, HF ratio | logged only | typing take: on the proxy neither separates clicks from speech |
| `laugh_jump_st` | 9 st (0.75 oct) | laugh take (laugh F0 vs the register; creak at 67 Hz also trips it) |
| `laugh_flip_rate` | 10 flips/s, 1 s mean | laugh take: speech here reaches 8.3, a laugh ~8.8 by the literature (P4) |
| `laugh_ceiling_hz`, lift glide | 600 Hz, ~100 ms | laugh take and the ear: "clamped laugh" vs "cracked laugh" (19 §3.3) |
| `PASS_*` (burst ratio 4 / 30 ms, breath 150 ms / 2.5 kHz, sibilant 4 kHz, floor -55 dBFS, band 150 Hz-10 kHz), `passthrough_mix` 0.25 | as listed | kiss / breath take, and a "mouth" marker to keep keys out (15.4) |

### 15.7 Tests

`tests/unit/test_declick.py` (5: a clack in a vowel clamped to about the vowel + 6 dB and
the rest untouched; a /p/ burst followed by aspiration and voicing left alone; a click whose
decision lands in the next block clamped through the look-back, identical to processing the
whole signal at once; off by default; the look-back split across RNNoise's delay);
`tests/unit/test_gate.py` (+2: a click in a pause rejected and logged with crest / HF, a held
rise opens energy-only within 30 ms; a burst confirmed by a VAD rise);
`tests/unit/test_laugh_guard.py` (3: off is exact, a laugh barely moves the register,
flicker lifts the ceiling and speech lets it down); `tests/unit/test_passthrough.py` (3: off
by default, a breath passes and a sibilant does not, voiced frames never pass and a short
burst does); `tests/engine/test_engine.py` (+2: `apply_retro` reaches all three buffers;
laugh guard needs no recapture, passthrough does, mix 0 equals off). Full suite: all pass
except `tests/engine/test_worker.py`, which fails 1-2 of its tests per run on an underrun or
a rebuild timing **with these changes stashed too** (the same flakiness 14.5 saw under load;
the desktop had OBS, browsers and Broadcast running).

### 15.8 Not verified

- Anything on real keystrokes, laughs, kisses, breaths or whispers: no take exists. The
  proxy's clicks are synthetic and RNNoise removes them, so in the user's chain this
  session measured the flags' *cost*, not their benefit.
- Live (worker) behaviour of the look-back under real timing; the simulator path and the
  engine test cover it, no live soak was run.
- The audibility of the de-clicker's clamps on plosives (15.1): the ear.
- Whisper passthrough, a Whisper input mode, the laugh bank (19 §4 items 7-10): not built.
- UTMOS seed floor for these renders: the -0.03..-0.06 moves are not shown to be outside it.

**Defaults, one line each:** none changed. De-clicker, probation, laugh guard and
passthrough are flags (`--declick`, `--onset-confirm-ms 30`, `--laugh-guard`,
`--passthrough` on `vox run`; `--set` / variants on `vox bench paraling`) until the takes in
15.6 decide them.
