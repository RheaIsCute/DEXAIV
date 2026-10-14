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
