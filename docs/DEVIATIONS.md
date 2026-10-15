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
| S9 | Experimental `sine_phase_carry`: SineGen phase continuous across hops; persistent 0-dim buffer advanced by `shift_frames` each hop; reset on gate onset and recapture; graph-safe | As specified, but the buffer is the pipeline's (`HopPipeline.sine_phase`, `[1, 1, 1]`) and is passed down through `RvcModel.infer` -> `SineGen.forward(phase, phase_frames)`, which reads it and writes the phase at frame `shift_frames` in place. Off by default | The loaded model is shared with typed-clip renders on a second CUDA stream: a phase buffer stored on the model would be advanced by the clip render and raced by it |
| S10 | RNNoise (48 kHz, 10 ms frames) after the high-pass, before AGC and the gate; live toggle with ~20 ms crossfade; `suppression_mix` | `audio/suppression.py`: the `rnnoise.dll` of the **pyrnnoise 0.4.5** wheel (xiph RNNoise, installed `--no-deps`, `requirements/suppression.txt`) through ctypes on preallocated buffers, one state carried across hops; mix and toggle as specified, plus a 100 ms unheard warm-up after an idle period; the user's on/off is remembered per microphone (`gui_state.json`), so the Broadcast mic keeps its own default (off, TARGET.md) | pyrnnoise is maintained and ships a Windows DLL (the machine has no compiler); its Python layer allocates per frame and imports audiolab and matplotlib, so only the DLL is used. Measured: **20 ms delay** (960 samples, correlation 0.99) and **0.39 ms CPU per 10 ms frame** (8.5) |
| S11 | Sing preset: AGC max +10 dB, trim 0 dB; one-time toast when Broadcast is the input while singing | As specified; the input-gain slider is locked at 0 in Sing; the toast shows once per app run, on switching to Sing, choosing the Broadcast mic, or starting with both | |

Not followed from the report (hard rules or not approved): lowering SineGen's unvoiced
noise (`sine_amp / 3` is what the models were trained with); `f0_unvoiced = "interp"` as a
default (it disables protect, which keys on `pitchf < 1`); `decoder_context_ms = 60`,
`embed_pad_ms = 20`, `f0_bridge_ms = 50`, `noise_roll`, `volume_envelope = 0.4`, protect
0.35 / index 0.3 as engine defaults (no A/B approved: the Sing preset carries index and
protect only for singing); a 40 ms default hop; an onset margin of 3 dB (measured as an
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
the engine defaults unless a profile file sets them. New and opt-in: the Sing preset (Speak
is the old behaviour), the soft ceiling (off in Speak), suppression (off unless switched on
or `rnnoise_enabled` in calibration.json; off on the Broadcast mic). Off: `sine_phase_carry`.

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
