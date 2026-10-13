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
