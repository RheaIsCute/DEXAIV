# 17. Streaming engine research brief: why live sounds worse than whole-file, and what to build

**For:** a research assistant with read access to this repository (branch `feature/formant`)
and the web. **From:** the DEXAIV session of 2026-10-18. **Goal:** find out what it takes for
the live engine to sound like the offline whole-file render, survey how other real-time RVC
engines are built, and judge whether we should restructure ours or build a new one.

Read this file first. It points at the code and the documents that hold the details, states
what has been measured and tried, and lists the questions. Section 7 says what to deliver.

---

## 1. The problem in one paragraph

DEXAIV converts a microphone with an RVC v2 model in real time, in **hops** of 20 ms
(the GUI allows 20-100 ms). The user compared the same 45 s of their own raw microphone
(male, ~100-110 Hz, pitch +12 semitones, model `Mom`) rendered two ways: by the streaming engine,
and by converting the whole file at once. **By ear, whole-file is clearly better on both
models tested; live at 20 ms is "really stuttery".** Larger hops remove the stutter: 60 ms is
still stuttery with small "chopped" artifacts; 200 ms has no stutter and "sounds alright, but
there are still things to work on". So the stutter is mostly a function of how many
hop-to-hop joins there are. A smaller remaining gap is the lack of future context.
A 200 ms hop costs about 180 ms of extra latency, which is too much for conversation.

**Wanted:** whole-file quality, or close to it, at a conversational latency, ideally under
~150 ms end to end.

## 2. The evidence

All renders are in `out/ab/model_vs_engine/`. They are not in git, so ask the user if you
need them. Source: `recordings/matrix_20261017_192237/shure_gateoff_agcoff_hop20_mic_raw.wav`,
the first 45 s, levelled once by the GUI's input chain. Fixed pitch +12, index 0, formant 0,
seed 0, and every file set to -14 dBFS active level.

| Render | Hop | Crossfade | Extra latency vs hop 20 | User's verdict | Log-mel L1 vs whole-file (speech frames) |
|---|---|---|---|---|---|
| `Mom_wholefile.wav` | whole file | n/a | n/a | "sounds great" | 0 (reseed floor 0.25) |
| `Mom_live.wav` | 20 ms | 20 ms | 0 | "really stuttery, pretty bad" | 0.59 |
| `Mom_live_hop20_blend30.wav` | 20 ms | 30 ms (beyond the hop) | +10 ms | not heard yet | 0.60 |
| `Mom_live_hop40.wav` | 40 ms | 20 ms | +20 ms | not heard yet | 0.56 |
| `Mom_live_hop40_xf40.wav` | 40 ms | 40 ms | +40 ms | not heard yet | 0.55 |
| `Mom_live_hop60.wav` | 60 ms | 20 ms | +40 ms | still stuttery, small "chopped" artifacts | 0.55 |
| `Mom_live_hop100.wav` | 100 ms | 20 ms | +80 ms | not heard yet | 0.54 |
| `Mom_live_hop150.wav` | 150 ms | 20 ms | +130 ms | not heard yet | 0.53 |
| `Mom_live_hop200.wav` | 200 ms | 20 ms | +180 ms | no stutter, "alright, still things to work on" | 0.51 |
| `GoatMommy_wholefile.wav` / `GoatMommy_live.wav` | whole / 20 ms | | | whole-file better | 0.60 (floor 0.29) |

**The metric does not track the ear.** Log-mel L1 barely moves between hop 20 and hop 200
(0.59 to 0.51) while the user hears the stutter disappear. Earlier sessions found the same
thing (`docs/DEVIATIONS.md` R3, §8.4, §8.8). A metric that does predict "stutter" would be
valuable in itself; brief 16 (`docs/research/16_AUDIO_ANALYSIS_RESEARCH_BRIEF.md`) asks
for one, so don't duplicate that work, but say whether anything you find also serves it.

## 3. How the engine is built (where to read)

Start with `docs/ARCHITECTURE_AND_METHODS.md` §1-3 (one hop, the context window, the seam
problem). Layout rules are in `docs/research/12_PROJECT_STRUCTURE.md`. Measurements and every
departure from the earlier research live in `docs/DEVIATIONS.md`.

**Per hop** (`src/vox/engine/worker.py` drives it, `src/vox/engine/pipeline.py` implements it):

```
mic block 48 kHz
 -> input chain (CPU): high-pass, optional RNNoise, auto level, limiter   audio/conditioning.py
 -> silence gate (CPU): energy + Silero VAD + hangover                    engine/gate.py
 -> resample_in: 48k -> 16k into a rolling ~1.1 s analysis window          audio/resample.py
 -> f0:     RMVPE on the newest ~310 ms, last 2 frames dropped, pitch cache pitch/
 -> embed:  ContentVec (HuBERT, 768-d, 50 Hz) over the WHOLE window        features/contentvec.py
 -> index:  optional k-NN blend + protect                                  retrieval/
 -> synth:  text encoder -> flow -> NSF-HiFiGAN, decoding only the tail    synthesis/ (generator.py, synthesizer.py)
            plus 80 ms of decoder left context, sine phase carried across hops, noise rolled with the window
 -> post:   resample to 48k, optional RMS envelope                         engine/postchain.py
 -> sola:   offset search (10 ms) + sin^2/cos^2 crossfade (20 ms) against the previous tail   engine/sola.py
 -> output ring -> device
```

- **Sizes.** Every buffer size comes from `engine/chunker.py`, the "chunking law":
  `convert = extra + crossfade + sola_search + block`.
- **CUDA graphs.** Each GPU stage is a captured graph (`engine/graphs.py`). That forces static
  shapes, no host syncs and no random draws inside a stage.
- **Cost per hop.** About 15-19 ms of GPU work on the RTX 5060 Ti (target machine: `TARGET.md`).
  RMVPE alone is ~7.9 ms (`docs/research/15_MODERNIZATION_REPORT.md`). Hop 10 is infeasible.
- **End-to-end latency.** About 100-130 ms at hop 20 (`ARCHITECTURE_AND_METHODS.md` §1).
- **Offline paths.** The whole-file reference is `offline/convert.py` (`FullFileConverter`,
  `render_full`). The streaming simulator used for the table is `offline/simulate.py`
  (`StreamSimulator`).
- **Benchmarks.** `benchmarks/bench.py`: `vox bench accept`, `ab`, `quality`, `take`,
  `formant`, `index`. Metrics are in `offline/metrics.py`.

**The core mismatch, as we understand it.** Every output sample is a crossfade of two decoder
renders of the same audio, made from windows shifted by one hop. The generator is extremely
sensitive to its input: at the same seed, a one-sample input delay moves the render by 0.10
log-mel L1, a third of the gap between two seeds (R3). So consecutive renders disagree, and
SOLA has to hide the disagreement 50 times a second at hop 20. What has already been done
about it (`ARCHITECTURE_AND_METHODS.md` §3, `DEVIATIONS.md` §8.8):

- `noise_roll`: the noise moves with the window instead of being redrawn each hop.
- `decoder_context_ms = 80`.
- `sine_phase_carry`.
- `f0_trim_tail = 2`.
- HuBERT phase-stable hops (multiples of 20 ms).
- The formant work (§12) found one more mechanism of this kind: a hop's right edge rendered
  differently from its interior. SOLA then drifted, pushing the pitch sharp.

**Already ruled out as the main cause of the gap** (R3, one experiment each):
- Hop size 20-250 ms, by the metric (the ear disagrees, see above)
- Context `extra` beyond 500 ms
- Oracle full-file F0
- A longer RMVPE tail
- Decoder left context
- Lookahead of 20-250 ms (0.535-0.593)
- Latent noise switched off

## 4. Constraints

- **Hardware:** exactly one target machine (`TARGET.md`): RTX 5060 Ti 16 GiB (sm_120),
  Ryzen 9 7900X, Windows 11, torch 2.7.1+cu128, Virtual Audio Cable output. No other hardware.
- **Latency:** conversational. ~100-150 ms end to end is the target. Every ms of algorithmic
  latency you propose must be named.
- **Compute:** ~16 ms of GPU per 20 ms hop leaves little headroom. Proposals that add compute
  must say where it comes from; the report 15 Phase 1 FCPE swap would free ~4-5 ms.
- **Models:** users bring ordinary RVC v2 `.pth` files (40k/48k, sometimes 32k), often with a
  `.index`. Options that need retraining are allowed, but must be marked as such. The user is
  open to a new engine, or a decoder trained once and reused across voices.
- **Style:** read `docs/research/15_MODERNIZATION_REPORT.md` first, so you don't redo its
  survey. Build on it, and say where you disagree with it.

## 5. Research questions

### A. Joining hops (the stutter)

1. **How the other real-time RVC engines join chunks.** For each, give the default chunk size,
   crossfade and SOLA method and length, extra context, and any trick that keeps consecutive
   chunks consistent. Cite the source file and line. Engines:
   - w-okada voice-changer
   - RVC-WebUI realtime (`gui_v1.py`, `infer/lib/rtrvc.py`)
   - Applio realtime
   - Deiteris's voice-changer fork
   - anything else current
   Note in particular which chunk sizes their users actually run live.
2. **Joins beyond SOLA plus a sin²/cos² crossfade.**
   - Longer overlap-add windows.
   - WSOLA or phase-vocoder joins.
   - Crossfading in the latent, mel or STFT domain rather than in the waveform.
   - Averaging several overlapping renders of the same output region (render each region
     k times in successive hops and blend).

   For each: expected effect, latency cost, compute cost.
3. **Why a bigger hop helps by ear when the metric says it barely does.** Is "stutter"
   periodic modulation at the hop rate (50 Hz at hop 20)? Point to literature on perceiving
   periodic micro-discontinuities. Would random or jittered seam positions mask it?

### B. Making the decoder streaming-consistent

4. **Stateful (cached-convolution) inference of HiFi-GAN or NSF-HiFiGAN**, so each output
   sample is rendered once instead of twice and crossfaded. Is it possible without retraining,
   given RVC's non-causal convolutions? What breaks (receptive field ~±90 ms here), and how
   much right context would it need?
5. **Streaming vocoders built for this.** Candidates to check, not to assume:
   - Causal or streaming HiFi-GAN variants
   - Vocos and other iSTFT heads
   - Streaming NSF decoders
   - The decoders inside StreamVC, LLVC, RT-VC, MeanVC and Seed-VC's streaming mode

   For each: can it replace RVC's decoder? Does it need retraining per voice, or once?
   What latency and compute on a 16 ms budget? What quality evidence is there?
6. **Deterministic rendering.** Can the excitation noise, the flow's noise and the decoder be
   made a function of absolute time (as we did for the formant stretch), so two windows
   rendering the same instant agree? What does the RVC flow contribute to the disagreement
   between hops?

### C. The front end over a sliding window

7. **ContentVec/HuBERT over a sliding ~1.1 s window.** How much do the features of the same
   instant change from one hop to the next? Look for research on streaming or causal SSL
   features: chunked HuBERT, causal distillations, the features StreamVC and LLVC use.
   Should features be computed once per instant and cached, rather than recomputed over the
   whole window every hop? The normalisation over the window (R4) matters here.
8. **F0 in streaming.** RMVPE runs on the newest 310 ms and drops the last 2 frames. Compare
   FCPE, streaming CREPE variants and others for frame-to-frame consistency at the right edge
   on male-to-female speech, and say how that consistency is measured.

### D. Architecture: how real-time voice-conversion engines are built, and should we build our own?

9. **A short architectural survey.** Cover: RVC-style "convert a window, keep the tail";
   fully causal streaming VC (StreamVC, LLVC and others); chunked models with KV or
   conv-state caches (MeanVC, Seed-VC streaming, and similar); and anything newer. For each:
   - latency
   - compute
   - quality evidence (with links)
   - whether it is trained per target voice or zero-shot
   - whether an RVC user's existing `.pth` files carry over
   - licence
10. **Your recommendation for DEXAIV**, one of:
    - (a) keep the RVC window engine and fix the joins (which techniques)
    - (b) keep RVC's encoder but swap in a streaming decoder (train once? per voice?)
    - (c) a new engine around a streaming model

    For each option give the effort, risk, expected gain against the hop-200 and whole-file
    renders, and what the user would lose. Check it against report 15's three phases.

### E. Measuring it

11. A test that predicts "stutter" and "chopped" from audio, validated against listeners if
    such work exists. Our current tools are log-mel L1, a seam click ratio, F0 flips/min,
    WER, ECAPA, UTMOS and blind A/B pairs. Coordinate with brief 16 rather than repeating it.

## 6. What not to spend time on

- GPU kernel speedups (report 15: the engine is at the GPU floor). Name them only where they
  pay for a quality technique.
- Voice-model shopping: the user is researching models separately.
- TTS (`vox say`).
- Index tuning: measured inert on `Mom` (`DEVIATIONS.md` §12.3).

## 7. Deliverable

One markdown report, structured as the questions above (A-E), ending with a ranked list of
concrete changes. For each change give:

- what it is
- the evidence (link, and what it showed)
- expected effect on stutter and on the gap to whole-file
- added latency in ms
- added GPU time per hop
- whether it needs retraining
- the files in this repo it would touch

Mark anything you could not verify in source code or a paper as **unverified**. Never invent
numbers, and say "not found" when you looked and found nothing.
