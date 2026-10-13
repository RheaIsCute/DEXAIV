# 08 — Voice Personalization: Single-User Calibration & Male→Female Tuning

Status: research complete · Scope: one user, one mic, convincing M→F conversion, Windows + CUDA
Code references: `Applio/rvc/realtime/pipeline.py`, `rvc-webui/infer/vc/pipeline.py`, `voice-changer/server/voice_changer/RVC/RVCSettings.py`, `Vonovox` (compiled; strings + README)

---

## 1. Why single-user tuning changes the design

Every public realtime RVC client (Applio realtime, w-okada voice-changer, Vonovox) is built for an
unknown user: anyone's mic, anyone's F0 range, anyone's room. That forces every tunable into a live
slider with a "safe" default — and safe defaults are mediocre for everyone.

We have exactly one user, one microphone, one room, one GPU, and a small set of target models. That
converts most sliders into **precomputed constants measured once during calibration**:

| Generic-client slider | Why it exists | Single-user replacement |
|---|---|---|
| Pitch / `f0_up_key` (±24 st) | unknown source & target F0 | constant per (user, model): `round(12*log2(F0_target_median / F0_source_median))` |
| Input gain / `silentThreshold` | unknown mic level & noise floor | measured RMS distribution + noise floor → fixed gain, fixed VAD threshold |
| F0 extractor `f0_min/f0_max` (Applio hardcodes 50–1100 Hz, `realtime/pipeline.py:171-172`) | must cover all humans | clamp to user's P1–P99 F0 ± margin → fewer octave errors, cheaper search |
| `index_rate`, `protect` | unknown model quality & articulation | per-model constants found by offline search + A/B (Section 6) |
| Block/context size | unknown GPU | benchmarked once on the actual GPU (doc 10) |
| VAD sensitivity | unknown noise floor | derived from measured noise floor + speech RMS gap |

Design consequence: **the runtime engine takes a profile, not a settings panel.** The GUI (later)
edits profiles; the hot path never branches on "what kind of user is this".

A second consequence: the calibration recordings double as the fixed quality corpus for the
benchmark system (doc 10), so every engine change is evaluated on the same voice it will serve.

---

## 2. Male→Female deep-dive

### 2.1 What the gender percept is made of (literature)

- **F0**: typical adult male conversational median ≈ 100–130 Hz, female ≈ 190–220 Hz. Listener
  studies place a **gender-ambiguous band at roughly 145–175 Hz, with ~165–220 Hz read reliably
  as female** when other cues agree. F0 alone is insufficient: a male voice pitched to 200 Hz with
  male formants reads as a male falsetto. See e.g. [Voice gender diversity: expression, perception
  and acoustics (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12539971/) and
  [Acoustic Features of Transfeminine Voices and Perceptions of Voice Femininity](https://digitalcommons.uri.edu/cmd_facpubs/41).
- **Formants / vocal-tract length**: female vocal tracts are ~10–15% shorter → formants ~15–20%
  higher (F1–F3). Listeners weight formants (especially F2/F3 region, vowel space expansion)
  heavily when F0 is ambiguous; formant biofeedback studies show formant raising alone shifts
  gender ratings ([UVA formant biofeedback study](https://sources.mandala.library.virginia.edu/node/33896)).
- **Breathiness / spectral tilt**: female phonation is on average breathier — higher H1–H2, more
  aspiration noise, steeper tilt. A secondary but audible cue.
- **Sibilant spectral centroid**: female /s/ centroids sit higher (~7–9 kHz vs ~5–7 kHz male)
  because of shorter front cavity. This is why mushy or darkened sibilants break the illusion.
- **Prosody**: wider F0 excursions, more rising contours; a monotone 200 Hz voice reads robotic or
  androgynous.

### 2.2 What RVC actually transfers vs what the user controls

RVC's pipeline: audio → ContentVec/HuBERT features (12.5 ms hop) → optional FAISS retrieval blend →
flow + HiFiGAN-NSF decoder trained **only on the target speaker**.

**Timbre and formants come from the trained decoder, not from knobs.** The ContentVec bottleneck is
trained with speaker-disentanglement objectives, so the 768-dim features carry mostly phonetic
content; the decoder re-renders that content with the target speaker's spectral envelope, formant
positions, sibilant spectra, and breathiness as learned from training data. This is why a good
female-target model feminizes formants and sibilants "for free" and why **no formant slider exists
in rvc-webui or Applio** — it would fight the decoder.

What *does* leak through the bottleneck (verified in code):

| Leaks through | Mechanism | Consequence |
|---|---|---|
| **F0 contour** | fed explicitly to the decoder as `pitch` (coarse, 1–255 mel bins) + `pitchf` (continuous, NSF excitation) — `rvc-webui/infer/vc/pipeline.py:130-139` | pitch must be transposed by the user; wrong transposition = wrong gender or falsetto artifacts |
| **Energy/loudness contour** | output RMS is the decoder's own, then optionally re-imposed from source via `rms_mix_rate` / `volume_envelope` | source dynamics (shouting, whisper) survive to the degree configured |
| **Prosody & timing** | F0 and phonetic timing are source-derived | RVC preserves the source's intonation shape (shifted) and speech rate — see 2.6 |
| **Residual accent/articulation** | ContentVec is not perfectly speaker-free | strong source habits (dark /l/, lazy sibilants) partially survive; `index_rate` pulls features toward target-training-set vectors to scrub this |
| **Unvoiced consonant character** | with `protect` > 0 the raw (non-retrieved) features are used on unvoiced frames | sibilant clarity vs. source-timbre leakage trade-off (2.5) |

User-controlled knobs and their real meaning:

| Knob | Code | What it does |
|---|---|---|
| `f0_up_key` | `f0 *= pow(2, f0_up_key / 12)` (`pipeline.py:130`) | multiplies the extracted F0 contour; the only gender-relevant knob |
| `index_rate` | `feats = retrieved * index_rate + (1-index_rate) * feats` (`pipeline.py:194-195`, k=8 FAISS search, inverse-square-distance weights) | blends content features toward nearest target-speaker training features; ↑timbre fidelity/accent scrub, ↓intelligibility & adds compute |
| `protect` | see 2.5 | preserves unvoiced-frame features from index blending |
| `rms_mix_rate` / `volume_envelope` | `change_rms`: output scaled by `rms_in^(1-rate) * rms_out^(rate-1)` (`pipeline.py:23-41`; Applio applies it only when `volume_envelope < 1`, `realtime/pipeline.py:580-587`) | rate=1: keep model's own loudness; rate→0: impose source envelope. Lower values make dynamics track the user but re-expose mic/room noise during pauses |

### 2.3 Why "+12" is the wrong universal answer

"+12 semitones for M→F" assumes every male median is exactly one octave below every female model's
median. Reality: a 95 Hz bass shifted +12 lands at 190 Hz — fine; a 135 Hz tenor shifted +12 lands
at 270 Hz — above almost any natural female conversational range, and the NSF decoder is now driven
outside the F0 distribution it saw in training → thin, whiny, unstable output. Conversely +12 on a
deep voice into a low-F0 female model (median ~170 Hz) overshoots too.

**Correct approach:**

```
shift_semitones = round( 12 * log2( F0_model_median / F0_user_median ) )
```

- `F0_user_median`: median voiced F0 over the calibration script (Section 4).
- `F0_model_median`: median F0 of the model's training data if known; otherwise measured from a
  rendered preview (run neutral text through the model at shift=0..+12 is circular — instead
  estimate from the model's F0 statistics by converting a reference clip and measuring where output
  F0 quality is best, or simply from metadata/the voice actor). Practical fallback: target
  200 Hz for a generic female model.
- Round to integer (RVC's coarse pitch is quantized anyway; fractional shifts buy little).
- **Validate by listening** at shift−1, shift, shift+1. The formula gets you within ±1 st; ears pick
  the winner, because decoder comfort zone matters more than arithmetic.

This is exactly what Applio's `proposed_pitch` feature computes per-chunk
(`realtime/pipeline.py:349-378`): `up_key = clamp(round(12*log2(threshold/median_f0)), ±12)` with
default `proposed_pitch_threshold=155 Hz`. Doing it **per chunk** at runtime is the wrong place —
median over a 100 ms chunk is noisy and makes pitch wander. We do the same math **once, offline,
over 60 s of speech**, and ship a constant. (Keep Applio's per-chunk mode as an emergency fallback
only.)

### 2.4 F0 extractor errors on low male voices

Low male voices are the worst case for every extractor:

- **Octave errors**: strong subharmonics/second harmonic cause period-doubling/halving. A frame
  read an octave high, then shifted +7, drives the decoder to ~500 Hz for one hop → audible chirp.
- **Vocal fry / creak**: ends of phrases drop to 50–80 Hz irregular pulses. RMVPE often returns 0
  (unvoiced) or jittery values there; rvc-webui then **linearly interpolates through unvoiced gaps**
  (`pipeline.py:126-128`), which invents a smooth contour through fry — usually benign, sometimes a
  downward "yawn" artifact into pauses.
- Extractor choice: **RMVPE** is the robustness king on low/creaky voices (threshold 0.03 in
  rvc-webui); **FCPE** (`torchfcpe`, local_argmax decoding, threshold 0.006) is cheaper/lower
  latency and fine on clean mids but flips octaves more on fry; Vonovox additionally ships
  **SwiftF0** (CPU, 4 threads — strings in `cuvc.pyd`) as an ultra-cheap option.

Single-user mitigations (all PROVEN concepts):
1. **Clamp `f0_min/f0_max` to the calibrated range** (e.g. user P1 ≈ 70 Hz, P99 ≈ 260 Hz → clamp
   60–300 Hz instead of Applio's 50–1100). Removes the entire upper octave-error space.
2. Median filter (3–5 frames) on `pitchf` before transposition — stock in several pipelines.
3. During calibration, *measure* each extractor's octave-flip rate on the user's own fry-heavy
   sentences and pick the winner per profile (Section 6).

### 2.5 `protect` — exact math and why it owns your sibilants

From `rvc-webui/infer/vc/pipeline.py:175-217` (active only when `protect < 0.5`):

```python
feats0 = feats.clone()                 # features BEFORE index retrieval
...index blend happens on feats...
pitchff = pitchf.clone()
pitchff[pitchf > 0] = 1                # voiced frames  -> weight 1
pitchff[pitchf < 1] = protect          # unvoiced (f0==0) -> weight = protect
feats = feats * pitchff + feats0 * (1 - pitchff)
```

So on **voiced** frames the index-blended features pass through untouched; on **unvoiced** frames
(plosives, /s z ʃ f/, breaths) the final features are
`protect * blended + (1-protect) * original`. `protect = 0.5` disables the branch entirely
(pure blended everywhere); `protect = 0` means unvoiced frames use 100% original ContentVec
features, bypassing the index.

Consequences for M→F:
- **Low protect (0–0.2)**: crisp consonants and clean sibilant timing (no index-retrieval smearing
  on noise-like frames), but unvoiced frames carry more *source* feature character — the decoder
  still renders them with target-trained filters, so sibilants usually still sound female, yet
  "accenty" articulation leaks more.
- **High protect (0.5)**: maximum target-timbre consistency, but noisy/ambiguous unvoiced frames get
  index-pulled toward whatever training vectors are nearest → occasionally lisped or "underwater"
  /s/, softened plosives.
- Sweet spot reported across communities and defaults: **0.33** (rvc-webui default), with 0.2–0.5
  the meaningful search range. Note `protect` matters **only when `index_rate > 0`** in the voiced
  path sense — strictly, the blend also protects against the retrieval step, so with
  `index_rate = 0` the protect branch is a near-no-op (feats ≈ feats0).
- w-okada default: `protect = 0.5`, `indexRatio = 0` (`RVCSettings.py`) — i.e. it ships with both
  mechanisms off, trusting the model. A good null hypothesis for strong models.

### 2.6 Prosody

RVC does not generate prosody; it **replays the source F0 contour, multiplied by 2^(shift/12)**.
Multiplicative shift preserves *relative* excursions (same ratio range), so a monotone male speaker
produces a monotone "female" voice — the single biggest naturalness gap that no parameter fixes.
Mitigation is behavioral (user speaks with more intonation) plus optional mild contour expansion
(`f0 = median * (f0/median)^k`, k≈1.05–1.15) — EXPERIMENTAL, risks instability on fry, keep off by
default but expose in the profile.

### 2.7 External formant shifting pre/post RVC — verdict from Vonovox

Vonovox exposes a realtime **Formant** slider ("adjust the formant of your voice without affecting
the pitch", README). The core is Cython-compiled, but symbol strings pin the implementation:
`core.inference.cuvc.RVC.change_formant`, `formant_shift`, `new_formant` live in **`cuvc.pyd`, i.e.
inside the RVC inference class itself**, and the package ships a `sifigan/cheaptrick` module
(`CheapTrick`, `AdaptiveWindowing`, `AdaptiveLiftering` — a differentiable WORLD CheapTrick spectral
envelope) plus SiFiGAN generator modules, and detects "SiFiGAN model with ..." at load. Reading:
Vonovox's formant control operates on the **source-filter–factorized decoder path (SiFiGAN-style
models)**, where the spectral envelope is explicit and can be frequency-warped independently of F0;
for stock RVCv2 models the classic trick is resampling the input by a ratio while compensating F0
(w-okada lineage — note `extract_pitch_wokref_style` in the same binary). Applio takes a different
route: an optional **pedalboard `PitchShift` post-effect** (`realtime/core.py:203-205`) and a
training-time formant augmentation via `stftpitchshift` (in requirements) — i.e. formant handling
pushed to pre/post FX, not the core.

**Conclusion:** with a **strong target model, external formant shifting is unnecessary and net
harmful** — the decoder already renders target formants; warping the *input* only perturbs
ContentVec features (which are designed to ignore the envelope anyway → little effect until the
warp is big enough to break phoneme recognition), and warping the *output* detunes the very formants
the model was trained to produce, adding chipmunk coloration and ~1 STFT of latency. External
formant shift earns its place only in two cases: (a) weak/androgynous models that need +5–10% help
(post-RVC, small ratios ≤1.08), (b) non-RVC fallback mode when the GPU is saturated. Recommendation:
implement a post-RVC formant knob as an **optional, default-off** profile field; do not build a
Vonovox-style envelope-warping path unless we adopt SiFiGAN decoders later. (LIKELY — grounded in
architecture reasoning + both repos' choices; Vonovox internals partially inferred from symbols.)

### 2.8 Breathiness and fry handling

- **Breathiness**: comes from the target model. If more is wanted: pick/train a breathier target;
  post-hoc breathiness (mixing filtered noise keyed to voicing) is EXPERIMENTAL and usually reads
  as hiss. Don't build it in v1.
- **Fry going in**: handled at the F0 stage (2.4): clamped range + RMVPE + median filter, and VAD
  (Silero, as in Vonovox/Applio) prevents trailing fry/breath frames from being rendered as low
  growls. Vonovox's "VAD Noise Reduction" mutes output when no speech is detected with a **400 ms
  release window** (README) — adopt the same release concept so word endings aren't clipped.
- **Breaths**: either muted by VAD or — often better for naturalness — passed through at low
  `volume_envelope` so the model renders soft breath noise; test both in A/B.

---

## 3. What this means for the engine defaults

| Parameter | Generic default | Our starting point (per calibration) |
|---|---|---|
| f0 method | rmvpe | rmvpe (fcpe as latency fallback), chosen by measured octave-flip rate |
| f0_up_key | 0 | `round(12*log2(F0_model/F0_user))`, validated ±1 by ear |
| f0_min/f0_max | 50/1100 Hz | user P1/P99 ± 20% |
| index_rate | 0.5 (webui) / 0 (w-okada) | per-model, from tuning loop; 0 if model is strong |
| protect | 0.33–0.5 | per-model, from tuning loop; 0.33 start |
| volume_envelope / rms_mix_rate | 1.0 | 1.0 unless A/B says dynamics feel flat, then 0.7–0.9 |
| silence threshold | 1e-5 (w-okada `silentThreshold`) | noise_floor_rms × margin (measured) |
| input gain | OS default | set so speech peaks ≈ −12 dBFS, from calibration |

---

## 4. Calibration design (30–60 s, run once per mic setup)

### 4.1 Guided script

Four passes, ~45 s total, recorded through the exact input chain (same device, same WASAPI config,
RNNoise/VAD **off** during capture):

1. **Silence** (5 s): "Don't speak, sit normally." → noise floor.
2. **Read passage** (20–25 s): phonetically balanced text (e.g. Rainbow Passage excerpt) at normal
   conversational effort → F0 distribution, RMS distribution, voiced/unvoiced ratio, spectral tilt.
3. **Dynamics** (10 s): one sentence quiet, one normal, one projected/loud → RMS extremes,
   clipping check.
4. **Edge material** (10 s): sustained /s/ and /ʃ/, "she sells seashells", a sentence allowed to
   trail into fry, a long vowel glide low→high → sibilant centroid, fry behavior, F0 extremes.

Analysis stack: `parselmouth` (Praat) or `librosa.pyin`/RMVPE itself for F0; plain numpy for RMS.
Using **the same RMVPE that the runtime uses** for the F0 stats is important — we calibrate the
extractor we ship, not Praat's idealized tracker. (PROVEN tooling.)

### 4.2 Measurements: keep vs skip

| Measurement | Keep? | Feeds | Justification |
|---|---|---|---|
| Median / P10 / P90 voiced F0 | **KEEP** | pitch shift formula; extractor clamps | directly determines the one gender-critical constant |
| Full F0 histogram | **KEEP** (cheap) | `f0_min/f0_max` = P1·0.8 / P99·1.25; sanity plots | octave-error suppression (2.4) |
| Speech RMS distribution (P10/median/P95) | **KEEP** | input gain target, `volume_envelope` sanity, benchmark normalization | one mic, one room → stable |
| Noise floor RMS + spectrum | **KEEP** | VAD/silence threshold = floor + 10–15 dB; decide if RNNoise is needed at all | w-okada's 1e-5 default is meaningless for an unknown mic |
| Clipping check (|x| ≥ 0.99 count) | **KEEP** | gain correction / "lower your mic" prompt | clipped input destroys ContentVec features silently |
| Voiced/unvoiced ratio | **KEEP** (bonus: free from F0 pass) | protect-search prior; sanity flag (<0.55 voiced on read speech → tracking problem) | tells us how much audio the `protect` path governs |
| Octave-flip rate per extractor | **KEEP** | F0 method choice | measured on the fry sentence; the decisive extractor metric for this user |
| Spectral tilt / H1–H2, sibilant centroid | **OPTIONAL** | diagnostics only | informative for "why does /s/ sound off", but no runtime knob consumes it in v1 |
| "Attack behavior" / onset sharpness | **SKIP** | — | no pipeline parameter consumes it; onsets are governed by block size + SOLA crossfade, which the benchmark (doc 10) tunes directly |
| Phrase duration / speaking rate | **SKIP** | — | RVC is frame-synchronous; rate passes through unchanged, nothing to tune |
| Jitter/shimmer | **SKIP** | — | clinical metrics; decoder re-synthesizes its own source signal, source micro-perturbations don't map to any knob |
| Room reverb RT60 | **SKIP** (note only) | — | would matter for denoise strategy, but RNNoise on/off A/B answers it faster |

---

## 5. Profile schema

`profiles/<name>/` — five small JSON files, split by *consumer* so modules load only what they own
and the GUI can regenerate one file without touching the others.

### `voice_profile.json` — consumed by: pipeline (F0 stage), tuning loop
```json
{
  "schema": 1,
  "created": "2026-10-13T18:00:00Z",
  "source_voice": {
    "f0_median_hz": 118.4, "f0_p10_hz": 92.1, "f0_p90_hz": 168.0,
    "f0_p01_hz": 74.0, "f0_p99_hz": 241.0,
    "voiced_ratio": 0.68, "spectral_tilt_db_per_oct": -11.2
  },
  "f0_extractor": {
    "method": "rmvpe",
    "f0_min_hz": 60.0, "f0_max_hz": 300.0,
    "median_filter_frames": 3,
    "octave_flip_rate": {"rmvpe": 0.004, "fcpe": 0.021}
  }
}
```

### `calibration.json` — consumed by: input stage (gain/VAD), calibration UI (re-run detection)
```json
{
  "schema": 1,
  "mic_fingerprint": "Shure MV7 (WASAPI) @ 48000",
  "noise_floor_dbfs": -62.5,
  "speech_rms_dbfs": {"p10": -31.0, "median": -24.5, "p95": -16.8},
  "clipping_detected": false,
  "input_gain_db": 3.0,
  "vad": {"engine": "silero", "threshold": 0.5, "release_ms": 400},
  "silence_gate_dbfs": -48.0,
  "rnnoise_enabled": false,
  "raw_recordings": "calibration/2026-10-13/"
}
```

### `audio_device.json` — consumed by: audio I/O layer only
```json
{
  "schema": 1,
  "input":  {"name": "Microphone (Shure MV7)", "api": "WASAPI", "exclusive": false, "sr": 48000, "channels": 1},
  "output": {"name": "Line 1 (Virtual Audio Cable)", "api": "WASAPI", "exclusive": false, "sr": 48000, "channels": 1},
  "monitor": {"name": "Headphones (Realtek)", "enabled": false}
}
```

### `runtime.json` — consumed by: engine scheduler (per-GPU, from benchmark doc 10)
```json
{
  "schema": 1,
  "gpu": "NVIDIA GeForce RTX 4070",
  "block_ms": 15, "context_extra_ms": 100, "sola_search_ms": 10,
  "precision": "fp16", "use_cuda_graphs": true,
  "benchmark_ref": "benchmarks/2026-10-13_rtx4070.json"
}
```

### `model_overrides.json` — consumed by: pipeline parameter loader (Section 7)
```json
{
  "schema": 1,
  "default_model": "aria_48k",
  "models": {
    "aria_48k": {
      "f0_up_key": 7, "index_rate": 0.25, "protect": 0.33,
      "volume_envelope": 1.0, "target_f0_median_hz": 196.0,
      "formant_post_shift": 1.0, "tuned": "2026-10-13", "tuning_report": "tuning/aria_48k.json"
    },
    "mika_40k": {"f0_up_key": 9, "index_rate": 0.0, "protect": 0.5, "target_f0_median_hz": 224.0}
  }
}
```

JSON (not TOML) because these files are machine-written by calibration/tuning and only read by
humans occasionally — see doc 11 for the config-format decision.

---

## 6. Automatic tuning loop (offline, same engine)

Principle: replay the calibration recordings through the **exact realtime engine in stream-simulation
mode** (doc 10) — never through the batch pipeline, whose padding/context behavior differs — and
score candidates.

### 6.1 Search space (coarse grid, ~2–4 h unattended on one GPU)

| Axis | Values | Count |
|---|---|---|
| `f0_up_key` | estimate −2 … +2 | 5 |
| `index_rate` | 0, 0.25, 0.5, 0.75 | 4 |
| `protect` | 0.2, 0.33, 0.5 | 3 |
| F0 method | rmvpe, fcpe | 2 |
| block/context | 2–3 winners from the perf benchmark (fixed before quality sweep) | 2 |

≈ 240 runs × ~60 s corpus. Stage it: sweep pitch first with index_rate=0/protect=0.5, lock the
pitch winner, then sweep index/protect — cuts to ~40 runs.

### 6.2 Objective proxies

| Proxy | How | Feasibility | Caveat |
|---|---|---|---|
| **Output F0 statistics** vs target band | RMVPE on output; score = distance of output median to `target_f0_median_hz`, penalty for P99 > 350 Hz or flip-rate > baseline | **PROVEN** | validates pitch + stability only, says nothing about timbre |
| **Speaker similarity** | resemblyzer (GE2E d-vector) or WavLM/ECAPA embedding; cosine(output, target-model preview renders) | **LIKELY** | discriminates good vs broken settings well; near the top, differences fall inside embedding noise — cannot rank the final 2–3 candidates |
| **Intelligibility (ASR WER)** | faster-whisper small/int8 on output vs the known script text | **PROVEN** mechanics / **LIKELY** sensitivity | Whisper is robust enough that WER only flags gross damage (high index_rate smearing, octave chaos); use as a *gate* (WER ≤ source-WER + 3 pts), not a ranker |
| Sibilant centroid of output /s/ segments | band 4–10 kHz centroid on the sibilant sentence | EXPERIMENTAL | cheap protect-setting signal; segmentation is fiddly |
| UTMOS / MOS predictors | pretrained MOS net | EXPERIMENTAL | trained on TTS artifacts, poorly correlated with VC gender-credibility — informational only |

Composite: hard gates (F0 flip-rate, WER, no NaN/silence) → rank survivors by speaker-cosine →
export **top 3 candidates**.

### 6.3 Mandatory human A/B

The final pick is by ear, no exceptions: blind A/B/X of the top 3 over the full corpus (especially
sibilants, laughter, fry sentences) plus 2 minutes of live mic. Automation reliably finds the
*plausible region*; it cannot judge "reads as a woman to a human on Discord". What automates
reliably: pitch offset, F0 clamps, extractor choice, gain/VAD thresholds, catastrophic-setting
elimination. What needs ears: final protect/index_rate pick, volume_envelope feel, breath handling,
any formant_post_shift ≠ 1.0.

Re-run triggers: new mic or interface, new model, engine DSP change (regression gate in doc 10),
or user reports drift ("sounds off today" → quick 15 s re-check of gain + noise floor only).

---

## 7. Per-model overrides concept

Every tunable that depends on the *pair* (user, model) lives under `model_overrides.json → models.<id>`,
resolved as: hardcoded engine defaults ← profile globals (`voice_profile`/`runtime`) ← per-model
override. Rationale: `f0_up_key` is pair-specific by definition (target medians differ per model);
`index_rate`/`protect` depend on each model's training quality and index file; a model trained on
breathy whisper data wants different `volume_envelope`. Switching models at runtime = swapping one
dict — no recalibration, no re-benchmark (block size is GPU-bound, not model-bound, unless the model
sample rate changes, which the benchmark table flags). Each override records its `tuning_report`
provenance so stale settings are auditable.

---

## Key takeaways

1. **One user → constants, not sliders.** Calibration measures F0 distribution, RMS/noise floor,
   clipping, voiced ratio, and per-extractor octave-flip rate; everything else (attack, rate,
   jitter) has no consumer and is skipped.
2. **Timbre/formants come from the target model** via the ContentVec bottleneck + trained decoder;
   what leaks through is F0, energy, prosody, and some articulation — exactly the things the knobs
   (`f0_up_key`, `volume_envelope`, `index_rate`, `protect`) exist to manage.
3. **Transposition is computed, not guessed**: `round(12*log2(F0_model_median/F0_user_median))`,
   then ±1 by ear. Applio's `proposed_pitch` already ships this formula (per-chunk); we move it
   offline. "+12" is a special case, not a rule.
4. **`protect` is a voiced/unvoiced feature crossfade** (`feats*pitchff + feats0*(1-pitchff)`,
   unvoiced weight = protect, 0.5 = off) — it is the sibilance/consonant-clarity knob and only
   meaningful with `index_rate > 0`.
5. **Skip external formant shifting by default.** Vonovox's formant slider rides a SiFiGAN/CheapTrick
   envelope path (compiled symbols) that stock RVCv2 doesn't have; with a strong model, input-side
   warping is ignored by ContentVec and output-side warping detunes trained formants. Keep an
   optional post-shift (≤1.08) for weak models only.
6. **Low male voices need F0 hygiene**: RMVPE + calibrated f0_min/f0_max clamps + short median
   filter + Silero VAD with ~400 ms release (Vonovox pattern) handle fry and octave errors.
7. **Automate the search, not the verdict**: F0 stats and WER gates are PROVEN, speaker-embedding
   ranking is LIKELY, MOS predictors are EXPERIMENTAL; the final pick among the top 3 is a blind
   human A/B, always.
