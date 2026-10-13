# 14 — Open Questions

Status: catalog complete · The unresolved engineering questions left by docs 01–11, each with a
concrete, benchmarkable test and the roadmap phase (doc 13) by which it must be decided.
Phases: P0–P9 per doc 13.

---

## Q1 — Minimum stable hop on the target GPU

- **QUESTION:** What is the smallest hop (model processing interval) at which `p99(T_compute) < hop`
  holds with ≥1.5× headroom on the user's actual GPU, with and without CUDA graphs?
- **WHY IT MATTERS:** Hop accumulation is the largest controllable latency term (doc 04 §4 #1);
  the whole aggressive budget (~110 ms) assumes hop 20 ms, the stretch (75–90 ms) assumes 16 ms.
  If the GPU can't sustain it, the latency target moves.
- **CURRENT EVIDENCE:** Doc 04 §3.6 estimates 10–20 ms total compute at extra=0.5 s on 3060/4080
  class GPUs (extrapolated from reference behavior, not measured on our code). Applio's floor is
  100 ms by policy, not physics (doc 03 §2.2). No reference runs hops < ~20 ms.
- **PROPOSED TEST:** Phase-2 perf sweep (throughput + paced modes), block ∈ {10, 16, 20, 32, 50} ms
  × extra ∈ {100, 300, 500} ms, eager; repeat in Phase 6 with graphs + fp16 HuBERT. Gate:
  p99 headroom > 1.5×, zero underruns over a 10-min paced soak.
- **DECISION REQUIRED BY:** provisional at P2 (sets conservative plan), final at P6.

## Q2 — FCPE vs RMVPE quality on this specific low male voice

- **QUESTION:** Does FCPE's lower cost justify its slightly weaker octave stability on *this user's*
  voice (fry, low register), or is RMVPE mandatory?
- **WHY IT MATTERS:** Octave errors post-transposition are catastrophic for M→F (register flips
  mid-word, doc 05 §4); FCPE saves ~1–2 ms/hop and has crisper V/UV.
- **CURRENT EVIDENCE:** Doc 05 §5: RMVPE most octave-stable (BiGRU long context), FCPE "very good,
  slightly weaker on creak/noise" — generic corpora, not this voice. Doc 08 §4.2 makes
  per-extractor octave-flip rate a calibration measurement.
- **PROPOSED TEST:** Calibration (P5): run both extractors over the `pitch_low`/fry corpus clip,
  count octave flips/min and V/UV disagreements vs a Praat-checked reference; pick per
  `voice_profile.json`. Re-A/B rendered output of the top candidates.
- **DECISION REQUIRED BY:** P5 (profile ships the choice).

## Q3 — Does EXTRA = 0.5 s suffice vs 2.5 s for output quality?

- **QUESTION:** How much context does HuBERT really need before the *kept* frames stop improving —
  is Applio's 0.5 s enough, or does the rvc-webui 2.5 s habit (and Vonovox's "2.0 recommended")
  buy audible quality?
- **WHY IT MATTERS:** EXTRA is zero algorithmic latency but linear-plus embedder compute
  (doc 04 §3.6: shrinking EXTRA is the single biggest compute cut); it directly bounds Q1.
- **CURRENT EVIDENCE:** Doc 07 §A.3: context frames stabilize kept frames via boundary-effect
  decay; Applio ships 0.5 s default (PROVEN workable), w-okada docs say longer = better accuracy
  with superlinear cost. No published quality-vs-extra curve exists.
- **PROPOSED TEST:** P2: render the full corpus at extra ∈ {100, 300, 500, 1000, 2500} ms with all
  else fixed; measure chunk-consistency (log-mel distance vs full-file render), speaker-cos, WER;
  blind A/B 500 vs 2500 on `slow_vowels` and `fast` clips.
- **DECISION REQUIRED BY:** P2 default; revisit at P6 when compute headroom is known.

## Q4 — Conv-frontend / feature caching feasibility

- **QUESTION:** Can the HuBERT conv frontend (strictly local, receptive field ≈400 samples) be
  cached across hops, and/or can a tail-recompute feature ring (mirroring the pitch cache) ship
  without audible shimmer?
- **WHY IT MATTERS:** The embedder is the dominant compute stage (doc 07 §A.3); frontend caching
  saves ~10–20%, a feature ring could cut the transformer cost several-fold — the path below the
  75 ms stretch target.
- **CURRENT EVIDENCE:** Doc 07 §A.4: no reference does either; bidirectional attention invalidates
  KV caching (settled, NOT open); frontend caching is exact-by-construction (EXPERIMENTAL); feature
  ring is approximate (boundary frames differ) and unproven.
- **PROPOSED TEST:** Post-V1 experiment: (a) frontend cache — verify bit-exactness vs full forward,
  measure ms saved; (b) feature ring at tail ∈ {0.3, 0.5, 1.0} s — measure feature L2 drift of kept
  frames vs full recompute, then listening test on `slow_vowels` (shimmer is tonal).
- **DECISION REQUIRED BY:** deferred backlog (post-P9); only revisit if Q1 fails the aggressive target.

## Q5 — ONNX decoder gains

- **QUESTION:** Does Applio-style decoder-only ONNX (IO binding + ORT CUDA graphs) beat
  torch-under-CUDA-graph on our fixed shapes?
- **WHY IT MATTERS:** The decoder renders `return_length` only; if it dominates the post-graph
  profile, 10–30% there is the next lever (doc 06 §7).
- **CURRENT EVIDENCE:** Doc 06 §7/§11: "roughly a wash versus PyTorch-under-CUDA-graph"; Applio
  ships it env-gated experimental. No measurement on our shapes.
- **PROPOSED TEST:** After P6 profiling: if decoder GPU-event time > ~30% of hop compute, implement
  `synthesis/onnx_backend.py` from the Applio template and A/B p99 + artifact check (log-mel diff,
  listen) at the shipped plan.
- **DECISION REQUIRED BY:** gate evaluated at P6 exit; implementation deferred unless triggered.

## Q6 — FP16 decoder artifacts

- **QUESTION:** Is FP16 on the HiFi-GAN conv stack (SineGen excitation kept FP32) artifact-free on
  this model/voice, and how much does it save?
- **WHY IT MATTERS:** Decoder is per-hop tail cost; fp16 could halve it on tensor-core GPUs,
  helping Q1.
- **CURRENT EVIDENCE:** Doc 06 §5: Applio `decoder_fp16` shipped experimental/default-off; NSF
  cumsum FP32 rule is settled; conv-stack artifact risk unquantified.
- **PROPOSED TEST:** P6: render corpus with decoder fp16 on/off; log-mel diff, sibilant-centroid
  delta, blind A/B on `sibilants` + `slow_vowels`; CUDA-event decoder time both ways.
- **DECISION REQUIRED BY:** P6 (ship flag default).

## Q7 — Phase-vocoder crossfade vs plain SOLA: audible?

- **QUESTION:** At our short overlaps (20–32 ms), does Applio's phase-vocoder crossfade audibly
  reduce stitch artifacts vs plain sin² SOLA, and can overlap shrink to 16–20 ms with it?
- **WHY IT MATTERS:** Overlap is pure algorithmic latency (doc 03 §4.4 note); if PV lets us run
  20 ms instead of 32 ms it buys ~12 ms end-to-end.
- **CURRENT EVIDENCE:** Doc 01 §7 tags it LIKELY — "smoother joins at short crossfade", worth A/B;
  only Applio implements it (at 100 ms overlap, a different regime). Doc 04 §3.5 flags the ~16 ms
  overlap quality floor as a guess.
- **PROPOSED TEST:** P2 (offline): stitch `slow_vowels` + `fast` at overlap ∈ {16, 20, 32, 40} ms ×
  {plain, PV}; chunk-consistency metric + SOLA-offset-histogram stability + blind A/B. Pick default
  and minimum overlap.
- **DECISION REQUIRED BY:** P2 (default), re-confirm live at P3.

## Q8 — WASAPI exclusive mode stability with VB-Cable

- **QUESTION:** Is exclusive-mode WASAPI (mic and/or cable side) stable enough on the user's
  devices to ship as more than an expert flag — and does VAC beat VB-Cable enough to recommend it?
- **WHY IT MATTERS:** Exclusive saves ~5–20 ms device latency (doc 03 §8); cable choice affects
  another ~5–15 ms (Vonovox prefers VAC). These are the cheapest milliseconds left after Phase 6.
- **CURRENT EVIDENCE:** Doc 03 §8: references expose exclusive as an option; rate-match
  requirements and device-hogging risks noted; Vonovox field guidance favors VAC pinned 48 k.
  No systematic stability data.
- **PROPOSED TEST:** P3: 30-min soak matrix {shared, exclusive-mic, exclusive-cable} ×
  {VB-Cable, VAC}: underruns, measured `stream.latency`, loopback round-trip measurement, device
  recovery after sleep/unplug.
- **DECISION REQUIRED BY:** P3 (default), documented recommendation by P9.

## Q9 — protect × index interactions on sibilance for this voice

- **QUESTION:** With the chosen model, what (index_rate, protect) region keeps /s/ crisp and
  gender-consistent — and is the index worth enabling at all?
- **WHY IT MATTERS:** Sibilant character is a top gender cue (doc 08 §2.1); protect only matters
  with index_rate > 0 and trades consonant clarity vs timbre consistency (doc 08 §2.5); index also
  costs VRAM + graph complexity (doc 07).
- **CURRENT EVIDENCE:** Doc 08 §2.5: community sweet spot protect ≈ 0.33, w-okada ships both off;
  doc 07 §B.3: benefit depends on model training quality — unknowable without the actual model.
- **PROPOSED TEST:** P7 quality sweep on `sibilants` + `plosives`: index_rate {0, 0.25, 0.5, 0.75}
  × protect {0.2, 0.33, 0.5}; sibilant-centroid trend + WER gate + blind A/B of top-3 (the doc 08
  §6 loop). Default remains index off until this proves otherwise.
- **DECISION REQUIRED BY:** P7 (writes `model_overrides.json`).

## Q10 — Auto-tuning objective validity

- **QUESTION:** Do the automated proxies (F0 stats, WER gate, speaker-cosine ranking) actually
  rank candidates in the same order as human A/B near the optimum — i.e., can the tuning loop be
  trusted to deliver a top-3 containing the true winner?
- **WHY IT MATTERS:** The single-user design converts sliders to tuned constants (doc 08 §1); if
  the proxies mis-rank, the loop wastes GPU-hours and ships worse defaults.
- **CURRENT EVIDENCE:** Doc 08 §6.2 / doc 10 §6: F0 stats + WER PROVEN as *gates*; speaker-cos
  LIKELY as a ranker but "top candidates differ by less than embedding noise"; UTMOS/PESQ invalid
  or unverified for VC gender-credibility.
- **PROPOSED TEST:** P7: run one full sweep, have the user blind-rank 8 spread candidates; compute
  rank correlation (Spearman) proxy-vs-human; if the human winner is in the proxy top-3, the loop
  stands; otherwise demote speaker-cos to a gate and widen the A/B set.
- **DECISION REQUIRED BY:** P7.

## Q11 — Hop/callback decoupling in Python: does p99 hold?

- **QUESTION:** The decoupled design (≤10 ms callbacks + worker thread) is ours alone — none of the
  four references does it (doc 03 §2.5). Do Python/GIL tail effects stay under control at p99 with
  a GUI attached, or is the Applio process split needed?
- **WHY IT MATTERS:** The whole latency advantage over the references rests on this decoupling;
  p99 jitter converts directly to underruns.
- **CURRENT EVIDENCE:** Doc 03 §9: torch CUDA ops release the GIL (argument), Applio proves the
  process escape hatch; no measurement of thread-mode p99 under UI load.
- **PROPOSED TEST:** P3: 30-min soak with synthetic GIL load (a busy Python thread) and, at P8,
  the real UI; compare p99 wake-to-done and underrun counts thread-worker vs a prototype
  process-worker.
- **DECISION REQUIRED BY:** P3 provisional; final at P8.

## Q12 — Frozen noise latent under CUDA graphs: audible?

- **QUESTION:** Graph capture freezes one `randn` draw for the z_p latent (doc 02 §3.7); over long
  sessions, does the repeated noise pattern produce audible periodicity or timbre flatness?
- **WHY IT MATTERS:** If audible, graphs need a noise-refresh strategy (write noise into a static
  buffer each hop pre-replay), slightly complicating capture.
- **CURRENT EVIDENCE:** Doc 02 takeaway 5: both reference graph implementations ignore it,
  "acceptable in practice" — asserted, not tested.
- **PROPOSED TEST:** P6: A/B long renders graph vs eager (same settings) on `slow_vowels`; if
  flagged, implement external-noise-input capture (the w-okada ONNX wrappers already model `rnd`
  as an input) and re-measure cost.
- **DECISION REQUIRED BY:** P6.

## Q13 — Target-model median F0 estimation method

- **QUESTION:** How do we reliably obtain `F0_model_median` for the transposition formula when the
  model ships no metadata?
- **WHY IT MATTERS:** The formula `round(12*log2(target/source))` is only as good as the target
  median; the fallback (200 Hz generic) may miss by 2–3 st for low- or high-F0 models.
- **CURRENT EVIDENCE:** Doc 08 §2.3 notes the circularity (measuring from converted previews
  depends on the shift used) and offers heuristics, no validated procedure.
- **PROPOSED TEST:** P5: for 2–3 models, render a reference clip at shifts 0..+12, track output F0
  stability/artifact metrics vs shift, and compare the stability-optimum against by-ear winner;
  standardize whichever estimator lands within ±1 st.
- **DECISION REQUIRED BY:** P5.

---

## Summary table

| # | Question | Phase due | Risk if unresolved |
|---|---|---|---|
| Q1 | Minimum stable hop on target GPU | P2 / P6 | latency target slips |
| Q2 | FCPE vs RMVPE on this voice | P5 | register-flip artifacts |
| Q3 | EXTRA 0.5 s vs 2.5 s | P2 | wasted compute or quality loss |
| Q4 | Feature caching feasibility | post-V1 | sub-75 ms path blocked |
| Q5 | ONNX decoder gains | P6 gate | minor perf left on table |
| Q6 | FP16 decoder artifacts | P6 | artifacts or wasted headroom |
| Q7 | Phase vocoder vs plain SOLA | P2 | ~12 ms overlap latency |
| Q8 | Exclusive WASAPI + cable choice | P3 | 10–30 ms device latency |
| Q9 | protect/index sibilance | P7 | gender-cue degradation |
| Q10 | Tuning objective validity | P7 | bad shipped defaults |
| Q11 | Thread-worker p99 under load | P3 / P8 | underruns; process rewrite |
| Q12 | Frozen graph noise audibility | P6 | subtle periodic artifact |
| Q13 | Target median F0 estimation | P5 | wrong transposition |
| Q14 | Is a streamed render audibly equivalent to full-file? | P3 | quality target undefined |
| Q15 | Which unvoiced-F0 contract does the target model need? | P5 | wrong voicing on consonants |
| Q16 | Compute floor on this machine; whole-hop graph replay hang | **before P3** | no real-time at any small hop |
| Q17 | Decoder left context: any audible benefit? | P6 | wasted compute or lost quality |

---

# Phase 2 results (2026-10-13)

Recorded by implementing the PROPOSED TESTs above. Machine: RTX 5060 Ti 16 GB, driver 617.14,
Ryzen 9 7900X, Windows 11, torch 2.7.1+cu128, eager, TF32 on, FP32. Test voice
`base_f0_48k_v2` (public RVC v2 base generator), clip `libri_1272.wav`. Raw records:
`benchmarks/baselines/phase2_2026-10-13/*.jsonl`. Reproduce with `vox bench <name>`.

Not done, because they need tooling or ears this phase does not have: speaker-cosine, WER,
and every blind A/B. The user's own corpus does not exist until Phase 5, so the
`slow_vowels` / `fast` clips named in the tests are replaced by one LibriSpeech clip.

## Q1 - provisional result: no small hop is stable in eager mode

`vox bench perf --seconds 12` (throughput mode, eager, idle GPU, first 50 hops discarded).
Host wall time per hop in ms; "headroom" is hop / p99.

| hop | extra | f0 | f0 p50 | embed p50 | synth p50 | total p50 | p95 | p99 | headroom |
|---|---|---|---|---|---|---|---|---|---|
| 10 ms | 500 | rmvpe | 13.9 | 6.1 | 16.5 | 38.1 | 79.9 | 84.1 | 0.12x |
| 20 ms | 500 | rmvpe | 13.7 | 6.1 | 16.4 | 37.9 | 62.1 | 81.0 | 0.25x |
| 30 ms | 500 | rmvpe | 13.6 | 6.0 | 16.3 | 37.6 | 53.7 | 80.9 | 0.37x |
| 50 ms | 500 | rmvpe | 13.8 | 6.0 | 16.3 | 37.8 | 77.4 | 83.1 | 0.60x |
| 100 ms | 500 | rmvpe | 13.8 | 6.0 | 16.4 | 38.0 | 44.6 | 70.3 | 1.42x |
| 100 ms | 300 | rmvpe | 13.9 | 5.9 | 16.2 | 37.9 | 45.6 | 50.0 | 2.00x |
| 100 ms | 100 | rmvpe | 14.2 | 6.1 | 16.5 | 38.6 | 49.4 | 57.7 | 1.73x |
| 20 ms | 500 | fcpe | 3.6 | 6.6 | 18.5 | 30.6 | 44.8 | 61.8 | 0.32x |
| 50 ms | 500 | fcpe | 3.1 | 6.0 | 16.2 | 26.6 | 34.5 | 66.2 | 0.76x |
| 100 ms | 500 | fcpe | 3.1 | 6.0 | 16.2 | 26.8 | 31.2 | 40.8 | 2.45x |

(30 points in the full grid; the rows omitted repeat the same pattern.)

- **Cost does not depend on hop size or EXTRA.** ~38 ms p50 with RMVPE at every point, from
  a 150 ms window to a 640 ms one. The hop is bound by the number of operations launched,
  not by compute (DEVIATIONS R1).
- **The tail is roughly twice the median.** p99 sits near 80 ms for RMVPE and 60 ms for
  FCPE. The distribution is bimodal in eager mode; graph replay does not show it (stage
  probes: p95 within 15% of p50).
- **FCPE saves ~11 ms per hop here**, not the 1-2 ms doc 05 assumed (3.1 vs 13.8 ms). That
  raises the stakes of Q2.
- **Only the 100 ms hop reaches 1.5x headroom**, and not at every setting (1.42x with RMVPE
  at 500 ms extra). Nothing at 50 ms or below is close.

**Provisional decision:** in eager mode the conservative plan on this machine is a hop of
100 ms: about 140 ms of algorithmic latency (hop + 30 ms overlap + 10 ms search) before
device and ring terms, already near the 175 ms end-to-end budget. The paced-mode soak and
the 16 ms point of the test were not run: with compute at twice the hop there is nothing
to soak. The 20 ms target depends entirely on Q16. Final at P6 as planned.

## Q3 - provisional result: 500 ms is enough for speech frames, not for pauses

`vox bench extra` (hop 20 ms, 20 s clip, evaluated after start-up; log-mel L1 vs the
full-file render, noise floor 0.296 in speech frames):

| extra | speech L1 | x floor | quiet L1 | level bias (nepers) |
|---|---|---|---|---|
| 100 ms | 0.804 | 2.72 | 1.867 | +1.06 |
| 300 ms | 0.632 | 2.14 | 1.300 | +0.65 |
| 500 ms | 0.558 | 1.89 | 0.978 | +0.40 |
| 1000 ms | 0.543 | 1.83 | 0.731 | +0.16 |
| 2500 ms | 0.547 | 1.85 | 0.572 | -0.09 |

Speech-frame distance stops improving at 500 ms. Below it, quality falls quickly. Loudness
in pauses keeps improving out to 2500 ms: with short context a mostly-silent window is
normalised up into speech-like features (DEVIATIONS R4). Eager cost is flat across the
range because the hop is launch-bound (Q1), so the usual "extra is the compute knob"
argument does not apply until graphs are in.

**Provisional default: 500 ms**, with 1000 ms the first thing to try if pauses sound noisy
before the Phase 4 gate exists. Revisit at P6 as planned.

## Q7 - provisional result: plain sin^2, overlap 20-30 ms; phase vocoder shows no gain

`vox bench sola` (hop 20 ms). Overlaps snap to 10 ms units, so the test's 16/20/32/40 ms
became 10/20/30/40 ms.

| overlap | plain: speech L1 | pv: speech L1 | plain: quiet L1 | pv: quiet L1 |
|---|---|---|---|---|
| 10 ms | 0.604 | 0.605 | 1.041 | 1.052 |
| 20 ms | 0.562 | 0.563 | 0.954 | 0.980 |
| 30 ms | 0.557 | 0.557 | 0.947 | 0.992 |
| 40 ms | 0.543 | 0.536 | 0.853 | 0.909 |

SOLA offsets are spread across the whole 10 ms search window in every case (p50 5-6 ms, p95
10 ms): there is no stable offset to lock to. Plain and phase-vocoder crossfades are within
0.007 of each other in speech frames; pv is slightly worse in quiet frames. 10 ms overlap
is clearly worse; 20, 30 and 40 ms are within 3%.

**Provisional default: plain, 30 ms.** 20 ms would save 10 ms of latency for about 1% on
this metric and is the candidate for the blind A/B. The metric cannot hear comb filtering,
so the A/B (re-confirmed live at P3) still decides.

## Q14 (new) - Is a streamed render audibly equivalent to the full-file render?

- **QUESTION:** The streamed output differs from the full-file render by 1.9x the seed noise
  floor in speech-frame log-mel distance at the default plan. Is that audible, and if so is
  it objectionable?
- **WHY IT MATTERS:** Phase 2's acceptance criterion is "hop-streamed ~= full-file". The
  automated half (no boundary clicks, sync-free, stable allocation) passes; the equivalence
  half cannot be settled by the metric.
- **CURRENT EVIDENCE:** DEVIATIONS R3. The gap is deterministic (0.51 with latent noise off,
  where a stream differs from itself by only 0.06) and invariant to hop size, EXTRA beyond
  500 ms, F0 source, F0 tail length, decoder context and lookahead. The generator is highly
  sensitive to window position (a 10 ms input delay alone moves a full-file render by 0.21),
  and at a 20 ms hop every output sample is a crossfade of two differently-positioned
  renders. At rvc-webui's defaults (250 ms block, 2.5 s extra) the gap is 0.32.
- **PROPOSED TEST:** Listen to `out/accept_full.wav` vs `out/accept_stream.wav`, then repeat
  with the real target model. If the stream is audibly rougher: A/B hop 20 vs 50 vs 100 ms
  at fixed EXTRA to find where it stops being audible, since larger hops put fewer samples
  inside crossfades.
- **DECISION REQUIRED BY:** P3 (it sets the hop / quality trade-off live audio is tuned to).

## Q15 (new) - Which unvoiced-F0 contract does the target model need?

- **QUESTION:** Should unvoiced frames carry `f0 = 0` (coarse bucket 1) or a pitch
  interpolated across the gap?
- **WHY IT MATTERS:** It changes the NSF excitation and the protect mask on every consonant
  and breath. The wrong choice feeds the model conditions it never trained on.
- **CURRENT EVIDENCE:** Current rvc-webui interpolates in both training
  (`train/dataset/extract_f0.py:103`) and inference (`infer/vc/pipeline.py:127`,
  `infer/rtrvc.py:203`). Applio and older RVC releases do not. Docs 02 and 05 each describe
  one of the two. Both are implemented (`f0_unvoiced`), default `zero`.
- **PROPOSED TEST:** Establish which tool and version trained the target model; render the
  `sibilants` and `breaths` clips both ways and A/B.
- **DECISION REQUIRED BY:** P5 (stored per model in `config.json` defaults).

## Q16 (new) - Compute floor on this machine, and the whole-hop graph replay hang

- **QUESTION:** How low can per-hop compute go here, and why does replaying the whole hop
  as one CUDA graph never return?
- **WHY IT MATTERS:** Eager compute is ~40 ms per hop regardless of hop size (Q1), so live
  audio needs graphs before it can meet any small-hop target. The roadmap has graphs in
  Phase 6, after live audio.
- **CURRENT EVIDENCE:** `vox bench graph-probe --stage S`: per-stage replay p50 is RMVPE
  7.3 ms, ContentVec 2.6 ms, synth 4.5 ms, SOLA 0.13 ms: about 15 ms per hop (17 ms summing
  the p99s), 2-4x better than eager rather than the order of magnitude doc 06 anticipated. Whole-hop capture
  succeeds; replay hangs with one core spinning (reproduced twice). The GPU sits in power
  state P3 at ~1860 MHz during these runs.
- **PROPOSED TEST:** (a) bisect the hang by capturing stage pairs; (b) re-measure eager and
  replay with the NVIDIA power-management mode set to maximum performance; (c) fold RMVPE's
  batch norms into its convolutions and re-measure (it is the largest stage under replay,
  and batch norms are by estimate about a third of its kernels); (d) FP16 ContentVec.
- **DECISION REQUIRED BY:** before P3. Either Phase 6's capture work moves ahead of live
  audio, or P3 is accepted at a hop of roughly 100 ms.

## Q17 (new) - Decoder left context

- **QUESTION:** Does decoding extra frames ahead of the tail (and discarding them) improve
  what is heard at small hops?
- **WHY IT MATTERS:** At a 20 ms hop the decoder renders 60 ms, well inside its own
  receptive field, so the whole chunk is rendered against zero padding on the left.
- **CURRENT EVIDENCE:** 0 vs 80 ms of context moved the log-mel distance to the full-file
  render by less than 0.01. Implemented as `decoder_context_ms`, default 0 (the upstream
  behaviour).
- **PROPOSED TEST:** Blind A/B at 0 / 80 ms on sustained vowels once per-hop cost is known
  under graphs.
- **DECISION REQUIRED BY:** P6.
