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
