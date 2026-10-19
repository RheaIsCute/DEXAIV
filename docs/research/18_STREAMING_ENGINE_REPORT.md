# 18. Streaming engine report: why live sounds worse than whole-file, and what to build

**Answers brief 17** (`17_STREAMING_ENGINE_RESEARCH_BRIEF.md`). Written 2026-10-19 against
`feature/formant` @ 5672cea, with the web. Builds on report 15 (`15_MODERNIZATION_REPORT.md`);
§D10 says where it disagrees. Every number is either from this repository's code or
measurements, or from a linked source. Anything not verified in code or a paper is marked
**[unverified]**; "not found" means it was looked for and not found.

**The short version.** The stutter is structural, not a bug. At a 20 ms hop every emitted
sample is a crossfade of two renders that *must* disagree, because (1) every frame's
ContentVec features and text-encoder output depend on the whole sliding window (GroupNorm
over time, global attention), (2) the flow is non-causal by ±240 ms and the decoder by
~±100 ms, so the kept 50 ms tail is always rendered against zero padding on its right, and
(3) SOLA then re-times the output by up to 10 ms, fifty times a second. The coherent-hop
work already removed every *avoidable* source (noise, phase, left padding). What is left
is exactly the set that costs latency or retraining to remove. The 50 Hz seam train sits
in the ear's roughness band; log-mel L1 cannot see it. No open engine runs below ~30-60 ms
chunks, and none has solved this: they run 130-500 ms hops. The ranked plan (§F): first the
cheap unheard experiments (right context 20-40 ms, hop 40, 30 ms blend, fill-target fix,
SOLA offset damping), then a **commit-once + exactly streamed decoder** that removes the
crossfade at +~100 ms, and in parallel the only path to whole-file quality under 150 ms:
a causal decoder (or a distilled causal student) trained once on this user's data.

---

## 0. What the code says before any literature

### 0.1 The latency ledger, live (not the simulator's)

`api/control.py:150` sets the output-ring fill target to **2 × hop + callback (10 ms)**;
`engine/worker.py` keeps the ring near it. The live end-to-end cost of a hop H is roughly

    H (accumulate) + 16-19 (compute) + (2H + 10) (ring fill) + 20 (overlap) + 10 (search)
    + 11 exclusive / 33 shared (device)  [+ 20 with RNNoise]

| hop | algorithmic (`ChunkPlan.algorithmic_latency_ms`) | ring fill at the default rule | live total, exclusive, approx. |
|---|---|---|---|
| 20 | 50 ms | 50 ms | ~127 ms (§1 of the architecture doc: 100-130 measured) |
| 40 | 70 | 90 | ~187 |
| 60 | 90 | 130 | ~247 |
| 200 | 230 | 410 | ~667 |

The brief's "+180 ms for hop 200" is the simulator's algorithmic view. Live, the 2×hop fill
rule roughly triples the cost of a bigger hop. The rule was sized for hop 20, where compute
(16 ms) is nearly a whole hop; at hop ≥ 40 compute is a fraction of the hop and a fill of
**hop + ~20 ms** would carry the same margin. Not a quality technique, but it changes the
price of every "bigger hop" option below: hop 40 at a hop+20 fill is ~147 ms live, not ~187.
**[unverified live]**: `vox bench underrun` with `fill_target_ms` set is the test.

### 0.2 Receptive fields: how much of each render is edge-impaired

**Decoder** (`synthesis/nets/generator.py`): conv_pre k7 at 100 Hz; per upsample stage three
parallel ResBlock1 branches (k 3/7/11; each branch three sequential pairs Conv(k, d∈{1,3,5})
→ Conv(k, 1)); 'same' zero padding, non-causal. One-sided reach per branch = 6(k−1) → 12 /
36 / 60 samples at that stage's rate; branches are parallel, so the stage reach is the max,
60. Propagated to output samples (the transposed-conv reach is ≤ 1 input right / ≤ 2 left):

| model rate | upsample rates / kernels | one-sided decoder RF (exact interval propagation, all in-frame offsets) |
|---|---|---|
| 48 kHz v2 | [12,10,2,2] / [24,20,4,4] | 4,526-5,005 samples = **94-104 ms**; frame N depends on frames [N−10, N+10] |
| 40 kHz | [10,10,2,2] / [16,16,4,4] | 4,158-4,557 = **104-114 ms**; [N−11, N+11] |
| 32 kHz v2 (Mom) | [10,8,2,2] / [20,16,4,4] | ≈ 3,800 = **~119 ms** (upper bound by the same arithmetic) |

Dominant terms: conv_pre (3 frames = 30 ms) + first-stage MRF (60 samples at 1.0-1.2 kHz =
50-60 ms) + the first transposed conv (10-20 ms). `ARCHITECTURE_AND_METHODS.md` §3 quotes
"~±90 ms"; the arithmetic is a little larger. The 80 ms `decoder_context_ms` is therefore
also short of the *left* RF by 15-40 ms.

**Flow** (`synthesizer.py`, `flow.py`): 4 coupling layers × WN(kernel 5, 3 layers, dilation
1) → ±6 frames each, **±24 frames = ±240 ms** one-sided. `FLOW_CONTEXT_FRAMES = 24` supplies
exactly that on the left; nothing can be supplied on the right.

**Text encoder** (`attentions.py`): six layers of self-attention over the **whole window**;
`window_size = 10` only clips the relative-position *bias*; the softmax is global; the FFN
is k3 non-causal; its LayerNorm is over channels per frame (local). So m_p/logs_p for any
frame depend on the window's extent.

**ContentVec** (`features/contentvec.py` loads the HF `HubertModel`; the shipped RVC
ContentVec config has `feat_extract_norm: "group"` —
[Politrees/RVC_resources embedders/transformers/contentvec/config.json](https://huggingface.co/Politrees/RVC_resources/blob/refs%2Fpr%2F24/embedders/transformers/contentvec/config.json)).
HF: "group" = "group normalization of only the first 1D convolutional layer"
([HubertConfig](https://huggingface.co/docs/transformers/v5.14.0/model_doc/hubert));
fairseq builds it as `Fp32GroupNorm(dim, dim)` for `mode == "default" and i == 0`
([fairseq wav2vec2.py, mirror](https://huggingface.co/spaces/wsntxxn/STAR/blob/main/fairseq/models/wav2vec/wav2vec2.py)).
A GroupNorm with one group per channel normalises **each channel over the time axis of the
whole input**: the first layer's statistics are computed over the entire 1.1 s window, so
every frame's activations shift whenever the window's content changes, even for frames
whose waveform did not. That is the mechanism behind R4 (pauses inflated at short context)
and a per-hop drift of every frame's features as the window slides. Twelve layers of global
attention sit on top; HuBERT's measured effective context is ~2 s each side
([Meng, Goldwater, Tang 2025](https://arxiv.org/pdf/2505.22487)).

**Already removed in this engine** (and *not* in any open engine, §A1): sine phase restarts
(`sine_phase_carry`, `harmonic_num = 0` so there is no random harmonic phase), redrawn
latent and excitation noise (`noise_roll`), the zero-padded left decoder edge
(`decoder_context_ms`), the RMVPE right edge (`f0_trim_tail = 2`).

**What remains, per emitted sample** (hop 20: tail = hop 20 + overlap 20 + search 10 = 50 ms):
decoded with 54-119 ms of its right context missing; its flow latent with up to 240 ms
missing; its features computed over a window that will be different next hop; RMVPE over a
310 ms tail whose BiGRU backward state is the initial state at the edge. The saved SOLA tail
(10-30 ms from the render's end) and the next hop's render of the same instant (30-50 ms
from its end) are *both* right-edge-impaired, by different amounts. §12 F3 measured the
residue: +3.1 samples of hop-to-hop lag at formant 0, the stitched stream ~11 cents flat.

### 0.3 Why a bigger hop helps, mechanically

| hop | seams/s | share of emitted audio ≥ ~104 ms from its render's right edge (fully decoder-contexted) | SOLA re-timing events/s |
|---|---|---|---|
| 20 | 50 | **0 %** (the whole 50 ms tail is inside the impaired zone) | 50 |
| 60 | 16.7 | 0 % (90 ms tail) | 16.7 |
| 100 | 10 | ~20 % | 10 |
| 200 | 5 | ~60 % | 5 |
| whole file | 0 | 100 % | 0 |

Hop 200 is better for three stacked reasons: five seams a second instead of fifty, most of
each block decoded with full right context, and the output re-timed five times a second
instead of fifty. Log-mel L1 barely moves (0.59 → 0.51) because it is dominated by the
encoder-level window dependence (GroupNorm + global attention), which R3 showed is of the
order of the seed floor and inaudible "as such"; the audible part is the seam/jitter
pattern, which a 25 ms-frame average cannot see (§A3). R3's lookahead experiment (20-250
ms) was judged by that metric only; **no lookahead render has ever been listened to.**

### 0.4 What SOLA adds: a 50 Hz time-base jitter

`engine/sola.py` picks an offset in [0, 10 ms] per hop and emits the render *shifted by that
offset*. §8.4 measured offsets p50 5-6 ms, mean hop-to-hop jumps 1-3 ms, p95 ~7 ms, and on
held notes jumps of whole F0 periods (2.5-4 ms). Each jump is a local time-scale
modification of a few ms inside a 20 ms block: a 50 Hz-rate jitter of the output's time
base, i.e. periodic frequency modulation. Hop 60 cuts the rate to 16.7 Hz, hop 200 to 5 Hz.

---

## A. Joining hops (the stutter)

### A1. How the other real-time RVC engines join chunks

Read from pinned commits: w-okada/voice-changer `master` @ d8ef157 (the 1.5.3.x engine; the
VCClient 2.x engine is a binary on Hugging Face and **not on GitHub**), deiteris/voice-changer
`master-custom` @ b9cd071, RVC-WebUI `main` @ 81eed5e (`realtime_gui.py`, `infer/rtrvc.py`,
new `RVCRealtimeVST/`) and legacy `20240604` @ d60cd3d (`gui_v1.py`), Applio `main` @ 5379cf2
(`rvc/realtime/`), DDSP-SVC @ b9d4e6a (the SOLA code every RVC engine credits), Vonovox @
02316ad (compiled; strings only).

| engine @ commit | chunk default / range | crossfade (shape) | SOLA search | extra (left ctx) | tail-only decode | consistency tricks |
|---|---|---|---|---|---|---|
| w-okada 1.5.3.x | 192 × 128 samples = **512 ms** (server mode 683 ms); 21 ms-44 s (`client/lib/src/const.ts#L532`, `102-2_ConvertArea.tsx#L17`) | 1024 samples = 21 ms, `min(cf, block)`, sin²/cos² (`VoiceChangerV2.py#L180-L198`, `#L235`) | 12 ms (`#L233`) | 85 ms default, up to 2.73 s (`RVCSettings.py#L12`) | Torch: full decode + crop for stock resblock-1 models; ONNX: features front-cropped (`Pipeline.py#L242-L244`) | cached pitchf + `silenceFront`; RMS gate 1e-5 → zeros through SOLA; `sqrt(vol)` scaling; tail saved pre-weighted by the fade-out (`#L278-L284`) |
| deiteris fork | 192 × 128 = **512 ms**; 2.7 ms-2.73 s slider (`VoiceChangerSettings.py#L77`) | **0.1 s** default, 0.05-0.20 s, sin², not clamped to block (`VoiceChangerV2.py#L27`, `#L97-L112`) | 10 ms (`#L29`) | 0.5 s default, 0.1-5 s | yes: flow gets 24 frames (240 ms), decoder **0** context (`models.py#L728-L749`) | F0 ring cache (last block + 50 ms); −90 dB gate with GPU busy-wait; hop-aligned buffers |
| RVC `realtime_gui.py` | **0.25 s**; 0.02-1.5 s (`#L53-L59`, `#L333-L339`) | 0.05 s (0.01-0.15), **overlap capped at 40 ms** (`#L607`), sin² | 10 ms (`#L608`) | 2.5 s | yes: flow 24 frames, decoder 0 context, CUDA graphs per shape (`rtrvc.py#L307-L343`) | F0 cache (last block + 50 ms, frames `[3:-1]`); input resampled with 20 ms history; per-10 ms dB gate; RMS mix; `use_pv` removed from main |
| RVC `gui_v1.py` (2024) | 0.25 s (shipped json **0.15 s**) | 0.05 s (json 0.08), cap 40 ms; sin² or phase vocoder (`use_pv`, off) | 10 ms | 2.5 s (json 2.0) | yes | same + `use_pv` |
| RVC Realtime VST (new) | **130 ms**; 20-1000 (`RVCRealtime.cpp#L150-L156`) | 80 ms → effective 40 ms | 10 ms | 2000 ms | yes (same worker) | README: "reported plug-in latency: twice the Block duration" |
| Applio realtime | **250 ms**; 2.7-2730 ms, 10 ms grid; auto-block clamps to 100-1000 ms (`realtime.py#L1381-L1391`, `utils/chunks.py#L4-L7`) | 0.05 s; **phase-vocoder crossfade always on** (`core.py#L515`, `#L24-L75`), sin² fallback with onset fade | 10 ms | 2.5 s | yes, cropped **before** the flow: 0 context for flow and decoder (`synthesizers.py#L239-L265`) | F0 cache (FCPE forced ≥ 300 ms analysis); VAD + −60 dB gate zeroing all buffers, tail drained with the fade |
| DDSP-SVC `gui_reflow.py` | 0.5 s; 0.05-3 s | 0.04 s, sin² or PV (off) | 10 ms | 2 s | no (full window, crop) | **20 ms right-context discard** (`last_delay_frame`, `#L363`, `#L461-L462`): the only open engine with lookahead |
| Vonovox (closed) | not found | SOLA + PV (symbols `process_sola`, `phase_vocoder`, `sola_search_multiplier`) | configurable multiplier | "Lookahead buffer … recommended 2.0" (README) | `skip_head`/`return_length` symbols | VAD with 400 ms release; `ringformer`/`sifigan`/`bigvgan` modules with `fund_phase_cum`/`phase_inc` symbols (a phase-accumulating source) **[unverified]** |

Across all of them: **none renders a region more than once and averages, none crossfades in a
non-waveform domain** (the phase vocoder is still a time-domain mix plus a magnitude/phase-
aligned sinusoidal top-up over the overlap), **none carries decoder state, none carries the
sine phase, none rolls noise** (`rand_ini` and `randn_like` per chunk everywhere), **none caches
HuBERT** (the whole window is re-embedded every chunk; only F0 is cached). RVC/deiteris give
the flow 240 ms of left context and the decoder zero; Applio gives both zero. So the stutter
mechanism in §0 is present in every one of them; they avoid hearing it by running big hops.

**What users actually run.** The AI Hub guide for the deiteris fork: "start with 500 ms …
go closer to [your perf number] but not lower"; crossfade "0.1 or 0.15"; extra "2.7 s is the
max"; known-good chunks by GPU class: xx90/xx80 Ti **30-60 ms** + 2.7 s extra, xx70 50-80 ms,
**xx60 Ti 50-90 ms**, GTX 10xx 200 ms
([docs.aihub.gg](https://docs.aihub.gg/realtime-voice-changer/local/deiteris-w-okada-fork/)).
w-okada issues show users at 293-512 ms chunks with 131072 extra and still "choppy + robotic"
([#1374](https://github.com/w-okada/voice-changer/discussions/1374),
[#1418](https://github.com/w-okada/voice-changer/issues/1418),
[#1525](https://github.com/w-okada/voice-changer/issues/1525)). Vonovox's README: "the
optimal block size is the lowest you can get without audio being choppy … at extremely low
block sizes, quality may be reduced." **Nobody ships or recommends a 20 ms hop.** The
practical floor elsewhere is ~30-60 ms with 100-150 ms crossfades, i.e. 2-5× the crossfade
DEXAIV uses, on hardware faster than the target machine.

### A2. Joins beyond SOLA + sin²/cos²

| technique | what it does to the two mechanisms (render disagreement; time-base jitter) | latency | GPU/hop | evidence |
|---|---|---|---|---|
| Longer overlap at the same hop (30-40 ms blend; three-render mix) | spreads each disagreement over more samples; does not reduce it; jitter unchanged | +10-20 ms | ~0 | §Q7 reopened: click ratio 1.39 → 1.00 at 30 ms; **unheard** (`Mom_live_hop20_blend30.wav`) |
| Phase-vocoder crossfade (Applio default, RVC `use_pv`, our `sola_mode="pv"`) | phase-aligns partials inside the overlap so they do not cancel; nothing for jitter | 0 | ~0.1 ms | Q7 provisional: no log-mel gain at 20-40 ms; Applio ships it on by default with 50 ms fades. No listening test anywhere. Worth one ear pass at 30-40 ms overlap, not at 20 |
| WSOLA-style join (search tolerance ≥ half a frame, template from the *input*) | better alignment on periodic signals; WSOLA "can only preserve the most prominent periodic pattern", leaves "warbling" FM and transient doubling ("stuttering") on harmonic signals ([Driedger & Müller 2016](https://www.mdpi.com/2076-3417/6/2/57)) | search ≥ one period of the *converted* voice (at 200 Hz, 5 ms; at 100 Hz, 10 ms — our 10 ms is at the limit) | ~0 | TSM literature; no VC data |
| **SOLA offset damping** (penalise offset change hop to hop; or hold the offset while NCC of the held offset is within ε of the max) | leaves disagreement alone; **removes most of the 50 Hz time-base jitter** | 0 | ~0 | Our own §8.4: 19 % of hops already keep the offset with the coherent set; offsets jump whole periods on held notes. Pulse-train jitter is detectable below 0.1 % of the interval ([Pollack 1968](https://labs.sonicfield.org/library/detection-and-relative-discrimination-of-auditory-jitter-1968)); ours is 5-15 %. Cheap, testable in a day |
| Crossfade in the latent / mel / STFT domain | the decoder is deterministic given z, so blending z across hops would make *one* render per sample; but z of the overlap frames differs per hop for the same reasons the audio does, and the decoder is not linear — a z-blend is a different approximation, not a smaller one. STFT-domain magnitude blending with phase from one side is what the PV crossfade already approximates | 0 | small | not found in any VC engine or paper; not recommended over commit-once (§B6) |
| Averaging k successive renders of the same region | each region rendered k times from windows shifted by one hop and averaged (in the waveform, after alignment) — a low-pass over the disagreement | +k×hop | ×k decode | none found anywhere (§A1). Equivalent in latency to right context but strictly worse in information (the k-th render is the one with the most right context; averaging dilutes it). Not recommended |
| **Right context / lookahead** (`extra_right_ms`: decode the tail L ms before the window's end) | directly removes the right-edge impairment up to L; at L = 100 ms both renders are fully contexted and the crossfade joins two near-identical renders | +L | +L of decode, ~+0.4 ms per 20 ms at today's 4.8 ms synth | DDSP-SVC ships 20 ms; Seed-VC's GUI `extra_time_right 0.02`; report 15 Phase 1.6 proposes 20-40 ms; Conan: removing right context raised WER 6.02 → 8.57 % ([Conan](https://arxiv.org/html/2507.14534v4)); Zero-VC needs 40-60 ms lookahead without input anonymisation ([Zero-VC](https://arxiv.org/pdf/2606.20218)). R3 measured L = 20-250 ms as 0.535-0.593 log-mel L1 (the metric that cannot see seams); **never heard** |
| Jittered seam positions | spreads the 50 Hz line into noise-like modulation; same energy | 0 | 0 | granular-synthesis experience (asynchronous grains lose the pitched buzz; [Roads via granularsynthesis.com](https://www.granularsynthesis.com/hthesis/sync.html), [Truax](https://www.sfu.ca/~truax/gran.html)); no VC listening data. Palliative; our hop timing is fixed by the callback, the only free jitter is the crossfade position inside the search region |

### A3. Why a bigger hop helps by ear when the metric says it barely does

**The percept moves along the modulation axis.** The seam train is a periodic
amplitude/phase modulation at the hop rate. Psychoacoustics (Zwicker & Fastl, codified in
ECMA-418-2 4th ed. 2025): **roughness** is the percept for envelope modulation at ~20-300 Hz,
maximal at **70 Hz**; **fluctuation strength** for 0.25-20 Hz, maximal at **4 Hz**
([ECMA-418-2](https://ecma-international.org/wp-content/uploads/ECMA-418-2_4th_edition_june_2025.pdf);
[Siemens summary](https://community.sw.siemens.com/articles/en_US/Knowledge/sound-modulation-metrics-fluctuation-strength-and-roughness);
[Salford](https://acoustics.salford.ac.uk/?p=716)). Click/pulse trains: below ~20 Hz listeners
hear discrete events ("a fast tapping rhythm"); above ~50 Hz they fuse into a steady,
pitched buzz; 20-33 Hz is a transitional gap
([Ungan & Yagcioglu 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4264405)).

| hop | seam rate | regime | expected percept | user's word |
|---|---|---|---|---|
| 20 ms | 50 Hz | roughness band, on the rising flank toward the 70 Hz peak; fused | buzzy/rough, tone-like | "really stuttery" |
| 60 ms | 16.7 Hz | transition: heard as discrete events | individual chops | "chopped" |
| 200 ms | 5 Hz | fluctuation-strength peak, but 10× fewer seams and most audio fully contexted | slow, low-depth | "alright" |

So the hop sweep did not remove the artifact, it moved it from "buzz" to "chop" to "rare",
and at hop 200 also removed most of its *depth* (§0.3). The fix is depth (consistency of
the two renders, §B) and the jitter (§A2 offset damping), not rate. Note the corollary: a
5 Hz seam train with *large* per-seam depth would be the worst possible slow modulation.

**The SOLA jitter is a second, independent percept.** Driedger & Müller describe plain
OLA's "phase jump artifacts" on harmonic signals as "a characteristic warbling sound, which
is a kind of periodic frequency modulation" ([2016](https://www.mdpi.com/2076-3417/6/2/57));
Pollack's jitter thresholds for pulse trains are below 0.1 % of the interval
([1968](https://labs.sonicfield.org/library/detection-and-relative-discrimination-of-auditory-jitter-1968)).
Ours is 1-3 ms per 20 ms. On speech the threshold is certainly higher **[no speech number
found]**, but two orders of magnitude is not a marginal effect.

**Why log-mel L1 is blind.** A 50 Hz low-depth seam pattern puts energy in ±50 Hz sidebands
around each harmonic and a 50 Hz envelope component. Mel filters above ~500 Hz are wider
than 50 Hz, so the sidebands sum into the carrier's bin; a 25 ms frame spans 1.25 seam
periods, so the envelope dip averages out inside the frame; log-magnitude discards the
phase jumps SOLA introduces. The metric measures the average spectral envelope per 25 ms,
which is what a bigger hop does *not* change. Independent evidence that spectral distances
miss discontinuity-type artifacts: on a TSM dataset with listener labels, the best spectral
measure correlates |r| = 0.37 with MOS and the authors call such measures "poor indicators"
for buzz/transient-doubling artifacts ([Roberts & Paliwal 2020](https://arxiv.org/pdf/2006.00848));
SVCC 2023: MCD vs MOS Spearman −0.24 to −0.38, mostly not significant, while UTMOS reached
0.72-0.77 ([SVCC 2023](https://arxiv.org/pdf/2306.14422)).

**Jittered seams** would de-tune the 50 Hz line into noise-like roughness (asynchronous
granular synthesis loses its pitched buzz), not remove the energy; no VC listening evidence
exists. Low priority.

---

## B. Making the decoder streaming-consistent

### B4. Stateful (cached-convolution) inference of the existing decoder, no retraining

**Possible, and exact.** A non-causal conv stack with symmetric 'same' padding streams
exactly if each layer keeps a ring buffer of its last (k−1)d inputs and emits only outputs
whose full window is present; the output then lags the input by the one-sided receptive
field, and no zero padding ever enters the interior. Caillon & Esling formalise it as
re-padding the trained net causally with per-layer delays ("behaves exactly like the
original model in terms of audio quality"; strided layers need an input delay that makes the
total lag a multiple of the stride; parallel branches need re-synchronising delays)
([arXiv 2204.07064](https://ar5iv.labs.arxiv.org/html/2204.07064)); their library
[acids-ircam/cached_conv](https://github.com/acids-ircam/cached_conv) implements
`CachedConv1d`, `CachedConvTranspose1d` and `AlignBranches` (the MRF's three branches have
reach 12/36/60 and need exactly that) — **licence CC BY-NC 4.0**, so a pattern to copy, not
a dependency. Kyutai's `StreamingConv1d`/`StreamingConvTranspose1d`
([moshi conv.py](https://github.com/kyutai-labs/moshi)) and TRT-SE
([Xiaobin-Rong/TRT-SE](https://github.com/Xiaobin-Rong/TRT-SE)) are **causal-only** (asserts
in code) but show the transposed-conv partial overlap-add and TensorRT state passing.
Yoneyama et al. cache "all convolutional, STFT and iSTFT layers" with ring buffers and
replace LayerNorm by BatchNorm because LN "computes statistics over the whole sequence"
([arXiv 2506.03554](https://arxiv.org/html/2506.03554v1)).

**ConvTranspose1d** streams exactly: run it on the chunk with padding 0, add the previous
chunk's saved tail to the first (k−u) outputs, save the new last (k−u) outputs, emit L·u.
The 'same' padding (k−u)/2 becomes a fixed delay of u/2 samples. Pitfalls: add the bias once
per output sample; chunk boundaries on input-frame boundaries (our 480-sample hop is a
multiple of every `noise_convs` stride {40, 4, 2}); never trigger the `F.interpolate`
fallback; fixed shapes (CUDA graphs) and one TF32 policy, or "exact" is only to float
reassociation.

**Right context an exact stream needs: 10 latent frames = 100 ms at 48 kHz, 11 = 110 ms at
40 kHz (~12 at 32 kHz), plus the hop.** That is the whole price: with it, each output sample
is rendered **once**, from a decoder state identical to the whole-file decode of the same
latent; SOLA and the crossfade disappear (and with them 30 ms of today's latency), so the
net algorithmic cost is **+70-90 ms** against today (hop 20 + 100-120 lookahead vs hop 20 +
30). Compute: with caching, the decoder computes only the activations behind 480 new
samples per hop (≈ 15 % of today's tail-decode FLOPs) but is launch-bound at that
granularity; it must be one CUDA graph with state tensors. Uncached-but-exact (decode hop +
10 frames right + 10 left ≈ 220 ms per hop) is ~1.7× today's 130 ms tail decode (~+3 ms).

**What breaks without the full right context:** emitting samples whose right side is zero
padded and then never revising them makes the padding error permanent at every chunk
boundary. CSSinger measured exactly this: a chunk-streamed VITS-class model with ordinary
zero padding at chunk edges scored MOS 2.415 against 3.165 with "Natural Padding" (pad
with real latent values, then trim) ([CSSinger](https://arxiv.org/pdf/2412.08918)). So a
*partial* lookahead only works together with a crossfade (today's design, §A2 right
context), or with a decoder trained to be causal (§B5).

**What an exact decoder does not fix: the latent.** z_N = flow⁻¹(m_p + exp(logs_p)·ε·0.667)
depends on the text encoder (global attention) and the flow (±240 ms). Rolling ε to absolute
time (done) is necessary but not sufficient: m_p/logs_p for the same absolute frame change
as the window slides. The decoder can only be streamed over a latent that is **committed
once per frame** (computed when frame N first has its chosen right context r, stored, never
recomputed). Committing with r < 24 frames is a one-time approximation at the commit, not
a per-hop inconsistency; nothing re-renders the frame. The same holds one level up for the
features and F0 (§C).

### B5. Streaming vocoders built for this

Constraint first: RVC's `net_g.dec` consumes the 192-channel flow latent z plus f0 and the
speaker embedding, and every voice is a separately fine-tuned synthesizer with its own
latent space, so **no mel vocoder plugs in without training** (report 15 said the same).
Routes: (a) stream the existing decoder exactly (B4); (b) train a causal decoder on the
frozen encoder+flow of **each** voice; (c) make the causal decoder part of the RVC
*pretrained* generator once, so each voice's ordinary fine-tune inherits it; (d) a different
model family.

| vocoder / system | replaces RVC's decoder? | retrain | lookahead | compute | quality evidence | licence |
|---|---|---|---|---|---|---|
| RVC NSF-HiFi-GAN streamed exactly (B4) | it *is* it | none | 100 ms @48k / 110 @40k (+hop) | ≈15 % of the tail decode if cached | unchanged (bit-equal to whole-file given the latent) | MIT |
| Causal HiFi-GAN on z ([Zero-VC](https://arxiv.org/pdf/2606.20218)) | via retrain | per voice (b) or once (c) | 0 ms ("one frame in, one out"), 20 ms frames | RTF 0.063 on a Xeon CPU | whole-system NMOS 3.81 / SMOS 3.88 vs Seed-VC-small 3.77 / 3.62 | no code |
| Causal BigVGAN by non-causal→causal transfer ([Shi & Fingscheidt](https://arxiv.org/pdf/2408.11842)) | via retrain | per voice / once | 0 frames; 32 ms algorithmic | 13.7 M params | PESQ 3.96 causal + teacher/student vs 3.64 non-causal small (VCTK); no MOS | — |
| SSMGAN causal StyleMelGAN ([arXiv 2108.04051](https://arxiv.org/pdf/2108.04051)) | via retrain | — | 0 (+10 ms PQMF) | 4.8 GMAC | MUSHRA in a figure only | — |
| MS-Wavehax causal, LA = 1 ([Yoneyama et al.](https://arxiv.org/html/2506.03554v1)) | via retrain (mel + F0) | once / per voice | 10 ms (LA=1) or 0 | 0.33 M params, 1.58 GMAC/s | "causal MS-Wavehax with LA=1 achieved the highest MOS" (values in a figure) | [unverified] |
| Vocos ([arXiv 2306.00814](https://arxiv.org/html/2306.00814v3)) | via retrain (mel) | once / per voice | non-causal ConvNeXt k7 + channel LayerNorm → needs lookahead or retrain | 13.5 M; xRT 6696 on A100 | MOS 3.62 vs HiFi-GAN 3.54 (GT 3.81) | MIT |
| HiFTNet ([arXiv 2309.09493](https://arxiv.org/abs/2309.09493)) | via retrain (mel + F0) | once / per voice | non-causal [unverified] | ~1/6 BigVGAN params | abstract claims only [unverified] | CC BY-NC-SA |
| SiFi-GAN ([arXiv 2210.15533](https://arxiv.org/pdf/2210.15533v3)) | via retrain (mel + F0) | once / per voice | non-causal | 11.3 M | MOS 3.90 vs HiFi-GAN+Sine 3.86; at 2× F0 3.05 vs 2.69 | — |
| StreamVC SoundStream-style causal decoder ([arXiv 2401.03078](https://arxiv.org/pdf/2401.03078)) | no (whole system) | once (zero-shot) | 60 ms architectural (2 frames) | 10.8 ms per 20 ms chunk on a Pixel 7 core | DNSMOS 4.07, WER 6.22 %; no MOS | no code |
| LLVC Waveformer causal enc/dec ([arXiv 2311.00873](https://arxiv.org/abs/2311.00873), [KoeAI/LLVC](https://github.com/KoeAI/LLVC)) | no (any-to-one system) | per voice | 16 samples = 1 ms @16 kHz; 19.7 ms e2e on an i9 | RTF 2.77 CPU | MOS 3.78 / SMOS 3.83 vs RVC 3.58 / 3.35 | MIT |
| RT-VC causal DDSP ([arXiv 2506.10289](https://arxiv.org/pdf/2506.10289)) | no | once | 32 ms + 15 ms chunk + 14 ms = 61 ms on M3 | — | MOS 3.87, SMOS 3.59 | checkpoints "will not be made open source" |
| MeanVC / MeanVC2 ([arXiv 2510.08392](https://arxiv.org/pdf/2510.08392v3); [HF](https://huggingface.co/ASLP-lab/MeanVC2)) | no | once | 160 ms chunk / "40 ms chunk + 40 ms future", 110 ms first packet | 14-18 M; CPU RTF < 0.63 | NMOS 3.82 / SMOS 3.87 vs Seed-VC 3.76 / 3.62 | Apache-2.0 |
| Seed-VC real-time ([real-time-gui.py @ 51383ef](https://github.com/Plachtaa/seed-vc)) | no | once | GUI defaults block 0.3 s, crossfade 0.04, `extra_time_right 0.02`, 20 ms SOLA search; CosyVoice HiFT vocoder | 10 diffusion steps per block | — | GPL-3.0 |
| SynthVC ([arXiv 2510.09245](https://arxiv.org/html/2510.09245v1)) / TVTSyn ([arXiv 2602.09389](https://arxiv.org/html/2602.09389v1)) | no | per target / zero-shot | 50 ms chunk + 27 ms = 77 ms CPU / 60 ms chunk, 78.5 ms GPU | 14.7 M / 86 M | N-MOS 3.51 / 3.82 | no code |
| X-VC ([arXiv 2604.12456](https://arxiv.org/pdf/2604.12456)) | no | once | [PDF unreadable through the proxy in this session; report 15: 240 ms] | — | — | "will be released" |

On the streaming-vocoder comparison paper specifically: it has **no per-vocoder
latency/lookahead table**; Table 1 gives params/MACs (HiFi-GAN 13.8 M / 28 GMAC/s, Vocos
13.5 M / 1.35, MS-Wavehax 0.33 M / 1.58 at 24 kHz, 10 ms shift), all RTFs are single-thread
CPU, MOS is a plot ([arXiv 2506.03554](https://arxiv.org/html/2506.03554v1)).

Takeaway: every causal replacement with published numbers targets CPU/mobile and reports
0-60 ms lookahead; none consumes an RVC latent, so all imply route (b)/(c) training. Exact
streaming of the existing decoder costs ~100 ms of lookahead and zero training. **The only
no-crossfade path with ≤ 20 ms lookahead is a causal (or 1-2-frame-lookahead) HiFi-GAN-class
decoder trained on z**, and the literature (Shi & Fingscheidt, Zero-VC, SSMGAN, CSSinger,
MS-Wavehax LA=1) says such decoders lose little against their non-causal parents.

### B6. Deterministic rendering: what still makes two windows disagree

| source | bounded or global | state in DEXAIV | remaining fix |
|---|---|---|---|
| latent noise ε | — | rolled with the window (`noise_roll`) | done |
| SineGen phase, excitation noise | global in time (an integral from the window start) | carried (`sine_phase_carry`); noise rolled; `harmonic_num = 0` so no random harmonic phase | done (every open engine still restarts both, §A1) |
| decoder left edge | ±94-119 ms | 80 ms context | +15-40 ms of left context makes it exact; compute only |
| **decoder right edge** | ±94-119 ms | zero padding | 100-120 ms right context, or a causal decoder |
| **flow** | ±240 ms | 24 frames left; zero on the right | commit-once latent; exact only with 240 ms right context |
| **text-encoder attention** | global | — | commit-once latent; or banded attention via the dormant `block_length` mask (train/test mismatch unless fine-tuned); or chunk-wise attention training (CSSinger Emformer-style, StreamFlow block masks, Incremental FastPitch static chunk masks) |
| encoder FFN, LayerNorm | ±6 frames; per-frame over channels | — | nothing once attention is handled |
| **ContentVec** | global (GroupNorm over time + 12 attention layers; ~2 s effective context) | recomputed every hop over 1.1 s | §C7 |
| RMVPE | whole 310 ms window (five 2×2 poolings collapse it; BiGRU) | `f0_trim_tail = 2` | §C8 |
| numerics | per shape | graphs fix shapes | one TF32 policy for both paths |

**What the flow contributes:** a bounded ±240 ms dependence — second only to attention and
HuBERT in span, and larger than the decoder's. It is deterministic given z_p, so it adds no
randomness, only edge sensitivity; the first 24 frames of every tail-only decode are its
zero-padded left margin (supplied) and the newest 24 frames its unsupplied right margin.

Literature on window-invariant VITS-class inference: Incremental FastPitch
(receptive-field-constrained chunk attention masks + fixed KV cache; conv state of size
kernel−1 "adequate to attain equivalence with parallel inference"; MOS 4.178 vs 4.185;
[arXiv 2401.01755](https://arxiv.org/html/2401.01755v1)); StreamFlow (block-wise attention
masks; context blocks "prevent phase conflicts and popping sounds"; NMOS 4.153 vs 4.213;
[arXiv 2506.23986](https://www.arxiv.org/pdf/2506.23986)); CSSinger (causal posterior
encoder + causal HiFi-GAN incl. transposed convs + Emformer chunked attention + Natural
Padding; [arXiv 2412.08918](https://arxiv.org/pdf/2412.08918)). All three retrain; none is
applicable to an existing `.pth` without it. Nothing found for VITS2 / MB-iSTFT-VITS
streaming specifically.

---

## C. The front end over a sliding window

### C7. ContentVec/HuBERT over a sliding ~1.1 s window

- **The features of one instant change every hop, by construction** (§0.2: GroupNorm over
  time in conv layer 0, global attention). Meng, Goldwater & Tang: HuBERT's phone error
  matches full context only at ±2.0 s windows and the ℓ2 feature distance only at ±4.0 s;
  "the effective context of HuBERT is about 2.0 s on either side"; streaming HuBERT without
  retraining: PER 11.9 % (full) → 12.5 % (unlimited history + 400 ms lookahead) → 13.7 %
  (2 s history + 400 ms lookahead) ([arXiv 2505.22487](https://arxiv.org/pdf/2505.22487)).
  Our kept frames have 1.1 s of history and **0 ms of lookahead**, inside the region where
  both the vectors and the phonetic content of a frame still move as the window slides.
  Direct hop-to-hop drift of the same instant: **not measured by anyone**; `vox bench` can
  (cosine between the feature a frame had when it was the newest and when it was 20/40/80
  ms old; the §12.3 tooling already computes feature cosines).
- **The measured penalty of naive sliding-window SSL in streaming VC:** Conan's ablation,
  replacing its Emformer student with offline HuBERT run chunk-wise, raised WER from
  **6.02 % to 26.92 %** ([Conan](https://arxiv.org/html/2507.14534v4)). Chunked SSL in ASR:
  WER +37 % at 320 ms chunks, +20 % at 640, +10 % at 1280 vs offline, with full left
  context; +101 % with one left chunk ([XLSR-Transducer](https://arxiv.org/pdf/2407.04439)).
- **How the streaming systems get consistent content features:** compute once per instant
  with a causal or chunked encoder, never recompute. StreamVC: causal conv encoder trained
  to predict k-means(100) of HuBERT-Base **layer 7**, 2-frame lookahead (60 ms); Conan:
  Emformer distilled from offline HuBERT, 80 ms chunks + 2 right-context chunks (139.7 ms)
  or 20 ms chunks with none (36.9 ms; WER 10.72 vs 6.02 %, SIM 78.3 vs 85.7 %); Zero-VC:
  distilled streaming w2v-bert, 20 ms frames, zero lookahead, causal HiFi-GAN with an RF
  cache, 20 ms algorithmic latency; with speaker-anonymised input the gains beyond 0-20 ms
  lookahead are < 3 % ([Zero-VC](https://arxiv.org/pdf/2606.20218)); MeanVC: streaming-ASR
  bottleneck features in 160 ms chunks with a chunk-wise causal mask; LLVC: no SSL at
  inference at all (HuBERT only in the loss). Seed-VC's "XLSR-tiny": **not found** in any
  primary source this session.
- **Should features be computed once and cached?** Yes, that is the entire literature's
  answer, but *with this encoder* a cached feature is a feature computed at the edge
  (0 ms lookahead, GroupNorm statistics of that moment) and then frozen: consistent, but
  the edge version. Two cheap mitigations inside the current model, both **untested
  anywhere** (my suggestions): (i) **freeze the first-layer GroupNorm statistics** (running
  statistics from a long buffer, or a fixed per-channel affine), which makes the conv stack
  shift-invariant and removes the R4 level dependence at the root; (ii) commit features
  with a small lookahead r (20-40 ms) rather than at the edge. Either is a train/test
  mismatch of unknown size; measure feature cosine vs the full-file features first.
- **Window normalisation (R4):** the GroupNorm above *is* the window normalisation. At 1.1 s
  it biases pauses by +0.16 (R4); a frozen statistic removes it; a longer window dilutes it
  (2.5 s: gone) at no latency cost but at ContentVec cost (3.0 ms today at 1.1 s).

### C8. F0 in streaming

| estimator | hop / context | causal? | right-edge behaviour | consistency evidence |
|---|---|---|---|---|
| RMVPE ([arXiv 2306.15412](https://arxiv.org/pdf/2306.15412)) | mel hop 20 ms, 128 ms Hann; U-Net with five 2×2 poolings + **BiGRU**; trained on 2.56 s | no | on a 310 ms input the bottleneck collapses to ~1 step, so every frame depends on the whole 310 ms; the BiGRU's backward state at the last frame is its initial state; the last ~64 ms of mel frames are zero-padded. `f0_trim_tail = 2` removes the worst frames, not the window dependence | ours: p95 seam step 21 → 10 cents (§8.4); no published number |
| FCPE ([arXiv 2509.15140](https://arxiv.org/html/2509.15140v1)) | depthwise-conv stack; bidirectional (ablation speaks of succeeding frames) | no | no streaming mode; "real-time SVC" is future work | none; 5.3× cheaper than RMVPE |
| SwiftF0 ([arXiv 2508.18440](https://arxiv.org/html/2508.18440v1)) | STFT hop 16 ms; 5× 2-D conv 5×5 → ~±10 frames | not stated | — | none |
| PESTO ([arXiv 2508.01488](https://arxiv.org/pdf/2508.01488)) | single VQT frame | **yes** (frame-wise; streamable VQT); lag = half the largest kernel (~85 ms at 48 kHz) or "buffer refilling" at RPA 97.7 → 97.4 % | stated lag model | the only streaming-degradation number found |
| FCNF0++/PENN ([arXiv 2301.12258](https://arxiv.org/html/2301.12258v3)) | per-frame, no zero padding | "permitting low-latency and streaming" (unquantified) | — | none |
| torchcrepe streaming | not found | | | |

How the field measures streaming F0 consistency: it does not. Standard metrics are RPA/RCA
at 50 cents, Δ¢ and voicing F1 on offline tracks; **no paper reports streamed-vs-offline
track agreement, octave-flip rate or frame-to-frame smoothness**; male-to-female streaming
F0 evidence: not found. The measurement is ours to define: per voiced frame, cents between
the F0 a frame had when it was committed and the whole-file F0 of the same frame (RMSE,
p95, flip rate), plus the hop-to-hop step at seams (§8.4 already does the latter). Report
15's FCPE swap remains a compute decision, not a consistency one; a 310 ms RMVPE tail could
also be committed with a 20-40 ms lookahead like the features, at the same latency.

---

## D. Architecture: how real-time VC engines are built, and should we build our own?

### D9. Survey

Verified against sources in October 2026; items that correct report 15 are marked **(corrects 15)**.

**(i) Sliding window + SOLA ("convert a window, keep the tail").** RVC-WebUI realtime, w-okada,
deiteris, Applio realtime, RVC's new VST, DDSP-SVC, and **Seed-VC's real-time GUI** (`real-time-gui.py`:
20 ms grid, SOLA overlap capped at 80 ms, one-frame search, the whole rolling window re-diffused
every block with 10 steps; **(corrects 15)**: it is this family, not a KV-cache streamer —
[real-time-gui.py](https://raw.githubusercontent.com/Plachtaa/seed-vc/main/real-time-gui.py)).
Compute scales with the window, not the hop; joins depend on SOLA finding a phase-coherent
offset inside one frame; every member re-renders the overlap and crossfades. Details in §A1.

**(ii) Fully causal streaming VC with carried state** (constant per-hop compute, no SOLA):
StreamVC (60 + 10.8 ms on a Pixel 7; no code), LLVC (MIT; 15 ms algorithmic + compute = 19.7 ms
on an i9 CPU; any-to-one; §D9-LLVC below), RT-VC (61.4 ms on an M3; a
[Berkeley-Speech-Group/RT-VC](https://github.com/Berkeley-Speech-Group/RT-VC) repo now exists,
9 commits, no licence, checkpoints "will not be made open source" per the paper, contents
**[unverified]**), SynthVC (77 ms CPU; no code), TVTSyn (78.5 ms GPU; no code), ALO-VC (55 ms
CPU; no code), and two new ones: **Zero-VC** (Jun 2026: strictly causal, zero lookahead, 20 ms
frames, distilled streaming w2v-bert on speaker-anonymised input, causal HiFi-GAN with conv-state
cache, RTF 0.063 on a Xeon core, WER 3.96 / SMOS 3.88 / NMOS 3.81; **no code**;
[arXiv 2606.20218](https://arxiv.org/html/2606.20218v1)) and **Beatrice 2** (Aug 2025, MIT code and
models, per-speaker trained, < 30 MB, RTF 0.2 target on one laptop-CPU thread, ~50 ms measured
with its VST, ~40 min training on a 4090; inference lives in the VST / VCClient, its openness
**[unverified]**; [trainer README](https://huggingface.co/fierce-cats/beatrice-trainer/raw/main/README.md)).

**(iii) Chunked models with KV / conv-state caches and bounded lookahead:** DualVC 1/2/3 (DualVC 3:
20 ms chunk + 2 predicted pseudo-future frames, 44-56 ms on one i5 core, fixed targets, no code;
[arXiv 2406.07846](https://arxiv.org/html/2406.07846v1)), IBF-VC (no code), Conan (MIT, weights on
Google Drive; README "~80 ms" / "~37 ms fast"; hardware and metrics not on the arXiv page;
[repo](https://github.com/User-tian/Conan)), MeanVC (Apache), **MeanVC 2 (corrects 15: code and
weights are released**, Apache-2.0, `meanvc2_40ms_40ms` / `_120ms_40ms`, JIT Vocos, Windows
executables through VB-CABLE, speaker-specific fine-tune scripts; 18 M params; 109.9 ms first
packet on one EPYC core; 16 kHz; trained on 10,000 h Mandarin Emilia, English coverage
**[unverified]**; NMOS 3.81 / SMOS 3.89 / CER 7.44 % zh; torch 2.5.1 cu121 + py3.11 pinned;
[arXiv 2606.09050](https://arxiv.org/html/2606.09050v1), [HF](https://huggingface.co/ASLP-lab/MeanVC2)),
**X-VC (corrects 15:** total latency is T_model 240 + T_compute 58 ms ≈ **298 ms** on an RTX 3090
by the paper's own Eq. 5; MIT; weights on HF; 539 M params; streaming en WER 3.14 / SIM 0.62;
torch 2.5.1 + deepspeed + GLM-4-Voice tokenizer; no fine-tune recipe;
[arXiv 2604.12456v3](https://arxiv.org/html/2604.12456v3), [repo](https://github.com/Jerrister/X-VC)).
FastVoiceGrad / FasterVoiceGrad are offline only.

**(iv) Codec-token / LM-based streaming VC:** StreamVoice / StreamVoice+ (111.6 ms on an A100, 153 M,
no code; MeanVC 2 re-measured it at 1,258 ms first packet on a CPU core), Stream-Voice-Anon
(anonymisation, not chosen-target VC; no code), Hibiki (Kyutai; S2ST with voice transfer only;
not a VC tool), CosyVoice VC (offline only), OneVoice (non-streaming; code "soon"), StyleStream
(~1 s). Nothing Mimi/Moshi-based for same-language VC with code was found.

**(v) Newer or adjacent:** Zero-VC and Beatrice 2 are the two genuinely new low-latency designs
since report 15; MeanVC 2's release and X-VC v3 are the availability changes.

| system | year | latency (what it includes; hardware) | compute | quality | target | fine-tune 10-30 min? | RVC `.pth` carries over | licence | code + weights | Win / cu128 / py3.13 |
|---|---|---|---|---|---|---|---|---|---|---|
| RVC-WebUI realtime | 2023- | block + crossfade + device (default 250 + 50 ms) + whole-window compute | HuBERT + synthesizer on the window | none published | per target | yes | yes | MIT | yes | yes |
| DEXAIV (this engine) | 2026 | ~127 ms live at hop 20 (§0.1), 72-110 ms RTT measured | 16 ms/hop on the 5060 Ti | §DEVIATIONS | per target | yes | yes | — | yes | yes |
| Seed-VC RT GUI | 2024-25 (archived 2025-11-21) | README table 430 ms; text "300 + 100"; inconsistent; RTX 3060 Laptop | 25 M, 10 diffusion steps per block | X-VC: streaming en WER 3.31 / SIM 0.40 | zero-shot (+ ft) | "2 min on a T4" claimed | no | GPL-3.0 | yes | py3.10 |
| Beatrice 2 | 2025 | ~50 ms e2e with the VST (external recorder) | < 30 MB; RTF 0.2 on one CPU thread | none public | per target | ~40 min on a 4090 | no | MIT | trainer yes; inference via VST / VCClient | trainer has a `cu128` extra; inference **[unverified]** |
| LLVC | 2023 | 19.7 ms e2e = 15 algorithmic + compute, i9-10850K CPU | RTF 2.77 CPU | MOS 3.78 / SMOS 3.83 vs **no-F0** RVC 3.58 / 3.35 | any-to-one | n/a (360 h of source speech, 3 days on a 3090 per target) | **yes, as the teacher** | MIT | yes | py3.11; torch 2.7 compatibility of its `TransformerDecoderLayer` override **[unverified]** |
| StreamVC | 2024 | 70.8 ms, Pixel 7 one core | — | WER 6.22, sim 77.8 %, DNSMOS 4.07 | zero-shot | n/a | no | — | **no** | — |
| RT-VC | 2025 | 61.4 ms, Apple M3 | — | MOS 3.87, SMOS 3.59 | zero-shot | n/a | no | not stated | repo; checkpoints no | — |
| SynthVC | 2025 | 72-96 ms, one i5 core | 8-15 M | N-MOS 3.51 vs teacher 4.02; SPK-COS 0.626 vs 0.611 | fixed set | n/a | no | — | **no** | — |
| TVTSyn | 2026 | 78.5 ms GPU (RTX 500 Ada) | 86 M | MOS 3.82 | reference | n/a | no | — | **no** | — |
| Zero-VC | 2026 | 20 ms algorithmic; RTF 0.063 CPU | — | WER 3.96, SMOS 3.88, NMOS 3.81 | zero-shot | n/a | no | — | **no** | — |
| DualVC 3 | 2024 | 44-56 ms, one i5 core | 12-23 M | NMOS 3.66, SMOS 3.76, CER 14.4 % | fixed set | n/a | no | — | **no** | — |
| Conan | 2025 | "~80 / ~37 ms" (README; hardware not stated) | — | numbers not on the arXiv page | zero-shot | not documented | no | MIT | yes | py3.10+ **[unverified]** |
| MeanVC 2 | 2026 | 109.9 ms first packet (40 chunk + 40 future + 80 ASR), one EPYC core | 18 M; pipeline RTF 0.63 CPU | NMOS 3.81, SMOS 3.89, CER 7.44 (zh) | zero-shot + ft scripts | script exists; data amount not stated | no | Apache-2.0 | yes + Windows exe | py3.11, torch 2.5.1 cu121 pinned |
| X-VC | 2026 | ≈ 298 ms (240 model + 58 compute), RTX 3090 | 539 M | en WER 3.14 / SIM 0.62 / UTMOS 3.07 | zero-shot | not documented | no | MIT | yes | py3.10, torch 2.5.1, deepspeed |
| StreamVoice+ | 2024 | 111.6 ms on an A100 | 153 M | NMOS 3.75 | zero-shot | n/a | no | — | **no** | — |

**LLVC, read from the clone** ([KoeAI/LLVC](https://github.com/KoeAI/LLVC), last commit Nov 2023):
causal conv prenet → `in_conv` (stride L = 16) → 8-layer dilated causal conv encoder (512-d) →
1-layer causal chunked transformer decoder (256-d, 13-frame chunks) → ConvTranspose1d; carried
`enc_buf` / `dec_buf` / `out_buf`; chunk 208 samples = 13 ms, lookahead 2 ms; 16 kHz only.
Training data: LibriSpeech train-clean-360 converted by an **RVC v2 32k** teacher fine-tuned on
39 min of one speaker, RMVPE, **index disabled (it hurt intelligibility)**, and **+12 st verified**
in the README recipe (`--f0_up_key 12`). 500 k steps, ~3 days on one 3090. MOS 3.78 / 3.83 vs a
*no-F0* RVC baseline 3.58 / 3.35; Resemblyzer 0.829 vs RVC 0.846; **the F0 RVC teacher itself is
never scored**, so "student beats teacher" is not shown. No fork or extension found.

**The distil-a-causal-student recipe, evidence:** students reach or beat the teacher's *speaker
similarity* when the target is fixed (SynthVC SPK-COS 0.626 vs 0.611; X-VC SIM 0.63 vs 0.56
offline; LLVC SMOS ≈ ground truth) and lose 0.1-0.5 MOS and 1-4× CER on intelligibility unless the
pair set is very large (X-VC: 20 k h). Data: LLVC 360 h of source speech, SynthVC 88 k
utterances × 6, X-VC 20 k h; **no recipe under ~100 h of source speech is reported.** Each new
target is a full training run (3 GPU-days on a 3090 for LLVC). Pitch/formant controls disappear
(no explicit F0 path). The only recipe with code is LLVC, frozen since 2023.


### D10. Recommendation for DEXAIV

**Recommendation: (a) now, (b) as the real bet, (c) not yet.** Reasoning, then the three
options against report 15's phases.

The brief's own evidence settles one thing report 15 left open: with a raw Shure, gate off, AGC
off, the stutter persists and scales with the hop. **It is a streaming problem** (report 15 §1
said "mostly not a streaming problem" and leaned toward gate/AGC; that is now wrong for this
artifact, though its Phase 0/1 items stand). The survey settles a second: no open engine has
solved it; they run 130-500 ms chunks with 50-150 ms fades and have *fewer* coherence
mechanisms than DEXAIV already has. So the comparison "DEXAIV live vs w-okada live" is not
the bar; "DEXAIV live vs DEXAIV whole-file" is, and the gap to it is the §0.2 list.

**(a) Keep the window engine and fix the joins** (§F 1-8, 10). Effort ~1-3 weeks. Risk low.
Expected gain: right context at 40-100 ms plus offset damping should move hop 20 toward what
hop 200 sounds like, because they attack the two mechanisms hop 200 attacks (edge
impairment, seam/jitter rate) without the seam-density change; whether 40 ms is enough is
exactly the unheard experiment. Loses nothing. Latency: +20-100 ms for right context, part of
it paid back by the fill-target fix (#4). Ceiling: the crossfade remains; whole-file is not
reachable this way, only approached.

**(b) Keep RVC's encoder, make the decoder stream** — in two steps. **(b1) commit-once +
exactly streamed decoder** (§F 9, §B4): no retraining, every `.pth` works, the crossfade and
SOLA disappear, the decoder output is bit-equal to a whole-file decode of the committed
latent. Cost: ~+70-90 ms algorithmic over today (→ ~200 ms live unless the periphery pays it
back: fill rule, IAudioClient3 input), 3-5 weeks of careful engineering with a sample-exact
test against `FullFileConverter`. Risk medium: launch-bound cached convs must be one graph.
This is a "quality mode" at conversation's upper limit, and it is the runtime that (b2) needs
anyway. **(b2) a causal decoder trained once on z** (§F 11): the pretrained 48k generator's
decoder replaced by a causal / one-frame-lookahead NSF-HiFi-GAN and every voice fine-tuned on
it as usual; this is the only path to no crossfade **and** ≤ 20 ms lookahead, and the
literature says causal HiFi-GAN-class decoders lose little. Needs the voices' datasets (Q15)
and a training stack (Applio fork); 6-12 weeks; risk medium-high (no published system does
exactly this; nearest analogues DualVC's distilled causal branch and IBF-VC, no code). Keeps
`.pth` as the encoder side and the teacher, keeps F0/formant/presets, 48 kHz.

**(c) A new engine around a streaming model.** Not now. The only code bases are MeanVC 2
(16 kHz, Mandarin-trained, no F0 control, torch 2.5.1/py3.11, English quality unreported),
X-VC (~298 ms, 539 M, heavier stack), LLVC (3 GPU-days per voice, 16 kHz, no F0 path, frozen
code) and Beatrice 2 (inference in a VST). Every one loses the `.pth` collection as runtime
models, the index, the formant and pitch controls and 48 kHz. Revisit as report 15's Phase 3
(the distilled student, §F 12) once (b2) exists as a teacher and the user's own mic data
has been collected; LLVC's +12 st recipe is the template, and it is the only route to sub-50 ms.

**Where this report disagrees with report 15:** (1) the stutter is a streaming artifact, not
a gate/AGC one (its Phase 1.1 gate redesign may still help pauses, not this); (2) its Phase 2
"stateful streaming HiFi-GAN … −10 to −30 ms" is wrong in sign for the *existing* decoder:
exact streaming costs +100-120 ms of right context; latency is only saved with a retrained
causal decoder; (3) its right-context item (Phase 1.6) is the single most important
experiment and must be judged by ear, not by R3; (4) X-VC is ~298 ms total, MeanVC 2 now has
code and weights, Seed-VC's real-time mode is a SOLA window engine; (5) its claim that the
≥ 1 s window "should not be shortened" is right, but the reason (HuBERT's ~2 s effective
context) also says the window should be *committed*, not re-encoded, and that its first-layer
GroupNorm is a per-hop drift source worth freezing (§F 10).


---

## E. Measuring it

### E11. A test that predicts "stutter" and "chopped"

- **No off-the-shelf metric is validated for periodic seam artifacts.** NISQA's
  *Discontinuity* dimension is trained on packet loss, silence insertion and time-varying
  codec artifacts ([NISQA](https://arxiv.org/pdf/2104.09494); dimension defined in
  [Wältermann et al. 2010](https://www.isca-archive.org/interspeech_2010/waltermann10_interspeech.pdf)):
  plausible for the 16.7 Hz "chop", unlikely to isolate the 50 Hz buzz (raters would file it
  under noisiness/coloration). DNSMOS and UTMOS move with chunk size in MeanVC and Conan but
  were never validated on seams; SpeechBERTScore is frame-averaged SSL cosine and
  alignment-free, i.e. blind to both mechanisms by construction. Audiobox Aesthetics lists
  "buzzing" in its rater brief; no seam validation ([arXiv 2502.05139](https://arxiv.org/pdf/2502.05139)).
- **The physically matched measurement is the envelope-modulation spectrum within
  critical bands**: specific roughness (20-300 Hz; our 50 Hz) and fluctuation strength
  (0.25-20 Hz; our 16.7 and 5 Hz), per ECMA-418-2 or Daniel & Weber 1997
  ([MoSQITo implementation](https://mosqito.readthedocs.io/en/latest/source/reference/mosqito.sq_metrics.roughness.roughness_dw.roughness_dw.html)).
  Concretely, and consistent with brief 16's "hop-rate envelope energy ratio": per-band
  Hilbert envelopes, envelope spectrum over ≥ 200 ms blocks, the peak at the hop rate and
  its harmonics relative to the neighbouring modulation floor ("seam-rate modulation
  prominence"), reported against the whole-file render of the same take. Calibration: ECMA
  roughness JNDs are ~0.025-0.05 asper, modulation-depth JND 14-20 % for AM tones
  ([Forum Acusticum 2026](https://dael.euracoustics.org/confs/fa2026/data/articles/000235.pdf)),
  validated for AM tones and noise, **not for VC seams**.
- **The jitter component is measured from the engine, not the audio**: the hop-to-hop SOLA
  offset change in ms (already in `Sola.offset_hist`) and its distribution; after offset
  damping (§A2) it should collapse.
- **Validation is in-house or nowhere**: render the same take at hop 20/60/200 and with
  artificially imposed 50 Hz and 16.7 Hz AM of controlled depth, run the 16-trial ABX the
  index work already scripted, and regress the seam-rate prominence against the hits.
  Nothing published replaces this; MeanVC's chunk sweep (DNSMOS 3.56 → 3.83 from 80 to 200
  ms chunks) and Conan's right-context ablation are the only chunk-size-vs-quality numbers
  in the VC literature, and neither is seam-specific.
- Serves brief 16 directly: the seam-rate prominence is the metric it asked for, and the
  whole-file render is the reference it proposed.

---

## F. Ranked list of concrete changes

Ordered by expected effect per unit of latency and effort; "stutter" = the 50 Hz seam
buzz + SOLA jitter; "gap" = distance to the whole-file render. Latency is algorithmic
unless noted. GPU is per 20 ms hop against today's ~16 ms.

| # | change | evidence | stutter | gap | +latency | +GPU/hop | retrain | files |
|---|---|---|---|---|---|---|---|---|
| 1 | **Listen to what exists**: `Mom_live_hop20_blend30`, `hop40`, `hop40_xf40` in `out/ab/model_vs_engine/`, plus `sola_mode="pv"` at 30-40 ms overlap | §Q7 click ratio 1.39 → 1.00 at 30 ms; Applio ships PV by default | unknown until heard | 0 | +10-40 ms | ~0 | no | none |
| 2 | **Right context** `extra_right_ms` 20 / 40 / 80 / 100: decode the tail L ms behind the window's end; render for the ear, not for R3 | §0.2 (every kept sample is edge-impaired); CSSinger padding MOS 3.17 vs 2.42; Conan WER 6.0 → 8.6 % without right context; DDSP-SVC and Seed-VC ship 20 ms; report 15 Phase 1.6 | high at ≥ 80 ms (both renders become near-identical); partial below | large part of the audible gap | +L | +L of decode (~+0.4 ms per 20 ms) | no | `engine/chunker.py` (window = extra + right + crossfade + search + block; return region shifted left by right), `pipeline.py` (`_stage_synth` slice), `config/schemas.py`, `offline/simulate.py`, `benchmarks/bench.py` |
| 3 | **SOLA offset damping**: keep the previous offset while its NCC is within ε of the max, or add a change penalty | §0.4, §8.4 (19 % held, jumps of whole periods); Pollack 1968 | removes most of the jitter percept | 0 | 0 | ~0 | no | `engine/sola.py` (`find_offset`), a `RuntimeParams` knob, `tests/unit/test_sola.py` |
| 4 | **Fill-target rule** `hop + 20 ms` for hops ≥ 40 (today 2×hop + 10) | §0.1 | 0 | 0 | **−(hop − 10) ms live** | 0 | no | `api/control.py:150`, `vox bench underrun` |
| 5 | **Hop 40 + overlap 40 + right context 40** as the "conversation" preset, measured live with #4 | halves the seam rate (25 Hz, out of the fused-buzz region toward the gap at 20-33 Hz) and gives every kept sample ≥ 40 ms of right context; §A1: no engine runs below 30-60 ms | medium-high | medium | ~+60 ms vs today (≈ 150-170 ms live) | same (compute does not scale with hop) | no | config/presets only, after #2 |
| 6 | **Decoder left context to the full RF** (`decoder_context_ms` 80 → 120) | §0.2: 80 < 94-119 | small | small | 0 | +0.3 ms | no | `config/schemas.py` default |
| 7 | **Seam-rate modulation prominence** metric + the AM-depth calibration ABX | §E11 | (measurement) | — | — | — | no | `offline/metrics.py`, `benchmarks/bench.py` (`quality`, `--pairs`), brief 16 |
| 8 | **Feature-drift and F0-drift measurement**: cosine of a frame's features at ages 0/20/40/80 ms vs the full-file features; cents of committed F0 vs full-file F0 | §C7, §C8: no published numbers exist | (measurement; decides #9 and #10) | — | — | — | no | `offline/simulate.py`, `offline/metrics.py` |
| 9 | **Commit-once latent + exactly streamed decoder** ("render once"): features/F0/latent committed per frame at lookahead r; decoder run with cached state and 10-12 frames of right context; SOLA and crossfade removed | §B4 (Caillon & Esling; cached_conv pattern; transposed-conv OLA); §B6 | **removes the crossfade mechanism entirely**; what remains is the one-time commit approximation | large (decoder bit-equal to whole-file given the latent; residue = commit edge) | **+70-90 ms vs today** (hop 20 + 100-120 lookahead − 30 overlap/search) → ~200 ms live unless #4 and an IAudioClient3 input path pay it back | ≈ −3 to −4 ms if cached in one graph (15 % of the tail-decode FLOPs); ~+3 ms uncached | no | new `synthesis/streaming_decoder.py` (cached Conv1d/ConvTranspose1d/MRF with branch alignment, state tensors, graph capture), `pipeline.py` (commit ring for feats/pitch/z), `chunker.py`, `sola.py` bypass, `tests/engine` sample-exact test vs `FullFileConverter` |
| 10 | **Frozen GroupNorm statistics in ContentVec** (running stats from a long buffer), A/B'd by #8 and the ear | §0.2, §C7 (my suggestion; untested anywhere) | small-medium (removes a per-hop drift source and R4) | small | 0 | 0 | no (train/test mismatch of unknown size) | `features/contentvec.py` (patched conv-layer-0 norm) |
| 11 | **Causal decoder trained once on z** (route (c): make a causal / LA-1 NSF-HiFi-GAN the decoder of the 48k pretrained generator, then fine-tune the user's voices as usual) | §B5: Shi & Fingscheidt, Zero-VC, MS-Wavehax LA=1, CSSinger lose little vs non-causal; the only no-crossfade path under 150 ms | removes the mechanism at ≤ 20 ms lookahead | expected close to whole-file for the decoder's share | +0-20 ms | similar to today's decoder | **yes** (pretrain once + per-voice fine-tune; needs the voices' datasets, Q15) | training repo (Applio fork) + `synthesis/nets/generator.py` causal variant + #9's streaming runtime |
| 12 | **Distilled causal any-to-one student** (LLVC / SynthVC recipe) from the best RVC teacher on the user's own mic data | report 15 Phase 3; LLVC MOS 3.78 vs RVC 3.58 at 19.7 ms CPU; SynthVC fixed-target beats zero-shot on identity | none (no windows, no seams) | capped by the teacher, which is the whole-file render | **−50 to −80 ms vs today** | small | **new model** (10-30 h of parallel data rendered offline) | new engine path; the existing `.pth` survives as the teacher |
| 13 | Jittered crossfade position inside the search region | §A2/A3 (granular synthesis) | de-tunes the buzz, same energy | 0 | 0 | 0 | no | `engine/sola.py` |
| 14 | Averaging k successive renders | none found anywhere | — | — | +k×hop | ×k | no | not recommended |

**Do first (one session):** 1, 2, 3, 4 and the metric 7 — none needs retraining, all are
listenable in a day, and together they answer the one question the metric could not: how
much of the stutter is right-edge disagreement (2), how much is jitter (3), and how much is
seam density (5). **Then** 8 → 9 if the user accepts ~200 ms for a crossfade-free engine as
a "quality" preset, and in parallel the data question (Q15) that gates 11 and 12.

---

## Sources

Code read at pinned commits (all claims in §A1, §B4, §D9 carry file paths and lines):
w-okada/voice-changer `master` @ d8ef157 and `v.2` @ bec5bd0; deiteris/voice-changer
`master-custom` @ b9cd071; RVC-Project/Retrieval-based-Voice-Conversion-WebUI `main` @ 81eed5e
and `20240604` @ d60cd3d; IAHispano/Applio `main` @ 5379cf2; yxlllc/DDSP-SVC @ b9d4e6a;
dr87/Vonovox @ 02316ad (strings only); Plachtaa/seed-vc @ 51383ef; KoeAI/LLVC @ 1627c5d;
acids-ircam/cached_conv @ beb8a8a; kyutai-labs/moshi @ e6a55d2; Xiaobin-Rong/TRT-SE @ e35783a;
gemelo-ai/vocos @ eb39abf; this repository `feature/formant` @ 5672cea.

Papers and documents: Caillon & Esling 2022 (arXiv 2204.07064); Yoneyama et al. 2025 streaming
vocoders (2506.03554); Shi & Fingscheidt 2024 (2408.11842); Zero-VC 2026 (2606.20218); SSMGAN
(2108.04051); CSSinger (2412.08918); Incremental FastPitch (2401.01755); StreamFlow (2506.23986);
Vocos (2306.00814); SiFi-GAN (2210.15533); HiFTNet (2309.09493); StreamVC (2401.03078); LLVC
(2311.00873); RT-VC (2506.10289); SynthVC (2510.09245); TVTSyn (2602.09389); MeanVC (2510.08392);
MeanVC 2 (2606.09050); X-VC (2604.12456); Conan (2507.14534); DualVC 3 (2406.07846); IBF-VC
(2210.15158); ALO-VC (2306.01100); StreamVoice+ (2408.02178); Meng, Goldwater & Tang 2025
(2505.22487); XLSR-Transducer (2407.04439); RMVPE (2306.15412); FCPE (2509.15140); SwiftF0
(2508.18440); PESTO (2508.01488); PENN (2301.12258); Driedger & Müller 2016 (Appl. Sci. 6(2):57);
Ungan & Yagcioglu 2014 (Front. Psychol.); Pollack 1968 (JASA); Roberts & Paliwal 2020
(2006.00848); SVCC 2023 (2306.14422); SVCC 2025 (2509.15629); SpeechBERTScore (2401.16812);
NISQA (2104.09494); Wältermann et al. 2010; DNSMOS P.835 (2110.01763); UTMOS (2204.02152);
Audiobox Aesthetics (2502.05139); ECMA-418-2 4th ed. (2025); Forum Acusticum 2026 roughness JNDs;
HF `HubertConfig` docs; fairseq `wav2vec2.py` (mirror); Politrees/RVC_resources ContentVec
`config.json`; AI Hub real-time guides (deiteris, Applio); w-okada issues #1374, #1418, #1525.

Not found / unverified (collected): a listening test of jittered seam placement in OLA/SOLA VC;
streamed-vs-offline F0 consistency numbers anywhere; Seed-VC "XLSR-tiny"; torchcrepe streaming;
ViSQOL sensitivity to clicks; the exact Zwicker-Fastl roughness weight at 50 Hz; Vonovox's
defaults and whether its phase symbols mean carried phase; RT-VC repo contents; Beatrice 2
inference openness; MeanVC 2 English coverage; X-VC PDF (unreadable through this session's proxy).
