# 20. Stronger noise suppression for the live engine

Written 2026-10-24 against `main` @ 439a314 and `feature/paraling` @ 860bd7b (DEVIATIONS
§13-§15.9), with the web. Numbers are from this repository or a linked source; anything
else is **[unverified]** or **[derived]**; "not found" means looked for and not found.
The brief's noise list was left "[FILL IN]"; §0 says how to find out which noise it is,
and the ranking in §6 is written so the first three steps are right whichever it is.

**Short version.** With the gate closed the engine outputs digital silence, so the noise
the user hears is one of three things: residual noise *re-voiced by the model during
speech* (RNNoise at mix 0.85 leaves at least 15 % of the dry noise, −16.5 dB, before the
model's own residue), room noise rendered during the **500 ms hangover** after every
phrase, or an energy-only open on a loud noise. The take JSON separates them in minutes;
do that before swapping anything. Then: the literature is unanimous that a VC/SSL model
is hurt more by a denoiser's *artifacts* than by its residual noise (Iwamoto 2023, N2N-VC,
Igarashi 2024), so "stronger" must mean a VAD-gated mix with a gain floor, suppressing
hard only when the gate's own VAD says there is no speech (hangover, probation), and
lightly inside speech. That alone removes the hangover tail and costs nothing. For a
better model, two candidates install today on Windows/py3.13 and run on the idle CPU
cores: **DPDFNet-2 48k** (Apache-2.0, `pip install dpdfnet`, streaming API, the best
published DNS4 scores of the real-time set) and **DeepFilterNet3** via its C runtime
(MIT, `deepfilternet-slim` + ctypes, the cheapest per frame, a 20 ms low-latency model).
Both add 10-20 ms over RNNoise. Do not put a denoiser on the GPU: every candidate under
3 M params costs ≤ 1 ms per 10 ms frame on a laptop CPU, and the Broadcast precedent
already showed the shared-GPU cost. Every DNS-trained model treats **breath as noise and
laughter as speech**, and NVIDIA's lists "body noises" as a target: the kiss/breath work
of §15 needs the gate-driven mix, not a stronger mask. Background voices and TV are a
different problem (personalised enhancement, 10-30 s enrollment), listed but not first.

---

## 0. Where the noise gets in, and how to tell in one take

`engine/gate.py` emits zeros while closed; `audio/conditioning.py` runs HPF → RNNoise
(mix 0.85 live) → AGC (≤ +18 dB, frozen below floor + 6) → limiter; the model renders
whatever the gate lets through. Three doors:

| door | when | what it sounds like | measured here |
|---|---|---|---|
| **A. inside speech** | always | noise the model re-voices as breathy texture under the voice; at mix 0.85 the dry path alone is −16.5 dB of the raw noise, before RNNoise's residue | §14.4: RNNoise removes synthetic bumps/taps; passes in-speech clacks |
| **B. hangover** | 500 ms after every phrase (`hangover_ms`, 14.7) | a tail of rendered room noise, "breathy garbage" fading as the drain kicks in | §14.1: the hangover's render read **−68 dBFS** on the quiet Shure take; a fan or TV would be louder |
| **C. energy-only open** | a noise ≥ threshold + 6 and ≥ 12 dB over the 200 ms background | 160 ms of rendered noise at hop 80 | §14.4: none on the proxy with RNNoise on; 18/18 with it off |

**Protocol (no code):** one Record-button take with the user silent for 20 s, speaking for
20 s, then silent again, in the room as it is (fan, PC, whatever hums). From the aligned
`mic_raw / mic_conditioned / output` and the gate trace: noise level (dBFS) in the raw
mic, after the chain, and in the output, split by gate state (closed / hangover / speech /
energy-only). If the output noise is in **B**, §3.1 fixes it for free; if in **A**, the
denoiser matters; if in **C**, the §19 probation work matters. Also log which it *is*
(the spectrum: < 300 Hz = hum/fan, broadband = hiss, bursty = keyboard, speech-shaped =
other people), because the denoiser ranking in §1 depends on it: every model handles
stationary noise; only personalised models handle other voices (§3.5).

The brief's "input, output, or both": the user hears the **output**; the question is which
door. The two live knobs that already exist and cost nothing are `suppression_mix`
(0.85 → 1.0 removes the dry floor; the kiss take showed 0.5 keeps smacks, §15.9, so this
is the conflict §3.1 resolves) and `hangover_ms` (500 ms chosen by ear for sentence ends,
14.7; it is also 500 ms of rendered noise per phrase).

---

## 1. Denoiser candidates that run live at 48 kHz on this machine

Only models with carried streaming state and a Windows/Python 3.13 path today are ranked;
offline or unavailable ones are listed at the end. DNSMOS numbers are comparable **only
within one paper** (two DNSMOS versions give DeepFilterNet2 OVRL 3.88 and 3.02 on the same
set), so the quality column names its source.

| # | candidate | algorithmic latency | compute (published) | quality (published) | streaming state in Python | install (Win / py3.13) | licence |
|---|---|---|---|---|---|---|---|
| 1 | **DPDFNet-2 48k hr** (Ceva, Dec 2025; DeepFilterNet2 + dual-path RNN) | 20 ms window / 10 ms hop, 2-frame lookahead → **30-40 ms [derived]**; first output after ~20 ms | 2.58 M params, 2.42 GMAC/s; x86 RTF **not published** (NPU only) | 16 kHz DNS4 blind SIG/BAK/OVL **3.39 / 4.05 / 3.10 vs DFN3 3.28 / 3.95 / 2.96**; low-SNR set PESQ 2.79 vs DFN3 2.45; 48 kHz numbers not published ([DPDFNet paper](https://arxiv.org/pdf/2512.16420)) | yes: `dpdfnet.StreamEnhancer`, 480-sample blocks, `reset()`, `flush()` ([repo](https://github.com/ceva-ip/DPDFNet)) | `pip install dpdfnet` 0.6.0, py3-none-any, ≥ 3.11, CPU onnxruntime ([PyPI](https://pypi.org/project/dpdfnet/#files)); also in sherpa-onnx 2.0.0 cp313 win wheel | Apache-2.0 |
| 2 | **DeepFilterNet3** via libDF C API | 30 ms (10 OLA + 20 lookahead; the paper quotes 40 incl. the frame); **`DeepFilterNet3_ll` 20 ms** (no lookahead; weights' provenance flagged "unclear" by one downstream) | **0.33 ms per 10 ms frame** (tract, both stages, a recent Mac, [WorkAdventure PR](https://github.com/workadventure/noise-suppression/pull/21)); 0.3-1.7 ms on Windows ORT builds; 2.14 M params, 0.35 GMAC | VB-DMD PESQ 3.17, STOI 0.944 ([DFN3](https://arxiv.org/pdf/2305.08227)); DNS4 OVL 2.96 in the DPDFNet table | yes via ctypes: `df_create(path, atten_lim, …)`, `df_process_frame(st, in, out) → local SNR`, `df_set_atten_lim`, `df_set_post_filter_beta` ([capi.rs](https://github.com/Rikorose/DeepFilterNet/blob/main/libDF/src/capi.rs)) | `pip install deepfilternet-slim` 0.1.0 (py3-none-win_amd64, no torch; ships the DLL + model, `runtime_assets()` only) ([PyPI](https://pypi.org/project/deepfilternet-slim/#files)); the torch package needs `deepfilterlib`, wheels stop at cp311 | MIT / Apache-2.0 |
| 3 | **NVIDIA AFX denoiser 48k** (Maxine, v1 / BNR 2.0) | 10 or 20 ms frames; algorithmic latency **not published** | GPU: p50 0.374 ms / p95 0.41 ms per 10 ms frame, **16 kHz v2 on an RTX 3500 Ada**; 48k ≈ 3× heavier by its stream tables ([release readiness](https://docs.nvidia.com/maxine/afx/latest/WindowsAFXSDK/ReleaseReadiness.html)) | no PESQ/DNSMOS published; noise list includes keyboard, mouse clicks, babble, chatter, **"body noises"** ([supported noises](https://docs.nvidia.com/maxine/afx/Appendix/DenoiserSupportedNoises.html)); "loud laughing or screaming might not be retained" | C API via ctypes (`NvAFX_Run` per frame, internal state); `NVAFX_PARAM_USER_CUDA_CONTEXT`, `NVAFX_PARAM_DISABLE_CUDA_GRAPH` for coexistence | NGC zip + `download_features.ps1` with an NGC key; AI-Enterprise gating **ambiguous** (a 402 on Blackwell AEC in one thread, "free on Windows" in another); driver ≥ 572.61 | Maxine SDK licence (NVIDIA-GPU-only, attribution) |
| 4 | **FastEnhancer-S/M 48k** (ICASSP 2026) | hop 10.7 ms (S) / 6.7 ms (M), n_fft 1024 → ~21 ms window [derived]; lookahead not stated | RTF 0.042 (S) / 0.135 (M) single thread on an M5; 195 K-492 K params | 16 kHz VB-DMD: S PESQ 3.19, OVL 3.12 (≈ DFN3, > GTCRN); 48 kHz table not published ([paper](https://arxiv.org/html/2509.21867v1), [repo](https://github.com/aask1357/fastenhancer)) | streaming ONNX in releases; feature/STFT glue is yours | ONNX + onnxruntime cp313 win wheel; no pip package | MIT |
| 5 | GTCRN (`gtcrn_simple`) | 32 ms window / 16 ms hop at **16 kHz** | 33 MMAC/s, RTF 0.07 on an i5-12400 | VB-DMD PESQ 2.87, DNS3 OVRL 2.70: the lowest of the neural set ([repo](https://github.com/Xiaobin-Rong/gtcrn)) | yes: 3 cache tensors; sherpa-onnx online denoiser | onnxruntime / sherpa-onnx wheels | MIT |
| 6 | RNNoise v0.2 (current, pyrnnoise 0.4.5) | 20 ms | ~1.3 % of one Haswell core | old model: DNS4 SIG/BAK/OVL 3.88 / 3.69 / 3.38 vs noisy 4.14 / 2.94 / 3.29 ([DFN2 paper](https://arxiv.org/pdf/2205.05474)); v0.2 model numbers **not found**; the "little" model and the 2018 rnnoise-nu weights (one "Voice" set claims to keep laughter) have no numbers | yes (today) | installed | BSD-3 |
| — | DeepVQE / -S (20 ms, 0.14 ms/frame) | | | DNS 2023 OVRL 2.73 | **no released weights** | — | — |
| — | Fast-ULCNet (16 kHz, 0.34 M) | | | DNS2020 OVRL 3.09 | code, **weights not found** | — | MIT |
| — | DTLN (16 kHz, 2020), Demucs denoiser (CC-BY-NC, 41 ms), Silero denoise (CC-BY-NC, offline), ClearerVoice MossFormer2_SE_48K, Resemble Enhance, SEMamba, TF-GridNet, SepFormer, noisereduce, Krisp (closed SDK) | offline, non-commercial, 16 kHz-only or not real-time | | | | | |

**Ranking for this use:** DPDFNet-2 first on quality per effort (one pip install, a
streaming class, 48 kHz native, best same-table DNS4 scores), with the caveat that its
x86 cost is unmeasured (2.42 GMAC/s is ~7× DFN3; expect single-digit ms per 80 ms hop on
one 7900X core **[unverified]**). DFN3 second as the proven, cheapest, lowest-latency path
(the `_ll` model keeps today's 20 ms). NVIDIA AFX third: the strongest published noise
list (keyboard, babble, chatter) and the only one that cannot be put on the CPU; A/B it
for free first through the Broadcast virtual mic the machine already has, remembering
report 15's finding that it lowers SIG and carves harmonics. Nothing documents breath,
laugh or kiss preservation for any of them.

---

## 2. Where denoising should sit for voice conversion

**Input side, and artifact-light.** Noisy source audio wrecks a speech-trained VC in every
measurement found: Seed-VC CER 5.2 → 12.5 % and SECS 0.823 → 0.788 on real noisy clips,
with a "tendency to reconstruct background noise" ([REF-VC](https://arxiv.org/html/2508.04996v2));
HuBERT-base WER 6.7 → 10.2 % under MUSAN/reverb and 33 % on CHiME-3, speaker-ID 85 → 66 %
([Huang et al.](https://ar5iv.labs.arxiv.org/html/2203.16104)); a VQ-VAE VC fed noisy input
scores MOS 2.07 vs 3.49 clean and 3.08 with a DCCRN front end ([N2N-VC](https://arxiv.org/pdf/2109.10608)).
The VC does not clean noise; it re-synthesises it. "Hallucinated breathy garbage / fake
consonants" as a phrase is **not in any paper**; the mechanisms are REF-VC's noise
reconstruction and RVC's own `protect`, which exists because unvoiced frames and breaths
are where "conversion artefacts" appear ([Applio docs](https://mintlify.wiki/IAHispano/Applio/features/inference)).

**But a separately trained denoiser can hurt more than the noise.** With a WavLM-based ASR:
noisy input WER 8.4 / 4.8 % (sim/real CHiME-4), a separately trained enhancer **12.3 /
9.4 %**, and interpolating enhanced with noisy ("observation adding") 6.4 / 4.3 %; the
damage is *artifact* error, not residual noise ([Iwamoto et al. 2023](https://arxiv.org/pdf/2311.11599)).
Every DNS-challenge suppressor raised WER over the noisy input (27.9 → 30-31 %) while
raising BAK ~1.2 and lowering SIG ([Braun & Gamper](https://arxiv.org/pdf/2111.11606)).
N2N-VC: DCCRN's output "sounds clean but carries some distortion" that "propagates into
the VQ-VAE". Igarashi et al. state enhancer-then-VC pipelines perform worse for this
reason and train the VC on noisy input instead ([2406.07280](https://arxiv.org/html/2406.07280v1)).
WavLM was pre-trained with DNS noise mixed into 20 % of utterances, but **its paper reports
no noisy-test result**, so "WavLM is more robust than HuBERT" is a recipe inference, not a
measurement ([WavLM](https://arxiv.org/pdf/2110.13900v4)).

**Output side** is cosmetic: Applio's `clean_audio` runs `noisereduce` on the converted
output at strength 0.7 and warns it "can compress the audio and make it sound less
natural" ([infer.py](https://github.com/IAHispano/Applio/blob/main/rvc/infer/infer.py)).
Nothing in VC uses it for latency reasons; the NSF decoder's own floor is the model's.

**Implication:** keep suppression before the model, bound it (gain floor, observation
adding), and judge it by the model's output (ΔWER, Δspeaker cosine), never by the
denoiser's own BAK. This is why "stronger" needs the §3 design and not a harder mask.

---

## 3. Stronger without eating speech

### 3.1 Gate-driven mix with a gain floor (free, first)

The field's shipped answer is a floor plus a detector-driven blend: WebRTC NS caps
attenuation at 6 / 12 / 18 / 21 dB (`minimum_attenuating_gain` 0.5 / 0.25 / 0.125 / 0.09,
[suppression_params.cc](https://arai.searchfox.org/firefox-main/source/third_party/libwebrtc/modules/audio_processing/ns/suppression_params.cc));
DeepFilterNet's `atten_lim_db` mixes enhanced with noisy in the STFT domain
([enhance.py](https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/df/enhance.py));
VoiceFilter-Lite blends `w·enhanced + (1−w)·input` with `w` from a detector and an
asymmetric loss against over-suppression, at zero WER cost on clean speech
([VoiceFilter-Lite](https://arxiv.org/pdf/2009.04323)); RNNoise itself has **no gain floor**
(the paper notes "the lack of a floor on the gain", [Valin 2017](https://arxiv.org/pdf/1709.08243v3)),
so the engine's `suppression_mix` is the only floor it has.

The engine already computes the detector: `GateDecision.vad_prob`, `energy_only`, and
whether the hop is inside the hangover. Make the mix a function of it:

    mix(t) = 1.0           while the gate is closed, draining, in the hangover
                           (vad_prob < vad_threshold − hysteresis) or on an
                           energy-only probation hop               → full suppression
    mix(t) = mix_speech    while loud and voiced                    → the user's 0.5-0.85
    one-pole ramp between them (20-50 ms), applied per 10 ms RNNoise frame

Effects: door **B** (the hangover tail) is suppressed at the model's full strength instead
of 85 %; door **C** likewise; inside speech nothing changes, so the kiss settings of §15.9
(mix 0.5) survive. Cost: zero latency, zero GPU. A/B: the §0 take, output noise in the
hangover before/after, lost-speech ms unchanged. **The kiss conflict:** a quiet "mwa" is a
low-VAD event, so this rule would suppress its smack harder; the smack-onset flag of
§15.9 (`onset_confirm_ms` + margin 0) must hold the mix at `mix_speech` for that event.
That is the one interaction to test on the kiss take.

### 3.2 Attenuation limit instead of a harder mask

When a stronger model arrives (§1), expose its floor (`atten_lim_db` for DFN3, the
equivalent blend for DPDFNet) and start at **12-18 dB**, not unlimited: the downstream
HuBERT prefers residual noise to artifacts (§2). Raise it only with the ΔWER / Δcosine
check of §5 flat. "Max" strength in §14.4 (VAD-gated residual −20 dB) cost 150 ms of speech
per minute: that is the over-suppression the DNS organisers describe ("noise suppressors
tend to be aggressive and end up suppressing low-energy emotions and sounds",
[Interspeech 2021 DNS](https://ar5iv.labs.arxiv.org/html/2101.01902)).

### 3.3 Two stages: no, unless co-trained

Every DNS winner is two-stage *by design* (magnitude then complex; DFN's ERB + DF), trained
together. Cascading two independently trained suppressors compounds artifacts (the
Iwamoto decomposition; N2N-VC's propagation). A controlled "classical then neural" study:
not found. If DPDFNet or DFN3 goes in, **it replaces RNNoise**; do not stack them.

### 3.4 Breaths, laughs, kisses

DNS taxonomy: **"heavy breathing" is a dev-set noise type; laughter, yelling and crying are
clean speech** from 2021 on ([ICASSP 2023 DNS](https://arxiv.org/pdf/2303.11510),
[ICASSP 2021](https://ar5iv.labs.arxiv.org/html/2009.06122)); the noise set is AudioSet
minus clips with speech activity, so breath/cough classes plausibly sit on the noise side
**[unverified; not stated by the organisers]**. Field reports: Zoom removes breath at every
level; Krisp "mutes or distorts laughter". NVIDIA lists "body noises" as a target. A
denoiser that preserves breaths: **not found**. So: the gate-driven mix (§3.1) and the
smack flag are what keep kisses; breaths pass only if the floor is high enough (≥ 12 dB
limit) and the VC's `protect`/passthrough path (§19) carries them; laughs are safer.
Measure with the over-suppression metric of §5 (seconds of non-speech vocal events
attenuated > 6 / 12 dB, the E3Net "TSOS" idea, [E3Net](https://arxiv.org/pdf/2204.00771)).

### 3.5 Background voices and TV: a different tool

No single-channel suppressor removes another talker without knowing who the user is.
Personalised enhancement (pDNS) does, from an enrollment of the user's own speech: the
challenges used 10-30 s (P.835 test: 10 s; 2023: 30 s), "most models submitted … were
personalized", and quality collapses only below ~2 s ([ICASSP 2022](https://ar5iv.labs.arxiv.org/html/2202.13288),
[2023](https://arxiv.org/pdf/2303.11510), [SEF-PNet](https://arxiv.org/html/2505.12288v1)).
Cost: E3Net-class models (4.5 M params, RTF 0.04-0.07 on one Xeon thread) **worsen WER on
single-talker speech** (13.4 → 19.7 % on one set) and over-suppress 2-4 s per half hour
([E3Net](https://arxiv.org/pdf/2204.00771)); TEA-PSE 3.0 is 22 M params, RTF 0.46 on CPU.
No pDNS model with 48 kHz weights and a Windows wheel was found. If the §0 take shows
speech-shaped noise, this is the route, gated on a "second voice present" detector; it
is not in the first three steps.

---

## 4. GPU sharing: don't

| fact | source |
|---|---|
| Every real-time candidate ≤ 3 M params costs ≤ 1.2 ms per block on decade-old laptop CPUs (DTLN 0.36-1.13 ms; DFN3 0.33 ms per 10 ms; DFN2 RTF 0.04 on an i5-8250U) | §1 |
| The GPU already shows the cost of a co-resident denoiser: NVIDIA Broadcast's added 0.3-0.9 ms of hop p99 | §8.2 |
| Per-launch host cost 5-15 µs; "for models with many small kernels, that launch overhead can exceed the actual GPU work"; concurrent streams "sharing compute resources … can offset the benefit" | [TensorRT perf guide](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/performance/optimization.html) |
| ORT CUDA EP can share torch's stream (`user_compute_stream`) and capture its own graph (`enable_cuda_graph`: fixed shapes, IOBinding, single-threaded, **never inside a torch capture**); ORT GPU wheels pair with CUDA 12.8 / cuDNN 9 and torch must be imported first | [ORT CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) |
| DirectML EP: no CUDA-stream interop, single-thread `Run`, static shapes at session creation | [ORT DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) |
| WDDM batches launches; HAGS is "strongly recommended" for launch-bound work | [NVIDIA WSL2 blog](https://developer.nvidia.com/blog/leveling-up-cuda-performance-on-wsl2-with-new-enhancements) |

**Design:** the denoiser lives where RNNoise lives, in `InputConditioner` on the worker
thread (or a dedicated conditioning thread feeding it), 4-8 model frames per 80 ms hop
with carried state, one ORT/tract CPU session, one core. Its latency replaces RNNoise's
20 ms (DFN3_ll: 0 net; DFN3/DPDFNet: +10-20 ms). A GPU denoiser is justified only for
NVIDIA AFX (no CPU build): then `NVAFX_PARAM_USER_CUDA_CONTEXT` on the engine's context,
`NVAFX_PARAM_DISABLE_CUDA_GRAPH` off, launched between stage graphs, p99 measured with
`vox bench underrun` at hop 20 and 80 before anything else. The 7-18 ms of GPU per 80 ms
hop is headroom for the *model*, not for a second runtime's scheduling jitter.

---

## 5. Test protocol for `vox bench`

**Material.** One clean raw-mic take (the matrix take) × noise {DNS5 noise (181 h, AudioSet +
Freesound + DEMAND, [DNS](https://ar5iv.labs.arxiv.org/html/2202.13288)), MUSAN *speech* for
background talkers ([MUSAN](https://openslr.org/17/)), ESC-50 {keyboard typing, mouse click,
vacuum cleaner, washing machine, clock tick, breathing, laughing} ([ESC-50](https://gitee.com/beyaso/ESC-50)),
WHAM! babble} × SNR {0, 5, 10, 20 dB} (the ranges WavLM, Igarashi and Huang used); tune on
one corpus, test on another (Igarashi's held-out-noise discipline); plus the user's real
§0 take as the sanity set, because simulated and real gaps differ (FAT-HuBERT: an enhancer
helps hugely on simulated 0-5 dB, barely on real CHiME-4, [2311.17790](https://arxiv.org/pdf/2311.17790)).

**Three scoring points.**
1. *Denoiser output* (clean reference exists): SI-SDR, Iwamoto's SNR-vs-SAR split (catches
   artifacts), DNSMOS P.835 **as a per-condition average only** (per-clip SIG PCC is 0.71;
   it scores hallucinated speech *higher*, [URGENT 2025](https://arxiv.org/pdf/2505.23212),
   [2403.11732](https://arxiv.org/pdf/2403.11732)).
2. *VC output* (what the user hears): ΔWER vs the **clean-input conversion** (Whisper
   small.en, as `vox bench quality` does; the VC's own error cancels), Δ speaker cosine
   vs target (ECAPA/WavLM-SV, as 0.1 built), UTMOS (the one MOS predictor trained on VC
   output), seed floor alongside. A setting that raises DNSMOS and raises ΔWER is a
   regression.
3. *Non-speech preservation*: time-stamped breath / laugh / kiss regions in the clean
   take; energy ratio processed/clean per region, share attenuated > 6 / 12 dB (TSOS-style).
   Plus the §0 door split (output noise dBFS by gate state) on the real take.

**Listening.** Pre-registered ABX/preference with the existing `--pairs` writer: 16 trials
is the floor (≥ 12/16 one-sided p = 0.038; Hydrogenaudio's criterion **13/16**, α = 0.01,
[ABX](https://wiki.hydrogenaud.io/index.php?title=ABX)); for denoiser A vs denoiser B,
where the effect is subtle, 24-32 trials (≥ 17/24, ≥ 22/32) and no optional stopping.
An end-to-end "real-time denoiser → RVC" study: **not found**; this protocol is the first.

---

## 6. Ranked list of concrete changes

| # | change | evidence | effect | latency | GPU/hop | files |
|---|---|---|---|---|---|---|
| 1 | **Diagnose the door** (§0): the silent/speech/silent take, output noise by gate state and its spectrum | code (§0) | decides 2-6 | — | — | Record button; `benchmarks/` scratch over the take JSON |
| 2 | **Gate-driven suppression mix with a floor** (§3.1): mix 1.0 in hangover / probation / closed, `mix_speech` in speech, ramped; smack flag holds `mix_speech` | VoiceFilter-Lite, WebRTC floors, RNNoise's missing floor; §14.1 hangover render; §15.9 kiss settings | removes door B and C noise at zero cost; speech untouched | 0 | 0 | `audio/conditioning.py`, `audio/suppression.py` (`set` per frame), `engine/worker.py` (pass the gate state), `tests/unit/test_suppression.py` |
| 3 | **Hangover A/B** 500 vs 350 ms with #2 on | 14.7 chose 500 for sentence ends; #2 may make the shorter one unnecessary or the longer one harmless | less rendered noise per phrase | 0 | 0 | config only |
| 4 | **DPDFNet-2 48k behind the `Suppression` interface**, CPU, streaming, floor 12-18 dB, replacing RNNoise when on | §1 #1, §2 (artifact-light), §3.2 | stronger stationary + non-stationary suppression at the best published quality of the set | +10-20 ms over RNNoise (30-40 vs 20) | 0 (CPU; measure ms per hop on one core) | `audio/suppression.py` (second backend), `config/schemas.py` (`suppression_backend`), `benchmarks/bench.py` (`suppress` variants) |
| 5 | **DeepFilterNet3 (`deepfilternet-slim` + ctypes)**, including the `_ll` 20 ms model, same interface | §1 #2 | the cheapest neural option; `_ll` keeps today's latency | 0 (`_ll`) / +10 ms | 0 | same files |
| 6 | **Noise-mix bench + three-point scoring + non-speech preservation metric** (§5) | §5 | the decision tool for 4/5 and every floor setting | — | — | `benchmarks/bench.py` (`quality --noise`), `offline/metrics.py` |
| 7 | **A/B the NVIDIA denoiser for free** through the Broadcast virtual mic on the §0 take (engine suppression off) | §1 #3; report 15's SIG caveat | tells whether its keyboard/babble handling is worth the SDK work | its own (unpublished) | shared GPU (0.3-0.9 ms p99, §8.2) | none |
| 8 | NVIDIA AFX SDK via ctypes, own CUDA context, graphs off, p99 measured | §1 #3, §4 | only if #7 wins by ear | unpublished | measure | `audio/suppression.py` backend; `TARGET.md` (driver ≥ 572.61, NGC key) |
| 9 | Personalised enhancement (pDNS-class) with a 10-30 s enrollment, gated on a second-voice detector | §3.5 | the only fix for TV / other people | +20-40 ms | 0 (CPU) or GPU | new backend; needs a model with 48 kHz weights (none found with a Windows wheel) |
| 10 | Output-side spectral gate (Applio-style `noisereduce`) | §2 | cosmetic; lowers naturalness | 0 | small | not recommended |

**Do first:** 1 and 2 in one session (2 needs no take to build, 1 tells whether it was
the fix); then 6, then 4 with 5 as the fallback, each judged by the ear on blind pairs and
by ΔWER / Δcosine, with the kiss take re-run so the §15.9 settings still hold. 7 is a
ten-minute check the user can do alone. 8 and 9 only if 1 says keyboard/babble or other
voices are the noise.

---

## Sources

Repository: `main` @ 439a314, `feature/paraling` @ 860bd7b — `audio/conditioning.py`,
`audio/suppression.py`, `engine/gate.py`, `engine/worker.py`, `docs/DEVIATIONS.md`
§8.2, §14.1, §14.4, §14.7, §15.9.

Candidates: DPDFNet (arXiv 2512.16420; github.com/ceva-ip/DPDFNet; PyPI dpdfnet);
DeepFilterNet2/3 (2205.05474, 2305.08227; Rikorose/DeepFilterNet capi.rs, enhance.py,
ladspa README, issue #628; PyPI deepfilternet, deepfilterlib, deepfilternet-slim,
dfnstream-py; soniqo DFN3 ONNX; WorkAdventure PR #21; lightsound/noican PR #48;
shimondoodkin/deepfilter-rt); GTCRN (Xiaobin-Rong/gtcrn; andrelgomes/gtcrn-stream);
sherpa-onnx 2.0.0; FastEnhancer (2509.21867; aask1357/fastenhancer; 2607.25350);
Fast-ULCNet (2601.14925); DeepVQE (2306.03177); NVIDIA Maxine AFX docs (release
readiness, supported noises, set/get parameters, Studio Voice, install, licence PDF, NGC
collection, developer-forum threads 343193 / 372446 / 259707); RNNoise (1709.08243, v0.2
release, xiph README, pyrnnoise, GregorR/rnnoise-models, werman/noise-suppression-for-
voice); DTLN; facebookresearch/denoiser; Silero denoise; noisereduce; ClearerVoice
(2506.19398); RT-SEMamba (2608.12099); Resemble Enhance; Krisp SDK.

Placement and over-suppression: REF-VC (2508.04996); R2-SVC (2510.20677); Du et al.
(2201.10693); N2N-VC (2109.10608); Igarashi et al. (2406.07280); Huang et al. SSL
robustness (2203.16104); FAT-HuBERT (2311.17790); Iwamoto et al. (2311.11599); Braun &
Gamper (2111.11606); WavLM (2110.13900); kNN-VC (2305.18975); Applio infer.py and docs;
RVC / so-vits-svc / DDSP-SVC READMEs; DNS Challenge ICASSP 2021 / Interspeech 2021 /
ICASSP 2022 / ICASSP 2023; TEA-PSE 3.0 (2303.07704); WebRTC NS suppression_params.cc;
VoiceFilter-Lite (2009.04323); E3Net (2204.00771); SEF-PNet (2505.12288); Lu & Loizou
2008; breath-aware TTS (2402.00288); Zoom and Discord community threads.

GPU and metrics: ORT CUDA / DirectML / TensorRT-RTX EP docs; TensorRT 11.4 performance
guide; NVIDIA torch CUDA-graph integration; WDDM/HAGS (Microsoft DirectX blog, NVIDIA
WSL2 blog); DNSMOS P.835 (2110.01763); NISQA (2104.09494); UTMOS (2204.02152); URGENT
2025 (2505.23212); hallucination in metric-driven SE (2403.11732); MUSAN; ESC-50; WHAM!;
FSD50K; Hydrogenaudio ABX; Leventhal 1986.

Not found / unverified: DPDFNet x86 RTF and 48 kHz quality; NVIDIA AFX algorithmic
latency and any PESQ/DNSMOS; AFX access terms on Windows (402 vs "free"); RNNoise v0.2
model quality; which model pyrnnoise compiles; whether DNS noise sets contain
breath/cough as noise targets; any denoiser that preserves breaths; an end-to-end
real-time denoiser → RVC study; a pDNS model with 48 kHz Windows wheels.
