# 19. Paralinguistics and transients: keystrokes, whispers, breaths, kisses, laughs

Written 2026-10-23 against `main` @ 5e3f5b2 (Phase A/B merged), with the web. Every number
is from this repository's code or measurements or from a linked source; anything else is
marked **[unverified]**, "not found", or **[derived]** (arithmetic from cited numbers).
Where an earlier external note (Argon, 2026-10-22, read against the two-week-old
2b2cf93 snapshot) is corrected, it says so.

**Short version.** (1) The keystroke leak is mostly not an onset problem: with the 500 ms
hangover and the gate open while you talk, a clack inside or within half a second after a
phrase goes straight to the converter, and no probation rule can touch it. The fix is an
in-speech de-clicker that uses the 40 ms of right context the engine already buys, plus a
context-based probation rule for pauses. A single 10 ms frame cannot tell a clack from a
/p/ burst (both broadband, ~10 ms); what differs is what follows, so the detector must look
20-30 ms ahead, which the lookahead already pays for. (2) Keep RNNoise; there is no
published evidence DeepFilterNet3 handles keystrokes better, it costs 40 ms instead of 20,
and it has no Python 3.13 wheel. (3) Whisper is a decoder problem (an all-noise excitation
the generator only ever saw on fricatives) before it is a gate problem; kisses, clicks and
breaths are safe to pass through from the aligned mic, **sibilants are not** (they carry
sex at 63-90 % identification); laughs will pull the auto-pitch register down by ~5.6 st
for a few seconds and the 330 Hz ceiling clamps every voiced laugh call, both measurable on
the Phase B take before any code changes.

---

## 0. What the chain does today, and where a clack gets in

The user runs (DEVIATIONS §14 step 0): **hop 80 ms**, extra 2400, `extra_right_ms` 40,
gate −28 dBFS with Silero, hangover **500 ms** (14.7), RNNoise on (mix 1.0), auto level on
with +6 dB trim, Mom +12, protect 0.35, ceiling 330 Hz, formant −0.5, index off (inert).
Chain per block (`audio/conditioning.py`): 70 Hz HPF → RNNoise (20 ms delay) → AGC
(target −20 dBFS, ≤ +18 dB, rise 400 ms / fall 60 ms, frozen below floor + 6 dB) → trim →
soft limiter → gate (`engine/gate.py::decide`).

**Three ways in, by likelihood:**

1. **Gate already open.** Typing while talking, or within the 500 ms hangover after a
   phrase: the block is speech by hysteresis (`loud and voiced`, or `_remaining > 0`), the
   clack is in the window, ContentVec sees it (and, per Seltzer et al., treats a keystroke
   as "two closely spaced noiselike impulses" that look like consonant bursts), and the
   decoder renders a consonant that was never said. **No onset rule applies here.** This is
   the path that matches "typing after I speak leaks".
2. **Energy-only probation in a pause.** `strong` needs block RMS ≥ threshold + 6 =
   **−22 dBFS over the whole 80 ms block** and ≥ 12 dB above the quietest block of the last
   200 ms, and (flag, default 1) a rise persisting `onset_persist_chunks` VAD chunks. A
   10 ms clack reads ~9 dB lower as an 80 ms RMS **[derived]**, so it needs a 10 ms RMS of
   about −13 dBFS *after* the chain to open the gate: AGC/trim up to +24 dB means a raw
   clack of about −37 dBFS/10 ms suffices. Plausible at 30-60 cm on an MV6 **[unverified:
   no measurement of this keyboard exists]**. Once open, `ONSET_PROBATION_MS = 100` is
   `ceil(100/80) = 2` hops = **160 ms** rendered before the VAD can close it.
3. **Silero says speech.** 32 ms chunks, `vad_reduce = "max"` over the 2-3 chunks a hop
   holds. Silero's default `min_speech_duration_ms` is 250 and its documented behaviour on
   clicks is not published (ESC-50 has keyboard/mouse-click classes but no per-class result,
   [Silero quality wiki](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics)). The
   repo's own §14.4 proxy shows every phrase opened with the VAD and no synthetic transient
   did, so this path is the least likely, but it is unmeasured on a real clack.

**What RNNoise does to it.** §14.4 measured RNNoise removing synthetic bumps/taps/creaks
(−6…−14 dBFS/10 ms dry → −54…−81 after). RNNoise is a 22-band, per-10 ms gain model
trained on "computer fans, office, crowd, airplane, car, train, construction"
([Valin 2017](https://arxiv.org/pdf/1709.08243)); its README offers a *foreground noise*
file "e.g. keyboard sounds" for training, but whether the shipped model used one is not
stated ([xiph README](https://raw.githubusercontent.com/xiph/rnnoise/main/README)). A
10 ms broadband thump with most energy below ~1 kHz (keystroke energy is "within the lower
frequencies", [Harrison 2023](https://arxiv.org/pdf/2308.01074); 0.4-12 kHz,
[Zhuang 2005](https://www.cs.utexas.edu/~shmat/courses/cs6431/zhuang.pdf)) overlaps the
speech bands, and a band-gain model has little reason to zero it inside speech. So the
clacks that leak are the ones that arrive while RNNoise's gains are open for speech — path
1 again.

**Argon's "AGC boosts the click" (corrected).** The AGC is slow (400 ms rise, 60 ms fall,
frozen in pauses): it does not react to a click, it applies the *current speech gain* to
everything. A clack is amplified exactly as much as the voice around it. The fix is not a
faster AGC (that would pump on syllables) but a de-clicker *before* the gain, so the clack
never reaches speech level.

---

## 1. Keystroke and transient rejection (`gate.py`, `conditioning.py`)

### 1.1 What a keystroke sounds like (sourced), and the AULA WIN60HE

| property | value | source |
|---|---|---|
| structure | two impulses per key: push (louder) and release, ~90-100 ms apart | [Asonov & Agrawal 2004](https://web.eecs.umich.edu/~genkin/teaching/fall2019/EECS598-12_files/kbdacoustic.pdf); [Zhuang 2005](https://www.cs.utexas.edu/~shmat/courses/cs6431/zhuang.pdf); [Compagno 2017](https://www.arxiv.org/pdf/1609.09359) |
| push-peak active part | analysed in an 8-10 ms window; two 2-3 ms sub-events (touch, hit); "even 20 ms samples" carry the discriminative energy | Asonov; Compagno |
| spectrum | energy "mainly between 400 Hz and 12 kHz"; "majority of features … within the lower frequencies"; 300-3400 Hz band already classifies keys; structural plate modes at ~60 and ~125 Hz (rubber dome, 5 cm) | Zhuang; Harrison; Asonov; [Rustighi & Joseph](https://eprints.soton.ac.uk/372566/1/_soton.ac.uk_ude_personalfiles_users_erustigh_mydesktop_Paper_2001.pdf) |
| frame view | with 20 ms frames / 10 ms shift a keystroke "typically lasts three frames"; treated as spectrally flat by the de-keystroke system | [Seltzer, Acero, Droppo 2006](https://www.microsoft.com/en-us/research/wp-content/uploads/2007/05/2006-mseltzer-interspeech.pdf) |
| crest factor | **not found** in any keystroke paper (Argon's "> 15 dB" is unsourced) | — |
| Hall-effect / WIN60HE | no acoustic measurement exists. Linear HE switch: no tactile leaf or click bar, so the only impulses are stem bottom-out and top-out; the MAX variant review calls it "on the quieter side thanks to the internal dampening" **[unverified, review]** | [xiaomitoday review](https://xiaomitoday.com/aula-win60he-max-and-win68he-max-review-wired-magnetic-switch-keyboards) |

**Contrast with speech.** A /p/ burst peaks 5-6 ms after release and lasts ~8-10 ms to
−10 dB ([Poor 1995](https://www.coli.uni-saarland.de/Phonetics/icphs/ICPhS1995/13_ICPhS_1995_Vol_3/p13.3_444.pdf));
/t/ bursts peak near 4 kHz, /k/ near 2 kHz. **In one 10 ms frame a plosive burst and a
clack are the same object: broadband, ~10 ms, high crest.** What differs is the next
20-80 ms: a plosive is followed by aspiration and voicing (English aspirated VOT ~80 ms,
tenuis 20-30 ms), a clack by silence and, ~90 ms later, a smaller second impulse. Spectral
flatness is the worst feature for impulsive noise in the VAD literature ("machine gun"
noise 45 % error at 0 dB, [LSFM VAD](https://asmp-eurasipjournals.springeropen.com/track/pdf/10.1186/1687-4722-2013-21)).
So the discriminator Argon sketched (crest factor + HF ratio on the peak sub-frame) is not
supported by the literature as a *single-frame* decision; it is supported as the **onset
flag**, with the decision made on the following frames. Every keystroke-suppression system
that works does this: Seltzer's detector predicts each frame from ±2 frames (20 ms each
way); the keyboard-suppression patent states a frame-based detector needs "look-ahead …
between 20-30 ms" ([US 9,520,141](https://patents.justia.com/patent/9520141)); the Technion
two-stage OM-LSA used a 4 ms transient estimator plus a 40 ms anti-causal window for
+4.5 dB SegSNR on keyboard ([Hirszhorn et al. 2012](https://webee.technion.ac.il/Sites/People/IsraelCohen/Publications/IWAENC2012_Hirszhorn.pdf)).

### 1.2 The engine already owns the lookahead a detector needs

`extra_right_ms = 40`: the kept region is rendered 40 ms behind the window's end, so the
newest 40 ms of the 16 kHz ring and the newest 40 ms of `in_tail` have been uploaded but
**not yet converted**. A transient detected in block n can therefore be acted on before any
of its samples reach the decoder's kept region, with **zero added latency**, as long as the
decision is made within those 40 ms (two 20 ms hops at hop 20; inside one hop at hop 80).
This is the design freedom neither Argon nor the Phase B prompt used; it makes "detect
then gate/reconstruct with 20-40 ms look-ahead" (the literature's recipe) free here.

### 1.3 Design A: context-based probation (pauses)

Host side, in `SilenceGate.decide`, replacing the single-frame `strong`:

1. **Onset flag** (cheap, per block, on 10 ms sub-frames of the conditioned block): sub-frame
   RMS rise ≥ 12 dB over the 200 ms background (today's rule) **or** high-frequency content
   / first-difference energy ratio jump (HFC, "impulsive noise in time → wide band in
   frequency", [Bello et al. via NYU notes](https://bpb-us-e1.wpmucdn.com/wp.nyu.edu/dist/8/1691/files/2018/01/3-novelty.pdf)).
   Crest factor can be logged for calibration; **do not threshold on it until the user's
   clack take gives a distribution** (none exists in print).
2. **Confirmation, not opening**: the flag starts a 30 ms confirmation window (three 10 ms
   sub-frames), during which the gate stays closed and the samples sit in the lookahead.
   Open only if (a) sub-frame energy has **not** decayed by ≥ 8 dB from its peak within
   20 ms (a clack does; a plosive's aspiration/voicing holds), **or** (b) Silero's
   probability on the next completed chunk rises ≥ 0.25 **[thresholds to calibrate]**.
   This is Phase B's `onset_persist_chunks = 2` (which stopped the 25 ms taps and 60 ms
   bumps, not the 150 ms creaks, §14.4) made frame-rate and decay-aware instead of
   chunk-count.
3. **Pair check (optional)**: a second, smaller impulse 60-120 ms after a rejected one is a
   key release; extend the rejection window by one block. Costs nothing.
4. **Cost**: the energy-only open today has lag 0-5 ms (§14.4, thanks to the lookahead);
   with confirmation it becomes 20-30 ms **only for energy-only onsets** (every phrase in
   the §14.4 proxy opened through the VAD, lag 0). Soft entrances are unaffected: they open
   on the VAD, not on `strong`.

Acceptance (on the user's leak take, when recorded): leaked clacks per 60 s → 0 in pauses;
lost speech per 60 s unchanged (§14.4 baseline 200-270 ms); onset lag p95 ≤ 30 ms.

### 1.4 Design B: in-speech de-clicker (the main leak)

Where: `conditioning.py`, **after the HPF and before RNNoise and the AGC**, so the clack is
tamed before any gain and before RNNoise's gains see it. What: a level-independent
fast/slow envelope detector (the "transient designer" principle: two followers, the
difference drives the gain, "the idea of setting a level threshold … is meaningless",
[SPL](https://barryrudolph.com/mix/transient.html)) with a **look-ahead clamp**:

- E_fast (τ ≈ 1-2 ms) and E_slow (τ ≈ 20-30 ms) on the HPF'd block.
- Candidate when E_fast/E_slow ≥ 4 (+12 dB) **[Argon's number; plausible, unverified]**.
- **Decision 20 ms later** (the lookahead): if the candidate's energy fell back to within
  6 dB of E_slow inside 15-20 ms **and** no sub-frame in the 20 ms after it is above the
  speech level, it is a click → attenuate those 10-20 ms toward 2·E_slow with 2 ms ramps
  (a soft clamp, not a zero: Seltzer found full-frame deletion with naive reconstruction
  *worse* than doing nothing, DMOS −1.92; MAP reconstruction +1.77). Otherwise leave it.
- Because the decision uses 20 ms of future and the samples are still inside the 40 ms
  lookahead, the attenuation is applied **retroactively to the ring and `in_tail`** before
  the window is next consumed; nothing is delayed. At hop 80 the whole block is
  available at once, so the retroactive write is trivial; at hop 20 it spans two hops.
- Plosive safety: a /p/ burst also passes step 1, but its following 20 ms hold aspiration
  or voicing, so it fails step 2 and is untouched. Risk: a word-final unreleased stop
  followed by a pause looks like a click; the clamp toward 2·E_slow (not zero) bounds the
  damage to a few dB on a burst nobody hears anyway.

Acceptance: on the leak take, leaked clacks *inside speech* → audibly gone (ABX against
untreated), consonant WER unchanged (Whisper small.en moves 6-9 % between renders of the
same take by seed, §14.4, so pair it with the ear), sibilant centroid unchanged.

### 1.5 What not to do

Spectral-flatness thresholds (worst feature for impulsive noise); a faster AGC (pumping);
raising the HPF above 90 Hz (the user's F0 floor is ~100 Hz; §14.4 found the 90 Hz
high-pass stopped nothing); "max" suppression as the keystroke fix (it costs 150 ms of
speech per minute and RNNoise still passes in-speech clacks, §14.4).

---

## 2. Denoiser: keep RNNoise, add the detector, re-evaluate DFN3 only through ONNX

| method | algorithmic latency | cost | stationary | keystrokes | Windows / py3.13 | licence |
|---|---|---|---|---|---|---|
| RNNoise (current) | **20 ms** (10 ms hop + 10 ms look-ahead, [Valin demo](https://jmvalin.ca/demo/rnnoise/)); matches the repo's measured 20 ms | ~1.3 % of one Haswell core ([paper](https://arxiv.org/pdf/1709.08243)); Argon's "0.39 ms/frame" not found | yes | weak by §14.4; in-speech clacks pass | pyrnnoise 0.4.5 win_amd64 wheel | BSD-3 |
| **Host-side click detector + de-clicker (§1.3-1.4)** | 0 added (uses the 40 ms lookahead) | negligible | — | the only approach in the literature that uses the context speech has and clacks lack | trivial | — |
| DeepFilterNet3 (torch) | **40 ms** (20 ms window, 10 ms hop, 2-frame look-ahead, [DFN3](https://arxiv.org/pdf/2305.08227)) | RTF 0.19 single-thread i5-8250U (DFN2: 0.04) | better than RNNoise on STOI/PESQ (0.810 vs 0.743 on ESC-50 ambient, [comparison](https://publikasi.mercubuana.ac.id/index.php/sinergi/article/download/21917/9619)) | **no evidence either way**; not evaluated on typing in any paper | `deepfilterlib` wheels cp38-cp311 only ([PyPI](https://pypi.org/project/deepfilterlib/#files)); needs a Rust build for 3.13 | MIT/Apache-2.0 |
| DFN3 via C runtime / ONNX ([deepfilternet-slim](https://pypi.org/project/deepfilternet-slim/), [soniqo ONNX](https://huggingface.co/soniqo/DeepFilterNet3-ONNX)) | 40 ms | onnxruntime CPU | yes | no evidence | py3-none-win_amd64 wheels exist; the ERB/spec front end and streaming state are yours to write | MIT |
| GTCRN (ONNX) | ≥ 32 ms (512/256 at 16 kHz) | RTF 0.07 i5-12400 | yes | no evidence; **16 kHz only** | onnxruntime | MIT |
| OM-LSA + transient stage (Technion 2012) | +40 ms | low | yes | +4.5 dB SegSNR on keyboard | no code found; 16 kHz | — |
| NVIDIA Broadcast | not published | the RTX GPU the engine runs on | yes | reviewer: clicks "disappeared like magic" on a clicky board **[anecdotal]** ([Tom's Hardware](https://www.tomshardware.com/news/nvidia-broadcast-tested)) | installed | proprietary |

Verdict: **keep RNNoise** for room tone; the keystroke problem is solved by context, not by
a bigger band-gain model. If the user wants to try the strongest off-the-shelf typing
suppressor, it is the Broadcast mic they already have (report 15's caveat stands: it
degrades SIG and carves harmonics, and it shares the GPU). Argon's latency and cost table
is directionally right (20 vs 40 ms) but its CPU numbers and "PESQ 3.5-4.0+" are not in
the papers.

---

## 3. Paralinguistics

With the index off, every non-speech sound passes Silero → ContentVec → RMVPE +
RegisterTracker → ceiling → NSF-HiFi-GAN (SineGen: sine 0.1 + noise 0.003 when voiced;
**noise only, at 0.1/3, when F0 = 0**) → formant. Classified by which stage breaks it.

### 3.1 Whispers and breaths

**Acoustics.** Whisper: no F0; F1 shifts up 39 % (men) / 31 % (women), F2 up 11 %, F2 as
intense as F1 ([Sharifzadeh et al. 2010](https://www.lintech.org/webpapers/JVoice3Revised1.pdf)).
Speech inhalations: 410-535 ms, 11-14 dB quieter than /h/, spectral peaks ~0.5, 1.45-1.85
kHz ([Werner et al. 2021](https://www.coli.uni-saarland.de/~trouvain/docs/Werner_et_al_2021b.pdf),
[2023](https://www.coli.uni-saarland.de/~moebius/documents/werner_etal_jslhr2023.pdf)).

**Gate.** Silero on whisper: not evaluated anywhere; Praat's manual says it detects
"unvoiced parts of speech as well as voiced parts"; a phonated-speech VAD on whispered
ASMR fell to F1 75-81 % vs 91-96 % for a whisper-trained one ([Rao et al. 2023](https://arxiv.org/pdf/2303.07442)).
The likelier blocker is **the energy term**: a whisper at the user's mic sits far below the
−28 dBFS threshold until the AGC lifts it, and the AGC is frozen below floor + 6 dB.
Measure first: whisper 20 s with the Record button, read `rms_dbfs` and `vad_prob` per hop
from the take JSON. If energy is the blocker, a "Whisper" input mode (lower gate threshold,
AGC target raised, Silero threshold 0.35) is a config change.

**Features.** Content survives in deep SSL layers after fine-tuning (WavLM on wTIMIT 9.3 %
WER, [arXiv 2407.21211](https://arxiv.org/html/2407.21211v2); WESPER's unit distance
whisper↔normal shrinks with depth, [Rekimoto 2023](https://ar5iv.labs.arxiv.org/html/2303.01639)),
but a zero-shot VC roughly doubles its error on whisper (Seed-VC CER 46 %,
[WhisperVC](https://arxiv.org/html/2511.01056v2)). Expect intelligible but degraded
ContentVec features.

**Decoder: the real wall.** With F0 = 0 on every frame the generator gets an all-noise
excitation it only ever saw on fricatives and silence; the NSF authors report single-branch
NSF "may produce low-quality unvoiced sounds or sometimes silence for fricative consonants"
([Wang, Takaki, Yamagishi 2019](https://arxiv.org/pdf/1904.12088v2.pdf)). No aperiodicity
or breathiness control exists in RVC. **Every working whisper→voiced system predicts an F0
rather than extracting one** (WESPER, WhisperVC, Pascual et al.). Two routes:

- *Whispered output, no retraining*: a "whisper passthrough" — the aligned conditioned mic,
  formant-warped by the existing decode-stretch-resample path (~+1.27× sibilant centroid,
  §3.2) and level-matched, mixed in **instead of** the decoder on frames where F0 = 0 for
  ≥ 300 ms and the band energy is whisper-like. Keeps the breath texture; the voice stays
  "a whispering woman" only as far as the formant warp carries it. Cheap, testable, honest
  about what it is.
- *Voiced output from a whisper*: drive SineGen with a synthetic F0 from the tracker's last
  register (hot device scalar, branchless) during whispered stretches **[untested anywhere
  on NSF-HiFi-GAN]**; the ContentVec features of whisper then meet a voiced excitation,
  which is what WESPER's decoder learns to do, but RVC's never did. A/B only.
- *Retrain* with whispered data in the target set is the Phase 2 answer and needs Q15.

### 3.2 Kisses, lip smacks, tongue clicks (and why sibilants are different)

**Acoustics.** No measurement of a kiss/lip smack exists in print (iZotope describes them
only qualitatively); the nearest analogue, phonological clicks: alveolar transient ~2 ms
(peak ~2 kHz), dental ~12 ms (peak ~6 kHz), lateral ~23 ms ([Demolin et al. 2023](https://guarant.cz/icphs2023/916.pdf)).
A bilabial smack is a < 30 ms event: **shorter than one 20 ms ContentVec frame**, so the
feature stream cannot represent it; the decoder renders whatever consonant the frame's
features suggest.

**Kiss vs keystroke.** Both are < 30 ms broadband transients; by §1.1 they cannot be told
apart in one frame either. The difference is context again: a kiss happens with the gate
open (you are "talking"), is usually preceded by a lip closure (a dip) and followed by
breath or a vowel ("mwah"), and is at mouth distance, not desk distance (louder,
mid-band). A keystroke in a pause fails the §1.3 confirmation; a keystroke during speech
is clamped by §1.4; **a kiss during speech is also clamped by §1.4 unless the passthrough
below claims it first**. So the passthrough detector must run before the de-clicker's
decision and mark the event as "mouth": high-passed level ≥ speech level (close source)
and no second impulse 90 ms later **[heuristic, to calibrate on a take of kisses + typing]**.

**The passthrough (PostChain).** `pipeline.py` already keeps `in_tail`, the device-rate
conditioned mic sample-aligned under the decoded tail (it is the RMS-mix reference), so a
mix costs one band-pass and one gain per hop, graph-safe:

    out = rvc_out + w · g(t) · BP(in_tail, f_lo, f_hi)

with `g(t)` a 0-dim-device smoothed weight that ramps toward 1 on frames flagged unvoiced
(`pitchf == 0`) **and** event-like (high-band energy burst ≤ 30 ms, or breath-like
sustained band energy with centroid < 2.5 kHz), else toward 0; `w` ≈ 0.15-0.3 to taste.

**What must not pass through: sibilants.** The > 4 kHz band carries sex strongly. Adult
/s/ centroid **6277 Hz men vs 7946 Hz women**, /ʃ/ 4124 vs 5016
([Romeo, Hazan & Pettinato 2013](https://education.umd.edu/media/16240/download)); listeners
identify speaker sex from isolated /s, ʃ/ at **63-90 %** (Schwartz 1968 via
[Pépiot 2017](https://ntffpu.uni-plovdiv.bg/wp-content/uploads/2025/01/006_Erwan-Pepiot-ред.pdf));
sex ID stays above chance for speech high-passed at 8.5-12 kHz
([Donai & Halbritter 2017](https://sonicfield.org/library/gender-identification-using-high-frequency-speech-energy-effects-of-increasing-t));
a higher /s/ centroid makes men sound "less masculine" (Campbell-Kibler via
[Levon et al. 2017](https://boris.unibe.ch/152294/1/Levon_Maegaard___Pharao_2017.pdf)).
So Argon's "oral frication above 4 kHz carries almost zero male cues" is **wrong**; a raw
passthrough of the user's /s/ re-masculinises the voice. Rules: pass **breaths** (peaks
≤ 2 kHz, 400+ ms) and **clicks/smacks** (≤ 30 ms) raw; **never pass sustained energy with a
centroid > 4 kHz** (sibilants), or pass it only through the formant resample at ≈ 1.27×
(7946/6277 **[derived]**), which the formant path can do at no latency. The decoder's own
sibilants are the model's; leave them.

### 3.3 Laughs: what to measure on the Phase B take, and the branchless fix

**Acoustics** (Bachorowski, Smoski & Owren 2001 via [Makagon 2006](https://ecommons.cornell.edu/bitstream/handle/1813/2175/MakagonThesis2006.pdf?sequence=1) and
[the Bonds review](https://arxiv.org/pdf/1010.5602); the JASA full text was behind a CAPTCHA):
bouts just under 1 s; 3.4-4.4 calls per bout at **~4.4 calls/s**; calls 110-170 ms with
~120 ms gaps; **30 % fully voiced, ~48 % unvoiced, 22 % mixed**
([Trouvain 2003](https://www.coli.uni-saarland.de/~trouvain/trouvain_2003_icphs.pdf)); men
produce mostly unvoiced "grunt" laughs; **male voiced-laugh F0 272-282 Hz mean (range
43-898 Hz), ≈ 2.5× male speech F0**; conversational laughter lower (171 Hz male, Vettin &
Todt); acted male laughs 199 Hz, with F1 up to 1300 Hz, i.e. vowels outside speech space
([Szameitat 2007](https://www.coli.uni-saarland.de/conf/laughter-07/files/SZAMEITAT_ET_AL.pdf)).
Laughter carries little speaker identity (verification EER 23 % laugh-laugh vs 0.8 %
speech, [Lin et al.](https://arxiv.org/pdf/2210.16028)). Speech-trained generative models
given a laugh do not reproduce it (ELaTE baseline Laughter-SIM 0.32 vs 0.80 conditioned,
[arXiv 2402.07383](https://arxiv.org/html/2402.07383v1); Hsu 2022: speech-only HuBERT
units "suboptimal" for NSVs, [arXiv 2206.12662](https://ar5iv.labs.arxiv.org/html/2206.12662)).
RMVPE/CREPE on laughter: **no measurement exists**.

**Predicted effects, to confirm on the take [derived]:**
1. **Tracker contamination.** `RegisterTracker` moves its log2 estimate toward each hop's
   mean voiced F0 by 1 − exp(−voiced_frames/200). One second of voiced laugh at 272 Hz on a
   ~110 Hz register is +1.3 oct; the tracker moves ≈ (1 − e^(−0.5)) · 1.3 ≈ **0.5 oct**, so
   the transposition drops ≈ **6 st** and decays back over the next 2-4 s of speech: the
   voice comes back "too low" after a laugh. Metric: `offset_st` per hop across the bout and
   the 3 s after (the Phase B harness `b2_laugh.py` reports the register move; add its
   recovery time).
2. **Ceiling.** 272 Hz × 2^(12/12) = 544 Hz, or ≈ 2.5× the 195 Hz target under auto pitch
   ≈ 490 Hz: every voiced call lands above the 330 Hz knee and is bent to a near-constant
   pitch. Natural female laugh F0 is 405-476 Hz, so the ceiling is *below* a natural laugh.
   Metric: output F0 p90 over laugh pulses vs the knee; count of frames with ceiling
   engagement.
3. **V/UV flicker.** Calls alternate voiced/unvoiced every ~115 ms (≈ 4.4 Hz); mixed calls
   (22 %) will sit at RMVPE's 0.03 voicing threshold. Metric: voicing flips per second
   inside the bout vs speech; unvoiced share of pulses (the harness reports it); whether
   `f0_trim_tail = 2` delays each call's voicing onset by its 20 ms.
4. **Gate.** A 500 ms hangover bridges 120 ms gaps; unvoiced "grunt" bouts may fail Silero
   (laughter is "speech absent" in VAD corpora, [Amazon VAD review](https://cdn.amazon.science/61/88/7d000a8041e1a0fd4a92976b1e86/a-comprehensive-empirical-review-of-modern-voice-activity-detection-approaches-for-movies-and-tv-shows.pdf)).
   Metric: gate closes inside bouts (the harness already counts them).

**Branchless, graph-safe fixes, once confirmed** (all hot device scalars inside the `f0`
stage; no op-sequence change):
- Tracker: weight each hop's update by `w = σ((Δ_max − |log2 f0 − level|)/s)` with Δ_max
  ≈ 0.75 oct (laugh F0 ≥ 2× speech per B2001; conversational laughs at 1.5× will leak,
  accepted), and multiply by a "flicker guard" `1 − σ((flips_per_s − 3)/1)` computed from
  the pitch cache's voicing run lengths. Both are `torch.sigmoid` on existing tensors.
- Ceiling: lift the knee for the duration of a flagged bout (same flags → `f0_ceiling_t`
  interpolated toward 600 Hz with a one-pole), so laugh calls keep their excursion; Mom at
  500+ Hz may crack (SiFi-GAN: HiFi-GAN+Sine MOS 2.69 at 2× F0, report 15) — the A/B decides
  between "clamped laugh" and "cracked laugh".
- Typed: no backend has a laugh tag (§14.2); "hahaha!" was the best spelling and still
  "needs improvements". The expressive route is a laugh *bank*: 3-5 of the user's own
  converted laughs (recorded, converted whole-file, polished) inserted by `[laugh]` through
  `override.py` like a typed clip. Zero research risk; the only honest fix until a TTS
  with laughter conditioning (ELaTE-style) is in the bake-off.

---

## 4. Ranked list of concrete changes

| # | change | evidence | effect | latency | GPU/hop | retrain | files |
|---|---|---|---|---|---|---|---|
| 1 | **Record the three takes**: 60 s talking + typing on the WIN60HE (in speech and in pauses), 20 s laughing + talking, 20 s whispering + a few kisses/smacks | nothing in print measures this keyboard, RMVPE on laughter, or Silero on whisper | decides everything below | — | — | no | Record button; `b2_laugh.py`, `b4_leaks.py` |
| 2 | **In-speech de-clicker with retroactive lookahead clamp** (§1.4) | leak path 1 is open-gate; literature: detect-then-gate with 20-30 ms look-ahead; the 40 ms right context is already paid | removes the main leak | 0 | 0 (host) | no | `audio/conditioning.py`, `engine/pipeline.py` (retroactive write into `ring_16k` / `in_tail`), `engine/worker.py` |
| 3 | **Context-based probation** (§1.3): onset flag → 30 ms confirmation on decay + VAD rise | §14.4: persist-2 stopped taps/bumps, not creaks; Seltzer/patent/Technion look-ahead | removes pause leaks incl. creaks | +20-30 ms on energy-only onsets only | 0 | no | `engine/gate.py`, `tests/unit/test_gate.py` |
| 4 | **Laugh metrics on the take** (§3.3): register move + recovery, ceiling engagements, flips/s, gate closes | B2001 numbers predict ≈ 6 st undershoot and total clamping | measurement | — | — | no | `benchmarks/b2_laugh.py` |
| 5 | **Tracker laugh guard + ceiling lift** (branchless) | §3.3, after #4 confirms | laughs keep pitch; speech after a laugh stays on register | 0 | ~0 | no | `pitch/base.py` (`RegisterTracker.ratio`), `engine/pipeline.py` `_stage_f0` |
| 6 | **Breath/click passthrough in PostChain** (§3.2), sibilant-safe (centroid gate or 1.27× warp) | `in_tail` is already aligned; sex-cue data for /s/ | kisses, smacks, breaths sound human | 0 | ~0.1 ms | no | `engine/postchain.py`, `pipeline.py` (voicing + band flags), `synthesis/formant.py` (warp reuse) |
| 7 | **Whisper measurement** (energy vs VAD) then a Whisper input mode | §3.1 | whispers reach the model | 0 | 0 | no | `api/controller.py` presets, `engine/gate.py` |
| 8 | **Whisper passthrough** (formant-warped mic on long unvoiced stretches) | §3.1 | whispering works as whispering | 0 | ~0.2 ms | no | `postchain.py`, `formant.py` |
| 9 | **Laugh bank for typed `[laugh]`** | §14.2 verdicts; ELaTE | convincing typed laughs | — | — | no (recordings) | `tts/`, `engine/override.py` |
| 10 | Synthetic-F0 voiced whisper (A/B only) | §3.1 [untested] | maybe a voiced whisper | 0 | 0 | no | `pipeline.py` `_stage_f0` |
| 11 | DeepFilterNet3 via ONNX, only if ambient noise becomes the complaint | §2 | better stationary SNR, +20 ms | +20 ms | CPU | no | `audio/suppression.py` |
| 12 | Retrain with whispered + laughed target data | §3.1, §3.3 | the proper fix for both | 0 | 0 | **yes** (Q15) | training stack |

**Do first:** 1, then 2 + 3 (one session, both flags, A/B on the typing take), 4 + 5 on
the laugh take, 6 on the kiss take. Items 7-9 after. Nothing here changes the hop, the
graphs' op sequence or the model.

---

## Sources

Repository: `main` @ 5e3f5b2 — `engine/gate.py`, `audio/conditioning.py`,
`audio/suppression.py`, `engine/postchain.py`, `engine/pipeline.py`, `pitch/base.py`,
`synthesis/nets/generator.py`, `docs/DEVIATIONS.md` §12-§14.

Keystrokes and detectors: Asonov & Agrawal 2004; Zhuang, Zhou & Tygar 2005; Compagno et
al. 2017 (arXiv 1609.09359); Harrison et al. 2023 (2308.01074); survey 2309.11012;
Rustighi & Joseph (ISVR); Seltzer, Acero & Droppo 2006; US 9,520,141; US 10,242,696; Poor
1995; Bello et al. 2005 (via NYU notes); LSFM VAD (EURASIP 2013); SPL Transient Designer;
Essentia ClickDetector; Godsill & Rayner; Hirszhorn et al. IWAENC 2012; Talmon et al. 2011,
2013; Cohen & Berdugo 2001; DNS Challenge 2021/2022.

Denoisers and VAD: Valin 2017 (1709.08243), RNNoise demo/README/W3C talk, pyrnnoise;
DeepFilterNet2 (2205.05474), DeepFilterNet3 (2305.08227), PyPI deepfilternet /
deepfilterlib / deepfilternet-slim, soniqo DFN3 ONNX, DFN3-vs-RNNoise study (Sinergi);
GTCRN; DeepVQE (2306.03177); Fast-ULCNet (2601.14925); Tom's Hardware on NVIDIA Broadcast;
Silero VAD README / utils_vad.py / releases / quality wiki; Praat Silero manual; Rao et al.
2023 whisper VAD (2303.07442); Gillick et al. 2021; Amazon VAD review.

Paralinguistics: Bachorowski, Smoski & Owren 2001 (via Makagon 2006, Bonds review
1010.5602, ScienceDaily); Trouvain 2003; Szameitat et al. 2007; Truong & van Leeuwen 2007;
Lin et al. 2210.16028; Hsu 2022 (2206.12662); Kreuk et al. (2111.07402); ELaTE
(2402.07383); Setoguchi & Arimoto 2025; Mori & Kimura (2306.03465); NonverbalTTS
(2507.13155); Affectron (2603.14432); Sharifzadeh et al. 2010; WavLM whispered ASR
(2407.21211); WESPER (2303.01639); WhisperVC (2511.01056); Pascual et al. (1808.10687);
NSF (1904.12088); uSFGAN-HN (2205.06053); Werner et al. 2021, 2023; Demolin et al. 2023;
iZotope Mouth De-click; VocalSound (2205.03433); VCC2018 (1804.04262); Huang et al. 2019
(1907.11898); Romeo, Hazan & Pettinato 2013; Fuchs & Toda 2010 (via Mays 2010); Levon et
al. 2017; Ford & Tabain 2019; Pépiot 2017; Donai & Halbritter 2017; Jacewicz, Fox & Holt
2023; RVC `en_US.json` (protect).

Not found / unverified: any acoustic measurement of the AULA WIN60HE or of Hall-effect
switches; keystroke crest factor; RMVPE/CREPE on laughter; Silero on whisper or clicks
per class; lip-smack acoustics; within-call laugh F0 slew; the Bachorowski 2001 full text;
Argon's RNNoise CPU and DFN3 PESQ figures; whether xiph's shipped RNNoise model used the
keyboard foreground set; RVC/so-vits issues on whisper/laugh (GitHub search blocked).
