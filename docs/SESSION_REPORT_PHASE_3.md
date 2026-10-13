# Session report: Phase 3 reordered (CUDA graphs, silence gate, live audio)

2026-10-13. Reference machine: RTX 5060 Ti 16 GB, driver 617.14, Ryzen 9 7900X, Windows 11,
torch 2.7.1+cu128. Test voice `base_f0_48k_v2`, clip `libri_1272.wav`. Every number below is
reproducible from the commands named; raw records are under
`benchmarks/baselines/phase3_2026-10-13/`. Full write-ups: `docs/DEVIATIONS.md` §4 and
`docs/research/14_OPEN_QUESTIONS.md` (Phase 3 results).

## Step 1: per-stage CUDA graphs (accepted)

`vox bench perf --sweep benchmarks/sweeps/perf_graphs.yaml`, RMVPE, throughput mode:

| hop | extra | eager p50 / p99 | graphs p50 / p99 | headroom (graphs) |
|---|---|---|---|---|
| 20 ms | 500 | 46.2 / 118.4 ms | **16.9 / 19.1 ms** | 1.04x |
| 30 ms | 500 | 48.8 / 120.3 ms | **16.2 / 20.8 ms** | 1.44x |
| 20 ms | 1000 | 41.7 / 114.0 ms | **15.3 / 20.7 ms** | 0.96x |
| 30 ms | 1000 | 42.4 / 97.6 ms | **15.3 / 20.1 ms** | 1.49x |

- Target (p50 <= 18, p99 <= 25 at hop 20-30, extra 1000): met at every point.
- Seven graphs: resample_in, F0, ContentVec, index, synth, post, SOLA. Replay vs eager from
  identical state and noise: log-mel L1 0.0006-0.0011. Pool allocations per hop: 0.
- `vox bench accept --graphs on`: all five checks pass (streamed within 2.2x floor at 1.89x,
  no boundary clicks, sync-free, pointer-stable, no allocator growth).
- GPU time per hop under replay: RMVPE 7.9 + ContentVec 3.0 + synth 4.8 + SOLA 0.2 = ~16 ms.
  The hop is GPU-bound now; launch count no longer matters.
- Whole-hop single graph: still hangs, bisected to **any graph that contains both
  ContentVec and the synthesizer** (every other fusion replays fine). Parked under Q16; the
  fusion would save ~0.2 ms.

## Step 2: silence gate (accepted)

`vox bench gate` (hop 20, extra 1000, threshold -48 dBFS + Silero 0.5, hangover 200 ms):

- All six checks pass: gate closes/reopens (7 transitions on the clip), transition click
  ratio 0.00, hop-boundary click ratio 1.01, closed hops 0.41 ms host (0.04 ms GPU: idle),
  speech-frame distance unchanged (0.545 either way).
- Pause-region level: full-file -43.1 dBFS, stream gate off -41.6 (+1.5 dB; +2.8 dB at the
  old 500 ms extra), stream gate on -44.9 (-1.8 dB: gated pauses are digital silence).
- R4 re-measured: level bias at extra 1000 ms is +0.13 nepers gate off, -0.32 gate on
  (Phase 2: +0.16 at 1000, +0.40 at 500). Default is now extra 1000 ms, gate on.
- Unit test `tests/unit/test_gate.py::test_drain_and_onset_are_click_free` streams
  speech-silence-speech through SOLA and asserts no sample step above a sine's own slope;
  `tests/engine/test_engine.py::test_gate_streams_speech_silence_speech_without_clicks`
  does it through the real engine.

## Step 3: live audio

`vox run --model base_f0_48k_v2 --input 22 --output 19 --gate off --minutes 10`
(microphone -> Virtual Audio Cable, WASAPI shared, hop 20, overlap 30, fill target 50 ms):

| | value |
|---|---|
| duration / hops | 601 s / 30,047 |
| output underruns (ring) / PortAudio underflow flags | **0 / 0** |
| refills / drift trims / input overruns / rebuilds | 0 / 0 / 0 / 0 |
| hop compute p50 / p95 / p99, median window | 15.7 / 16.7 / 17.1 ms |
| hop compute p99, worst window | 23.7 ms |
| end-to-end estimate | 143-159 ms |

- **Zero underruns at hop 20 ms: met.** Estimate <= 90 ms: **not met in the shared-mode
  default**; the budget is hop 20 + compute 17 + output ring ~30 + SOLA 40 + devices 44.
- Loopback round trip (`scripts/measure_rtt.py`, file-backed input -> engine -> cable ->
  recorder, speech bursts, gate on), measured median vs estimate:
  shared / overlap 30 / fill 50: **110 ms** vs 122; shared / overlap 20 / fill 40: **88 ms**
  vs 102; **exclusive** / overlap 20 / fill 40: **72 ms** vs 90. The estimate is 13-18 ms
  conservative (within the +-15 ms acceptance for the shared runs; 2.6 ms outside for
  exclusive).
- So the way to <= 90 ms is already in the flags: `--exclusive --crossfade-ms 20
  --fill-ms 40` (estimate 90 with a file-backed input, measured 72). With the real
  microphone the input stream must stay in shared mode: in exclusive mode the Realtek
  microphone delivers 72,000 samples/s at a nominal 48 kHz (DEVIATIONS R9), which the new
  `INPUT RATE` check on the stats line now flags. A 2-minute gated soak with exclusive
  output and shared input ran clean (6,009 hops, 0 underruns, 0 trims); its estimate is
  108 ms because the microphone's 22 ms shared buffer stays in the sum.
- Hot changes (`pitch`, `index`, `protect`, `envelope`, `gate on|off`) apply between hops;
  restart-class changes (`hop`, `extra`, `crossfade`, `f0 rmvpe|fcpe`) rebuild in place in
  ~1.5 s with the gate closed and the input backlog dropped
  (`tests/engine/test_worker.py::test_worker_rebuilds_shape_live_and_keeps_running`).
- The worker reproduces the stream simulator's output (same seed) within log-mel L1 0.08
  (`test_worker_reproduces_the_simulator_stream_without_underruns`): P13/P17 hold.

## Not done / needs the user

- **NVIDIA power state (Q16):** tested with `nvidia-smi -lgc 2000,3090` instead of the
  control panel (the shell had the rights; clocks were reset afterwards): no delta (graphs
  16.3 / 19.1 ms locked vs 16.1 / 20.5 unlocked; eager 41.2 vs 40.9 p50). The control-panel
  setting is not needed.
- **Listening:** `out/accept_full.wav` vs `out/accept_stream.wav` and `out/gate_full.wav` vs
  `out/gate_on.wav` (Q14, and the gate's onsets). The soak ran on a silent microphone.
- **Phase 5 (calibration) needs a 45 s recording of the user's voice** following the script
  in `docs/research/08_VOICE_PERSONALIZATION.md`: sustained vowels, a read passage at
  normal level, the softest and loudest they will use, and ~5 s of room silence, recorded
  through the microphone and chain they will use live, 48 kHz mono WAV, no processing.
  From it: F0 range and median (sets `f0_min_hz` / `f0_max_hz` and the transpose), the
  noise floor and speech level (sets the gate threshold instead of today's -48 dBFS), and
  the VAD threshold.
- **Headphone monitoring:** the soak rendered to the virtual cable so nothing played on
  the speakers; `--output 20` is the Realtek headphone output.
