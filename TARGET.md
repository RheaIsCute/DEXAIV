# TARGET.md — the only supported machine

This project is tuned for exactly one computer. Do not add compatibility paths,
device-capability branching, or conservative defaults for hardware not listed here.
Every benchmark number in docs/ was or must be measured on this machine.

## Hardware (verified 2026-10-13)

| Component | Value |
|---|---|
| OS | Windows 11 Pro 26200 |
| CPU | AMD Ryzen 9 7900X · 12c/24t |
| RAM | 32 GB |
| GPU (compute) | **NVIDIA GeForce RTX 5060 Ti · 16311 MiB · compute capability 12.0 (Blackwell)** · driver 617.14 |
| Other display adapters | AMD Radeon iGPU (7900X), AskLink virtual display — never compute targets |
| Python | 3.13.15 |
| Torch | 2.7.1+cu128 (pinned) |
| Disks | KIOXIA 954 GB NVMe (primary), ADATA LEGEND 700 477 GB |

## Audio endpoints present

- Realtek HDA (onboard) — mic input lives here unless a USB mic is attached
- **Virtual Audio Cable (VAC)** — the installed virtual cable. All docs/prompts that
  said "VB-Cable" mean VAC on this machine; the output device is the VAC input line.
- NVIDIA Broadcast — present. Its denoised virtual mic is an available input; if the
  user selects it, the app's own Suppression toggle should default OFF (double
  denoising hurts). The engine's RNNoise path remains for the raw-mic case.
- AMD/NVIDIA HDMI audio endpoints — ignore.

## Rules this implies

1. Compute device is `cuda:0` = the RTX 5060 Ti, asserted at startup
   (name contains "5060 Ti", cc (12,0), VRAM ≥ 15 GiB, torch cu128). Hard error
   otherwise; CPU path exists only for unit tests behind an env flag. The iGPU and
   virtual display adapters must never be auto-picked for anything.
2. Blackwell defaults: TF32 on; HuBERT FP16/BF16 autocast are default candidates once
   they pass the eager-match check; no FP32-only conservatism kept for older cards.
3. Memory guards sized to 16 GiB: index load caps, buffer pools, and warmup budgets
   assume ~12 GiB usable for the engine after Windows/driver overhead.
4. Fixed shapes per profile → run cuDNN autotune once at capture warmup and persist
   the result with the profile; startup must not re-benchmark.
5. Power state: nvidia-smi shows P1/2700 MHz under real load, but tiny-kernel streams
   were observed at P3/~1860 MHz (DEVIATIONS R2). Test "Prefer maximum performance"
   for the Python process and record the per-hop delta (open question Q16).
6. Audio defaults: input = user's mic endpoint, output = VAC input line, monitoring
   via a second output stream to headphones. 48 kHz pinned end to end on the cable.
7. 12 physical cores: the inference worker, audio callbacks, and GUI get distinct
   cores' worth of headroom; no need to economize threads, but keep the hot path on
   one worker as designed.
