# Pilot experiments

Run on 2026-11-10 on an Apple M4 Pro (24 GB, MPS). Pilots A, B and D use `segmind/tiny-sd`
with DDIM and guidance 7.5, measured against the same-seed 50-step image. Pilot C uses
Stable Diffusion 1.5 with LCM-LoRA.

| Pilot | Script | Output | Split |
|---|---|---|---|
| A1 phase attribution | `scripts/pilots/a1_phase_attribution.py` | `results/pilots/a1_phase_attribution.json` | 24 dev |
| A2 step allocation | `scripts/pilots/a2_step_allocation.py` | `results/pilots/a2_step_allocation.json` | dev, then 72 sealed |
| B block clocks | `scripts/pilots/b_block_clocks.py` | `results/pilots/b_block_clocks.json` | dev, then 72 sealed |
| C adapter anatomy | `scripts/pilots/c_lora_anatomy.py` | `results/pilots/c_lora_anatomy.json`, `.png` | 16 dev |
| D compass search | `scripts/pilots/d_compass_search.py` | `results/pilots/d_compass_search.json` | 8 dev, then 72 sealed |

Run any of them with `.venv/bin/python scripts/pilots/<name>.py`. They take 10 to 25
minutes each. Pilot C downloads a 135 MB adapter on first use.

## How to read the numbers

- **Texture (T):** mean absolute deficit in the 1/16-1/4 cycles-per-pixel bands, in dB.
- **LPIPS (L):** perceptual distance to the reference.
- **Composition:** RMS difference after a Gaussian blur with sigma 16, on a 0-255 scale.
- **Score:** `T/T_base + L/L_base`, where 2.0 is unedited 6-step sampling.
- **Steps matched:** the unedited step count that gives the same value.
- Lower is better for all of them.

## Method notes

- Pilots A, B and D use a hand-written DDIM loop in `src/wspd/lab.py`. It differs from the
  diffusers loop by 0.64/255 mean pixel value on the trailing schedule, because diffusers
  reads the next alpha one index off. Dev numbers therefore show 4.57 for the pilot loop
  and 4.52 for the cached baseline.
- The per-block clock is byte-identical to a single shifted timestep when all blocks share
  one clock.
- Selection rules were written into each script before it ran: lowest dev score, then one
  evaluation on the sealed prompts.
- Brackets in the idea files are 95% paired bootstrap intervals over prompts.

## Limits

- The sealed prompts were already used once by the main experiment and are reused here.
- Pilot C measures closeness to the full adapter's image, not image quality, on 16 prompts
  and one adapter.
- No pilot was repeated with other seeds.
