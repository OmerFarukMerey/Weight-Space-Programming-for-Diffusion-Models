## op2 report

I sealed edit `795a1d69e96c94bb` from `runs/op2/c2.py`; the sealed copy is in `runs/op2/final`. It greatly improves few-step texture statistics on the dev prompts, but only modestly improves LPIPS. It also makes sampling much worse at 24 steps.

### 1. Weights changed
421,120 of 323,384,964 weights (0.13%), in 2 parameters:

- **`conv_out.weight` (11,520 weights):** multiplied by 0.9875, which scales the predicted noise at every t.
- **`time_embedding.linear_1.weight` (409,600 weights, rel L2 change 0.207):** replaced by a fit, so that `W' s(t) ≈ W s(f(t))` for integer t in 0 to 999.
  - `s` is the fixed sinusoidal embedding.
  - `f(t) = t - 40 * rise(t/100) * (1 - rise((t-300)/250))`, with `rise` a clamped smoothstep.
  - For 100 < t < 300 the UNet reads about t-40. At t ≥ 550 it is effectively unchanged.
  - The fit is ridge-regularized weighted least squares, with λ = 1 and weights 0.02 for t < 100, 16 for t ≥ 550, and 8× more at t ≥ 995.
  - The fit error is about 0.08 or less at t ≥ 550, against an embedding norm of about 50. At low t it is about 30% of the wanted change (1.1–1.3 of 3.6–3.8), and up to 2.4 at t=374, so the shift there is only approximate.

### 2. Hypothesis
**The network looks fine at each t, and the loss comes from large sampler steps.**
- In the free probes, module RMS and spectrum at matching t are nearly the same at 6 and 50 steps (ratios mostly within ±3%).
- What differs is `x_t` energy, which falls behind as t drops (`conv_in` RMS ratio 0.996, 0.984, 0.972 at t=499, 332, 166). The 6-step run also makes its final jump from t=166, whereas the 50-step run gets down to t=19.

**A smaller noise prediction should retain that energy.** A Gaussian toy model of DDIM loses energy per frequency at each step, and a constant noise gain restores it across all frequencies.

**The toy was right about direction but wrong about size.** It predicted a gain of about 0.92, while the real best is about 0.98. A gain of 0.92 overshot coarse by 13 dB.

My untested explanation for the extra sensitivity has two parts:
- The network feeds the change back: at gain 0.985, `eps_rms` at t=666 rose 2.7% despite the 1.5% scale-down.
- The VAE seems to amplify it: a deficit of about 1.5 dB in the latent high band goes with about 4.5 dB in the image.

**The time warp came from experiment, not theory.** Warping low t alone improved all 8 prompts in the coarse band without visibly changing composition, and it reduced grain error.

### 3. Evidence
All 24 dev prompts, baseline → edited, measured against the 50-step reference:

| Metric | 5 steps | 6 steps | 8 steps |
|---|---|---|---|
| texture | 4.76 → 2.05 | 4.52 → 1.71 | 3.73 → 1.39 |
| signed coarse deficit (dB) | 4.67 → 1.12 | 4.64 → 0.86 | 3.71 → -0.28 |
| signed fine deficit (dB) | 4.85 → 1.26 | 4.40 → 0.67 | 3.73 → -0.49 |
| abs grain deficit | 2.33 → 1.84 | 1.82 → 1.32 | 1.48 → 0.88 |
| LPIPS | 0.460 → 0.406 | 0.406 → 0.353 | 0.318 → 0.296 |
| SWD (floor 40.7) | 226 → 77 | 201 → 52 | 165 → 56 |
| CLIP | 0.327 → 0.333 | 0.327 → 0.333 | 0.331 → 0.334 |
| contrast ratio | 0.88 → 0.99 | 0.88 → 0.98 | 0.89 → 1.02 |
| luma shift | -0.016 → -0.017 | -0.012 → -0.013 | -0.009 → -0.009 |

- **Images:** I viewed the 6-step sheet for c2 (first 8 prompts). The added detail looks like real texture, not noise, but it is often not the same texture as the reference. I did not open the 5- and 8-step c2 sheets.
- **Step equivalents (my interpolation of the baseline curve):**
  - Texture at 6 steps matches about a 16-step baseline.
  - LPIPS at 6 steps matches only about a 7-step baseline, because composition still differs from the reference.
- **Value of the warp:** it adds a modest amount.
  - Gain 0.9825 alone gives texture 2.13, LPIPS 0.355 and SWD 71 at 6 steps.
  - c1 (gain 0.99 plus warp) has about the same signed deficits, with texture 1.79, LPIPS 0.355 and SWD 67.
  - So the warp lowers texture at a matched deficit and leaves LPIPS unchanged. c2 is just c1 with a stronger gain, and it adds the extra 409,600 changed weights.

**What failed or was not fixed**

| Trial | Images | Result at 6 steps |
|---|---|---|
| Gain 0.92 | 8 | Severe overshoot: contrast 2.05, LPIPS 0.84, composition changed |
| Constant t-shift -100 | 8 | LPIPS worse (0.44), colors oversaturated |
| Constant t-shift -40 | 8 | Partial fix only: texture 3.08, LPIPS 0.383 |
| High-t warp, -100 | 8 | LPIPS worse (0.432); per-prompt changes erratic |
| High-t warp, +60 | 8 | LPIPS worse (0.419), texture worse; per-prompt changes erratic |
| Gain 0.98 + t-shift +20 | 8 | No LPIPS gain (0.367); luma shift +0.018 |
| `attn2.to_out` × 1.10 | 8 | Almost no effect (texture 4.32) |
| `conv_out` response weighted to high frequencies | 8 | Slightly worse than uniform gain |
| `conv_out` response strongly weighted to low frequencies | 8 | Contrast and SWD better, but LPIPS and texture worse than uniform |
| `conv_out` response mildly weighted to low frequencies | 8 | About equal to uniform, so not adopted |

- **Serious side effect:** at 24 steps on 8 images, the edit overshoots badly: texture 0.74 → 3.48, LPIPS 0.091 → 0.276, SWD 65 → 252, contrast ratio 1.14. It already crosses zero at 8 steps, and I did not measure between 8 and 24. It should only be used for few-step sampling.
- **Not fixed:**
  - Luma shift is unchanged or very slightly worse.
  - Prompts with the largest deficits stay under-corrected; denim and snake still show roughly 3–5 dB in coarse at c1 and about the same on the c2 sheet.

**Judgment calls to check**
- **Time-embedding fit:** fitting the warp multiplies `linear_1`'s matrix by the sinusoid basis, which is that one layer's linear map. It uses no SiLU, later layers or images. I judged this to be weight analysis, not a forward pass under rule 2, but you may read it differently.
- **Files read:** I read `src/wspd/{config,edits,metrics,model}.py` and the model's `unet/config.json` in the Hugging Face cache, neither of which the brief forbids. `config.py` has a comment about the sampler pilot. I did not open `runs/cache/pilot.json`, `runs/pilot.log`, or the other operators' folders.

### 4. Budget and confidence
**I used 37 of 40 units.**

**Confidence that it generalizes:**
- **High** that few-step texture statistics, SWD and contrast improve on unseen prompts and categories. The edit depends only on t, has few tuned constants, and works across all three step counts.
- **Moderate** that LPIPS improves by about 0.02–0.05, depending on step count.
- **Lower** on the size of the texture gain, so expect less than the table shows. Most tuning used the first 8 prompts, and c1 scored about 1.16 on those against an estimated 2.1 on the other 16 (the 1.16 comes from e15, a near-identical fit of the same edit). LPIPS held up better, improving about 0.05 on both groups.
- **Caveats:**
  - All 24 prompts informed the final choice, so there is no true holdout.
  - Unseen categories are untested.
  - This restores detail statistics and does not recover the reference's actual composition or texture.
