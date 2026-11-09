**op3 report: submitted edit `5b70b287a9fead33`, 38/40 units used.**

The edit clearly improves texture and SWD on dev at 5, 6 and 8 steps. The LPIPS gain is modest, and the edit makes 16-step sampling worse.

## 1. Weights changed
- **Parameter:** only `time_embedding.linear_1.weight` (1280x320). All 409,600 entries change, with ||dL||/||L|| = 0.26.
- **Operation:** `L -> L + dL`, where dL is a closed-form ridge solution, with no gradients and no search over the weights. It makes `(L+dL) S(t) ≈ L S(t - delta(t))` for integer t in 0..999, where S is the fixed sinusoidal embedding.
- **Fit settings:** ridge is 10, regularised towards dL = 0. The 16 timesteps visited by 5, 6 and 8-step sampling get weight 200. At those timesteps the time-embedding error is at most 0.004, against a shift of about 1.3-2.0.
- **delta(t):** piecewise linear through knots (t, delta): (0,0), (124,60), (166,85), (199,100), (249,70), (400,70), (470,0), (1000,0).
- **Effect:** timesteps at or above 470 are unchanged, so the early steps behave as before.
- **Script:** `/Users/ofm/Desktop/my_repostation/Weight-Space-Programming-for-Diffusion-Models/runs/op3/c4.py`, sealed in `runs/op3/final/`.

## 2. Hypothesis
DDIM takes one epsilon estimate at the start of each big step and holds it for the whole step. The conditional-mean estimate removes too much, so modes get under-amplified, and the last step is worst because it jumps straight to x0.

- **Weight structure:** the sinusoidal embedding is a fixed input to `linear_1`, so a linear map on its output can remap the timestep the UNet perceives, as a function of t only. No prompt or step-count dependence is needed.
- **Gaussian model:** I worked out each step's exact gain per mode, assuming Gaussian data. The ideal perceived-time shift came out nearly independent of the mode on all steps but the last, about 0.28 of the step size. The last step needs more, and its t differs by step count (199, 166, 124), so a rising delta adapts to step count without being told it. This predicted a deficit of about 5-7 dB, in line with the measured coarse and fine deficits of about 4-5 dB.
- **Probe under an earlier warp (t1, same last-step delta):** at t=166, `eps_rms` fell from 0.82 to 0.75, about what the 50-step run shows near the perceived t=81. So the network follows the time embedding more than the input noise level. `x0_rms` rose from 0.65 to 0.77, against about 0.72 near the end of the 50-step run.
- **What did not hold:** the model also called for a shift at middle and high t. Measurement did not support keeping one, so that part of the theory did not carry over. I don't have a mechanism for why.

## 3. Evidence
Submitted edit on dev, against the 50-step reference (baseline -> edited):

| Steps | n | texture | lpips | swd | clip | contrast | abs grain |
|---|---|---|---|---|---|---|---|
| 5 | 24 | 4.76 -> 1.75 | .460 -> .419 | 226 -> 63 | .327 -> .333 | .88 -> .99 | 2.33 -> 1.78 |
| 6 | 21 | 4.66 -> 1.50 | .414 -> .373 | 208 -> 60 | .327 -> .335 | .87 -> .98 | 1.74 -> 1.08 |
| 8 | 24 | 3.73 -> 1.21 | .318 -> .296 | 165 -> 42 | .331 -> .333 | .89 -> .99 | 1.48 -> .75 |

- **Signed deficits:** all within about 0.7 dB of zero in the coarse, fine and grain bands, with no meaningful overshoot on average.
- **Swd floor:** the floor is 40.7 at n=24, so the 8-step swd is at the floor. The 5 and 6-step values are near it, so they should not be over-read.
- **Images:** on the sheets I viewed, the added detail looks like structured scales, grass and strata, not noise. Composition stays close to the baseline, so it still differs from the reference.
- **Check on images not used for tuning:** tuning used the first 10 images at 6 steps. Comparing a near-identical candidate (c2/c3) at n=10 and n=21 implies about -2.9 texture and -.036 lpips on the 11 added images, against -3.4 and -.053 on the first 10. So the gain persists but shrinks. These images come from the same categories, so this says nothing about unseen categories.

**What failed or was rejected** (6 steps, n=10 unless noted):
- **Constant shifts:** 50, 75 and 100 give texture 2.84, 1.69 and 1.47, and lpips .394, .399 and .427. At 8 steps, a shift of 75 made lpips worse (.334 -> .364).
- **Shift only at t>=450:** texture 3.85 with no lpips gain (.429).
- **Adding a shift at t=666 and 499:** lpips got worse (.386 -> .409).
- **Shifting the highest timesteps too:** at n=10, adding a shift at t>=760 improved lpips from .386 to .371, which looked real. At n=21 the lpips gap was only .003 against the low-t warp (c3 .370, c4 .373). It also raised abs grain (1.66 vs 1.08), so I dropped it.
- **`conv_in` skip in `up_blocks.2.resnets.1.conv_shortcut`:**
  - This knob is very sensitive, at about 1.2 dB per 1%.
  - x1.05 on top of the warp wiped out most of the gain (texture 4.0).
  - x0.93 alone overshot badly (lpips .469).
  - x0.96 alone is a real but weaker alternative (texture 2.11, lpips .393, swd 82).
- **Mid-resolution skip x0.98:** no measurable effect.

**Process errors:**
- My first try failed the lint because a comment contained a library name, at no cost.
- I once ran two tries in parallel and they collided on a folder. The c3 n=21 sheet was overwritten, but its metrics survive in the ledger and the command output. The c4 sheet is intact.
- Metrics are summaries only, so I have no per-image significance tests.
- For analysis I evaluated the two time-embedding layers on CPU. No UNet or VAE forward pass ran outside `wsp`. This is a judgement call under rule 2.

## 4. Units and confidence
**Units:** 38 of 40 used.

**Confidence:**
- **High (about 85-90%)** that texture and SWD improve substantially on unseen prompts at 5, 6 and 8 steps, because the edit depends only on t and helps across all 8 dev categories in aggregate. I expect somewhat smaller gains than on dev.
- **Moderate (about 65-70%)** that the LPIPS improvement holds. It is small, and it may partly reflect tuning on these images.
- **Not fixed:** composition and structural mismatch with the 50-step image. LPIPS stays far from the 16-step baseline (about .14).
- **Uncertain:** the exact knot values, since differences among nearby variants were within the noise. Prompts with little texture could overshoot.

**Known harm:** the edit is tuned to few steps and hurts above that range. At 16 steps, n=8:
- texture 1.61 -> 2.51
- lpips .153 -> .239
- swd 79 -> 179

The likely cause is that the needed shift scales with step size, and a static edit can't know the step size. 50-step behaviour with the edit was not measured, but I expect it to be worse too.
