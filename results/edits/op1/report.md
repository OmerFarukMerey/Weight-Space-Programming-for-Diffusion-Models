**op1 report: edit submitted, sealed as `a4cd4c96d203e765`.** It restores texture energy and statistics at 5, 6 and 8 steps on the dev prompts. It does not fix composition errors. All 40 units are used.

## 1. Weights changed
421,120 weights in 2 parameters, from `/Users/ofm/Desktop/my_repostation/Weight-Space-Programming-for-Diffusion-Models/runs/op1/V3.py`:

- **`conv_out.weight` (11,520 weights, rel_l2 0.0115):** multiplied by 0.9885. The bias is untouched.
- **`time_embedding.linear_1.weight` (409,600 weights, rel_l2 0.208):** replaced with `W' = W + W (Eg-E)^T E (E^T E + 10 I)^-1`.
  - `E` holds the sinusoidal embeddings for t = 0..999, and `Eg` holds those for g(t) = t - delta(t).
  - Delta is linearly interpolated from these knots (t:delta): 0:0, 60:13, 124:26, 166:50, 199:70, 249:60, 332:54, 399:54, 499:0, 999:0.
  - The network therefore reads t as: 124→98, 166→116, 199→129, 249→189, 332→278, 374→320, 399→345. From t=499 upward the label is unchanged.

## 2. Hypothesis
- **Starting observation:** at 6 steps the final x0 comes from t=166, with rms 0.653 and high-frequency fraction 0.433. The 50-step run ends at rms 0.729 and fraction 0.487.
- **Proposed cause:** predicted noise rms falls as t falls (0.85 at t≈199, 0.57 at t=19 on the 50-step path). A large DDIM step uses the noise estimate from the start of its interval, so it removes too much and shrinks detail. A mean-direction argument suggests using a smaller eps, or a later t label, within each step.
- **Why these weights:**
  - A shift in t is a rotation of the (cos, sin) pairs, and the remap's effect stays low-dimensional, so `linear_1` can express it. A ridge fit reproduced the target labels closely.
  - Labels at high t appear to affect composition. Changing them in either direction raised LPIPS, while changing only low t lowered it.
  - The uniform scale supplies the small eps reduction at high t without changing labels.
- **How far to trust this:** the mechanism is only partly established. A plain `conv_out` scale did nearly as well on mean deficits and better on LPIPS than the pure remap, so reduced eps amplitude may be what matters rather than the label. I found the hybrid by experiment and explained it afterwards, and the knot values are fitted, not derived.

## 3. Evidence
**Submitted edit, all 24 dev prompts (baseline → edited):**

| Metric | 5 steps | 6 steps | 8 steps |
|---|---|---|---|
| texture | 4.76 → 1.94 | 4.52 → 1.63 | 3.73 → 1.46 |
| abs_deficit_structure | 3.04 → 1.48 | 3.23 → 1.49 | 2.55 → 1.37 |
| abs_deficit_grain | 2.33 → 1.86 | 1.82 → 1.30 | 1.48 → 0.87 |
| lpips | 0.460 → 0.410 | 0.406 → 0.354 | 0.318 → 0.303 |
| swd (floor 40.7) | 225.8 → 44.5 | 200.6 → 44.8 | 164.7 → 62.4 |
| clip | 0.3266 → 0.3341 | 0.3265 → 0.3356 | 0.3306 → 0.3339 |
| contrast_ratio | 0.88 → 1.03 | 0.88 → 1.00 | 0.89 → 1.02 |

- **Tone guard:** luma_shift barely changes at any step count.
- **Latent check (probe, 8 prompts, 6 steps):**
  - Final x0 rms is 0.7235 (reference 0.7289, baseline 0.653).
  - High fraction is 0.500 (reference 0.487, baseline 0.433), a slight excess.
  - The endpoint statistics now match, but intermediate x0 estimates are more high-frequency than the 50-step path at similar t. It reaches the right statistics by a different route.
- **Closest thing to held-out evidence:** I tuned the previous version, V2, on the first 8 prompts. Arithmetic on the means gives the other 16 prompts at 6 steps: texture 4.52 → 2.00 and LPIPS 0.405 → 0.353. Its mean deficits were about 1 dB under-corrected there compared with the tuning prompts.
- **No clean held-out set for the final edit:** I raised the strength to V3 after seeing all 24 prompts, chosen on the 6-step result alone. V2 was never run on all 24 at 5 or 8 steps.

**What failed or was weaker (8 prompts, 6 steps unless noted; baseline texture 4.52, LPIPS 0.407):**

| Attempt | texture | lpips | Outcome |
|---|---|---|---|
| Power remap, gamma 1.5 | 2.84 | 0.451 | Overshoot, SWD worse than baseline (223 vs 189) |
| Power remap, gamma 1.25 | 1.77 | 0.392 | Works, but changes high-t labels |
| Constant shift t-80 | 1.72 | 0.406 | No LPIPS gain, contrast 1.05 |
| Remap at t≥499 only | 3.56 | 0.449 | Composition damage |
| Low-t remap plus opposite-sign remap at high t | 1.54 | 0.427 | LPIPS worse than low-t alone (0.379) |
| `conv_out` x0.97 | 3.35 | 0.402 | Overshoot, SWD 186 |
| `conv_out` x0.98 | 1.75 | 0.364 | Good, but abs_deficit_grain worse than baseline |
| Low-res ResNet time projections only | 4.45 | 0.409 | Essentially no effect |

- **Where the remap acts:** the same remap on only the three 64x64 ResNet time projections gave texture 1.33. `up_blocks.2.resnets.1` alone gave 2.17. So the effect sits almost entirely in the high-resolution blocks. I still edited the shared `linear_1`, because it is one tensor and needs no MLP evaluation. I did not test a hybrid using high-res projections only.

**Limits:**
- Composition still differs from the reference. LPIPS at 6 steps is 0.354, against 0.14 for unedited 16-step sampling.
- The added detail is plausible, not the reference's actual detail, and it partly comes from noise the edit no longer removes. The SWD and LPIPS gains argue against plain noise, but do not show it matches the reference. I also saw some invented detail, such as extra creases in the denim row of the low-t remap sheet.
- Mean deficits are near zero, but the abs deficits (1.3 to 2.2 for coarse and fine) show that individual images still over- or under-correct.
- 8 steps overshoots slightly (coarse -0.32, fine -0.56), and its SWD stays well above the floor.
- My visual checks were limited. Contact sheets show only the first 8 prompts and are colour-quantized, so I never saw the edited images for the other 16. For the final edit I viewed only the 6-step sheet.
- Other step counts were not tested. I expect the edit to get worse at high step counts, because the eps scale applies at every step.

**Things for you to judge:**
- **Rule 2 grey area:** I evaluated the time-embedding MLP (`linear_1`, silu, `linear_2`) on CPU with synthetic t values. That was to check the fit, and it also ran inside the attribution-only per-block edits. No UNet or VAE ran outside the metered commands, no images were involved, and the submitted script uses only linear algebra on `linear_1`.
- **Possible hint:** I read parts of `src/wspd/` (`edits.py`, `config.py`, the sampling loop in `model.py`) and the cached Hugging Face configs. A comment in `config.py` says energy is lost on every prompt with this sampler. I read it after the first probe analysis and before the first metered try.
- **Not opened:** `runs/cache/pilot.json`, or any file the brief forbids.

## 4. Units and confidence
40 of 40 units used: 39 on `try` and 1 on the probe.

These confidence figures are my own judgement, not computed:

| Claim on unseen prompts | Confidence |
|---|---|
| Texture and SWD beat baseline at 5, 6 and 8 steps | About 90% |
| LPIPS and clip do not get worse | About 80% (the 8-step LPIPS gain is small) |
| Mean deficits stay within about 1 dB of zero in unseen categories | About 65% |

I expect the texture gain to shrink somewhat, to about 2 at 6 steps, because the strength was set on dev prompts and those 8 categories may not represent the unseen ones. Smooth subjects are the most likely to be over-textured.

Scripts and all try and probe outputs are in `/Users/ofm/Desktop/my_repostation/Weight-Space-Programming-for-Diffusion-Models/runs/op1/`; the sealed edit is in its `final` subfolder.
