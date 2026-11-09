# Pre-registration

Written on 2026-11-06, after the dev pilot and before any weight edit was scored
by anyone. The one edit run so far was `conv_out.weight * 0.9`, used to check that
installing and reverting an edit restores identical output, and no metric was
computed on it. The SHA-256 of this file is recorded in
`results/preregistration.sha256` at the same moment, so later changes are detectable.
Any deviation from this plan is listed in the README rather than folded in silently.

## Question

If a diffusion model loses fine texture in a similar way across prompts when it
samples in 5 to 8 steps instead of 50, can an LLM identify the denoiser weights
related to that error and edit them without gradients? Does one edit carry over to
new prompts, and does it reach the same quality in fewer steps?

## Setup

- **Model:** `segmind/tiny-sd` at revision `cad0bd74`: 323.4M UNet weights.
- **Sampling:** DDIM with trailing spacing, guidance 7.5, 512x512.
- **Reference:** the unedited model at 50 steps, same prompt and same seed.
- **Target:** 6 steps is primary; 5 and 8 are secondary.

### Why this sampler

The pilot (8 dev prompts, 6 steps) found that the direction of the error depends
on the schedule:

| Sampler | Coarse deficit | Fine deficit | Prompts with lost energy | LPIPS |
|---|---|---|---|---|
| DDIM linspace | -2.42 dB | -2.12 dB | 12% | 0.441 |
| DDIM trailing | +4.85 dB | +4.20 dB | 100% | 0.407 |
| DPM++ 2M Karras | -2.57 dB | -2.14 dB | 12-25% | 0.579 |

DDIM trailing is the only one of the three where the premise holds on every prompt,
and it is also the closest to the reference at 6 steps. So it is a fair baseline
rather than a handicapped one. This choice was made on dev data only.

### Splits

| Split | Prompts | Categories | Who sees it |
|---|---|---|---|
| dev | 24 | 8 | operator |
| test_seen | 40 | the same 8, different prompts | supervisor, after sealing |
| test_new | 32 | 4 held-out categories | supervisor, after sealing |

Assignment comes from a fixed RNG seed, not a human choice. Every test prompt has
its own seed. Test metrics are never computed until the edit being scored is sealed.

## Arms

| Arm | What it is | Budget |
|---|---|---|
| LLM operators | 3 independent Claude Opus 5.5 agents with no shared context, given `OPERATOR_BRIEF.md` | 40 units each |
| Blind hill climb | 3 seeds. Scale one random weight tensor by a random factor and keep it if the dev score improves | 40 units each |
| Random-location null | The operator's own delta values, scattered at random inside the same tensors; 20 draws | n/a |
| Unsharp mask | No weight edit at all: an image filter tuned on dev | n/a |

The hill climb answers whether the LLM's reasoning adds anything over dumb search
with the same compute. The null answers whether the location of the edit matters,
or only its size. The filter answers whether this needed weights at all.

## Metrics

Per prompt, against the same-seed 50-step image:

- **T (texture deficit):** the mean of `|deficit_coarse|` and `|deficit_fine|` in dB,
  covering 1/16 to 1/4 cycles per pixel. The absolute value penalizes overshoot.
  These bands were chosen because the pilot shows the loss there; the grain band
  above them shows little loss.
- **L (perceptual distance):** LPIPS (AlexNet).

T alone can be gamed, since noise adds energy. L alone can punish a plausible texture
that merely differs from the reference. So both are required.

Secondary: SWD on Laplacian patches, and the CLIP similarity to the prompt. CLIP
barely moves with step count in the pilot (0.328 at 5 steps, 0.329 at 50), so it
serves only as a guard against semantic damage, not as a measure of quality.

The dev score for the blind control and the null is `T/T_base + L/L_base`, where 2.0
is the unedited model and lower is better.

## Hypotheses and success criteria

All of these are evaluated at 6 steps on **both** test splits, with 95% paired
bootstrap intervals over prompts. There is no multiple-comparison correction, so
the verdict requires every split to pass and every arm is reported, not just the best.

| ID | Claim | Passes if |
|---|---|---|
| H1 | The error is consistent across prompts | Coarse and fine deficits are positive on at least 80% of prompts, and the interval excludes 0 |
| H2 | The edit generalizes on texture | The change in T is negative and the interval excludes 0 |
| H3 | The recovered texture is the right texture | The change in L is negative and the interval excludes 0 |
| H4 | It saves steps | Edited 6-step matches baseline at 8 or more steps on **both** T and L |
| H5 | It keeps meaning | The change in CLIP is at least -0.01 |
| H6 | The LLM adds value | The LLM median score beats the blind-climb median, and each LLM edit beats at least 95% of its random-location draws |
| H7 | The edit is causal | T moves monotonically from strength 0 to 1, and strength -1 does not improve it |

### Verdict

- **Yes:** H1 to H7 all pass for at least 2 of the 3 operators, and the unsharp mask
  does not match the LLM median score.
- **Partial:** H2 passes for at least 2 operators, but H3, H4, H5, H7 or the null half
  of H6 fails, or the unsharp mask matches the LLM median. The edit restores energy
  without restoring the right image, or an image filter could have done the same.
- **No:** H2 fails for most operators, or the blind climb median is at least as good
  as the LLM median.

"Matches" and "as good" compare the score at 6 steps on the sealed prompts, lower is better.

## What this cannot show

- FID is out of reach here: 72 test images are far too few for it. Reference-based
  metrics stand in for distribution-level quality.
- All operators are the same model. Three agents show repeatability, not a general
  fact about LLMs.
- One small model on one sampler. A result here does not transfer automatically.
- A local machine cannot truly seal files from an agent with shell access. The rules
  forbid reading test data, and I check the harness ledger and transcripts afterward,
  but this is an audit and not a technical guarantee.

## Amendments

1. **2026-11-06, while operators were running and before any edit had been scored on
   any split.** The first version said a matching filter meant "No" in one sentence and
   "reported separately" in another. Resolved as above: a matching filter caps the
   verdict at Partial. "Partial" was also made exhaustive, so that H5, H7 and the null
   half of H6 are covered and every outcome maps to exactly one verdict. The hash of the
   first version is kept in `results/preregistration.sha256`.
