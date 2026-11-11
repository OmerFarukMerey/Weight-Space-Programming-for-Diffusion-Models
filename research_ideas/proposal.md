# Project proposal: Anatomy of acceleration adapters

**Where does few-step ability live inside a distilled diffusion model?**

| | |
|---|---|
| Author | Ömer Faruk Merey |
| Course | Generative Models for Multimedia, METU |
| Date | 11 November 2026 |
| Status | Draft for discussion with the course advisor |

## Summary

Diffusion models can be distilled to sample in 4 steps, not 50, and the result is often
shipped as a small low-rank weight delta added to the base model. I propose to take these
deltas apart in weight space and find which parts carry the few-step ability. A pilot on
one adapter shows it is far more compact than its file suggests: cutting LCM-LoRA from
rank 64 to rank 4 keeps 96% of its effect, and 82% of its energy sits in the decoder half
of the network. The project tests whether this holds across distillation methods and
models, and whether it can be used to build much smaller adapters.

## Motivation

Step distillation is the main route to fast diffusion sampling. Methods such as LCM,
Hyper-SD, TCD and DMD2 each train a student to imitate many teacher steps in a few. They
differ in their training objective, and their papers compare them on image quality.

What the trained student changed is rarely examined. That matters for three reasons:

- **Cost.** If the few-step ability needs only a handful of directions in a few layers,
  distillation could train far fewer parameters.
- **Deployment.** Adapters of 100-400 MB are shipped and combined with style adapters. A
  16-times smaller one is easier to distribute and to merge.
- **Understanding.** Methods with different objectives may be learning the same change.
  Knowing that would say something about distillation itself.

A low-rank adapter makes the question tractable. The whole few-step ability is a set of a
few hundred small matrix pairs that can be measured directly.

## Background

- **Few-step sampling.** A diffusion model denoises in steps. With few steps each jump is
  large and a base model produces blurry, off-prompt images.
- **Distillation.** A student is trained to match the teacher's multi-step result in one
  jump (consistency distillation) or to match its output distribution (DMD).
- **LoRA.** A weight change `ΔW` is stored as a product of two thin matrices, `B·A`, of
  rank `r`. LCM-LoRA stores the 4-step ability for Stable Diffusion 1.5 this way, with
  rank 64 in 278 modules.
- **Singular value decomposition.** `ΔW` can be split into independent directions sorted
  by strength. Keeping the strongest `k` is the best rank-`k` approximation.

## Research questions

1. **How low is the true rank?** How many directions per module does an acceleration
   adapter need?
2. **Where is it?** Which blocks and module types are necessary, and which are sufficient?
3. **Is it shared?** Do adapters from different distillation methods occupy the same
   subspace for the same base model?
4. **Can it be used?** Does a compact adapter keep image quality, and can one be trained
   directly?

## Hypotheses

Each is written so that it can fail.

| | Hypothesis | Supported if | Rejected if |
|---|---|---|---|
| H1 | Acceleration adapters are effectively very low rank | Rank 8 or less keeps FID within 5% of the full adapter, for at least 3 of 4 adapters | Most adapters need rank 32 or more |
| H2 | The decoder carries most of the ability | Decoder-only recovery is at least twice encoder-only recovery, for at least 3 of 4 adapters | Recovery is balanced or encoder-heavy |
| H3 | Different methods learn overlapping subspaces | Subspace overlap between two adapters is clearly above a random-subspace baseline of the same rank | Overlap equals the random baseline |
| H4 | A compact adapter can be trained directly | A rank-4, decoder-only adapter reaches FID within 10% of the standard one | It fails to converge or is clearly worse |

The thresholds are first guesses. I will fix them with the advisor before running step 2.

## Preliminary results

Run on 10 November 2026: Stable Diffusion 1.5 with LCM-LoRA, 4 steps, no guidance, 16
prompts. "Recovery" is 1 minus the LPIPS distance to the full adapter's same-seed image,
divided by the distance of the base model without the adapter (0.972).

**Truncation and location**

| Part of the adapter kept | Recovery | CLIP |
|---|---:|---:|
| Nothing (base model at 4 steps) | 0.00 | 0.218 |
| Full adapter, rank 64 | 1.00 | 0.320 |
| Rank 16 | 0.99 | 0.320 |
| Rank 8 | 0.98 | 0.320 |
| Rank 4 | 0.96 | 0.320 |
| Decoder (up) blocks only | 0.75 | 0.320 |
| Encoder (down) blocks only | 0.01 | 0.221 |
| Middle block only | 0.00 | 0.219 |

**Module types**

| Module type | Share of adapter energy | Recovery when kept alone | Recovery when removed |
|---|---:|---:|---:|
| ResNet convolutions | 40% | 0.15 | 0.39 |
| Feed-forward layers | 38% | 0.04 | 0.59 |
| Attention input/output projections | 7% | 0.04 | 0.76 |
| Self-attention | 7% | 0.02 | 0.82 |
| Cross-attention | 6% | 0.02 | 0.75 |
| Time-embedding projections | 1% | 0.00 | 0.94 |

What this suggests:

- **H1 looks plausible.** The median module needs 2 singular directions for 90% of its
  energy. Rank 4 images look identical to the full adapter's.
- **H2 looks plausible.** The decoder holds 82% of the energy.
- **No module type is sufficient alone.** Convolutions are the most necessary.
- **The time-conditioning path is nearly idle.** This is unexpected, because a recent
  method (MTEO) reaches 3-step sampling by training only time embeddings.

Limits of the pilot: one adapter, 16 prompts, no repeated seeds, and a metric that measures
closeness to the full adapter, not image quality. Code is in
`scripts/pilots/c_lora_anatomy.py`.

## Method

### Step 1: Dissect more adapters

Apply the pilot's analysis to Hyper-SD, TCD and DMD2 adapters on Stable Diffusion 1.5,
then to LCM-LoRA and SDXL-Lightning on SDXL. For each adapter:

- singular-value spectrum per module, and the rank needed for 90% and 99% of the energy;
- energy share by block and by module type;
- keep-one and remove-one ablations by block and module type;
- rank truncation at 32, 16, 8, 4, 2 and 1.

### Step 2: Compare adapters

For two adapters on the same base model, compute the principal angles between their top-`k`
subspaces in each module. Compare against random subspaces of the same rank. Also test
whether one adapter's delta projected onto another's subspace still works.

### Step 3: Measure quality

Evaluate full and truncated adapters on 5,000 COCO captions with FID, CLIP score and a
preference metric (HPSv2 or ImageReward). This replaces the pilot's recovery metric.

### Step 4: Train a compact adapter

Train a rank-4, decoder-only adapter with the public LCM distillation script, and compare
its quality and training cost with the standard rank-64 adapter. This step needs a GPU and
is the first to drop if time runs out.

### Step 5 (stretch): Explain the time-path result

Check whether MTEO-style time-embedding training and LCM-style distillation reach the same
output by different routes, by comparing their effect on the network's activations.

## Baselines and controls

- **Random truncation.** Keep 4 random directions per module instead of the top 4. If this
  works as well, the finding is about redundancy, not structure.
- **A non-acceleration adapter.** Run the same analysis on a style LoRA for the same base
  model. If it shows the same rank and location, the finding is about LoRA in general.
- **Matched parameter count.** Compare "decoder only" with an encoder subset of equal size.
- **Seeds.** Repeat generation with 3 seeds and report intervals.

## Timeline

Assumes the term ends in mid-January 2027. Dates shift if it does not.

| Weeks | Dates | Work | Output |
|---|---|---|---|
| 1 | 11-17 Nov | Read LoRA, LCM, LCM-LoRA, WaDi, MTEO. Agree thresholds with the advisor | Fixed hypotheses |
| 2-3 | 18 Nov - 1 Dec | Step 1 on SD 1.5 adapters; add controls | Anatomy tables for 4 adapters |
| 4 | 2-8 Dec | Step 2 | Subspace-overlap results |
| 5-6 | 9-22 Dec | Step 3; step 1 on SDXL if a GPU is available | Quality table |
| 7-8 | 23 Dec - 5 Jan | Step 4, or deeper analysis if no GPU | Compact adapter or extended analysis |
| 9 | 6-12 Jan | Write the term report | Report and presentation |
| After | Feb - Apr | Extend to a paper if H1-H3 hold | Submission draft |

**Decision point, 1 December.** If the other adapters do not show low rank or decoder
concentration, the project becomes a case study of LCM-LoRA and the paper plan is dropped.

## Resources

- **Available:** an Apple M4 Pro laptop with 24 GB. It runs SD 1.5 at 4 steps in about 2
  seconds per image, so steps 1 and 2 fit.
- **Needed:** a GPU with 16-24 GB for SDXL and for step 4. FID on 5,000 images for about 30
  adapter variants is roughly 4 days on the laptop, so step 3 also benefits from a GPU.
- **Software:** PyTorch, diffusers, and the harness in this repository. All models and
  adapters are public.

## Risks

| Risk | Likelihood | Response |
|---|---|---|
| The result is specific to LCM-LoRA | Medium | Decision point on 1 December |
| "LoRA ranks can be reduced" is seen as known | High | Rest the claim on the cross-method and module findings; include the style-LoRA control |
| A closely related paper already exists or appears | Medium | WaDi reports a similar rank result for DMD2; position against it early |
| Recovery does not predict quality | Medium | Step 3 replaces it with FID and a preference metric |
| No GPU access | Unknown | Steps 1-2 and a reduced step 3 still make a complete term report |

## Expected outcomes

- **Minimum (term report):** a dissection of four acceleration adapters with controls, and
  a quality table for truncated versions.
- **Target (workshop paper):** the above plus evidence for or against a shared subspace.
- **Stretch (conference paper):** a directly trained compact adapter, on two base models.
  Candidate venues are ICCV 2027 and NeurIPS 2027; their deadlines are not yet announced.

## Related work

The 2026 entries were found by an automated literature search and I have not yet read them.

- **Closest.** WaDi (Wang et al., CVPR 2026, [2603.08258](https://arxiv.org/abs/2603.08258))
  analyses student-minus-teacher weights for DMD2 and reports that 30% of the rank recovers
  93% of the residual. It has no module or block breakdown and no cross-method comparison.
- **Time-path distillation.** MTEO (Cai et al., [2603.22375](https://arxiv.org/abs/2603.22375))
  trains only per-layer time embeddings.
- **Mostly-frozen distillation.** Zheng and Yang ([2405.20750](https://arxiv.org/abs/2405.20750))
  distil to one step with most convolutions frozen.
- **Adapters.** LoRA ([2106.09685](https://arxiv.org/abs/2106.09685)); LCM-LoRA
  ([2311.05556](https://arxiv.org/abs/2311.05556)).
- **Distillation methods.** LCM ([2310.04378](https://arxiv.org/abs/2310.04378)); DMD2
  ([2405.14867](https://arxiv.org/abs/2405.14867)); Hyper-SD
  ([2404.13686](https://arxiv.org/abs/2404.13686)); TCD
  ([2402.19159](https://arxiv.org/abs/2402.19159)); SDXL-Lightning
  ([2402.13929](https://arxiv.org/abs/2402.13929)).
- **Weight-space analysis.** Task arithmetic ([2212.04089](https://arxiv.org/abs/2212.04089));
  intrinsic dimension of fine-tuning ([2012.13255](https://arxiv.org/abs/2012.13255)).

## Relation to the original proposal

The first proposal asked whether an LLM could find and edit the denoiser weights behind a
few-step failure. I tested that in this repository. Three LLM agents each found a working
edit, but all three rediscovered published sampler corrections, and a blind four-parameter
search matched them. The advisor's doubt was justified. This proposal keeps the
weight-space question and drops the LLM: it studies what trained few-step weights look like.
