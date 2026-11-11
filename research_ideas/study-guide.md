# Study guide: anatomy of acceleration adapters

A deeper explanation of [the idea](01-anatomy-of-acceleration-adapters.md): the background
you need, what the pilot showed, what the paper would claim, and what to read. Written
2026-11-11.

**Check every citation before you use it.** The arXiv numbers for older papers were written
from memory. The 2026 papers came from an automated literature search and were not read by
the author of this guide. Open each link first.

## Background

Learn these first if any are shaky.

- **How sampling works.** A diffusion model starts from noise and denoises in steps. Fewer
  steps means bigger jumps and more error. Read DDPM
  ([2006.11239](https://arxiv.org/abs/2006.11239)), then DDIM
  ([2010.02502](https://arxiv.org/abs/2010.02502)), which is the sampler this repo uses.
- **Latent diffusion and guidance.** Stable Diffusion denoises in a compressed latent space
  ([2112.10752](https://arxiv.org/abs/2112.10752)), and classifier-free guidance runs the
  network twice per step ([2207.12598](https://arxiv.org/abs/2207.12598)). That is why
  "6 steps" costs 12 network evaluations here.
- **The metrics.** LPIPS ([1801.03924](https://arxiv.org/abs/1801.03924)), FID
  ([1706.08500](https://arxiv.org/abs/1706.08500)) and CLIP
  ([2103.00020](https://arxiv.org/abs/2103.00020)). Know what each one cannot see.

## The idea

### Where it comes from

There are two ways to make sampling faster. One changes the sampler and leaves the model
alone, which is what the agents in this repo ended up doing. The other is **distillation**:
train a student model to do in 4 steps what the teacher does in 50.

The key fact for this idea is how distilled models are shipped. Fine-tuning changes every
weight matrix `W` to `W + ΔW`. LoRA forces `ΔW` to be a product of two thin matrices,
`B·A`, with a small rank `r`. LCM-LoRA showed that the whole 4-step ability can be stored
this way, as a 135 MB file you add to Stable Diffusion 1.5. Its authors called it an
"acceleration vector".

So the few-step ability is a concrete object: 278 small matrix pairs. You can open it and
measure it without running the model. That makes it a weight-space project that does not
need an LLM.

### What the pilot did

1. **Measured the delta.** For each of the 278 modules, compute `ΔW` and its singular
   values. A singular value decomposition splits a matrix into independent directions
   sorted by strength. If a few directions hold most of the energy, the matrix is
   effectively lower rank than its nominal 64.
2. **Truncated it.** Keep only the top 4 directions per module and generate images. They
   were 96% as close to the full adapter's images as possible, and looked identical.
3. **Ablated it.** Keep one part of the adapter and zero the rest, then do the reverse.
   "Only X works" means X is sufficient; "everything except X fails" means X is necessary.

| Finding | Evidence | Why it matters |
|---|---|---|
| Effectively rank 4 or lower | The median module needs 2 directions for 90% of its energy | The adapter is about 16 times bigger than it needs to be |
| The decoder carries it | Up blocks hold 82% of the energy and alone recover 75% | Distillation mostly rewrites the half of the UNet that draws detail |
| The time path is idle | Time layers alone recover 0%; removing them costs 6% | Conflicts with MTEO, which trains only time embeddings |
| No single module type suffices | The best single type (convolutions) recovers 15% | The ability is spread across types |

Code: `scripts/pilots/c_lora_anatomy.py`. Output: `results/pilots/c_lora_anatomy.json`.

### What the paper would claim

"Few-step ability, as learned by distillation, lives in a very low-dimensional,
decoder-heavy subspace, and this is true across distillation methods."

The second half is untested, and the paper depends on it. If Hyper-SD, TCD and DMD2
adapters show the same structure, or share the same directions, you have a finding about
distillation itself. If only LCM-LoRA does, you have a footnote.

The open questions, in the order to answer them:

1. **Does it replicate?** Run the same dissection on other adapters and on SDXL.
2. **Do methods agree?** Measure the overlap between the top directions of two adapters
   for the same base model.
3. **Does quality survive?** The pilot measured closeness to the full adapter, not quality.
   You need FID and a preference metric at rank 4.
4. **Can you use it?** Train a rank-4, decoder-only adapter directly and compare its cost.

### What to read

| Topic | Paper | Why |
|---|---|---|
| Low-rank adaptation | LoRA, Hu et al., [2106.09685](https://arxiv.org/abs/2106.09685) | Read first. Its argument for why low rank works is your starting hypothesis |
| Fine-tuning moves in few directions | Aghajanyan et al., [2012.13255](https://arxiv.org/abs/2012.13255) | The general version of your claim |
| Weight deltas as vectors | Task arithmetic, Ilharco et al., [2212.04089](https://arxiv.org/abs/2212.04089) | The vocabulary for comparing adapters |
| Distillation | Progressive distillation, [2202.00512](https://arxiv.org/abs/2202.00512); consistency models, [2303.01469](https://arxiv.org/abs/2303.01469) | How a student learns to skip steps |
| The adapter in the pilot | LCM, [2310.04378](https://arxiv.org/abs/2310.04378); LCM-LoRA, [2311.05556](https://arxiv.org/abs/2311.05556) | What you dissected |
| Other adapters to dissect | DMD, [2311.18828](https://arxiv.org/abs/2311.18828); DMD2, [2405.14867](https://arxiv.org/abs/2405.14867); Hyper-SD, [2404.13686](https://arxiv.org/abs/2404.13686); TCD, [2402.19159](https://arxiv.org/abs/2402.19159); SDXL-Lightning, [2402.13929](https://arxiv.org/abs/2402.13929) | Step 1 of the plan |
| Linear algebra | Eckart–Young theorem; principal angles between subspaces | Why SVD truncation is optimal, and how to compare two adapters |
| Closest competitors | WaDi, [2603.08258](https://arxiv.org/abs/2603.08258); MTEO, [2603.22375](https://arxiv.org/abs/2603.22375) | Read before anything else. They decide how you position the paper. Not read by the author of this guide |

## Suggested reading order

1. DDIM, then classifier-free guidance.
2. LoRA, then LCM and LCM-LoRA.
3. WaDi and MTEO, to see how close the competition is.
4. Task arithmetic and the intrinsic-dimension paper.
5. The other distillation papers, as you reach each adapter.
