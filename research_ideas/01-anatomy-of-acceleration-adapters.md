# Anatomy of acceleration adapters

**Pitch.** Few-step distillation is shipped as a low-rank weight delta (LCM-LoRA, Hyper-SD,
TCD, SDXL-Lightning, DMD2). Take those deltas apart in weight space and ask where the
few-step ability actually lives. The pilot says: in very few directions, mostly in the
decoder, and not in the time-conditioning path.

## Research question

Which parts of an acceleration adapter are necessary and sufficient for few-step sampling,
how low is its true rank, and do different distillation methods learn the same directions?

## Pilot evidence (pilot C)

SD 1.5 with LCM-LoRA (278 modules, rank 64), 4 steps, no guidance, 16 prompts. "Recovery"
is 1 minus the LPIPS distance to the full adapter's same-seed image, divided by the
distance of the unadapted model (0.972).

| Part of the adapter kept | Recovery | CLIP |
|---|---:|---:|
| Nothing (base SD 1.5 at 4 steps) | 0.00 | 0.218 |
| Full adapter | 1.00 | 0.320 |
| Rank 16 of 64 | 0.99 | 0.320 |
| Rank 8 | 0.98 | 0.320 |
| **Rank 4** | **0.96** | **0.320** |
| Decoder (up) blocks only | 0.75 | 0.320 |
| Encoder (down) blocks only | 0.01 | 0.221 |
| Time projections only | 0.00 | 0.218 |
| Everything except time projections | 0.94 | 0.320 |
| ResNet convolutions only | 0.15 | 0.264 |
| Everything except convolutions | 0.39 | 0.301 |

- **The adapter is effectively rank 4 or lower.** The median module needs 2 singular
  directions for 90% of its energy, and 1 for convolutions and cross-attention. Rank 4
  would shrink the 135 MB file about 16 times with no retraining.
- **The decoder carries it.** Up blocks hold 82% of the delta's energy and alone restore
  full prompt alignment.
- **No single module type is sufficient**, but convolutions are the most necessary.
- **The time path does almost nothing here.** That is in tension with MTEO (below), which
  gets 3-step sampling by training only time embeddings. Explaining that tension is a
  ready-made paper question.

`results/pilots/c_lora_anatomy.png` shows rank 4 is visually identical to the full adapter.

## Plan

1. **Replicate across adapters** (2 weeks): Hyper-SD, TCD and DMD2 LoRAs on SD 1.5 and SDXL.
   Is "rank 4, decoder-heavy" universal or specific to consistency distillation?
2. **Compare directions** (1 week): subspace overlap between adapters for the same base
   model. If two methods share a subspace, that is the headline.
3. **Use it** (3 weeks): either ship truncated adapters with a quality table, or train a
   rank-4 decoder-only adapter from scratch and compare cost and quality.
4. **Explain it** (stretch): relate the top directions to the sampler corrections this
   repo found (noise gain, timestep relabel).

Metrics: FID and CLIP on COCO prompts, a preference metric (HPSv2 or ImageReward), and
LPIPS to the full adapter. Compute: steps 1-2 fit this laptop for SD 1.5; SDXL and any
training need a GPU with 16-24 GB.

## Closest prior work

| Paper | Overlap | Read? |
|---|---|---|
| WaDi, Wang et al., CVPR 2026, [2603.08258](https://arxiv.org/abs/2603.08258) | Analyses student-minus-teacher weights for DMD2; 30% of rank recovers 93% of the residual. No module or block breakdown, no cross-method comparison | Full text, by the search agent |
| MTEO, Cai et al., [2603.22375](https://arxiv.org/abs/2603.22375) | Trains only per-layer time embeddings for 3-step sampling | Abstract |
| Zheng & Yang, [2405.20750](https://arxiv.org/abs/2405.20750) | One-step distillation with most convolutions frozen | Abstract |
| LCM-LoRA, Luo et al., [2311.05556](https://arxiv.org/abs/2311.05556) | Coined "acceleration vector"; no anatomy | Snippet |
| Wang et al., ICML 2026, [2605.01929](https://arxiv.org/abs/2605.01929) | Spectral analysis of step-distilled video weights, for LoRA transfer | Abstract |

Verdict from the search: pieces exist, the module-and-block dissection and the
cross-method comparison were not found. Confidence medium.

## Risks

- **"LoRA ranks can be reduced" is known folklore**, and WaDi reports a similar number.
  The paper has to rest on the cross-method and module findings, not on truncation alone.
- **One adapter, 16 prompts, and a reference metric.** Recovery measures closeness to the
  full adapter, not quality. FID may tell a different story at rank 4.
- **Kill criterion:** if Hyper-SD and DMD2 adapters need rank 32 or more, or show no shared
  structure, this becomes a one-adapter observation. Step 1 answers that in two weeks.

**Venue fit:** ICCV or NeurIPS 2027 main track if steps 1-3 hold; a workshop otherwise.
