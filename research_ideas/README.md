# Research direction: anatomy of acceleration adapters

The term project now follows one idea. This folder holds everything needed to start.

| File | What it is |
|---|---|
| [proposal.md](proposal.md) | The project proposal: question, hypotheses, plan, timeline |
| [01-anatomy-of-acceleration-adapters.md](01-anatomy-of-acceleration-adapters.md) | The idea in brief, with the pilot table and prior work |
| [study-guide.md](study-guide.md) | Background explanation and a reading list |
| [pilots.md](pilots.md) | How the pilot experiments were run |

## The idea in one paragraph

Few-step distillation is shipped as a low-rank weight delta on a base model, such as
LCM-LoRA. The project takes such deltas apart in weight space and asks where the few-step
ability lives. In the pilot, LCM-LoRA for Stable Diffusion 1.5 kept 96% of its effect when
cut from rank 64 to rank 4, and 82% of its energy sat in the decoder half of the UNet.

## How this follows from the first experiment

The first experiment in this repo asked whether an LLM could edit denoiser weights to fix
few-step sampling. It could, but only by rediscovering known sampler corrections, and a
blind search did as well. The weight-space question that survived is a different one: not
"can an LLM read weights" but "what does a trained few-step delta look like". That needs no
LLM and no claim the course advisor doubted.

## Deadlines

From a literature search on 2026-11-10. Only SIGGRAPH is official; the rest are estimates
from earlier years.

| Venue | Full paper |
|---|---|
| SIGGRAPH 2027 | 28 Jan 2027 |
| ICML 2027 | late Jan to early Feb 2027 |
| ICCV 2027 | first week of Mar 2027 |
| NeurIPS 2027 | early to mid May 2027 |

## Caveats

- Papers dated 2025-2026 come from an automated literature search. The idea file marks
  which were read. Read them yourself before writing related work.
- The pilot used one adapter and 16 prompts, and measured closeness to the full adapter's
  image, not image quality.
- Four other directions were considered and dropped on 2026-11-11. Their pilot scripts and
  outputs are still in `scripts/pilots/` and `results/pilots/`, described in pilots.md.
