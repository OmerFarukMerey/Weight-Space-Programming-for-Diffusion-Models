"""Frozen experiment constants. Anything that affects results lives here."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("WSP_RUNS_DIR", ROOT / "runs"))
CACHE = RUNS / "cache"
RESULTS = ROOT / "results"
PROMPT_POOL = ROOT / "data" / "prompts_pool.json"

MODEL_ID = "segmind/tiny-sd"
MODEL_REVISION = "cad0bd7495fa6c4bcca01b19a723dc91627fe84f"
CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

SIZE = 512
LATENT = SIZE // 8
GUIDANCE = 7.5
BATCH = 1

# Chosen from a dev pilot before any edit existed. docs/PREREGISTRATION.md has why.
SAMPLER = os.environ.get("WSP_SAMPLER", "ddim_trailing")
REF_STEPS = 50
PRIMARY_STEPS = 6
TARGET_STEPS = (5, 6, 8)
SWEEP_STEPS = (5, 6, 8, 10, 12, 16, 24, 50)

# One unit is 8 images at 8 sampler steps or fewer. Longer or larger runs cost
# proportionally, so a budget means the same amount of compute for every arm.
BUDGET_UNITS = 40

SPLIT_SEED = 20261106
N_SEEN_CATEGORIES = 8
DEV_PER_CATEGORY = 3


def unit_cost(n_images: int, steps: int) -> int:
    from math import ceil

    return ceil(n_images / 8 * steps / 8)
