"""Deterministic prompt splits: dev for the operator, two sealed test splits."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class Item:
    id: str
    prompt: str
    category: str
    seed: int


def _seed(split: str, category: str, index: int) -> int:
    key = f"{config.SPLIT_SEED}:{split}:{category}:{index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def build_splits() -> dict[str, list[Item]]:
    pool = json.loads(config.PROMPT_POOL.read_text())
    rng = np.random.default_rng(config.SPLIT_SEED)
    categories = sorted(pool)
    order = rng.permutation(len(categories))
    seen = sorted(categories[i] for i in order[: config.N_SEEN_CATEGORIES])
    new = sorted(categories[i] for i in order[config.N_SEEN_CATEGORIES :])

    out: dict[str, list[Item]] = {"dev": [], "test_seen": [], "test_new": []}
    for category in seen:
        prompts = [pool[category][i] for i in rng.permutation(len(pool[category]))]
        for i, prompt in enumerate(prompts):
            split = "dev" if i < config.DEV_PER_CATEGORY else "test_seen"
            out[split].append(Item(f"{split}-{category}-{i}", prompt, category, _seed(split, category, i)))
    for category in new:
        prompts = [pool[category][i] for i in rng.permutation(len(pool[category]))]
        for i, prompt in enumerate(prompts):
            out["test_new"].append(Item(f"test_new-{category}-{i}", prompt, category, _seed("test_new", category, i)))
    # Interleave categories so that the first 8 dev items span all of them.
    out["dev"].sort(key=lambda it: (int(it.id.rsplit("-", 1)[1]), it.category))
    return out


def dev_items(n: int | None = None) -> list[Item]:
    items = build_splits()["dev"]
    return items if n is None else items[:n]


def as_json(items: list[Item]) -> list[dict]:
    return [asdict(it) for it in items]
