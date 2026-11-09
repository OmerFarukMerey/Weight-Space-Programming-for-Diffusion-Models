"""Image cache, per-run budget ledger, and the device lock shared by all arms."""
from __future__ import annotations

import fcntl
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from . import config
from .splits import Item


@contextmanager
def device_lock():
    """Serialise model use so that parallel operators queue instead of exhausting memory."""
    config.RUNS.mkdir(parents=True, exist_ok=True)
    with open(config.RUNS / ".device.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _image_dir(split: str, sampler: str, steps: int) -> Path:
    return config.CACHE / "images" / split / sampler / str(steps)


def baseline_images(denoiser, split: str, items: Sequence[Item], steps: int, sampler: str | None = None) -> np.ndarray:
    """Unedited images, generated once. A caller must not have an edit installed."""
    sampler = sampler or config.SAMPLER
    folder = _image_dir(split, sampler, steps)
    paths = [folder / f"{it.id}.png" for it in items]
    missing = [(it, p) for it, p in zip(items, paths) if not p.exists()]
    if missing:
        if denoiser is None:
            raise FileNotFoundError(f"baseline images are not cached for {split}/{sampler}/{steps}")
        if denoiser._applied:
            raise RuntimeError("baseline_images called with an edit installed")
        folder.mkdir(parents=True, exist_ok=True)
        images = denoiser.generate([it for it, _ in missing], steps, sampler)
        for (_, path), image in zip(missing, images):
            Image.fromarray(image).save(path)
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])


class Ledger:
    """Append-only record of every metered action in a run."""

    def __init__(self, run: str, budget: int = config.BUDGET_UNITS):
        self.dir = config.RUNS / run
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ledger.jsonl"
        self.budget = budget

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    @property
    def used(self) -> int:
        return sum(e.get("cost", 0) for e in self.entries())

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def new_folder(self, prefix: str) -> Path:
        """A fresh numbered folder. Creating it is atomic, so parallel commands cannot share one."""
        index = len(self.entries())
        while True:
            folder = self.dir / f"{prefix}_{index:03d}"
            try:
                folder.mkdir()
                return folder
            except FileExistsError:
                index += 1

    def require(self, cost: int) -> None:
        if cost > self.remaining:
            raise SystemExit(f"over budget: this action costs {cost}, {self.remaining} of {self.budget} units remain")

    def record(self, **fields) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}) + "\n")


def contact_sheet(rows: Sequence[Sequence[np.ndarray]], path: Path, crop: int = 128, zoom: int = 2, thumb: int = 256) -> None:
    """Each row is [baseline, edited, reference]: a thumbnail of each, then a magnified centre crop of each."""
    def thumbnail(a: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(a).resize((thumb, thumb), Image.LANCZOS))

    def magnified(a: np.ndarray) -> np.ndarray:
        h, w = a.shape[:2]
        top, left = (h - crop) // 2, (w - crop) // 2
        patch = Image.fromarray(a[top : top + crop, left : left + crop])
        return np.asarray(patch.resize((crop * zoom, crop * zoom), Image.NEAREST))

    lines = [np.concatenate([thumbnail(a) for a in row] + [magnified(a) for a in row], axis=1) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(lines, axis=0)).save(path)
