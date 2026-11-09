"""Weight edits: a pure function of the weights, stored as replacement tensors.

An edit script defines ``apply(w)``, where ``w`` maps parameter names to CPU
float32 tensors that it mutates in place. Nothing else is passed in: no images,
no prompts, no model object. Scripts run with autograd disabled and are linted
so that a gradient, a model load or an optimizer cannot appear in one.
"""
from __future__ import annotations

import hashlib
import json
import re
import runpy
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from . import config

FORBIDDEN = re.compile(
    r"\b(backward|autograd|requires_grad|grad|optim|optimizer|functorch|vmap|jacrev|jacfwd|"
    r"diffusers|from_pretrained|UNet2DConditionModel|lpips|sample_latents)\b|torch\.func\b|wspd\.(model|metrics)"
)


class EditError(ValueError):
    pass


def pristine_path() -> Path:
    return config.CACHE / "unet_pristine.safetensors"


def load_pristine() -> dict[str, torch.Tensor]:
    """Unedited UNet weights as CPU float32 tensors. Reading these costs no budget."""
    if not pristine_path().exists():
        raise FileNotFoundError(f"{pristine_path()} is missing; run `wsp-sup prepare` first")
    return load_file(pristine_path(), device="cpu")


def lint(source: str) -> None:
    match = FORBIDDEN.search(source)
    if match:
        raise EditError(f"edit script uses forbidden token {match.group(0)!r}: edits must be gradient-free")


@dataclass
class Edit:
    tensors: dict[str, torch.Tensor]
    summary: dict
    sha256: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.clone() for k, v in self.tensors.items()}, path)
        path.with_suffix(".json").write_text(json.dumps({"sha256": self.sha256, **self.summary}, indent=2))


def diff(new: dict[str, torch.Tensor], pristine: dict[str, torch.Tensor]) -> Edit:
    """Validate an edited state dict and reduce it to the parameters that changed."""
    if set(new) != set(pristine):
        raise EditError(f"edit added or removed parameters: {sorted(set(new) ^ set(pristine))[:5]}")
    tensors: dict[str, torch.Tensor] = {}
    per_param = {}
    n_changed = 0
    n_total = 0
    hasher = hashlib.sha256()
    for name in sorted(pristine):
        old, cur = pristine[name], new[name]
        n_total += old.numel()
        if cur.shape != old.shape:
            raise EditError(f"{name}: shape changed from {tuple(old.shape)} to {tuple(cur.shape)}")
        if not torch.isfinite(cur).all():
            raise EditError(f"{name}: edit produced non-finite values")
        changed = int((cur != old).sum())
        if changed:
            n_changed += changed
            tensors[name] = cur.detach().clone().float()
            hasher.update(name.encode())
            hasher.update(tensors[name].numpy().tobytes())
            per_param[name] = {
                "changed": changed,
                "of": old.numel(),
                "rel_l2": float((cur - old).norm() / old.norm().clamp_min(1e-12)),
            }
    return Edit(
        tensors,
        {
            "weights_changed": n_changed,
            "weights_total": n_total,
            "fraction_changed": n_changed / n_total,
            "params_touched": len(tensors),
            "per_param": per_param,
        },
        hasher.hexdigest()[:16],
    )


def from_script(path: Path, pristine: dict[str, torch.Tensor] | None = None) -> Edit:
    pristine = pristine if pristine is not None else load_pristine()
    source = Path(path).read_text()
    lint(source)
    namespace = runpy.run_path(str(path))
    if "apply" not in namespace:
        raise EditError(f"{path} must define apply(w)")
    work = {k: v.clone() for k, v in pristine.items()}
    with torch.no_grad():
        namespace["apply"](work)
    return diff(work, pristine)


def load(path: Path, pristine: dict[str, torch.Tensor] | None = None) -> Edit:
    pristine = pristine if pristine is not None else load_pristine()
    merged = dict(pristine)
    merged.update(load_file(path, device="cpu"))
    return diff(merged, pristine)


def scaled(edit: Edit, alpha: float, pristine: dict[str, torch.Tensor]) -> Edit:
    """The same edit at strength alpha: old + alpha * (new - old)."""
    merged = dict(pristine)
    for name, value in edit.tensors.items():
        merged[name] = pristine[name] + alpha * (value - pristine[name])
    return diff(merged, pristine)
