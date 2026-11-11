"""Exploratory sampler-level interventions: custom schedules, relabelled time, per-block time.

Nothing here changes a weight. It exists to pilot follow-up ideas on the same splits and
metrics as the weight-edit experiment, so the numbers are comparable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from . import config
from .metrics import Evaluator, paired_bootstrap, step_equivalent, summarize, swd
from .model import Denoiser
from .splits import Item, build_splits
from .store import baseline_images

# tiny-sd has no mid block: three resolutions on the way down, the same three on the way up.
GROUPS = {
    "high": ("down_blocks.0", "up_blocks.2"),  # 64x64 latents
    "mid": ("down_blocks.1", "up_blocks.1"),  # 32x32
    "low": ("down_blocks.2", "up_blocks.0"),  # 16x16
}

# The time shift op3 sealed in the weight-edit experiment, as (t, delta) knots.
OP3_KNOTS = ((0, 0), (124, 60), (166, 85), (199, 100), (249, 70), (400, 70), (470, 0), (1000, 0))


def trailing(n: int) -> list[int]:
    """The timesteps diffusers visits with trailing spacing."""
    return [int(round(1000 * (1 - i / n))) - 1 for i in range(n)]


def warped(n: int, rho: float) -> list[int]:
    """n timesteps from 999 down. rho < 1 spends more steps at high noise, rho > 1 at low noise."""
    return [max(int(round(1000 * (1 - i / n) ** rho)) - 1, 0) for i in range(n)]


@dataclass(frozen=True)
class Plan:
    """One sampling recipe. The defaults are plain DDIM with full guidance."""

    timesteps: tuple[int, ...]
    gain: float = 1.0  # multiplies the guided noise prediction
    relabel: float = 0.0  # the network is told t - relabel * (t - t_next) ...
    relabel_below: int = 1000  # ... for t below this
    relabel_steps: tuple[float, ...] | None = None  # or one such fraction per step
    knots: tuple[tuple[float, float], ...] | None = None  # or an explicit delta(t) curve, which wins
    block_scale: tuple[float, float, float] | None = None  # multiplier on the shift for (high, mid, low)
    guided: tuple[bool, ...] | None = None  # per step: run the unconditional pass

    def delta(self, t: int, t_next: int, i: int = 0) -> float:
        if self.knots is not None:
            xs, ys = zip(*self.knots)
            return float(np.interp(t, xs, ys))
        if self.relabel_steps is not None:
            return self.relabel_steps[i] * (t - t_next)
        return self.relabel * (t - t_next) if t < self.relabel_below else 0.0

    @property
    def nfe(self) -> int:
        """UNet evaluations per image: two for a guided step, one for an unguided step."""
        flags = self.guided or (True,) * len(self.timesteps)
        return sum(2 if g else 1 for g in flags)


class BlockClock:
    """Lets each resolution group of the UNet receive its own time embedding."""

    def __init__(self, unet):
        self.unet = unet
        self.override: dict[str, torch.Tensor] = {}
        modules = dict(unet.named_modules())
        self.handles = [
            modules[name].register_forward_pre_hook(self._hook(name), with_kwargs=True)
            for names in GROUPS.values()
            for name in names
        ]

    def _hook(self, name: str):
        def fn(_module, args, kwargs):
            emb = self.override.get(name)
            if emb is not None and "temb" in kwargs:
                kwargs = dict(kwargs, temb=emb)
            return args, kwargs

        return fn

    def embed(self, t: float, like: torch.Tensor) -> torch.Tensor:
        steps = torch.full((like.shape[0],), float(t), device=like.device)
        return self.unet.time_embedding(self.unet.time_proj(steps).to(like.dtype))

    def set(self, times: dict[str, float] | None, like: torch.Tensor) -> None:
        self.override = {}
        for group, t in (times or {}).items():
            emb = self.embed(t, like)
            for name in GROUPS[group]:
                self.override[name] = emb

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def composition_error(images: np.ndarray, refs: np.ndarray, sigma: float = 16.0) -> np.ndarray:
    """RMS difference after removing all detail, per image, on a 0-255 scale."""
    low = lambda a: gaussian_filter(a.astype(np.float32), sigma=(0, sigma, sigma, 0))
    return np.sqrt(((low(images) - low(refs)) ** 2).mean(axis=(1, 2, 3)))


class Bench:
    """A loaded model plus the cached references, for running plans on dev or on the sealed prompts."""

    def __init__(self):
        self.den = Denoiser()
        self.ev = Evaluator(self.den.device)
        self.clock = BlockClock(self.den.unet)
        splits = build_splits()
        self.sets = {
            "dev": [("dev", it) for it in splits["dev"]],
            "test": [("test_seen", it) for it in splits["test_seen"]] + [("test_new", it) for it in splits["test_new"]],
        }
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        self._per: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    def cached(self, name: str, steps: int) -> np.ndarray:
        """Unedited images from the main experiment (diffusers DDIM, trailing spacing)."""
        key = (name, steps)
        if key not in self._cache:
            self._cache[key] = np.concatenate([baseline_images(None, s, [it], steps) for s, it in self.sets[name]])
        return self._cache[key]

    def items(self, name: str) -> list[Item]:
        return [it for _, it in self.sets[name]]

    @torch.no_grad()
    def latents(self, items: Sequence[Item], plan: Plan) -> torch.Tensor:
        den = self.den
        alphas = den.alphas
        noise = [
            torch.randn((4, config.LATENT, config.LATENT), generator=torch.Generator("cpu").manual_seed(it.seed))
            for it in items
        ]
        z = torch.stack(noise).to(den.device)
        cond = den._encode([it.prompt for it in items])
        uncond = den._encode([""] * len(items))
        ts = list(plan.timesteps)
        for i, t in enumerate(ts):
            last = i + 1 == len(ts)
            t_next = 0 if last else ts[i + 1]
            delta = plan.delta(t, t_next, i)
            guided = plan.guided is None or plan.guided[i]
            inp = torch.cat([z, z]) if guided else z
            emb = torch.cat([uncond, cond]) if guided else cond
            if plan.block_scale is None:
                self.clock.set(None, inp)
                seen = t - delta
            else:
                self.clock.set({g: t - m * delta for g, m in zip(GROUPS, plan.block_scale)}, inp)
                seen = t
            out = den.unet(inp, torch.tensor(float(seen), device=den.device), encoder_hidden_states=emb).sample
            if guided:
                u, c = out.chunk(2)
                eps = u + config.GUIDANCE * (c - u)
            else:
                eps = out
            eps = plan.gain * eps
            a_t = alphas[t]
            a_prev = alphas[0] if last else alphas[t_next]
            x0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            z = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        self.clock.set(None, z)
        return z

    def run(self, name: str, plan: Plan) -> np.ndarray:
        items = self.items(name)
        out = [self.den.decode(self.latents(items[i : i + config.BATCH], plan)) for i in range(0, len(items), config.BATCH)]
        return np.concatenate(out)

    def per_image(self, name: str, images: np.ndarray) -> dict[str, np.ndarray]:
        refs = self.cached(name, config.REF_STEPS)
        per = self.ev.evaluate(images, refs, [it.prompt for it in self.items(name)])
        per["composition"] = composition_error(images, refs)
        return per

    def baseline(self, name: str, steps: int = config.PRIMARY_STEPS) -> dict[str, np.ndarray]:
        key = (name, steps)
        if key not in self._per:
            self._per[key] = self.per_image(name, self.cached(name, steps))
        return self._per[key]

    def measure(self, name: str, images: np.ndarray, nfe: int | None = None) -> dict:
        """Summary metrics, the fixed score, and paired changes against the unedited 6-step sampler."""
        per, base = self.per_image(name, images), self.baseline(name)
        out = {k: float(np.mean(per[k])) for k in ("texture", "lpips", "composition", "clip", "contrast_ratio", "deficit_coarse", "deficit_fine", "deficit_grain")}
        out["swd"] = swd(images, self.cached(name, config.REF_STEPS))["mean"]
        out["score"] = out["texture"] / base["texture"].mean() + out["lpips"] / base["lpips"].mean()
        for key in ("texture", "lpips", "composition"):
            out[f"delta_{key}"] = paired_bootstrap(per[key] - base[key])
        out["steps_matched"] = self.steps_matched(name, out)
        if nfe is not None:
            out["nfe"] = nfe
        out["_per"] = {k: per[k] for k in ("texture", "lpips", "composition")}
        return out

    def steps_matched(self, name: str, summary: dict) -> dict[str, float]:
        """The unedited step count that gives the same value, for each metric."""
        grid = [s for s in config.SWEEP_STEPS if s != config.REF_STEPS]
        curves = {s: self.baseline(name, s) for s in grid}
        return {
            key: step_equivalent(summary[key], grid, [float(curves[s][key].mean()) for s in grid])
            for key in ("texture", "lpips", "composition")
        }


def compare(a: dict, b: dict) -> dict:
    """Paired change of arm a against arm b on the same prompts, with 95% intervals."""
    return {k: paired_bootstrap(a["_per"][k] - b["_per"][k]) for k in ("texture", "lpips", "composition")}


def public(measured: dict) -> dict:
    return {k: v for k, v in measured.items() if not k.startswith("_")}


def line(name: str, m: dict) -> str:
    eq = m["steps_matched"]
    return (
        f"{name:26s} nfe {m.get('nfe', 12):3d} | T {m['texture']:.2f} (={eq['texture']:4.1f} st) | L {m['lpips']:.3f} (={eq['lpips']:4.1f} st) "
        f"| comp {m['composition']:5.2f} (={eq['composition']:4.1f} st) | swd {m['swd']:4.0f} | clip {m['clip']:.3f} | contrast {m['contrast_ratio']:.2f} | score {m['score']:.3f}"
    )
