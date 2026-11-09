"""Image-quality metrics, all measured against the same-seed 50-step reference.

Several metrics are needed because they fail differently. Band deficits see
lost energy but cannot tell real texture from added noise. LPIPS sees structure
but also punishes a texture that is plausible and merely not the same one. SWD
compares texture distributions without alignment. CLIP guards the semantics.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import config

# Cycles per pixel at 512px. The VAE latent Nyquist limit sits at 1/16.
BANDS = {
    "structure": (1 / 64, 1 / 16),
    "coarse": (1 / 16, 1 / 8),
    "fine": (1 / 8, 1 / 4),
    "grain": (1 / 4, 1 / 2),
}


def luminance(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float64) @ np.array([0.299, 0.587, 0.114]) / 255.0


def band_power_db(images: np.ndarray) -> dict[str, np.ndarray]:
    """Mean radial power in each band, per image, in dB."""
    gray = luminance(images)
    h, w = gray.shape[1:]
    window = np.outer(np.hanning(h), np.hanning(w))
    gray = (gray - gray.mean(axis=(1, 2), keepdims=True)) * window
    power = np.abs(np.fft.fft2(gray)) ** 2
    radius = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    return {
        name: 10 * np.log10(power[:, (radius >= lo) & (radius < hi)].mean(axis=1) + 1e-12)
        for name, (lo, hi) in BANDS.items()
    }


def band_deficit_db(images: np.ndarray, refs: np.ndarray) -> dict[str, np.ndarray]:
    """Reference power minus image power: positive means detail is missing."""
    a, b = band_power_db(images), band_power_db(refs)
    return {name: b[name] - a[name] for name in BANDS}


def _laplacian(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    out = []
    for _ in range(levels):
        down = F.avg_pool2d(x, 2)
        out.append(x - F.interpolate(down, scale_factor=2, mode="bilinear", align_corners=False))
        x = down
    return out


def swd(images: np.ndarray, refs: np.ndarray, seed: int = 0, patches: int = 128, size: int = 7, projections: int = 256) -> dict[str, float]:
    """Sliced Wasserstein distance between Laplacian patch distributions (Karras et al. 2018), times 1000."""
    gen = torch.Generator().manual_seed(seed)
    to_tensor = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).float() / 127.5 - 1
    result = {}
    for level, (x, y) in enumerate(zip(_laplacian(to_tensor(images), 3), _laplacian(to_tensor(refs), 3))):

        def sample(t: torch.Tensor) -> torch.Tensor:
            n, _, h, w = t.shape
            top = torch.randint(0, h - size, (n, patches), generator=gen)
            left = torch.randint(0, w - size, (n, patches), generator=gen)
            cells = [t[i, :, top[i, j] : top[i, j] + size, left[i, j] : left[i, j] + size] for i in range(n) for j in range(patches)]
            return torch.stack(cells).flatten(1)

        px, py = sample(x), sample(y)
        mean, std = py.mean(0, keepdim=True), py.std(0, keepdim=True).clamp_min(1e-6)
        px, py = (px - mean) / std, (py - mean) / std
        dirs = F.normalize(torch.randn(px.shape[1], projections, generator=gen), dim=0)
        result[f"level{level}"] = float((px @ dirs).sort(0).values.sub((py @ dirs).sort(0).values).abs().mean() * 1000)
    result["mean"] = float(np.mean(list(result.values())))
    return result


class Evaluator:
    """Loads LPIPS and CLIP lazily so that a spectrum-only caller stays cheap."""

    def __init__(self, device: str):
        self.device = device
        self._lpips = None
        self._clip = None

    def lpips(self, images: np.ndarray, refs: np.ndarray, batch: int = 8) -> np.ndarray:
        if self._lpips is None:
            import lpips

            self._lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        out = []
        for i in range(0, len(images), batch):
            a = torch.from_numpy(images[i : i + batch]).permute(0, 3, 1, 2).float().to(self.device) / 127.5 - 1
            b = torch.from_numpy(refs[i : i + batch]).permute(0, 3, 1, 2).float().to(self.device) / 127.5 - 1
            with torch.no_grad():
                out.append(self._lpips(a, b).flatten().cpu().numpy())
        return np.concatenate(out)

    def clip_similarity(self, images: np.ndarray, prompts: list[str], batch: int = 16) -> np.ndarray:
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor

            kw = dict(revision=config.CLIP_REVISION, use_safetensors=False)
            self._clip = (
                CLIPModel.from_pretrained(config.CLIP_ID, **kw).to(self.device).eval(),
                CLIPProcessor.from_pretrained(config.CLIP_ID, revision=config.CLIP_REVISION),
            )
        model, processor = self._clip
        out = []
        for i in range(0, len(images), batch):
            inputs = processor(
                text=prompts[i : i + batch], images=list(images[i : i + batch]), return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                res = model(**inputs)
            im = F.normalize(res.image_embeds, dim=-1)
            tx = F.normalize(res.text_embeds, dim=-1)
            out.append((im * tx).sum(-1).cpu().numpy())
        return np.concatenate(out)

    def evaluate(self, images: np.ndarray, refs: np.ndarray, prompts: list[str]) -> dict[str, np.ndarray]:
        """Per-image metric arrays, keyed by metric name."""
        deficits = band_deficit_db(images, refs)
        gray_i, gray_r = luminance(images), luminance(refs)
        result = {f"deficit_{k}": v for k, v in deficits.items()}
        result["texture"] = (np.abs(deficits["coarse"]) + np.abs(deficits["fine"])) / 2
        result["lpips"] = self.lpips(images, refs)
        result["clip"] = self.clip_similarity(images, prompts)
        result["luma_shift"] = gray_i.mean(axis=(1, 2)) - gray_r.mean(axis=(1, 2))
        result["contrast_ratio"] = gray_i.std(axis=(1, 2)) / gray_r.std(axis=(1, 2)).clip(1e-6)
        return result


def summarize(per_image: dict[str, np.ndarray]) -> dict[str, float]:
    out = {k: float(np.mean(v)) for k, v in per_image.items()}
    for k, v in per_image.items():
        if k.startswith("deficit_"):
            out[f"abs_{k}"] = float(np.mean(np.abs(v)))
    return out


def paired_bootstrap(delta: np.ndarray, n: int = 10_000, seed: int = 0) -> dict[str, float]:
    """Mean of a per-prompt difference with a 95% percentile interval."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(delta.mean()), "lo": float(lo), "hi": float(hi)}


def step_equivalent(value: float, steps: list[int], curve: list[float]) -> float:
    """The baseline step count whose metric equals value, for a lower-is-better metric.

    Interpolates in log steps over the curve made monotone. Values worse than
    the first point return its step count, values better than the last return
    the last, so the result always stays inside the measured range.
    """
    best = np.minimum.accumulate(np.asarray(curve, dtype=float))
    if value >= best[0]:
        return float(steps[0])
    if value <= best[-1]:
        return float(steps[-1])
    for i in range(len(steps) - 1):
        if best[i] >= value >= best[i + 1] and best[i] > best[i + 1]:
            frac = (best[i] - value) / (best[i] - best[i + 1])
            return float(np.exp(np.log(steps[i]) + frac * (np.log(steps[i + 1]) - np.log(steps[i]))))
    return float(steps[0])
