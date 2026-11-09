"""The denoiser, a few-step sampler, and activation probes. Everything runs without autograd."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel
from safetensors.torch import save_file
from transformers import CLIPTextModel, CLIPTokenizer

from . import config
from .edits import Edit, load_pristine, pristine_path
from .splits import Item

torch.set_grad_enabled(False)

# "ddim" uses the spacing tiny-sd ships with. The pilot compares all three, because
# the direction of the few-step error turns out to depend on this choice.
SAMPLERS = {
    "ddim": lambda cfg: DDIMScheduler.from_config(cfg, timestep_spacing="linspace", steps_offset=0),
    "ddim_trailing": lambda cfg: DDIMScheduler.from_config(cfg, timestep_spacing="trailing", steps_offset=0),
    "dpm2m": lambda cfg: DPMSolverMultistepScheduler.from_config(cfg, algorithm_type="dpmsolver++", use_karras_sigmas=True),
}


def best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_pristine() -> None:
    """Materialise the UNet weights once as a float32 safetensors file."""
    if pristine_path().exists():
        return
    unet = UNet2DConditionModel.from_pretrained(
        config.MODEL_ID, subfolder="unet", revision=config.MODEL_REVISION, torch_dtype=torch.float32, use_safetensors=False
    )
    pristine_path().parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.clone().contiguous() for k, v in unet.state_dict().items()}, pristine_path())


class Denoiser:
    def __init__(self, device: str | None = None):
        self.device = device or best_device()
        kw = dict(revision=config.MODEL_REVISION, torch_dtype=torch.float32, use_safetensors=False)
        self.unet = UNet2DConditionModel.from_pretrained(config.MODEL_ID, subfolder="unet", **kw).to(self.device).eval()
        self.vae = AutoencoderKL.from_pretrained(config.MODEL_ID, subfolder="vae", **kw).to(self.device).eval()
        self.text = CLIPTextModel.from_pretrained(config.MODEL_ID, subfolder="text_encoder", **kw).to(self.device).eval()
        self.tok = CLIPTokenizer.from_pretrained(config.MODEL_ID, subfolder="tokenizer", revision=config.MODEL_REVISION)
        sched = DDIMScheduler.from_pretrained(config.MODEL_ID, subfolder="scheduler", revision=config.MODEL_REVISION)
        self.sched_config = sched.config
        self.alphas = sched.alphas_cumprod.to(self.device)
        self.pristine = load_pristine()
        self._applied: list[str] = []

    def set_edit(self, edit: Edit | None) -> None:
        """Install an edit, first restoring whatever was installed before."""
        state = self.unet.state_dict()
        for name in self._applied:
            state[name].copy_(self.pristine[name].to(self.device))
        self._applied = []
        if edit is not None:
            for name, value in edit.tensors.items():
                state[name].copy_(value.to(self.device))
                self._applied.append(name)

    def _encode(self, prompts: Sequence[str]) -> torch.Tensor:
        ids = self.tok(
            list(prompts), padding="max_length", max_length=self.tok.model_max_length, truncation=True, return_tensors="pt"
        ).input_ids
        return self.text(ids.to(self.device))[0]

    def sample_latents(
        self,
        items: Sequence[Item],
        steps: int,
        sampler: str | None = None,
        on_step: Callable[[int, int, torch.Tensor, torch.Tensor, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        sched = SAMPLERS[sampler or config.SAMPLER](self.sched_config)
        sched.set_timesteps(steps, device=self.device)
        noise = [
            torch.randn((4, config.LATENT, config.LATENT), generator=torch.Generator("cpu").manual_seed(it.seed))
            for it in items
        ]
        z = torch.stack(noise).to(self.device) * sched.init_noise_sigma
        emb = torch.cat([self._encode([""] * len(items)), self._encode([it.prompt for it in items])])
        for i, t in enumerate(sched.timesteps):
            inp = sched.scale_model_input(torch.cat([z, z]), t)
            uncond, cond = self.unet(inp, t, encoder_hidden_states=emb).sample.chunk(2)
            eps = uncond + config.GUIDANCE * (cond - uncond)
            if on_step is not None:
                alpha = self.alphas[int(t)]
                on_step(i, int(t), z, eps, (z - (1 - alpha).sqrt() * eps) / alpha.sqrt())
            z = sched.step(eps, t, z).prev_sample
        return z

    def decode(self, z: torch.Tensor) -> np.ndarray:
        x = self.vae.decode(z / self.vae.config.scaling_factor).sample
        return ((x.clamp(-1, 1) + 1) * 127.5).round().permute(0, 2, 3, 1).byte().cpu().numpy()

    def generate(self, items: Sequence[Item], steps: int, sampler: str | None = None) -> np.ndarray:
        out = []
        for i in range(0, len(items), config.BATCH):
            out.append(self.decode(self.sample_latents(items[i : i + config.BATCH], steps, sampler)))
        return np.concatenate(out)


def spectrum_fractions(x: torch.Tensor) -> tuple[float, float, float]:
    """Power split of an NCHW map into low (<1/8), mid (1/8-1/4) and high (>1/4) cycles per sample."""
    x = x.float().cpu()
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    power = torch.fft.rfft2(x).abs().pow(2).mean(dim=(0, 1))
    fy = torch.fft.fftfreq(x.shape[-2])
    fx = torch.fft.rfftfreq(x.shape[-1])
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    total = power.sum().clamp_min(1e-12)
    low = power[radius < 0.125].sum() / total
    mid = power[(radius >= 0.125) & (radius < 0.25)].sum() / total
    return float(low), float(mid), float(1 - low - mid)


class ActivationProbe:
    """Records RMS and spectral split of selected module outputs at each sampler step."""

    def __init__(self, unet: torch.nn.Module, names: Sequence[str]):
        self.step = 0
        self.records: list[dict] = []
        modules = dict(unet.named_modules())
        self.handles = [modules[n].register_forward_hook(self._hook(n)) for n in names]

    def _hook(self, name: str):
        def fn(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(out) and out.dim() == 4 and min(out.shape[-2:]) >= 4:
                low, mid, high = spectrum_fractions(out)
                self.records.append(
                    {
                        "module": name,
                        "step": self.step,
                        "shape": list(out.shape[1:]),
                        "rms": float(out.pow(2).mean().sqrt()),
                        "low": low,
                        "mid": mid,
                        "high": high,
                    }
                )

        return fn

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _mean_by(records: list[dict], keys: Sequence[str], values: Sequence[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        groups.setdefault(tuple(record[k] for k in keys), []).append(record)
    return [
        {**{k: group[0][k] for k in keys}, **{v: float(np.mean([r[v] for r in group])) for v in values}}
        for group in groups.values()
    ]


def run_probe(denoiser: Denoiser, items: Sequence[Item], steps: int, names: Sequence[str]) -> dict:
    """Activation and x0-estimate statistics per step, averaged over items."""
    probe = ActivationProbe(denoiser.unet, names)
    sampler_log: list[dict] = []

    def on_step(i, t, z, eps, x0):
        probe.step = i + 1  # the next UNet call belongs to the next step
        low, mid, high = spectrum_fractions(x0)
        sampler_log.append(
            {"step": i, "t": t, "eps_rms": float(eps.pow(2).mean().sqrt()), "x0_rms": float(x0.pow(2).mean().sqrt()),
             "x0_low": low, "x0_mid": mid, "x0_high": high}
        )

    for start in range(0, len(items), config.BATCH):
        probe.step = 0
        denoiser.sample_latents(items[start : start + config.BATCH], steps, on_step=on_step)
    probe.close()
    shapes = {r["module"]: r["shape"] for r in probe.records}
    records = _mean_by(probe.records, ("module", "step"), ("rms", "low", "mid", "high"))
    return {
        "steps": steps,
        "n_images": len(items),
        "modules": list(names),
        "shapes": shapes,
        "sampler": _mean_by(sampler_log, ("step", "t"), ("eps_rms", "x0_rms", "x0_low", "x0_mid", "x0_high")),
        "records": records,
    }
