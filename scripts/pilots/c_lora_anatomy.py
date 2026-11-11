"""Pilot C: where does few-step ability live inside an acceleration LoRA?

Takes LCM-LoRA for Stable Diffusion 1.5 apart in weight space, then tests each part at
4 steps: one module type alone, everything but that type, one block group alone, and
rank truncations of the whole adapter. The full adapter's same-seed image is the
reference, so "recovery" is how much of the adapter's effect a part reproduces.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch
from diffusers import AutoencoderKL, LCMScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

from wspd import config
from wspd.metrics import Evaluator, band_deficit_db, paired_bootstrap
from wspd.model import best_device
from wspd.splits import dev_items
from wspd.store import device_lock

torch.set_grad_enabled(False)
BASE, BASE_REV = "stable-diffusion-v1-5/stable-diffusion-v1-5", "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
LORA, LORA_REV = "latent-consistency/lcm-lora-sdv1-5", "cf2fced511dbe7e26c8d1d397e728fbab875db4b"
STEPS, N_PROMPTS = 4, 16
KINDS = ("time", "conv", "self_attn", "cross_attn", "ff", "attn_proj")
BLOCKS = ("down_blocks", "mid_block", "up_blocks")


def kind(name: str) -> str:
    if name.endswith("time_emb_proj"):
        return "time"
    if ".attn1." in name:
        return "self_attn"
    if ".attn2." in name:
        return "cross_attn"
    if ".ff." in name:
        return "ff"
    if name.endswith(("proj_in", "proj_out")):
        return "attn_proj"
    return "conv"


def factorize(up: torch.Tensor, down: torch.Tensor, scale: float):
    """SVD of scale * up @ down through two thin QRs, so truncation is exact and cheap."""
    qu, ru = torch.linalg.qr(up.flatten(1).double())
    qd, rd = torch.linalg.qr(down.flatten(1).double().T)
    u, s, vh = torch.linalg.svd(ru @ rd.T)
    return (qu @ u).float(), (scale * s).float(), (qd @ vh.T).T.float()


def main() -> None:
    device = best_device()
    kw = dict(revision=BASE_REV, torch_dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", **kw).to(device).eval()
    vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", **kw).to(device).eval()
    text = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder", **kw).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer", revision=BASE_REV)
    sched = LCMScheduler.from_pretrained(BASE, subfolder="scheduler", revision=BASE_REV)

    raw = load_file(hf_hub_download(LORA, "pytorch_lora_weights.safetensors", revision=LORA_REV))
    by_flat = {"lora_unet_" + n.replace(".", "_"): n for n, _ in unet.named_modules() if n}
    state = unet.state_dict()
    parts, pristine = {}, {}
    for key in sorted(k for k in raw if k.endswith(".alpha")):
        flat = key[: -len(".alpha")]
        name = by_flat[flat]
        up, down = raw[f"{flat}.lora_up.weight"], raw[f"{flat}.lora_down.weight"]
        parts[name] = factorize(up, down, float(raw[key]) / down.shape[0])
        pristine[name] = state[f"{name}.weight"].detach().cpu().clone()

    # --- Anatomy: how the adapter's energy is spread, with no forward pass.
    energy, base_norm, eff_rank, count = defaultdict(float), defaultdict(float), defaultdict(list), defaultdict(int)
    for name, (_, s, _) in parts.items():
        for group in (kind(name), name.split(".")[0], "all"):
            energy[group] += float((s**2).sum())
            base_norm[group] += float(pristine[name].pow(2).sum())
            count[group] += 1
        cumulative = torch.cumsum(s**2, 0) / (s**2).sum()
        eff_rank[kind(name)].append(int((cumulative < 0.9).sum()) + 1)
        eff_rank["all"].append(eff_rank[kind(name)][-1])
    anatomy = {
        g: {
            "modules": count[g],
            "energy_share": energy[g] / energy["all"],
            "relative_norm": (energy[g] / base_norm[g]) ** 0.5,
            **({"rank_for_90pct_energy_median": float(np.median(eff_rank[g]))} if g in eff_rank else {}),
        }
        for g in (*KINDS, *BLOCKS, "all")
    }
    for g, a in anatomy.items():
        print(f"{g:12s} {a['modules']:3d} modules | energy share {a['energy_share']:.3f} | ||dW||/||W|| {a['relative_norm']:.4f} | rank90 {a.get('rank_for_90pct_energy_median', '')}", flush=True)

    def install(keep, rank: int = 64) -> None:
        for name, (u, s, vh) in parts.items():
            weight = pristine[name]
            if keep(name):
                delta = (u[:, :rank] * s[:rank]) @ vh[:rank]
                weight = weight + delta.view_as(weight)
            state[f"{name}.weight"].copy_(weight.to(device))

    items = dev_items(N_PROMPTS)
    prompts = [it.prompt for it in items]

    def encode(p):
        ids = tok(p, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
        return text(ids.to(device))[0]

    conds = [encode(p) for p in prompts]

    def generate() -> np.ndarray:
        out = []
        for it, cond in zip(items, conds):
            sched.set_timesteps(STEPS, device=device)
            z = torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(it.seed)).to(device) * sched.init_noise_sigma
            noise = torch.Generator("cpu").manual_seed(it.seed + 1)
            for t in sched.timesteps:
                eps = unet(z, t, encoder_hidden_states=cond).sample  # guidance 1.0: no unconditional pass
                z, denoised = sched.step(eps, t, z, generator=noise, return_dict=False)
            x = vae.decode(denoised / vae.config.scaling_factor).sample
            out.append(((x.clamp(-1, 1) + 1) * 127.5).round().permute(0, 2, 3, 1).byte().cpu().numpy())
        return np.concatenate(out)

    arms = {"none": (lambda n: False, 64), "full": (lambda n: True, 64)}
    for k in KINDS:
        arms[f"only_{k}"] = (lambda n, k=k: kind(n) == k, 64)
        arms[f"drop_{k}"] = (lambda n, k=k: kind(n) != k, 64)
    for b in BLOCKS:
        arms[f"only_{b}"] = (lambda n, b=b: n.split(".")[0] == b, 64)
    for r in (32, 16, 8, 4):
        arms[f"rank_{r}"] = (lambda n: True, r)

    evaluator = Evaluator(device)
    images = {}
    for name, (keep, rank) in arms.items():
        install(keep, rank)
        images[name] = generate()
        print(f"generated {name}", flush=True)
    install(lambda n: False)

    full, none = images["full"], images["none"]
    worst = evaluator.lpips(none, full)
    results = {}
    for name, img in images.items():
        dist = evaluator.lpips(img, full)
        deficit = band_deficit_db(img, full)
        texture = (np.abs(deficit["coarse"]) + np.abs(deficit["fine"])) / 2
        results[name] = {
            "lpips_to_full": float(dist.mean()),
            "recovery": float(1 - dist.mean() / worst.mean()),
            "recovery_ci": paired_bootstrap(1 - dist / worst),
            "texture_gap_db": float(texture.mean()),
            "signed_fine_deficit_db": float(deficit["fine"].mean()),
            "clip": float(evaluator.clip_similarity(img, prompts).mean()),
            "weights_in_part": int(sum(pristine[n].numel() for n in parts if arms[name][0](n))),
        }
        r = results[name]
        print(f"{name:18s} LPIPS to full {r['lpips_to_full']:.3f} | recovery {r['recovery']:+.2f} | texture gap {r['texture_gap_db']:.2f} dB | clip {r['clip']:.3f}", flush=True)

    out = config.RESULTS / "pilots"
    from PIL import Image

    show = ["none", "only_time", "only_conv", "only_self_attn", "only_cross_attn", "rank_4", "rank_16", "full"]
    rows = [np.concatenate([np.asarray(Image.fromarray(images[a][i]).resize((192, 192), Image.LANCZOS)) for a in show], axis=1) for i in range(6)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(out / "c_lora_anatomy.png")
    (out / "c_lora_anatomy.json").write_text(
        json.dumps({"base": BASE, "lora": LORA, "steps": STEPS, "n_prompts": N_PROMPTS, "sheet_columns": show, "anatomy": anatomy, "arms": results}, indent=1)
    )


if __name__ == "__main__":
    with device_lock():
        main()
