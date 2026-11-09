"""Exploratory, not pre-registered: did these edits need to be weight edits at all?

The sealed edits amount to a noise gain and a timestep relabel. Both can be applied in
the sampling loop with no weight touched. This compares that replica to the sealed edit
on all sealed prompts, using the operators' own constants, so nothing is selected on
test data. It also measures how much composition error each arm leaves behind.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from wspd import config, edits
from wspd.metrics import Evaluator, paired_bootstrap, summarize, swd
from wspd.model import Denoiser
from wspd.splits import build_splits
from wspd.store import baseline_images, device_lock

N = config.PRIMARY_STEPS
OP1_KNOTS = [(0, 0), (60, 13), (124, 26), (166, 50), (199, 70), (249, 60), (332, 54), (399, 54), (499, 0), (999, 0)]
OP3_KNOTS = [(0, 0), (124, 60), (166, 85), (199, 100), (249, 70), (400, 70), (470, 0), (1000, 0)]


class CorrectedLoop:
    """The UNet as the sampler would call it, with the correction applied around it."""

    def __init__(self, unet, knots, gain: float):
        self.unet, self.gain = unet, gain
        self.xs, self.ys = zip(*knots) if knots else ((0, 1000), (0, 0))

    def __call__(self, x, t, encoder_hidden_states):
        relabeled = float(t) - float(np.interp(float(t), self.xs, self.ys))
        out = self.unet(x, torch.tensor(relabeled, device=x.device), encoder_hidden_states=encoder_hidden_states)
        return SimpleNamespace(sample=self.gain * out.sample)


def composition_error(images: np.ndarray, refs: np.ndarray, sigma: float = 16.0) -> np.ndarray:
    """RMS difference after removing all detail, per image, on a 0-255 scale."""
    low = lambda a: gaussian_filter(a.astype(np.float32), sigma=(0, sigma, sigma, 0))
    return np.sqrt(((low(images) - low(refs)) ** 2).mean(axis=(1, 2, 3)))


def main() -> None:
    splits = build_splits()
    items = {s: splits[s] for s in ("test_seen", "test_new")}
    flat = [(s, it) for s, its in items.items() for it in its]
    prompts = [it.prompt for _, it in flat]
    with device_lock():
        denoiser = Denoiser()
        evaluator = Evaluator(denoiser.device)
        real_unet = denoiser.unet

        def cached(steps):
            return np.concatenate([baseline_images(None, s, [it], steps) for s, it in flat])

        refs, base = cached(config.REF_STEPS), cached(N)
        arms: dict[str, np.ndarray] = {"baseline_6": base, "baseline_8": cached(8), "baseline_16": cached(16)}

        def weight_arm(name, edit):
            denoiser.set_edit(edit)
            arms[name] = denoiser.generate([it for _, it in flat], N)
            denoiser.set_edit(None)
            print(name, "done", flush=True)

        def loop_arm(name, knots, gain):
            denoiser.unet = CorrectedLoop(real_unet, knots, gain)
            arms[name] = denoiser.generate([it for _, it in flat], N)
            denoiser.unet = real_unet
            print(name, "done", flush=True)

        pristine = denoiser.pristine
        weight_arm("op1_weights", edits.load(config.RUNS / "op1/final/edit.safetensors", pristine))
        loop_arm("op1_in_sampler", OP1_KNOTS, 0.9885)
        weight_arm("op3_weights", edits.load(config.RUNS / "op3/final/edit.safetensors", pristine))
        loop_arm("op3_in_sampler", OP3_KNOTS, 1.0)
        # 0.9825 is the gain-only candidate op2 chose on dev.
        gain_only = dict(pristine)
        gain_only["conv_out.weight"] = pristine["conv_out.weight"] * 0.9825
        weight_arm("conv_out_gain_only", edits.diff(gain_only, pristine))
        loop_arm("gain_in_sampler", None, 0.9825)

        per = {name: evaluator.evaluate(images, refs, prompts) for name, images in arms.items()}
        report = {"note": "exploratory, not pre-registered", "n_prompts": len(flat), "swd_floor": swd(refs, refs, seed=1)["mean"], "arms": {}, "pairs": {}}
        for name, images in arms.items():
            summary = summarize(per[name])
            report["arms"][name] = {
                "texture": summary["texture"],
                "lpips": summary["lpips"],
                "clip": summary["clip"],
                "contrast_ratio": summary["contrast_ratio"],
                "swd": swd(images, refs)["mean"],
                "composition_rms": float(composition_error(images, refs).mean()),
                "delta_texture": paired_bootstrap(per[name]["texture"] - per["baseline_6"]["texture"]),
                "delta_lpips": paired_bootstrap(per[name]["lpips"] - per["baseline_6"]["lpips"]),
                "delta_composition": paired_bootstrap(composition_error(images, refs) - composition_error(base, refs)),
            }
        for a, b in (("op1_weights", "op1_in_sampler"), ("op3_weights", "op3_in_sampler"), ("conv_out_gain_only", "gain_in_sampler"), ("op1_weights", "baseline_6")):
            report["pairs"][f"{a} vs {b}"] = {
                "lpips_between": float(evaluator.lpips(arms[a], arms[b]).mean()),
                "mean_abs_pixel_diff": float(np.abs(arms[a].astype(float) - arms[b].astype(float)).mean()),
            }
    (config.RESULTS / "exploratory_sampler_equivalence.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
