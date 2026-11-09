"""`wsp-sup`: reference generation, sealed evaluation and controls. Not for the operator."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from . import config, edits
from .cli import evaluate_edit
from .metrics import Evaluator, paired_bootstrap, step_equivalent, summarize, swd
from .model import Denoiser, prepare_pristine, run_probe
from .splits import Item, as_json, build_splits, dev_items
from .store import Ledger, baseline_images, contact_sheet, device_lock

TEST_SPLITS = ("test_seen", "test_new")
PROBE_MODULES = (
    r"^(conv_in|conv_out|mid_block|down_blocks\.\d+|up_blocks\.\d+"
    r"|up_blocks\.\d+\.resnets\.\d+|up_blocks\.\d+\.attentions\.\d+|down_blocks\.\d+\.resnets\.\d+)$"
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sweep(denoiser, evaluator, split: str, items: list[Item], sampler: str, steps_list=config.SWEEP_STEPS) -> dict:
    """Baseline metrics across step counts, plus the per-image arrays that bootstraps need."""
    prompts = [it.prompt for it in items]
    refs = baseline_images(denoiser, split, items, config.REF_STEPS, sampler)
    curve, per_image = {}, {}
    for steps in steps_list:
        images = baseline_images(denoiser, split, items, steps, sampler)
        per = evaluator.evaluate(images, refs, prompts)
        summary = summarize(per)
        summary["swd"] = swd(images, refs)["mean"] if steps != config.REF_STEPS else 0.0
        curve[steps] = summary
        per_image[steps] = per
        log(f"{split}/{sampler} steps={steps}: {_brief(summary)}")
    return {"curve": curve, "per_image": per_image}


def _brief(s: dict) -> str:
    return (
        f"coarse {s['deficit_coarse']:+.2f}dB fine {s['deficit_fine']:+.2f}dB grain {s['deficit_grain']:+.2f}dB "
        f"lpips {s['lpips']:.3f} swd {s.get('swd', float('nan')):.1f} clip {s['clip']:.3f}"
    )


def _probe_modules(denoiser) -> list[str]:
    return [n for n, _ in denoiser.unet.named_modules() if n and re.search(PROBE_MODULES, n)]


def cmd_prepare(args) -> None:
    prepare_pristine()
    with device_lock():
        denoiser = Denoiser()
        evaluator = Evaluator(denoiser.device)
        items = dev_items()
        result = sweep(denoiser, evaluator, "dev", items, config.SAMPLER)
        refs = baseline_images(denoiser, "dev", items, config.REF_STEPS)
        floor = swd(refs, refs, seed=1)["mean"]
        log(f"swd noise floor on {len(items)} images: {floor:.1f}")
        (config.CACHE / "baseline_dev.json").write_text(json.dumps({"swd_floor": floor, "curve": result["curve"]}, indent=1))
        for steps in (config.PRIMARY_STEPS, config.REF_STEPS):
            data = run_probe(denoiser, items[:8], steps, _probe_modules(denoiser))
            (config.CACHE / f"probe_baseline_{steps}.json").write_text(json.dumps(data))
    log("prepared")


def cmd_pilot(args) -> None:
    """Does a consistent few-step error exist, and in which sampler and band?"""
    out = {}
    with device_lock():
        denoiser = Denoiser()
        evaluator = Evaluator(denoiser.device)
        items = dev_items(args.n)
        for sampler in args.samplers.split(","):
            result = sweep(denoiser, evaluator, "dev", items, sampler, tuple(int(s) for s in args.steps.split(",")))
            per = result["per_image"][config.PRIMARY_STEPS]
            consistency = {
                k: {"share_positive": float((v > 0).mean()), **paired_bootstrap(v)} for k, v in per.items() if k.startswith("deficit_")
            }
            out[sampler] = {"curve": result["curve"], "consistency_at_primary": consistency}
            log(f"{sampler} consistency at {config.PRIMARY_STEPS} steps: " + json.dumps({k: round(v['share_positive'], 2) for k, v in consistency.items()}))
    path = config.CACHE / "pilot.json"
    path.write_text(json.dumps(out, indent=1))
    log(f"wrote {path}")


def _split_baseline(denoiser, evaluator, split: str, items: list[Item]) -> dict:
    cached = config.CACHE / f"baseline_{split}.npz"
    if cached.exists():
        loaded = np.load(cached, allow_pickle=False)
        per_image: dict[int, dict] = {}
        for key in loaded.files:
            steps, metric = key.split("/")
            per_image.setdefault(int(steps), {})[metric] = loaded[key]
        return per_image
    result = sweep(denoiser, evaluator, split, items, config.SAMPLER)
    np.savez(cached, **{f"{s}/{m}": v for s, per in result["per_image"].items() for m, v in per.items()})
    return result["per_image"]


def cmd_baseline_test(args) -> None:
    """Generate and score the unedited model on the sealed splits. Only ever run after all operators have sealed."""
    unsealed = [r for r in args.require if not (config.RUNS / r / "final" / "edit.safetensors").exists()]
    if unsealed:
        raise SystemExit(f"refusing to touch the sealed splits: {unsealed} have not submitted")
    splits = build_splits()
    with device_lock():
        denoiser = Denoiser()
        evaluator = Evaluator(denoiser.device)
        for split in TEST_SPLITS:
            _split_baseline(denoiser, evaluator, split, splits[split])
    log("sealed baselines ready")


def cmd_final(args) -> None:
    """The only path that reads the sealed splits. Runs against an already sealed edit."""
    run = args.run
    path = config.RUNS / run / "final" / "edit.safetensors"
    if not path.exists():
        raise SystemExit(f"{run} has not submitted an edit")
    splits = build_splits()
    with device_lock():
        denoiser = Denoiser()
        evaluator = Evaluator(denoiser.device)
        edit = edits.load(path, denoiser.pristine)
        report = {"run": run, "edit": edit.sha256, "summary": {k: v for k, v in edit.summary.items()}, "splits": {}}
        sheets = config.RUNS / run / "final"
        for split in TEST_SPLITS:
            items = splits[split]
            prompts = [it.prompt for it in items]
            baseline = _split_baseline(denoiser, evaluator, split, items)
            per_split = {}
            for steps in config.TARGET_STEPS:
                res = evaluate_edit(denoiser, evaluator, edit, items, steps, split)
                deltas = {k: paired_bootstrap(res["per_edit"][k] - res["per_base"][k]) for k in res["per_edit"]}
                for band in ("structure", "coarse", "fine", "grain"):
                    key = f"deficit_{band}"
                    deltas[f"abs_{key}"] = paired_bootstrap(np.abs(res["per_edit"][key]) - np.abs(res["per_base"][key]))
                equivalent = {}
                sweep_steps = [s for s in config.SWEEP_STEPS if s != config.REF_STEPS]
                for metric in ("texture", "lpips", "abs_deficit_coarse", "abs_deficit_fine", "abs_deficit_grain"):
                    curve = [_metric_value(baseline[s], metric) for s in sweep_steps]
                    equivalent[metric] = step_equivalent(res["edit"][metric], sweep_steps, curve)
                per_split[steps] = {"base": res["base"], "edit": res["edit"], "delta": deltas, "step_equivalent": equivalent}
                rows = [[a, b, c] for a, b, c in zip(res["base_images"], res["images"], res["ref_images"])]
                contact_sheet(rows[:8], sheets / f"sheet_{split}_{steps}.png")
                log(
                    f"{run} {split} steps={steps}: T {res['base']['texture']:.2f}->{res['edit']['texture']:.2f} (equiv {equivalent['texture']:.1f}) "
                    f"L {res['base']['lpips']:.3f}->{res['edit']['lpips']:.3f} (equiv {equivalent['lpips']:.1f})"
                )
            report["splits"][split] = per_split
            del baseline, prompts
        if not args.quick:
            report["dose_response"] = _dose_response(denoiser, evaluator, edit, splits)
            report["side_effect_50"] = _side_effect(denoiser, evaluator, edit, splits)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS / f"final_{run}.json"
    out.write_text(json.dumps(report, indent=1))
    log(f"wrote {out}")


def _metric_value(per: dict, metric: str) -> float:
    if metric.startswith("abs_"):
        return float(np.abs(per[metric[4:]]).mean())
    return float(np.mean(per[metric]))


def _spread(items: list[Item], n: int) -> list[Item]:
    """Evenly spaced items: the sealed lists are ordered by category, so a prefix would be biased."""
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


def _subset(splits: dict, n: int) -> tuple[str, list[Item]]:
    return "test_seen", _spread(splits["test_seen"], n)


def _dose_response(denoiser, evaluator, edit, splits, alphas=(-1.0, -0.5, 0.25, 0.5, 1.0, 1.5, 2.0), n: int = 16) -> dict:
    """A causal edit should move the metrics smoothly with strength, and reverse sign."""
    split, items = _subset(splits, n)
    prompts = [it.prompt for it in items]
    denoiser.set_edit(None)
    refs = baseline_images(denoiser, split, items, config.REF_STEPS)
    base = baseline_images(denoiser, split, items, config.PRIMARY_STEPS)
    out = {"0.0": summarize(evaluator.evaluate(base, refs, prompts))}
    for alpha in alphas:
        denoiser.set_edit(edits.scaled(edit, alpha, denoiser.pristine))
        images = denoiser.generate(items, config.PRIMARY_STEPS)
        denoiser.set_edit(None)
        out[str(alpha)] = summarize(evaluator.evaluate(images, refs, prompts))
    return out


def _side_effect(denoiser, evaluator, edit, splits, n: int = 16) -> dict:
    """What the edit does to the slow path: edited at 50 steps against the unedited 50-step image."""
    split, items = _subset(splits, n)
    prompts = [it.prompt for it in items]
    denoiser.set_edit(None)
    refs = baseline_images(denoiser, split, items, config.REF_STEPS)
    denoiser.set_edit(edit)
    images = denoiser.generate(items, config.REF_STEPS)
    denoiser.set_edit(None)
    summary = summarize(evaluator.evaluate(images, refs, prompts))
    summary["swd"] = swd(images, refs)["mean"]
    return summary


def score(summary: dict, base: dict) -> float:
    """The fixed objective used by the blind control. Lower is better; 2.0 is the unedited model."""
    texture = summary["abs_deficit_coarse"] + summary["abs_deficit_fine"]
    base_texture = base["abs_deficit_coarse"] + base["abs_deficit_fine"]
    return texture / max(base_texture, 1e-6) + summary["lpips"] / base["lpips"]


def _weight_params(pristine: dict) -> list[str]:
    return [n for n, v in pristine.items() if v.dim() >= 2 and v.numel() > 1]


def cmd_climb(args) -> None:
    """Blind control: the operator's budget and objective, with no reasoning at all.

    Proposes a random multiplicative scale on one random weight tensor, keeps it
    if the dev objective improves, and accumulates accepted proposals.
    """
    rng = np.random.default_rng(args.seed)
    ledger = Ledger(args.run, args.budget)
    with device_lock():
        denoiser = Denoiser()
        evaluator = Evaluator(denoiser.device)
        items = dev_items(8)
        prompts = [it.prompt for it in items]
        pristine = denoiser.pristine
        names = _weight_params(pristine)
        refs = baseline_images(denoiser, "dev", items, config.REF_STEPS)
        base_images = baseline_images(denoiser, "dev", items, config.PRIMARY_STEPS)
        base = summarize(evaluator.evaluate(base_images, refs, prompts))
        current = dict(pristine)
        best = score(base, base)
        while ledger.remaining > 0:
            name = names[int(rng.integers(len(names)))]
            factor = float(np.exp(rng.normal(0, args.sigma)))
            proposal = dict(current)
            proposal[name] = current[name] * factor
            edit = edits.diff(proposal, pristine)
            denoiser.set_edit(edit)
            images = denoiser.generate(items, config.PRIMARY_STEPS)
            denoiser.set_edit(None)
            value = score(summarize(evaluator.evaluate(images, refs, prompts)), base)
            accepted = value < best
            if accepted:
                best, current = value, proposal
            ledger.record(cmd="climb", cost=1, param=name, factor=factor, score=value, accepted=accepted)
            log(f"{args.run}: {ledger.used}/{ledger.budget} {name} x{factor:.3f} score {value:.4f} {'ACCEPT' if accepted else ''}")
        final = edits.diff(current, pristine)
    folder = ledger.dir / "final"
    final.save(folder / "edit.safetensors")
    (folder / "note.txt").write_text(f"blind hill climb, seed {args.seed}, sigma {args.sigma}, dev score {best:.4f}\n")
    ledger.record(cmd="submit", cost=0, edit=final.sha256)
    log(f"{args.run}: sealed {final.sha256}, dev score {best:.4f}")


def cmd_null(args) -> None:
    """Random-location null: the operator's delta values, scattered inside the same tensors."""
    source = config.RUNS / args.like / "final" / "edit.safetensors"
    splits = build_splits()
    items = _spread(splits["test_seen"], args.n // 2) + _spread(splits["test_new"], args.n // 2)
    tags = ["test_seen"] * (args.n // 2) + ["test_new"] * (args.n // 2)
    rng = np.random.default_rng(args.seed)
    records = []
    with device_lock():
        denoiser = Denoiser()
        evaluator = Evaluator(denoiser.device)
        pristine = denoiser.pristine
        edit = edits.load(source, pristine)
        if not edit.tensors:
            raise SystemExit(f"{args.like} submitted a no-op")
        denoiser.set_edit(None)
        prompts = [it.prompt for it in items]
        refs = np.concatenate([baseline_images(denoiser, t, [it], config.REF_STEPS) for t, it in zip(tags, items)])
        base_images = np.concatenate([baseline_images(denoiser, t, [it], config.PRIMARY_STEPS) for t, it in zip(tags, items)])
        base = summarize(evaluator.evaluate(base_images, refs, prompts))
        denoiser.set_edit(edit)
        real_images = denoiser.generate(items, config.PRIMARY_STEPS)
        denoiser.set_edit(None)
        real = summarize(evaluator.evaluate(real_images, refs, prompts))
        real_score = score(real, base)
        for k in range(args.k):
            merged = dict(pristine)
            for name, value in edit.tensors.items():
                delta = value - pristine[name]
                perm = torch_permutation(delta.numel(), rng)
                merged[name] = pristine[name] + delta.flatten()[perm].reshape(delta.shape)
            denoiser.set_edit(edits.diff(merged, pristine))
            images = denoiser.generate(items, config.PRIMARY_STEPS)
            denoiser.set_edit(None)
            summary = summarize(evaluator.evaluate(images, refs, prompts))
            records.append({"score": score(summary, base), **summary})
            log(f"null {args.like} {k+1}/{args.k}: score {records[-1]['score']:.4f} (real {real_score:.4f})")
    scores = np.array([r["score"] for r in records])
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS / f"null_{args.like}.json"
    out.write_text(json.dumps({"like": args.like, "real_score": real_score, "real": real, "base": base, "share_null_at_least_as_good": float((scores <= real_score).mean()), "null": records}, indent=1))
    log(f"wrote {out}")


def _unsharp(images: np.ndarray, sigma: float, gain: float) -> np.ndarray:
    blurred = gaussian_filter(images.astype(np.float32), sigma=(0, sigma, sigma, 0))
    return np.clip(blurred + gain * (images - blurred), 0, 255).round().astype(np.uint8)


def cmd_postproc(args) -> None:
    """No weights touched: an unsharp mask tuned on dev, then scored on the sealed splits.

    If this matches an edit, the edit is not doing anything an image filter cannot.
    """
    splits = build_splits()
    with device_lock():
        evaluator = Evaluator(Denoiser().device)
        items = dev_items()
        prompts = [it.prompt for it in items]
        refs = baseline_images(None, "dev", items, config.REF_STEPS)
        base_images = baseline_images(None, "dev", items, config.PRIMARY_STEPS)
        base = summarize(evaluator.evaluate(base_images, refs, prompts))
        grid = [(sigma, gain) for sigma in (1.0, 2.0, 4.0, 8.0) for gain in (1.1, 1.25, 1.5, 2.0, 3.0)]
        scored = []
        for sigma, gain in grid:
            value = score(summarize(evaluator.evaluate(_unsharp(base_images, sigma, gain), refs, prompts)), base)
            scored.append({"sigma": sigma, "gain": gain, "score": value})
            log(f"postproc sigma={sigma} gain={gain}: dev score {value:.4f}")
        best = min(scored, key=lambda r: r["score"])
        report = {"dev_grid": scored, "chosen": best, "splits": {}}
        for split in TEST_SPLITS:
            items = splits[split]
            prompts = [it.prompt for it in items]
            refs = baseline_images(None, split, items, config.REF_STEPS)
            per_split = {}
            for steps in config.TARGET_STEPS:
                images = baseline_images(None, split, items, steps)
                per_base = evaluator.evaluate(images, refs, prompts)
                sharpened = _unsharp(images, best["sigma"], best["gain"])
                per_edit = evaluator.evaluate(sharpened, refs, prompts)
                base_s, edit_s = summarize(per_base), summarize(per_edit)
                base_s["swd"], edit_s["swd"] = swd(images, refs)["mean"], swd(sharpened, refs)["mean"]
                deltas = {k: paired_bootstrap(per_edit[k] - per_base[k]) for k in per_edit}
                for band in ("coarse", "fine"):
                    key = f"deficit_{band}"
                    deltas[f"abs_{key}"] = paired_bootstrap(np.abs(per_edit[key]) - np.abs(per_base[key]))
                per_split[steps] = {"base": base_s, "edit": edit_s, "delta": deltas}
                log(f"postproc {split} steps={steps}: {_brief(edit_s)}")
            report["splits"][split] = per_split
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "postproc.json").write_text(json.dumps(report, indent=1))
    log("wrote results/postproc.json")


def torch_permutation(n: int, rng: np.random.Generator):
    import torch

    return torch.from_numpy(rng.permutation(n))


def cmd_seal_info(args) -> None:
    """Print a fingerprint of the sealed splits without revealing them."""
    splits = build_splits()
    digest = hashlib.sha256(json.dumps({k: as_json(v) for k, v in splits.items() if k != "dev"}).encode()).hexdigest()[:16]
    print({k: len(v) for k, v in splits.items()}, "sealed sha", digest)


def cmd_report(args) -> None:
    from .report import build

    summary = build()
    print(json.dumps({k: summary[k] for k in ("H1_premise", "checks", "scores", "verdict")}, indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="wsp-sup", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").set_defaults(fn=cmd_prepare)
    p = sub.add_parser("pilot")
    p.add_argument("--samplers", default="ddim,ddim_trailing,dpm2m")
    p.add_argument("--steps", default="5,6,8,12,24,50")
    p.add_argument("-n", type=int, default=8)
    p.set_defaults(fn=cmd_pilot)
    p = sub.add_parser("baseline-test")
    p.add_argument("--require", nargs="+", default=["op1", "op2", "op3"], help="runs that must be sealed first")
    p.set_defaults(fn=cmd_baseline_test)
    p = sub.add_parser("final")
    p.add_argument("run")
    p.add_argument("--quick", action="store_true", help="skip dose-response and the 50-step side-effect check")
    p.set_defaults(fn=cmd_final)
    p = sub.add_parser("climb")
    p.add_argument("run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--budget", type=int, default=config.BUDGET_UNITS)
    p.set_defaults(fn=cmd_climb)
    p = sub.add_parser("null")
    p.add_argument("like")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("-n", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_null)
    sub.add_parser("postproc").set_defaults(fn=cmd_postproc)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    sub.add_parser("seal-info").set_defaults(fn=cmd_seal_info)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
