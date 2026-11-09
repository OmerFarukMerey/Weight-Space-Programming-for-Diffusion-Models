"""`wsp`: what the LLM operator is allowed to do.

Every command here sees the dev split only. Reading weights is free. Anything
that runs the denoiser draws from the run's budget and is written to the ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

from . import config, edits
from .splits import as_json, dev_items
from .store import Ledger, baseline_images, contact_sheet, device_lock


def _run_name(args) -> str:
    run = args.run or os.environ.get("WSP_RUN")
    if not run:
        raise SystemExit("set --run NAME or WSP_RUN")
    return run


def _load_tools():
    from .metrics import Evaluator
    from .model import Denoiser

    denoiser = Denoiser()
    return denoiser, Evaluator(denoiser.device)


def _table(rows: list[dict], columns: list[str]) -> str:
    def cell(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    text = [[c for c in columns]] + [[cell(r.get(c, "")) for c in columns] for r in rows]
    widths = [max(len(line[i]) for line in text) for i in range(len(columns))]
    return "\n".join("  ".join(v.ljust(w) for v, w in zip(line, widths)) for line in text)


def cmd_info(args) -> None:
    baseline = config.CACHE / "baseline_dev.json"
    print(f"model {config.MODEL_ID}@{config.MODEL_REVISION[:8]}  sampler {config.SAMPLER}  guidance {config.GUIDANCE}")
    print(f"reference: {config.REF_STEPS} steps; target steps: {config.TARGET_STEPS}; primary: {config.PRIMARY_STEPS}")
    print(f"dev images:  {config.CACHE}/images/dev/{config.SAMPLER}/<steps>/<id>.png")
    print(f"free probes: {config.CACHE}/probe_baseline_6.json and probe_baseline_50.json")
    if baseline.exists():
        data = json.loads(baseline.read_text())
        rows = [{"steps": int(s), **{k: v for k, v in m.items()}} for s, m in data["curve"].items()]
        cols = ["steps", "abs_deficit_coarse", "abs_deficit_fine", "abs_deficit_grain", "lpips", "swd", "clip"]
        print("\nbaseline on all 24 dev prompts, measured against the 50-step reference:")
        print(_table(rows, cols))
        print(f"swd noise floor on 24 images: {data['swd_floor']:.1f}")
    if args.run or os.environ.get("WSP_RUN"):
        ledger = Ledger(_run_name(args))
        print(f"\nrun {_run_name(args)}: {ledger.used}/{ledger.budget} units used, {ledger.remaining} left")


def cmd_prompts(args) -> None:
    print(json.dumps(as_json(dev_items(args.n)), indent=1))


def cmd_weights(args) -> None:
    weights = edits.load_pristine()
    pattern = re.compile(args.filter) if args.filter else None
    rows = []
    for name, value in weights.items():
        if pattern and not pattern.search(name):
            continue
        rows.append(
            {
                "name": name,
                "shape": "x".join(map(str, value.shape)) or "scalar",
                "numel": value.numel(),
                "l2": float(value.norm()),
                "mean": float(value.mean()),
                "std": float(value.std()) if value.numel() > 1 else 0.0,
            }
        )
    print(_table(rows, ["name", "shape", "numel", "l2", "mean", "std"]))
    print(f"\n{len(rows)} parameters, {sum(r['numel'] for r in rows):,} weights")


def _module_names(unet, pattern: str) -> list[str]:
    regex = re.compile(pattern)
    names = [n for n, _ in unet.named_modules() if n and regex.search(n)]
    if not names:
        raise SystemExit(f"no module matches {pattern!r}")
    if len(names) > 48:
        raise SystemExit(f"{pattern!r} matches {len(names)} modules; narrow it to at most 48")
    return names


def cmd_probe(args) -> None:
    from .model import run_probe

    run = _run_name(args)
    ledger = Ledger(run)
    n = args.n_images
    cost = config.unit_cost(n, args.steps)
    ledger.require(cost)
    items = dev_items(n)
    folder = ledger.new_folder("probe")
    with device_lock():
        denoiser, _ = _load_tools()
        edit = edits.from_script(Path(args.edit), denoiser.pristine) if args.edit else None
        denoiser.set_edit(edit)
        payload = run_probe(denoiser, items, args.steps, _module_names(denoiser.unet, args.modules))
    (folder / "probe.json").write_text(json.dumps(payload))
    ledger.record(cmd="probe", cost=cost, steps=args.steps, n_images=n, modules=args.modules, edit=edit.sha256 if edit else None, out=str(folder))

    print(_table(payload["sampler"], ["step", "t", "eps_rms", "x0_rms", "x0_low", "x0_mid", "x0_high"]))
    rows = []
    for name in payload["modules"]:
        records = [r for r in payload["records"] if r["module"] == name]
        if records:
            rows.append(
                {
                    "module": name,
                    "shape": "x".join(map(str, payload["shapes"][name])),
                    **{k: float(np.mean([r[k] for r in records])) for k in ("rms", "low", "mid", "high")},
                }
            )
    print()
    print(_table(rows, ["module", "shape", "rms", "low", "mid", "high"]))
    print(f"\nper-step records: {folder}/probe.json")
    print(f"cost {cost}; {ledger.used}/{ledger.budget} units used, {ledger.remaining} left")


def evaluate_edit(denoiser, evaluator, edit, items, steps: int, split: str = "dev") -> dict:
    """Baseline and edited metrics for one step count, against the unedited reference."""
    from .metrics import summarize, swd

    denoiser.set_edit(None)
    prompts = [it.prompt for it in items]
    refs = baseline_images(denoiser, split, items, config.REF_STEPS)
    base = baseline_images(denoiser, split, items, steps)
    denoiser.set_edit(edit)
    edited = denoiser.generate(items, steps)
    denoiser.set_edit(None)
    per_base = evaluator.evaluate(base, refs, prompts)
    per_edit = evaluator.evaluate(edited, refs, prompts)
    summary_base = summarize(per_base)
    summary_edit = summarize(per_edit)
    summary_base["swd"] = swd(base, refs)["mean"]
    summary_edit["swd"] = swd(edited, refs)["mean"]
    return {
        "images": edited,
        "base_images": base,
        "ref_images": refs,
        "per_base": per_base,
        "per_edit": per_edit,
        "base": summary_base,
        "edit": summary_edit,
    }


def cmd_try(args) -> None:
    run = _run_name(args)
    ledger = Ledger(run)
    steps_list = [int(s) for s in args.steps.split(",")]
    for steps in steps_list:
        if steps not in config.SWEEP_STEPS:
            raise SystemExit(f"--steps must come from {config.SWEEP_STEPS}")
    n = args.n_images
    cost = sum(config.unit_cost(n, s) for s in steps_list)
    ledger.require(cost)
    items = dev_items(n)
    folder = ledger.new_folder("try")
    shutil.copy(args.edit, folder / "edit.py")
    with device_lock():
        denoiser, evaluator = _load_tools()
        edit = edits.from_script(Path(args.edit), denoiser.pristine)
        results = {}
        for steps in steps_list:
            res = evaluate_edit(denoiser, evaluator, edit, items, steps)
            results[steps] = res
            rows = [[a, b, c] for a, b, c in zip(res["base_images"], res["images"], res["ref_images"])]
            contact_sheet(rows[:8], folder / f"sheet_{steps}.png")
    report = {str(s): {"base": r["base"], "edit": r["edit"]} for s, r in results.items()}
    (folder / "metrics.json").write_text(json.dumps({"summary": edit.summary, "note": args.note, "metrics": report}, indent=1))
    ledger.record(cmd="try", cost=cost, steps=steps_list, n_images=n, edit=edit.sha256, note=args.note, out=str(folder), metrics=report)

    print(f"edit {edit.sha256}: {edit.summary['weights_changed']:,} weights changed ({edit.summary['fraction_changed']:.4%}), {edit.summary['params_touched']} parameters")
    keys = ["texture", "abs_deficit_structure", "abs_deficit_coarse", "abs_deficit_fine", "abs_deficit_grain", "deficit_coarse", "deficit_fine", "deficit_grain", "lpips", "swd", "clip", "luma_shift", "contrast_ratio"]
    for steps, res in results.items():
        print(f"\nsteps={steps}, {n} dev images, against the {config.REF_STEPS}-step reference (lower is better except clip):")
        print(_table([{"metric": k, "baseline": res["base"][k], "edited": res["edit"][k], "change": res["edit"][k] - res["base"][k]} for k in keys], ["metric", "baseline", "edited", "change"]))
        print(f"sheet: {folder}/sheet_{steps}.png  (columns: baseline, edited, reference, then the same three magnified)")
    print(f"\ncost {cost}; {ledger.used}/{ledger.budget} units used, {ledger.remaining} left")


def cmd_submit(args) -> None:
    run = _run_name(args)
    ledger = Ledger(run)
    final = ledger.dir / "final"
    if (final / "edit.safetensors").exists():
        raise SystemExit(f"{run} has already submitted; a submission is final")
    edit = edits.from_script(Path(args.edit))
    if not edit.tensors:
        print("warning: this edit changes no weights; submitting a no-op means the operator found nothing worth keeping")
    final.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.edit, final / "edit.py")
    edit.save(final / "edit.safetensors")
    (final / "note.txt").write_text(args.note + "\n")
    ledger.record(cmd="submit", cost=0, edit=edit.sha256, note=args.note, summary={k: v for k, v in edit.summary.items() if k != "per_param"})
    print(f"sealed {edit.sha256}: {edit.summary['weights_changed']:,} weights in {edit.summary['params_touched']} parameters -> {final}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="wsp", description=__doc__)
    parser.add_argument("--run", help="run name (or set WSP_RUN)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="task constants, baseline curve, budget (free)").set_defaults(fn=cmd_info)

    p = sub.add_parser("prompts", help="list dev prompts (free)")
    p.add_argument("-n", type=int)
    p.set_defaults(fn=cmd_prompts)

    p = sub.add_parser("weights", help="list parameters with shape and norm (free)")
    p.add_argument("--filter", help="regex over parameter names")
    p.set_defaults(fn=cmd_weights)

    p = sub.add_parser("probe", help="record per-module activation stats during sampling (metered)")
    p.add_argument("--modules", required=True, help="regex over module names, at most 48 matches")
    p.add_argument("--steps", type=int, default=config.PRIMARY_STEPS)
    p.add_argument("--n-images", type=int, default=8)
    p.add_argument("--edit", help="optional edit script to install first")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("try", help="apply an edit script and measure it on dev (metered)")
    p.add_argument("edit")
    p.add_argument("--steps", default=str(config.PRIMARY_STEPS), help="comma-separated, e.g. 5,6,8")
    p.add_argument("--n-images", type=int, default=8, help="8 for 1 unit at 6 steps, up to 24 for 3 units")
    p.add_argument("--note", default="")
    p.set_defaults(fn=cmd_try)

    p = sub.add_parser("submit", help="seal one final edit (free, once only)")
    p.add_argument("edit")
    p.add_argument("--note", required=True, help="which weights, why, and what evidence supports it")
    p.set_defaults(fn=cmd_submit)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
