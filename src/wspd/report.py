"""Scores the pre-registered hypotheses mechanically from result files, then draws the figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config

SPLITS = ("test_seen", "test_new")
STEPS = str(config.PRIMARY_STEPS)
N_EQUIVALENT = 8

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"
COLORS = {"LLM": "#2a78d6", "Blind climb": "#eb6834", "Unsharp mask": "#1baf7a"}


def test_score(final: dict) -> float:
    """T/T_base + L/L_base at the primary step count, pooled over prompts. 2.0 is the unedited model."""
    total, weight = 0.0, 0
    for split, n in (("test_seen", 40), ("test_new", 32)):
        res = final["splits"][split][STEPS]
        total += n * (res["edit"]["texture"] / res["base"]["texture"] + res["edit"]["lpips"] / res["base"]["lpips"])
        weight += n
    return total / weight


def premise(baseline: dict[str, dict[str, np.ndarray]]) -> dict:
    """H1: is the energy loss one-sided across prompts, for each split and band?"""
    from .metrics import paired_bootstrap

    out = {}
    for split, per in baseline.items():
        out[split] = {}
        for band in ("coarse", "fine"):
            values = per[f"{STEPS}/deficit_{band}"]
            ci = paired_bootstrap(values)
            out[split][band] = {"share_positive": float((values > 0).mean()), **ci}
    out["passes"] = all(
        out[split][band]["share_positive"] >= 0.8 and out[split][band]["lo"] > 0 for split in baseline for band in ("coarse", "fine")
    )
    return out


def arm_checks(final: dict, null: dict | None = None) -> dict:
    """H2 to H5 and H7, plus the null half of H6. All splits must pass."""
    per = [final["splits"][s][STEPS] for s in SPLITS]
    checks = {
        "H2_texture_generalizes": all(r["delta"]["texture"]["hi"] < 0 for r in per),
        "H3_perceptual_improves": all(r["delta"]["lpips"]["hi"] < 0 for r in per),
        "H4_saves_steps": all(min(r["step_equivalent"]["texture"], r["step_equivalent"]["lpips"]) >= N_EQUIVALENT for r in per),
        "H5_keeps_meaning": all(r["delta"]["clip"]["mean"] >= -0.01 for r in per),
    }
    if null is not None:
        checks["H6_beats_null"] = null["share_null_at_least_as_good"] <= 0.05
    dose = final.get("dose_response")
    if dose:
        path = [dose[a]["texture"] for a in ("0.0", "0.25", "0.5", "1.0")]
        checks["H7_causal"] = all(b <= a for a, b in zip(path, path[1:])) and dose["-1.0"]["texture"] >= dose["0.0"]["texture"]
    return checks


def verdict(llm: dict[str, dict], llm_scores: list[float], climb_scores: list[float], filter_score: float | None) -> dict:
    """The verdict rules from the pre-registration, including the amendment on the filter."""
    majority = len(llm) // 2 + 1
    median_llm = float(np.median(llm_scores))
    median_climb = float(np.median(climb_scores)) if climb_scores else None
    beats_climb = median_climb is None or median_llm < median_climb
    h2 = sum(c["H2_texture_generalizes"] for c in llm.values())
    full = sum(all(c.values()) and beats_climb for c in llm.values())
    filter_matches = filter_score is not None and filter_score <= median_llm
    if h2 < majority or not beats_climb:
        label = "No"
    elif full >= majority and not filter_matches:
        label = "Yes"
    else:
        label = "Partial"
    return {
        "label": label,
        "operators_passing_H2": h2,
        "operators_passing_all": full,
        "median_llm_score": median_llm,
        "median_climb_score": median_climb,
        "H6_beats_climb": beats_climb,
        "filter_score": filter_score,
        "filter_matches_llm": filter_matches,
    }


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_curves(baseline: dict, arms: dict[str, tuple[str, dict]], path: Path) -> None:
    """Where each edit lands on the baseline quality-against-steps curve."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    steps = [s for s in config.SWEEP_STEPS if s != config.REF_STEPS]
    for ax, metric, label in zip(axes, ("texture", "lpips"), ("Texture deficit T (dB)", "Perceptual distance L (LPIPS)")):
        _style(ax)
        curve = [np.concatenate([baseline[s][f"{n}/{metric}"] for s in SPLITS]).mean() for n in steps]
        ax.plot(steps, curve, color=MUTED, linewidth=2, marker="o", markersize=5, zorder=2)
        ax.annotate("unedited", (steps[-2], curve[-2]), textcoords="offset points", xytext=(6, 8), color=MUTED, fontsize=9)
        ax.set_xscale("log")
        ax.set_xticks(steps)
        ax.set_xticklabels(steps)
        ax.minorticks_off()
        ax.set_xlabel("Sampler steps", color=MUTED, fontsize=9)
        ax.set_title(label, color=INK, fontsize=11, loc="left")
        ax.set_ylim(bottom=0)
        seen = set()
        for _name, (group, final) in arms.items():
            values = [(final["splits"][s][STEPS]["edit"][metric], n) for s, n in (("test_seen", 40), ("test_new", 32))]
            value = sum(v * n for v, n in values) / 72
            ax.scatter(
                [config.PRIMARY_STEPS], [value], s=64, color=COLORS[group], edgecolors=SURFACE, linewidths=2, zorder=3,
                label=group if group not in seen else None, alpha=0.9,
            )
            seen.add(group)
        ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper right")
    fig.suptitle("Edited 6-step results on the unedited step curve, all 72 sealed prompts", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figure_effects(arms: dict[str, tuple[str, dict]], path: Path) -> None:
    """Change from the unedited model, with 95% intervals, for each arm and split."""
    import matplotlib.pyplot as plt

    names = list(arms)
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(names) + 1.6), sharey=True, facecolor=SURFACE)
    for ax, metric, label in zip(axes, ("texture", "lpips"), ("Change in T (dB)", "Change in L (LPIPS)")):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.axvline(0, color=MUTED, linewidth=1)
        for i, name in enumerate(names):
            group, final = arms[name]
            for split, marker, offset in (("test_seen", "o", -0.14), ("test_new", "s", 0.14)):
                delta = final["splits"][split][STEPS]["delta"][metric]
                y = i + offset
                ax.plot([delta["lo"], delta["hi"]], [y, y], color=COLORS[group], linewidth=2, solid_capstyle="round")
                ax.scatter(
                    [delta["mean"]], [y], s=44, marker=marker, color=COLORS[group], edgecolors=SURFACE, linewidths=1.5, zorder=3,
                    label={"test_seen": "Seen categories", "test_new": "New categories"}[split] if i == 0 else None,
                )
        ax.set_title(label, color=INK, fontsize=11, loc="left")
        ax.invert_yaxis() if ax is axes[0] else None
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, color=INK, fontsize=10)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="best")
    fig.suptitle("Effect at 6 steps on sealed prompts, left of zero is better", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figure_dose(arms: dict[str, tuple[str, dict]], path: Path) -> None:
    """Metric against edit strength. Strength 0 is the unedited model and 1 is the submitted edit."""
    import matplotlib.pyplot as plt

    with_dose = {n: f for n, (_, f) in arms.items() if f.get("dose_response")}
    if not with_dose:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    tones = ["#2a78d6", "#6250d6", "#1baf7a"]
    for ax, metric, label in zip(axes, ("texture", "lpips"), ("Texture deficit T (dB)", "Perceptual distance L (LPIPS)")):
        _style(ax)
        for (name, final), tone in zip(with_dose.items(), tones):
            alphas = sorted(float(a) for a in final["dose_response"])
            values = [final["dose_response"][str(a)][metric] for a in alphas]
            ax.plot(alphas, values, color=tone, linewidth=2, marker="o", markersize=5, label=name)
        ax.axvline(0, color=GRID, linewidth=1)
        ax.axvline(1, color=GRID, linewidth=1)
        ax.set_xlabel("Edit strength", color=MUTED, fontsize=9)
        ax.set_title(label, color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    fig.suptitle("Dose response on 16 sealed prompts at 6 steps", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _load_baseline() -> dict:
    out = {}
    for split in SPLITS:
        loaded = np.load(config.CACHE / f"baseline_{split}.npz", allow_pickle=False)
        out[split] = {k: loaded[k] for k in loaded.files}
    return out


def _as_final(postproc: dict) -> dict:
    return {"splits": postproc["splits"]}


def _row(name: str, final: dict, checks: dict | None, score: float) -> str:
    def cell(split: str, metric: str, digits: int) -> str:
        res = final["splits"][split][STEPS]
        d = res["delta"][metric]
        return f"{d['mean']:+.{digits}f} [{d['lo']:+.{digits}f}, {d['hi']:+.{digits}f}]"

    def equivalents(split: str) -> str:
        eq = final["splits"][split][STEPS].get("step_equivalent")
        return f"{eq['texture']:.1f} / {eq['lpips']:.1f}" if eq else "n/a"

    weights = final.get("summary", {}).get("weights_changed")
    size = f"{weights:,}" if weights is not None else "0"
    return (
        f"| {name} | {size} | {score:.3f} | {cell('test_seen', 'texture', 2)} | {cell('test_new', 'texture', 2)} "
        f"| {cell('test_seen', 'lpips', 3)} | {cell('test_new', 'lpips', 3)} | {equivalents('test_seen')} | {equivalents('test_new')} |"
    )


def build() -> dict:
    results = config.RESULTS
    finals = {p.stem.removeprefix("final_"): json.loads(p.read_text()) for p in sorted(results.glob("final_*.json"))}
    llm = {k: v for k, v in finals.items() if k.startswith("op")}
    climbs = {k: v for k, v in finals.items() if k.startswith("climb")}
    nulls = {p.stem.removeprefix("null_"): json.loads(p.read_text()) for p in sorted(results.glob("null_*.json"))}
    postproc = json.loads((results / "postproc.json").read_text()) if (results / "postproc.json").exists() else None
    baseline = _load_baseline()

    llm_checks = {k: arm_checks(v, nulls.get(k)) for k, v in llm.items()}
    llm_scores = {k: test_score(v) for k, v in llm.items()}
    climb_scores = {k: test_score(v) for k, v in climbs.items()}
    filter_score = test_score(_as_final(postproc)) if postproc else None
    summary = {
        "H1_premise": premise({s: baseline[s] for s in SPLITS}),
        "checks": llm_checks,
        "scores": {**llm_scores, **climb_scores, **({"unsharp": filter_score} if postproc else {})},
        "verdict": verdict(llm_checks, list(llm_scores.values()), list(climb_scores.values()), filter_score),
        "nulls": {k: {"share_null_at_least_as_good": v["share_null_at_least_as_good"], "real_score": v["real_score"]} for k, v in nulls.items()},
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=1))

    arms: dict[str, tuple[str, dict]] = {k: ("LLM", v) for k, v in llm.items()}
    arms.update({k: ("Blind climb", v) for k, v in climbs.items()})
    if postproc:
        arms["unsharp"] = ("Unsharp mask", _as_final(postproc))
    figure_curves(baseline, arms, results / "fig_step_curve.png")
    figure_effects(arms, results / "fig_effects.png")
    figure_dose(arms, results / "fig_dose.png")

    header = (
        "| Arm | Weights changed | Score (2.0 = unedited) | Change in T, seen | Change in T, new | Change in L, seen | Change in L, new "
        "| Step equivalent T / L, seen | Step equivalent T / L, new |\n|" + "---|" * 9
    )
    rows = [_row(k, v, llm_checks.get(k), summary["scores"][k]) for k, v in {**llm, **climbs}.items()]
    if postproc:
        rows.append(_row("unsharp", _as_final(postproc), None, filter_score))
    (results / "summary.md").write_text(header + "\n" + "\n".join(rows) + "\n")
    return summary
