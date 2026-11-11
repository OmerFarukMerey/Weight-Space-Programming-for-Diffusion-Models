"""Pilot D: what does a blind search get with the LLM operators' budget, in a sensible space?

The whole-tensor hill climb in the main experiment was a weak control. This one searches
four sampler parameters (a noise gain, and a timestep relabel fraction for high-noise
steps, low-noise steps and the last step) with a compass search. It gets the operators'
budget of 40 units, each unit being 8 dev prompts at 6 steps, and the same objective.
The result is then scored once on the 72 sealed prompts.
"""
from __future__ import annotations

import json

import numpy as np

from wspd import config
from wspd.lab import Bench, Plan, line, public, trailing
from wspd.metrics import swd
from wspd.store import device_lock

BUDGET = 40
ts = tuple(trailing(config.PRIMARY_STEPS))
NAMES = ("gain", "c_high", "c_low", "c_last")
LOWER, UPPER = np.array([0.94, -0.3, -0.3, -0.3]), np.array([1.03, 0.9, 0.9, 0.9])


def plan(x) -> Plan:
    gain, c_high, c_low, c_last = (float(v) for v in x)
    fractions = tuple(c_last if i + 1 == len(ts) else (c_high if t >= 500 else c_low) for i, t in enumerate(ts))
    return Plan(ts, gain=gain, relabel_steps=fractions)


with device_lock():
    bench = Bench()
    items = bench.items("dev")[:8]
    prompts = [it.prompt for it in items]
    refs = bench.cached("dev", config.REF_STEPS)[:8]
    base = bench.ev.evaluate(bench.cached("dev", 6)[:8], refs, prompts)
    base_t, base_l = base["texture"].mean(), base["lpips"].mean()
    log = []

    def objective(x) -> float:
        images = np.concatenate([bench.den.decode(bench.latents([it], plan(x))) for it in items])
        per = bench.ev.evaluate(images, refs, prompts)
        value = float(per["texture"].mean() / base_t + per["lpips"].mean() / base_l)
        log.append({"x": [float(v) for v in x], "score": value})
        print(f"eval {len(log):2d}: " + " ".join(f"{n}={v:+.3f}" for n, v in zip(NAMES, x)) + f" -> {value:.4f}", flush=True)
        return value

    x = np.array([1.0, 0.0, 0.0, 0.0])
    step = np.array([0.01, 0.2, 0.2, 0.2])
    best = objective(x)
    while len(log) < BUDGET:
        improved = False
        for d in range(len(x)):
            for sign in (1, -1):
                if len(log) >= BUDGET:
                    break
                cand = np.clip(x + sign * step * np.eye(len(x))[d], LOWER, UPPER)
                if np.allclose(cand, x):
                    continue
                value = objective(cand)
                if value < best:
                    best, x, improved = value, cand, True
                    break
        if not improved:
            step = step / 2
    print("sealed:", dict(zip(NAMES, (float(v) for v in x))), "dev score", best, flush=True)

    final = plan(x)
    report = {"budget_units": BUDGET, "parameters": dict(zip(NAMES, (float(v) for v in x))), "dev8_score": best, "log": log}
    for split in ("dev", "test"):
        m = bench.measure(split, bench.run(split, final))
        report[split] = public(m)
        print(line(f"{split.upper()} compass", m), flush=True)
(config.RESULTS / "pilots" / "d_compass_search.json").write_text(json.dumps(report, indent=1))
