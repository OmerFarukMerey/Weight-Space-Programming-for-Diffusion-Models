"""Pilot A2: spend steps where composition is decided, and repair texture with a relabel.

Pilot A1 showed composition error is set above t=666 and texture below it. Each arm here
has 12 UNet evaluations per image, like plain 6-step sampling with guidance. Schedules put
more steps at high noise; the low-noise steps then get a timestep relabel (the network is
told t - c * step size). Some arms also drop the unconditional pass at low noise and
reinvest it in more steps. Selection rule, fixed before running: lowest dev score among
all arms; that arm and the best trailing-schedule arm are scored once on sealed prompts.
"""
from __future__ import annotations

import json
import sys

from wspd import config
from wspd.lab import Bench, Plan, compare, line, public, trailing, warped
from wspd.store import device_lock

SCHEDULES = {
    "trailing": tuple(trailing(6)),
    "rho0.7": tuple(warped(6, 0.7)),
    "rho0.5": tuple(warped(6, 0.5)),
    "front4": (999, 899, 799, 699, 499, 250),
}


def relabelled(ts, c, gain=1.0, guided=None):
    fractions = tuple(c if t <= 700 else 0.0 for t in ts)
    return Plan(ts, gain=gain, relabel_steps=fractions, guided=guided)


ARMS = {}
for name, ts in SCHEDULES.items():
    for c in (0.0, 0.3, 0.5, 0.7):
        ARMS[f"{name}_c{c}"] = relabelled(ts, c)
    ARMS[f"{name}_c0.5_g0.99"] = relabelled(ts, 0.5, gain=0.99)
# Same 12 evaluations, but the low-noise steps skip the unconditional pass.
eight = (999, 916, 832, 749, 666, 499, 332, 166)
seven = (999, 916, 832, 749, 666, 444, 222)
for c in (0.0, 0.5):
    ARMS[f"8step_4guided_c{c}"] = relabelled(eight, c, guided=(True,) * 4 + (False,) * 4)
    ARMS[f"7step_5guided_c{c}"] = relabelled(seven, c, guided=(True,) * 5 + (False,) * 2)
assert all(p.nfe == 12 for p in ARMS.values())

with device_lock():
    bench = Bench()
    report = {"dev": {}, "test": {}}
    dev = {}
    for name, p in ARMS.items():
        dev[name] = bench.measure("dev", bench.run("dev", p), nfe=p.nfe)
        report["dev"][name] = {**public(dev[name]), "timesteps": list(p.timesteps)}
        print(line(name, dev[name]), flush=True)
    best = min(ARMS, key=lambda k: dev[k]["score"])
    best_trailing = min((k for k in ARMS if k.startswith("trailing")), key=lambda k: dev[k]["score"])
    report["selected_on_dev"] = {"best": best, "best_trailing": best_trailing}
    print(f"\nselected on dev: best={best} best_trailing={best_trailing}\n", flush=True)
    if "--dev-only" not in sys.argv:
        test = {}
        for name in dict.fromkeys((best, best_trailing)):
            test[name] = bench.measure("test", bench.run("test", ARMS[name]), nfe=12)
            report["test"][name] = public(test[name])
            print(line(f"TEST {name}", test[name]), flush=True)
        if best != best_trailing:
            report["test"]["best_minus_best_trailing"] = compare(test[best], test[best_trailing])
            print(json.dumps(report["test"]["best_minus_best_trailing"]), flush=True)
(config.RESULTS / "pilots" / "a2_step_allocation.json").write_text(json.dumps(report, indent=1))
