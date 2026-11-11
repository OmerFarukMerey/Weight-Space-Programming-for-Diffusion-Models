"""Pilot A1: at which noise levels do extra steps buy composition, and at which texture?

Each arm keeps the coarse 6-step grid and replaces one third of the trajectory with the
50-step grid. If composition is decided at high noise, only the arm that is fine there
should repair it. Dev prompts only.
"""
from __future__ import annotations

import json

from wspd import config
from wspd.lab import Bench, Plan, line, public, trailing
from wspd.store import device_lock

fine, coarse = trailing(50), trailing(6)
ARMS = {
    "coarse6_custom_loop": coarse,
    "fine_high (t>666)": [t for t in fine if t > 666] + coarse[2:],
    "fine_mid (333-666)": coarse[:2] + [t for t in fine if 332 < t <= 666] + coarse[4:],
    "fine_low (t<=332)": coarse[:4] + [t for t in fine if t <= 332],
}

with device_lock():
    bench = Bench()
    report = {"split": "dev", "n": len(bench.items("dev")), "arms": {}}
    for steps in (6, 8, 12, 16, 24):
        m = bench.measure("dev", bench.cached("dev", steps), nfe=2 * steps)
        report["arms"][f"unedited_{steps}"] = public(m)
        print(line(f"unedited_{steps}", m), flush=True)
    for name, ts in ARMS.items():
        plan = Plan(tuple(ts))
        m = bench.measure("dev", bench.run("dev", plan), nfe=plan.nfe)
        report["arms"][name] = {**public(m), "timesteps": ts}
        print(line(name, m), flush=True)
(config.RESULTS / "pilots" / "a1_phase_attribution.json").write_text(json.dumps(report, indent=1))
