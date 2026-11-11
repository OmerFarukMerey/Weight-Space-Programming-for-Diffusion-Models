"""Pilot B: should different resolution levels of the UNet be told different times?

A sampler can pass only one timestep. Here the 64x64, 32x32 and 16x16 blocks each get
their own shifted time, using op3's sealed shift curve as the unit. Arms are compared on
dev; the best block-wise arm and the best global arm are then scored once on the 72
sealed prompts. Selection rule, fixed before running: lowest dev score, where
score = T/T_base + L/L_base.
"""
from __future__ import annotations

import json
import sys

from wspd import config
from wspd.lab import OP3_KNOTS, Bench, Plan, compare, line, public, trailing
from wspd.store import device_lock

ALL_T = ((0, 0), (124, 60), (166, 85), (199, 100), (249, 70), (1000, 70))  # op3's curve without the cut-off at t=470
ts = tuple(trailing(config.PRIMARY_STEPS))


def plan(scale, knots=OP3_KNOTS):
    return Plan(ts, knots=knots, block_scale=scale)


GLOBAL = {
    "global_x0.75": plan((0.75, 0.75, 0.75)),
    "global_x1": plan((1, 1, 1)),
    "global_x1.25": plan((1.25, 1.25, 1.25)),
    "global_allT": plan((1, 1, 1), ALL_T),
}
BLOCKWISE = {
    "high_only": plan((1, 0, 0)),
    "mid_only": plan((0, 1, 0)),
    "low_only": plan((0, 0, 1)),
    "high_mid": plan((1, 1, 0)),
    "high_x1.5": plan((1.5, 0, 0)),
    "high_x2": plan((2, 0, 0)),
    "high_x1.5_mid_x0.5": plan((1.5, 0.5, 0)),
    "global_low_neg": plan((1, 1, -0.5)),
    "high_x1.5_low_neg": plan((1.5, 0, -0.5)),
    "high_only_allT": plan((1, 0, 0), ALL_T),
    "high_x1.5_allT": plan((1.5, 0, 0), ALL_T),
}

with device_lock():
    bench = Bench()
    report = {"dev": {}, "test": {}}
    base = bench.measure("dev", bench.cached("dev", 6))
    print(line("unedited_6", base), flush=True)
    # The block-clock path with equal scales must agree with passing one shifted t.
    one_t = bench.measure("dev", bench.run("dev", Plan(ts, knots=OP3_KNOTS)))
    print(line("global_x1 (single t)", one_t), flush=True)
    report["dev"]["global_x1_single_t"] = public(one_t)
    dev = {}
    for name, p in {**GLOBAL, **BLOCKWISE}.items():
        dev[name] = bench.measure("dev", bench.run("dev", p))
        report["dev"][name] = {**public(dev[name]), "block_scale": p.block_scale}
        print(line(name, dev[name]), flush=True)
    best_global = min(GLOBAL, key=lambda k: dev[k]["score"])
    best_block = min(BLOCKWISE, key=lambda k: dev[k]["score"])
    report["selected_on_dev"] = {"global": best_global, "blockwise": best_block}
    print(f"\nselected on dev: global={best_global} blockwise={best_block}\n", flush=True)
    if "--dev-only" not in sys.argv:
        test = {}
        for name in (best_global, best_block):
            test[name] = bench.measure("test", bench.run("test", {**GLOBAL, **BLOCKWISE}[name]))
            report["test"][name] = public(test[name])
            print(line(f"TEST {name}", test[name]), flush=True)
        report["test"]["blockwise_minus_global"] = compare(test[best_block], test[best_global])
        print(json.dumps(report["test"]["blockwise_minus_global"]), flush=True)
(config.RESULTS / "pilots" / "b_block_clocks.json").write_text(json.dumps(report, indent=1))
