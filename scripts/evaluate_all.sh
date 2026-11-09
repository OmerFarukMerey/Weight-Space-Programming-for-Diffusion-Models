#!/usr/bin/env bash
# Scores every sealed arm on the sealed splits. Run it only after all operators have
# submitted. The climbs seal first, so no arm is chosen after test metrics exist.
set -euo pipefail
cd "$(dirname "$0")/.."
SUP=.venv/bin/wsp-sup

for seed in 1 2 3; do
  [ -f "runs/climb$seed/final/edit.safetensors" ] || $SUP climb "climb$seed" --seed "$seed"
done

$SUP baseline-test --require op1 op2 op3 climb1 climb2 climb3

for run in op1 op2 op3; do
  [ -f "results/final_$run.json" ] || $SUP final "$run"
done
for seed in 1 2 3; do
  [ -f "results/final_climb$seed.json" ] || $SUP final "climb$seed" --quick
done
for run in op1 op2 op3; do
  [ -f "results/null_$run.json" ] || $SUP null "$run" --k 20
done
[ -f results/postproc.json ] || $SUP postproc
$SUP report
