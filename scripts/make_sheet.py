"""A contact sheet of evenly spaced sealed prompts, so that the picture spans categories.

Spacing is fixed in advance and does not look at results. Columns: unedited 6 steps,
edited 6 steps, unedited 50 steps, then the same three magnified.
"""
from __future__ import annotations

import sys

import numpy as np
from wspd import config, edits
from wspd.model import Denoiser
from wspd.splits import build_splits
from wspd.store import baseline_images, contact_sheet, device_lock
from wspd.supervisor import _spread

run = sys.argv[1] if len(sys.argv) > 1 else "op1"
splits = build_splits()
pairs = [("test_seen", it) for it in _spread(splits["test_seen"], 4)] + [("test_new", it) for it in _spread(splits["test_new"], 4)]
with device_lock():
    denoiser = Denoiser()
    base = np.concatenate([baseline_images(None, s, [it], config.PRIMARY_STEPS) for s, it in pairs])
    refs = np.concatenate([baseline_images(None, s, [it], config.REF_STEPS) for s, it in pairs])
    denoiser.set_edit(edits.load(config.RUNS / run / "final" / "edit.safetensors", denoiser.pristine))
    edited = denoiser.generate([it for _, it in pairs], config.PRIMARY_STEPS)
contact_sheet([[a, b, c] for a, b, c in zip(base, edited, refs)], config.RESULTS / f"sheet_{run}_spread.png")
for s, it in pairs:
    print(s, it.category, "|", it.prompt)
