# Operator brief

This is the complete brief given to each LLM operator. It deliberately states
the failure and the tools, not any hypothesis about the cause or the fix.

## The task

`segmind/tiny-sd` is a small text-to-image latent diffusion model: a UNet
denoiser with 323,384,964 weights, trained to predict noise. With DDIM
(trailing timestep spacing) at 50
steps it produces images with visible fine texture such as weave, fur and leaf
structure. At 5 to 8 steps that texture is degraded, on many different prompts.

Find **one static edit to the UNet's weights** that makes few-step sampling
(6 steps is the primary target, 5 and 8 also count) closer to the 50-step
result, on prompts you have not seen. You work by reasoning about the weights
and by running a small number of measured experiments. You may not use gradients.

The edit is judged later, by someone else, on prompts and categories you will
never see. It has to generalize. Fitting these dev prompts does not count.

## Tools

Set `WSP_RUN=<your run name>` and call the CLI at `.venv/bin/wsp`.

Free, as often as you like:

- `wsp info`: constants, the baseline curve against step count, and your budget.
- `wsp prompts [-n N]`: the dev prompts.
- `wsp weights [--filter REGEX]`: parameter names, shapes and norms.
- Python over the raw weights, on CPU, using any numeric analysis you like:

  ```python
  from wspd.edits import load_pristine
  w = load_pristine()  # dict of parameter name -> float32 CPU tensor
  ```

- Viewing images. Unedited dev images are at
  `runs/cache/images/dev/ddim_trailing/<steps>/<id>.png` for steps in 5, 6, 8, 10, 12,
  16, 24 and 50, and you can open PNGs directly.
- Unedited activation statistics at 6 and 50 steps, in
  `runs/cache/probe_baseline_6.json` and `runs/cache/probe_baseline_50.json`.

Metered, drawn from a budget of **40 units**. One unit is 8 images at up to 8
steps, and larger runs cost proportionally:

- `wsp probe --modules REGEX [--steps N] [--n-images K] [--edit FILE]`: runs
  sampling and records RMS and spatial spectrum for up to 48 module outputs at
  each step.
- `wsp try EDIT.py [--steps 5,6,8] [--n-images 8] [--note TEXT]`: applies your
  edit, samples, and prints metrics beside the baseline. It also writes a
  contact sheet whose columns are baseline, edited and reference, followed by
  the same three magnified.

Finish with:

- `wsp submit EDIT.py --note TEXT`: seals your one final edit. It cannot be undone.

## Edit scripts

An edit script is a Python file defining `apply(w)`, which mutates the weights
in place. It receives nothing else: no images, no prompts, no model.

```python
def apply(w):
    w["some.parameter.weight"][:, 3] *= 1.1
```

The edit has to be a static change to weights. It cannot depend on the prompt or
on the step count at sampling time, and the sampler, guidance scale and text
encoder stay fixed. The denoiser receives the timestep as input, so anything the
weights can compute from it is available to you.

## Metrics

Every metric compares against the same-prompt, same-seed 50-step image.

- `deficit_<band>`: reference power minus image power, in dB. Positive means
  detail is missing, negative means excess. `abs_deficit_<band>` is the mean
  absolute value per image, so overshooting is penalized. Bands in cycles per
  pixel: structure 1/64-1/16, coarse 1/16-1/8, fine 1/8-1/4, grain 1/4-1/2.
- `texture`: the mean of `abs_deficit_coarse` and `abs_deficit_fine`.
- `lpips`: perceptual distance to the reference.
- `swd`: sliced Wasserstein distance over Laplacian patches, which compares
  texture statistics without pixel alignment. It has a noise floor that grows
  as the image count shrinks, shown by `wsp info`, so differences near the floor
  mean little.
- `clip`: similarity to the prompt. A drop means the edit is damaging meaning.
- `luma_shift`, `contrast_ratio`: guards against simply changing tone.

No single number decides success. Adding noise raises band energy and is not a
fix, so look at the images as well as the tables.

## Rules

1. No gradients of any kind: no autograd, no finite-difference optimizer run
   over the weights outside your budget, and no `torch.func`.
2. Do not run the UNet or VAE outside `wsp probe` and `wsp try`. Reading the
   weights is free, and computing with them on CPU is fine as long as it is not
   a forward pass of the network or a re-implementation of one.
3. Do not read `data/prompts_pool.json`, `results/`, `docs/PREREGISTRATION.md`,
   `README.md`, `runs/cache/images/test_*`, `runs/cache/baseline_test_*`, or any
   other run's folder under `runs/`.
4. Do not change anything in `src/` or `tests/`. Keep your scripts in your run's
   folder, `runs/<your run name>/`.
5. A negative result is a good result. If nothing you find is a real
   improvement, say so and submit a no-op or your best candidate with an honest
   note. Do not overstate what an edit does.

## Report back

When you have submitted, reply with:

1. The weights you changed: names, count, and the exact operation.
2. Your hypothesis about why those weights relate to the failure, and how you
   arrived at it from the weights and measurements.
3. The evidence, including what you tried that failed.
4. Units used, and your honest confidence that this will generalize.
