"""Fast checks that need no model download."""
import numpy as np
import pytest
import torch

from wspd import config, edits, metrics, splits


def make_weights():
    torch.manual_seed(0)
    return {"a.weight": torch.randn(4, 4), "b.bias": torch.randn(4)}


def test_splits_are_disjoint_and_complete():
    built = splits.build_splits()
    assert {len(built["dev"]), len(built["test_seen"]), len(built["test_new"])} == {24, 40, 32}
    prompts = [it.prompt for items in built.values() for it in items]
    assert len(prompts) == len(set(prompts)) == 96
    seen = {it.category for it in built["dev"]}
    assert seen == {it.category for it in built["test_seen"]}
    assert not seen & {it.category for it in built["test_new"]}
    assert len({it.category for it in built["dev"][:8]}) == 8


def test_splits_are_deterministic():
    assert splits.build_splits() == splits.build_splits()


def test_diff_counts_only_changed_weights():
    pristine = make_weights()
    new = {k: v.clone() for k, v in pristine.items()}
    new["a.weight"][0, 0] += 1.0
    edit = edits.diff(new, pristine)
    assert edit.summary["weights_changed"] == 1
    assert set(edit.tensors) == {"a.weight"}


def test_noop_edit_is_empty():
    pristine = make_weights()
    assert edits.diff({k: v.clone() for k, v in pristine.items()}, pristine).tensors == {}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda w: w.update({"a.weight": torch.zeros(3, 3)}),
        lambda w: w.update({"a.weight": w["a.weight"] * float("nan")}),
        lambda w: w.pop("b.bias"),
    ],
)
def test_diff_rejects_invalid_edits(mutate):
    pristine = make_weights()
    new = {k: v.clone() for k, v in pristine.items()}
    mutate(new)
    with pytest.raises(edits.EditError):
        edits.diff(new, pristine)


@pytest.mark.parametrize(
    "source",
    [
        "loss.backward()",
        "torch.autograd.grad(a, b)",
        "x.requires_grad = True",
        "torch.optim.SGD([])",
        "from diffusers import UNet2DConditionModel",
        "from wspd.model import Denoiser",
    ],
)
def test_lint_blocks_gradient_and_model_access(source):
    with pytest.raises(edits.EditError):
        edits.lint(source)


def test_lint_allows_plain_edits():
    edits.lint('def apply(w):\n    w["conv_out.weight"] *= 0.95\n')


def test_scaled_edit_interpolates():
    pristine = make_weights()
    new = {k: v.clone() for k, v in pristine.items()}
    new["a.weight"] = new["a.weight"] + 2.0
    half = edits.scaled(edits.diff(new, pristine), 0.5, pristine)
    assert torch.allclose(half.tensors["a.weight"], pristine["a.weight"] + 1.0)


def test_save_and_load_round_trip(tmp_path):
    pristine = make_weights()
    new = {k: v.clone() for k, v in pristine.items()}
    new["b.bias"] *= 2
    edit = edits.diff(new, pristine)
    edit.save(tmp_path / "edit.safetensors")
    assert edits.load(tmp_path / "edit.safetensors", pristine).sha256 == edit.sha256


def _noise_image(seed=0, blur=0):
    rng = np.random.default_rng(seed)
    x = rng.random((1, 256, 256, 3))
    if blur:
        k = np.ones(blur) / blur
        for axis in (1, 2):
            x = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis, x)
    return (x * 255).astype(np.uint8)


def test_blur_produces_positive_deficit_at_high_frequency():
    ref, soft = _noise_image(), _noise_image(blur=5)
    deficits = metrics.band_deficit_db(soft, ref)
    assert deficits["grain"][0] > deficits["structure"][0] and deficits["grain"][0] > 3


def test_deficit_of_identical_images_is_zero():
    ref = _noise_image()
    assert all(abs(v[0]) < 1e-9 for v in metrics.band_deficit_db(ref, ref).values())


def test_swd_separates_blur_from_identity():
    ref = np.concatenate([_noise_image(i) for i in range(4)])
    soft = np.concatenate([_noise_image(i, blur=5) for i in range(4)])
    other = np.concatenate([_noise_image(i + 10) for i in range(4)])
    assert metrics.swd(soft, ref)["mean"] > 5 * metrics.swd(other, ref)["mean"]


def test_paired_bootstrap_brackets_the_mean():
    delta = np.random.default_rng(0).normal(1.0, 0.5, 64)
    ci = metrics.paired_bootstrap(delta)
    assert ci["lo"] < ci["mean"] < ci["hi"] and ci["lo"] > 0.5


def test_step_equivalent_interpolates_and_clamps():
    steps, curve = [5, 10, 20], [4.0, 2.0, 1.0]
    assert metrics.step_equivalent(2.0, steps, curve) == pytest.approx(10)
    assert 10 < metrics.step_equivalent(1.5, steps, curve) < 20
    assert metrics.step_equivalent(9.0, steps, curve) == 5
    assert metrics.step_equivalent(0.1, steps, curve) == 20


def test_unit_cost_scales_with_images_and_steps():
    assert config.unit_cost(8, 6) == 1
    assert config.unit_cost(24, 6) == 3
    assert config.unit_cost(8, 50) == 7


def test_spectrum_fractions_separate_smooth_from_checkerboard():
    from wspd.model import spectrum_fractions

    y, x = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    smooth = torch.sin(2 * torch.pi * x / 32)[None, None]
    checker = (((x + y) % 2) * 2 - 1)[None, None]
    assert spectrum_fractions(smooth)[0] > 0.99
    assert spectrum_fractions(checker)[2] > 0.99
    assert sum(spectrum_fractions(smooth)) == pytest.approx(1.0)


def test_probe_labels_steps_per_image_and_averages():
    from wspd.model import ActivationProbe, _mean_by

    module = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1))
    probe = ActivationProbe(module, ["0"])
    for _image in range(2):
        probe.step = 0
        for step in range(3):
            module(torch.ones(1, 1, 8, 8))
            probe.step = step + 1
    probe.close()
    assert [r["step"] for r in probe.records] == [0, 1, 2, 0, 1, 2]
    merged = _mean_by(probe.records, ("module", "step"), ("rms",))
    assert [r["step"] for r in merged] == [0, 1, 2]


def test_unsharp_identity_at_unit_gain():
    from wspd.supervisor import _unsharp

    image = _noise_image()
    assert np.abs(_unsharp(image, 2.0, 1.0).astype(int) - image.astype(int)).max() <= 1


def test_texture_metric_is_mean_absolute_coarse_and_fine():
    class Stub(metrics.Evaluator):
        def lpips(self, images, refs, batch=8):
            return np.zeros(len(images))

        def clip_similarity(self, images, prompts, batch=16):
            return np.zeros(len(images))

    ref, soft = _noise_image(), _noise_image(blur=5)
    out = Stub("cpu").evaluate(soft, ref, ["x"])
    expected = (abs(out["deficit_coarse"][0]) + abs(out["deficit_fine"][0])) / 2
    assert out["texture"][0] == pytest.approx(expected)
    assert metrics.summarize(out)["texture"] == pytest.approx(expected)


def _fake_final(dt=-1.0, dl=-0.03, eq=9.0, clip=0.0):
    def interval(mean, width=0.5):
        return {"mean": mean, "lo": mean - abs(width * mean), "hi": mean + abs(width * mean)}

    split = {
        "base": {"texture": 4.0, "lpips": 0.4},
        "edit": {"texture": 4.0 + dt, "lpips": 0.4 + dl},
        "delta": {"texture": interval(dt), "lpips": interval(dl), "clip": {"mean": clip, "lo": clip, "hi": clip}},
        "step_equivalent": {"texture": eq, "lpips": eq},
    }
    return {
        "splits": {"test_seen": {"6": split}, "test_new": {"6": split}},
        "dose_response": {
            "-1.0": {"texture": 5.0}, "0.0": {"texture": 4.0}, "0.25": {"texture": 3.8},
            "0.5": {"texture": 3.5}, "1.0": {"texture": 3.0},
        },
    }


def test_arm_checks_pass_a_clean_edit():
    from wspd.report import arm_checks

    assert all(arm_checks(_fake_final(), {"share_null_at_least_as_good": 0.0}).values())


def test_arm_checks_catch_each_failure():
    from wspd.report import arm_checks

    assert not arm_checks(_fake_final(dt=0.2))["H2_texture_generalizes"]
    assert not arm_checks(_fake_final(dl=0.02))["H3_perceptual_improves"]
    assert not arm_checks(_fake_final(eq=7.0))["H4_saves_steps"]
    assert not arm_checks(_fake_final(clip=-0.02))["H5_keeps_meaning"]
    assert not arm_checks(_fake_final(), {"share_null_at_least_as_good": 0.2})["H6_beats_null"]
    broken = _fake_final()
    broken["dose_response"]["0.5"]["texture"] = 4.2
    assert not arm_checks(broken)["H7_causal"]


@pytest.mark.parametrize(
    "n_pass_h2,n_pass_all,llm,climb,filter_score,expected",
    [
        (3, 3, 1.6, 1.9, 1.8, "Yes"),
        (2, 2, 1.6, 1.9, 1.8, "Yes"),
        (3, 3, 1.6, 1.9, 1.5, "Partial"),  # a filter matches
        (3, 1, 1.6, 1.9, 1.8, "Partial"),  # energy back, too few pass everything
        (1, 0, 1.9, 1.95, 1.8, "No"),  # does not generalize
        (3, 3, 1.6, 1.5, 1.8, "No"),  # blind search does as well
    ],
)
def test_verdict_rules(n_pass_h2, n_pass_all, llm, climb, filter_score, expected):
    from wspd.report import verdict

    checks = {}
    for i in range(3):
        checks[f"op{i}"] = {"H2_texture_generalizes": i < n_pass_h2, "H3_perceptual_improves": i < n_pass_all}
    assert verdict(checks, [llm] * 3, [climb] * 3, filter_score)["label"] == expected


def test_ledger_never_reuses_a_folder(tmp_path, monkeypatch):
    from wspd.store import Ledger

    monkeypatch.setattr(config, "RUNS", tmp_path)
    ledger = Ledger("run")
    assert ledger.new_folder("try") != ledger.new_folder("try")


def test_trailing_matches_diffusers_spacing():
    from wspd.lab import trailing, warped

    assert trailing(6) == [999, 832, 666, 499, 332, 166]
    assert warped(6, 1.0) == trailing(6)
    assert warped(6, 0.7)[-1] > trailing(6)[-1] > warped(6, 1.4)[-1]


def test_plan_shift_and_nfe():
    from wspd.lab import OP3_KNOTS, Plan

    plain = Plan((999, 499, 166))
    assert plain.delta(499, 166) == 0 and plain.nfe == 6
    half = Plan((999, 499, 166), relabel=0.5, relabel_below=500)
    assert half.delta(999, 499) == 0 and half.delta(499, 166) == pytest.approx(166.5)
    assert Plan((999, 166), knots=OP3_KNOTS).delta(166, 0) == pytest.approx(85)
    assert Plan((999, 499, 166), guided=(True, True, False)).nfe == 5
    per_step = Plan((999, 499, 166), relabel_steps=(0.0, 0.5, 1.0))
    assert per_step.delta(999, 499, 0) == 0 and per_step.delta(166, 0, 2) == pytest.approx(166)
