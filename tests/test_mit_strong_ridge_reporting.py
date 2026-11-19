"""Regression checks for the retained stronger-ridge aggregate, without refitting."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "results/mit-strong-ridge-2026-11-19-v1.json"
LEGACY = ROOT / "results/study2-compact-2026-10-21-aggregate.json"
LEGACY_SHA256 = "c0560b284a30b3e10785027cd4ee28e09694218229abc6d44c13653e19bd13cd"
ARMS = {
    "natural_task",
    "visual_balanced_task",
    "audio_balanced_task",
    *(
        f"{proxy}_{exposure}_w1_lambda-{strength}"
        for proxy in ("visual", "audio")
        for exposure in ("natural", "balanced")
        for strength in ("0.002", "0.02", "0.2")
    ),
}
PRIMARY = {
    f"{proxy}_{exposure}_w1_lambda-0.02"
    for proxy in ("visual", "audio")
    for exposure in ("natural", "balanced")
}


@pytest.fixture(scope="module")
def evidence() -> dict:
    return json.loads(AGGREGATE.read_bytes())


def test_new_identity_and_source_bindings_preserve_the_original_aggregate(evidence: dict) -> None:
    assert evidence["schema_version"] == "mit-strong-ridge-aggregate-1"
    assert evidence["study_id"] == "mit-strong-ridge-2026-11-19-v1"
    assert evidence["study"] == "post_audit_exploratory_mit_strong_ridge_2026_11_19"
    assert evidence["status"] == "complete"
    assert evidence["ridge_alpha"] == evidence["method"]["ridge_alpha"] == 1000
    assert evidence["method"]["alpha_selection_scope"] == "existing_rq1_development_selection"
    assert evidence["coverage"] == {"n_arms": 15, "n_participants": 69, "n_predictions": 2070}
    assert evidence["input_hashes"]["prediction_lock_sha256"] == (
        "fef702149740a9d62e3ded023e1058774806994e13f8a194c65de465e11bfed3"
    )
    legacy_bytes = LEGACY.read_bytes()
    assert hashlib.sha256(legacy_bytes).hexdigest() == LEGACY_SHA256
    old = json.loads(legacy_bytes)
    for name in ("annotation_csv_sha256", "annotation_manifest_sha256"):
        assert evidence["input_hashes"][name] == old["input_hashes"][name]


def test_all_fixed_strengths_and_controls_have_finite_complete_utility(evidence: dict) -> None:
    assert set(evidence["utility"]) == ARMS
    assert set(evidence["method"]["primary_contrasts"]) == PRIMARY
    for arm, scopes in evidence["utility"].items():
        assert set(scopes) == {"pooled", "pre", "post"}
        for scope, metrics in scopes.items():
            assert metrics["n_sessions"] == (138 if scope == "pooled" else 69)
            assert metrics["n_participants"] == 69
            assert all(math.isfinite(metrics[key]) for key in ("mae", "rmse", "srcc", "pearson"))
            assert metrics["rmse"] >= metrics["mae"] >= 0
            assert -1 <= metrics["srcc"] <= 1
        if "lambda-" in arm:
            proxy, exposure = arm.split("_")[:2]
            expected = "natural_task" if exposure == "natural" else f"{proxy}_balanced_task"
            assert evidence["utility_contrasts"][arm]["control_arm_id"] == expected


def test_all_75_solver_receipts_and_the_20_primary_fits_are_retained(evidence: dict) -> None:
    fits = evidence["training_and_solver_diagnostics"]
    assert len(fits) == 75
    assert {(fit["arm_id"], fit["outer_fold"]) for fit in fits} == {
        (arm, fold) for arm in ARMS for fold in range(5)
    }
    assert (
        dict(Counter(fit["status"] for fit in fits))
        == evidence["fit_status_counts"]
        == {
            "converged": 62,
            "finite_with_termination": 13,
        }
    )
    primary = [fit for fit in fits if fit["arm_id"] in PRIMARY]
    assert len(primary) == 20 and all(fit["optimizer_converged"] for fit in primary)
    terminated = [fit for fit in fits if not fit["optimizer_converged"]]
    assert len(terminated) == 13
    assert all(fit["arm_id"].endswith("lambda-0.2") for fit in terminated)
    assert evidence["all_optimizers_converged"] is False
    for fit in fits:
        assert math.isfinite(fit["objective"]) and math.isfinite(fit["initial_objective"])
        assert fit["objective"] <= fit["initial_objective"] + 1e-14
        assert set(fit["training_proxy_w1"]) == {"visual", "audio"}


def test_phase_matched_audit_coverage_and_ordinal_skin_tone_are_preserved(evidence: dict) -> None:
    assert (
        evidence["method"]["skin_tone"]
        == "ordinal_1_to_5_categories_preserved_without_binarisation"
    )
    assert evidence["method"]["minimum_reportable_phase_category_participants"] == 10
    for phase, count in (("post", 69), ("pre", 57)):
        audit = evidence["perceived_group_audit"][phase]
        assert audit["n_participants"] == count
        assert set(audit["axes"]) == {
            "perceived_gender_presentation",
            "perceived_racialised_appearance",
            "perceived_skin_tone",
        }
        for axis in audit["axes"].values():
            assert set(axis["arms"]) == ARMS
            for result in axis["arms"].values():
                assert result["n_annotated_participants"] == count
                assert sum(result["all_category_counts"].values()) == count
        skin = audit["axes"]["perceived_skin_tone"]["arms"]["natural_task"]
        assert set(skin["all_category_counts"]) == {"1", "2", "3", "4", "5"}
        assert len(skin["supported_categories"]) >= 2


def test_natural_reference_uses_the_new_pooled_metrics_not_the_selected_fold_mean(
    evidence: dict,
) -> None:
    baseline = evidence["utility"]["natural_task"]["pooled"]
    assert baseline["mae"] == pytest.approx(0.08124787100183284, abs=1e-15)
    assert baseline["srcc"] == pytest.approx(0.4945637780043393, abs=1e-15)
    assert round(6 * baseline["mae"], 3) == 0.487
    assert round(baseline["srcc"], 3) == 0.495
    assert baseline["srcc"] != pytest.approx(0.567, abs=1e-3)
    old = json.loads(LEGACY.read_bytes())["utility"]["natural_task"]["pooled"]
    assert baseline["mae"] < old["mae"] and baseline["srcc"] > old["srcc"]


def test_primary_disparity_gains_use_the_correct_control_minus_arm_direction(
    evidence: dict,
) -> None:
    method = evidence["method"]
    assert method["bootstrap_replicates"] == 10000 and method["bootstrap_seed"] == 2026
    assert method["interval"] == "two_sided_percentile_95_conditional_on_fixed_fits"
    assert method["disparity_gain_direction"] == "control_minus_arm_positive_is_reduction"
    for axis in evidence["perceived_group_audit"]["post"]["axes"].values():
        for arm in PRIMARY:
            contrast = axis["contrasts"][arm]
            effect, control = contrast["disparity_gain"], contrast["control_arm_id"]
            expected = (
                axis["arms"][control]["max_pairwise_w1"] - axis["arms"][arm]["max_pairwise_w1"]
            )
            assert effect["estimate"] == pytest.approx(expected, abs=1e-14)
            lower, upper = effect["percentile_95"]
            assert math.isfinite(lower) and math.isfinite(upper) and lower < upper
            assert effect["valid_draws"] == 10000 and effect["undefined_draws"] == 0


def test_near_zero_primary_gender_interval_signs_are_not_rounded_into_equivalence(
    evidence: dict,
) -> None:
    contrasts = evidence["perceived_group_audit"]["post"]["axes"]["perceived_gender_presentation"][
        "contrasts"
    ]
    natural = contrasts["visual_natural_w1_lambda-0.02"]["disparity_gain"]
    balanced = contrasts["visual_balanced_w1_lambda-0.02"]["disparity_gain"]
    assert natural["estimate"] == pytest.approx(0.019013214703555555, abs=1e-15)
    assert natural["percentile_95"] == pytest.approx(
        [-0.0004330133709776244, 0.02368136957743996], abs=1e-15
    )
    assert balanced["percentile_95"] == pytest.approx(
        [0.00033860407470582703, 0.020607629738867435], abs=1e-15
    )
    assert natural["percentile_95"][0] < 0 < balanced["percentile_95"][0]
    assert round(natural["percentile_95"][0], 4) == -0.0004
    assert round(balanced["percentile_95"][0], 4) == 0.0003


def test_primary_utility_contrasts_retain_method_minus_control_signs(evidence: dict) -> None:
    for arm in PRIMARY:
        contrasts = evidence["utility_contrasts"][arm]
        control = contrasts["control_arm_id"]
        for scope in ("pooled", "pre", "post"):
            for metric in ("mae", "srcc"):
                expected = (
                    evidence["utility"][arm][scope][metric]
                    - evidence["utility"][control][scope][metric]
                )
                assert contrasts[scope][metric]["estimate"] == pytest.approx(expected, abs=1e-14)
    primary = evidence["utility_contrasts"]["visual_natural_w1_lambda-0.02"]["pooled"]
    assert primary["mae"]["estimate"] == pytest.approx(-0.00037732499528629615, abs=1e-15)
    assert primary["srcc"]["estimate"] == pytest.approx(-0.013224106596562257, abs=1e-15)


def test_readable_report_preserves_opposite_near_zero_gender_endpoint_signs() -> None:
    report = AGGREGATE.with_suffix(".md").read_text()
    gender = report.split("## Post perceived-group audit", 1)[1].split(
        "### perceived racialised appearance", 1
    )[0]
    assert "| visual_natural_w1_lambda-0.02 | 0.032 | 0.019 [-4.3e-4, 0.024] |" in gender
    assert "| visual_balanced_w1_lambda-0.02 | 0.034 | 0.016 [3.4e-4, 0.021] |" in gender
