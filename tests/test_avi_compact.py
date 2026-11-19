"""Focused numerical and role-boundary checks for the reconciled AVI comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.study2.compact import (
    RowSpaceDesign,
    canonical_bytes,
    comparison_arms,
    empirical_w1_gradient,
    file_sha256,
    fit_linear_arm,
    task_weights,
    write_json,
)
from src.study3.avi_compact import (
    PRIMARY_ARM,
    AlphaDesign,
    avi_utility,
    fit_proxies,
    intake_binding,
    load_inputs,
    read_locked_run,
    render_report,
    report_run,
    role_indices,
)


@pytest.mark.parametrize("alpha", [0.1, 1.0, 1000.0])
def test_alpha_coordinates_match_ridge_without_changing_compact_fitter(alpha: float) -> None:
    rng = np.random.default_rng(481)
    x, test, y = rng.normal(size=(18, 40)), rng.normal(size=(5, 40)), rng.normal(size=18)
    base = RowSpaceDesign.fit(x)
    design = AlphaDesign.from_base(base, alpha)
    theta, receipt = fit_linear_arm(design, y, task_weights(None, len(y)), None, 0.0)
    ridge = Ridge(alpha=alpha, solver="svd").fit(base.scaler.transform(x), y)
    np.testing.assert_allclose(
        design.transform(test) @ theta, ridge.predict(base.scaler.transform(test)), atol=1e-8
    )
    assert receipt["optimizer_converged"]


def test_exact_unequal_w1_regression() -> None:
    value, _ = empirical_w1_gradient(np.array([0.0, 1.0, 0.0, 0.5, 1.0]), np.array([0, 0, 1, 1, 1]))
    assert value == pytest.approx(1 / 6)


def test_proxy_fit_does_not_use_held_out_rows() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(15, 2560))
    train = np.arange(10)
    groups, receipt = fit_proxies(x, train)
    changed = x.copy()
    changed[10:] += 1000
    changed_groups, changed_receipt = fit_proxies(changed, train)
    assert changed_receipt == receipt
    for source in groups:
        np.testing.assert_array_equal(groups[source][train], changed_groups[source][train])


def _feature_fixture(root: Path) -> tuple[Path, Path]:
    release, shards = root / "release", []
    (release / "shards").mkdir(parents=True)
    outcomes = {}
    roles = ("Training", "Training", "Validation", "Testing")
    for index, role in enumerate(roles):
        participant = f"{index + 1:024x}"
        path = release / "shards" / f"{participant}.npz"
        np.savez(
            path,
            video=np.full(1024, index, dtype=np.float32),
            audio=np.full(1536, index, dtype=np.float32),
            video_mask=np.ones(6, dtype=bool),
            audio_mask=np.ones(6, dtype=bool),
            text_available=np.array(False),
        )
        shards.append({"participant_id": participant, "split": role, "sha256": file_sha256(path)})
        outcomes[participant] = {
            "split": {"Training": "train", "Validation": "validation", "Testing": "test"}[role],
            "outcomes": {"mean_rating_hirea": None if index == 0 else 3.0},
        }
    manifest = {
        "schema_version": "avi-participant-video-audio-aggregation-1",
        "status": "complete_video_audio_only",
        "clips_per_participant": 6,
        "outcome_or_audit_fields_accessed": [],
        "participant_shards": shards,
        "split_counts": {"Training": 2, "Validation": 1, "Testing": 1},
        "participant_shard_ledger_sha256": hashlib.sha256(canonical_bytes(shards)).hexdigest(),
    }
    write_json(release / "participant_feature_manifest.json", manifest)
    outcome_path = root / "outcomes.json"
    write_json(outcome_path, outcomes)
    return release, outcome_path


def test_cached_loader_needs_no_audit_and_preserves_missing_target(tmp_path: Path) -> None:
    release, outcomes = _feature_fixture(tmp_path)
    data = load_inputs(release, outcomes, require_official_counts=False)
    assert data.x.shape == (4, 2560)
    assert np.isnan(data.y[0]) and data.y[1] == 0.5
    assert len(role_indices(data)["Training"]) == 1
    assert data.provenance["audit_fields_accessed"] == []
    json.dumps(data.provenance)
    shard = next((release / "shards").glob("*.npz"))
    shard.write_bytes(b"not the accepted cached feature")
    with pytest.raises(ValueError, match="shard hash"):
        load_inputs(release, outcomes, require_official_counts=False)


def test_report_requires_prediction_lock_before_any_attribute_read(tmp_path: Path) -> None:
    audit_path = tmp_path / "must-not-open-audit.json"
    with pytest.raises(FileNotFoundError) as error:
        report_run(tmp_path, tmp_path / "outcomes.json", audit_path)
    assert str(error.value.filename).endswith("prediction-lock.json")


def _locked_fixture(
    root: Path, direction: float = 1.0, *, terminated_arms: set[str] | None = None
) -> tuple[Path, Path]:
    people = [f"{index + 1:024x}" for index in range(130)]
    outcomes, attributes, rows = {}, {}, []
    for index, participant in enumerate(people):
        code = 1 if index < 67 else 2 if index < 129 else 4
        target = 0.25 + index / 300
        outcomes[participant] = {"split": "test", "outcomes": {"mean_rating_hirea": 1 + 4 * target}}
        attributes[participant] = {"self_reported_gender_code": code}
        error = (0.1 if code == 1 else 0.04) if direction > 0 else (0.04 if code == 1 else 0.1)
        for arm in comparison_arms():
            rows.append({**arm, "participant_id": participant, "prediction": target + error})
    outcome_path, audit_path = root / "outcomes.json", root / "audit.json"
    write_json(outcome_path, outcomes)
    write_json(audit_path, attributes)
    manifest = {
        "inputs": {"outcomes_sha256": file_sha256(outcome_path)},
        "test_participants": people,
        "training_mean_target": 0.5,
        "selected_alpha": 1.0,
        "interpretation": "synthetic_test_fixture_only",
        "protocol": {"audit_attributes_sha256": file_sha256(audit_path)},
    }
    write_json(root / "manifest.json", manifest)
    write_json(root / "predictions.json", {"rows": rows})
    write_json(root / "proxy-mechanism.json", {"rows": []})
    (root / "fits").mkdir()
    statuses = {}
    for arm in comparison_arms():
        converged = arm["arm_id"] not in (terminated_arms or set())
        status = "converged" if converged else "finite_with_termination"
        statuses[status] = statuses.get(status, 0) + 1
        write_json(
            root / "fits" / f"{arm['arm_id']}.json",
            {
                **arm,
                "selected_alpha": 1.0,
                "status": status,
                "optimizer_converged": converged,
                "optimizer_status": 0 if converged else 1,
                "optimizer_message": "converged" if converged else "iteration limit",
                "iterations": 1 if converged else 1000,
                "function_evaluations": 2 if converged else 1100,
                "retained_initial": False,
                "initial_objective": 0.04,
                "objective": 0.03,
                "global_optimum_claimed": False,
            },
        )
    lock = {
        "state": "locked",
        "audit_fields_accessed": [],
        "arm_count": 15,
        "prediction_count": 1950,
        "test_participant_count": 130,
        "fit_status_counts": statuses,
        "all_optimizers_converged": not terminated_arms,
        "artifact_hashes": {
            str(path.relative_to(root)): file_sha256(path)
            for path in (
                root / "manifest.json",
                root / "predictions.json",
                root / "proxy-mechanism.json",
                *(root / "fits" / f"{arm['arm_id']}.json" for arm in comparison_arms()),
            )
        },
    }
    write_json(root / "prediction-lock.json", lock)
    return outcome_path, audit_path


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_paired_audit_keeps_signed_gap_and_uses_all_fifteen_arms(
    tmp_path: Path,
    direction: float,
) -> None:
    outcomes, attributes = _locked_fixture(tmp_path, direction)
    result = report_run(tmp_path, outcomes, attributes)
    assert result["coverage"]["prediction_rows"] == 1950
    assert result["coverage"]["group_counts"] == {"1": 67, "2": 62, "3": 0, "4": 1, "unknown": 0}
    primary = result["arms"][PRIMARY_ARM]
    signed = primary["signed_mae_gap_code_1_minus_2"]
    assert signed["estimate"] == pytest.approx(direction * 0.06)
    np.testing.assert_allclose(signed["percentile_95"], [direction * 0.06] * 2, atol=1e-12)
    assert primary["absolute_mae_gap_descriptive"] == pytest.approx(0.06)
    for effect in primary["matched_effects"].values():
        assert effect["estimate"] == pytest.approx(0)
        np.testing.assert_allclose(effect["percentile_95"], [0, 0], atol=1e-12)
    assert not primary["group_errors"]["4"]["reportable"]


def test_incomplete_arm_coverage_fails_even_with_updated_file_hash(tmp_path: Path) -> None:
    _locked_fixture(tmp_path)
    path = tmp_path / "predictions.json"
    payload = json.loads(path.read_bytes())
    payload["rows"].pop()
    path.write_bytes(canonical_bytes(payload))
    lock_path = tmp_path / "prediction-lock.json"
    lock = json.loads(lock_path.read_bytes())
    lock["artifact_hashes"]["predictions.json"] = file_sha256(path)
    lock_path.write_bytes(canonical_bytes(lock))
    with pytest.raises(ValueError, match="coverage"):
        read_locked_run(tmp_path)


def test_report_requires_all_bound_fit_receipts_before_any_attribute_read(tmp_path: Path) -> None:
    outcomes, _ = _locked_fixture(tmp_path)
    lock_path = tmp_path / "prediction-lock.json"
    lock = json.loads(lock_path.read_bytes())
    del lock["artifact_hashes"][f"fits/{PRIMARY_ARM}.json"]
    lock_path.write_bytes(canonical_bytes(lock))
    with pytest.raises(ValueError, match="complete fifteen-arm fit receipts"):
        report_run(tmp_path, outcomes, tmp_path / "must-not-open-audit.json")


def test_tampered_fit_receipt_fails_before_any_attribute_read(tmp_path: Path) -> None:
    outcomes, _ = _locked_fixture(tmp_path)
    (tmp_path / "fits" / f"{PRIMARY_ARM}.json").write_bytes(b"tampered receipt")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        report_run(tmp_path, outcomes, tmp_path / "must-not-open-audit.json")


def test_fit_receipt_identity_must_match_even_with_updated_hash(tmp_path: Path) -> None:
    _locked_fixture(tmp_path)
    name = f"fits/{PRIMARY_ARM}.json"
    path = tmp_path / name
    receipt = json.loads(path.read_bytes())
    receipt["arm_id"] = "natural_task"
    path.write_bytes(canonical_bytes(receipt))
    lock_path = tmp_path / "prediction-lock.json"
    lock = json.loads(lock_path.read_bytes())
    lock["artifact_hashes"][name] = file_sha256(path)
    lock_path.write_bytes(canonical_bytes(lock))
    with pytest.raises(ValueError, match="receipt identity"):
        read_locked_run(tmp_path)


@pytest.mark.parametrize("field", ["fit_status_counts", "all_optimizers_converged"])
def test_fit_receipt_dispositions_must_agree_with_lock(tmp_path: Path, field: str) -> None:
    _locked_fixture(tmp_path, terminated_arms={PRIMARY_ARM})
    lock_path = tmp_path / "prediction-lock.json"
    lock = json.loads(lock_path.read_bytes())
    lock[field] = {"converged": 15} if field == "fit_status_counts" else True
    lock_path.write_bytes(canonical_bytes(lock))
    with pytest.raises(ValueError, match="solver counts disagree"):
        read_locked_run(tmp_path)


def test_primary_termination_is_published_with_each_arm_disposition(tmp_path: Path) -> None:
    outcomes, attributes = _locked_fixture(tmp_path, terminated_arms={PRIMARY_ARM})
    result = report_run(tmp_path, outcomes, attributes)
    assert result["fit_status_counts"] == {"converged": 14, "finite_with_termination": 1}
    assert not result["all_optimizers_converged"]
    assert all("solver" in arm for arm in result["arms"].values())
    solver = result["arms"][PRIMARY_ARM]["solver"]
    assert solver["status"] == "finite_with_termination"
    assert solver["optimizer_status"] == 1 and solver["iterations"] == 1000
    assert solver["optimizer_message"] == "iteration limit"
    assert not solver["optimizer_converged"] and not solver["global_optimum_claimed"]
    report = render_report(result)
    assert "| Arm | Solver disposition |" in report
    assert f"| {PRIMARY_ARM} | finite_with_termination |" in report
    assert "Some bounded solver fits terminated without reported convergence" in report


def test_avi_rating_difference_conversion_is_four_not_six() -> None:
    metrics = avi_utility(np.array([0.2, 0.8]), np.array([0.3, 0.7]))
    assert metrics["mae_raw_scale"] == pytest.approx(0.4)
    assert metrics["rmse_raw_scale"] == pytest.approx(0.4)


def test_intake_binding_resolves_separated_sources_without_opening_them(tmp_path: Path) -> None:
    outcome, outcome_hash = intake_binding(tmp_path, "participant-outcomes.json")
    audit, audit_hash = intake_binding(tmp_path, "audit-attributes.json")
    assert outcome.parent == audit.parent
    assert outcome.is_relative_to(tmp_path / "validation")
    assert not outcome.exists() and not audit.exists()
    assert outcome_hash == "2c0ac7bea1cab8e9898a4214fa842c8a714c6329e4789bde47854b5766c804ef"
    assert audit_hash == "7488ae7aac68aa9b91002024375be1002fe881b43c7182a542afe656d84257e9"
