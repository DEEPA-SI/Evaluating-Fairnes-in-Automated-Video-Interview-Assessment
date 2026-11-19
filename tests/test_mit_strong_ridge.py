"""Focused algebra, namespace and no-clobber checks for the stronger-ridge extension."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from src.study2 import compact, strong_ridge
from src.study2.compact import (
    AlphaRowSpaceDesign,
    CachedInputs,
    RowSpaceDesign,
    canonical_bytes,
    comparison_arms,
    file_sha256,
    fit_linear_arm,
    task_weights,
)


@pytest.mark.parametrize("balanced", [False, True])
def test_alpha_1000_matches_ridge_and_unpenalised_intercept(balanced: bool) -> None:
    rng = np.random.default_rng(512)
    x, test, y = rng.normal(size=(17, 30)), rng.normal(size=(5, 30)), rng.normal(size=17) + 4
    groups = np.r_[np.zeros(5, dtype=int), np.ones(12, dtype=int)]
    base = RowSpaceDesign.fit(x)
    design = AlphaRowSpaceDesign.from_base(base, 1000)
    weights = task_weights(groups if balanced else None, len(y))
    theta, receipt = fit_linear_arm(design, y, weights, groups if balanced else None, 0)
    reference = Ridge(alpha=1000, solver="svd").fit(
        base.scaler.transform(x), y, sample_weight=len(y) * weights
    )
    np.testing.assert_allclose(
        design.transform(test) @ theta, reference.predict(base.scaler.transform(test)), atol=1e-10
    )
    beta = base.basis @ theta[:-1] / np.sqrt(1000)
    np.testing.assert_allclose(
        base.scaler.transform(test) @ beta + theta[-1], design.transform(test) @ theta, atol=1e-12
    )
    assert receipt["final_diagnostics"]["regularisation"] == pytest.approx(
        1000 * np.dot(beta, beta) / len(y)
    )
    np.testing.assert_array_equal(design.train[:, -1], np.ones(len(y)))
    constant = np.full(len(y), 4.25)
    constant_theta = design.exact_task_solution(constant, weights)
    assert constant_theta[-1] == pytest.approx(4.25)
    np.testing.assert_allclose(constant_theta[:-1], 0, atol=1e-12)


def test_alpha_one_coordinates_and_numerical_path_are_unchanged() -> None:
    rng = np.random.default_rng(101)
    x, y, held = rng.normal(size=(14, 21)), rng.normal(size=14), rng.normal(size=(3, 21))
    base = RowSpaceDesign.fit(x)
    design = AlphaRowSpaceDesign.from_base(base, 1)
    np.testing.assert_array_equal(design.train, base.train)
    np.testing.assert_array_equal(design.transform(held), base.transform(held))
    groups = np.r_[np.zeros(6, dtype=int), np.ones(8, dtype=int)]
    weights = task_weights(groups, len(y))
    expected, _ = fit_linear_arm(base, y, weights, groups, 0.02)
    actual, _ = fit_linear_arm(design, y, weights, groups, 0.02)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf")])
def test_alpha_rejects_invalid_values(alpha: float) -> None:
    base = RowSpaceDesign.fit(np.arange(12).reshape(4, 3))
    with pytest.raises(ValueError, match="finite and positive"):
        AlphaRowSpaceDesign.from_base(base, alpha)


def test_nonfinite_optimizer_candidate_keeps_finite_exact_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(55)
    base = RowSpaceDesign.fit(rng.normal(size=(10, 15)))
    design = AlphaRowSpaceDesign.from_base(base, 1000)
    y = rng.normal(size=10)
    groups = np.r_[np.zeros(5, dtype=int), np.ones(5, dtype=int)]
    weights = task_weights(groups, len(y))

    def failed_minimize(
        _function: object, initial: np.ndarray, **_kwargs: object
    ) -> SimpleNamespace:
        return SimpleNamespace(
            x=np.full_like(initial, np.nan),
            fun=float("nan"),
            success=False,
            status=2,
            message="fixture nonfinite termination",
            nit=1,
            nfev=2,
        )

    monkeypatch.setattr(compact, "minimize", failed_minimize)
    theta, receipt = fit_linear_arm(design, y, weights, groups, 0.02)
    np.testing.assert_array_equal(theta, design.exact_task_solution(y, weights))
    assert receipt["retained_initial"] is True
    assert receipt["status"] == "finite_with_termination"
    assert np.isfinite(receipt["objective"])


def test_protocol_rejects_selection_or_comparison_changes(tmp_path: Path) -> None:
    original = strong_ridge.load_protocol(strong_ridge.PROTOCOL_PATH)
    for field, changed in (
        ("ridge_alpha", 1),
        ("strengths", [0.02]),
        ("development_selection_file_sha256", "0" * 64),
    ):
        payload = copy.deepcopy(original)
        payload[field] = changed
        path = tmp_path / f"{field}.yaml"
        path.write_text(yaml.safe_dump(payload))
        with pytest.raises(ValueError):
            strong_ridge.load_protocol(path)


@pytest.fixture
def complete_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = tuple(
        {
            "participant_id": str(person + 1),
            "session_id": f"{'P' if phase == 'pre' else 'PP'}{person + 1}",
            "source_session_id": f"{'P' if phase == 'pre' else 'PP'}{person + 1}",
            "phase": phase,
            "fold": person % 5,
        }
        for person in range(69)
        for phase in ("pre", "post")
    )
    data = CachedInputs(np.zeros((138, 3328)), np.full(138, 0.5), rows, {"fixture_only": True})

    def fixed_proxies(_data: CachedInputs, _train: np.ndarray) -> tuple[dict, dict]:
        groups = np.repeat(np.arange(69) % 2, 2)
        return {"visual": groups, "audio": 1 - groups}, {"fixture_only": True}

    monkeypatch.setattr(compact, "fit_fold_proxies", fixed_proxies)
    output = tmp_path / "strong-run"
    with threadpool_limits(limits=1):
        strong_ridge.run_strong_ridge(data, output)
    return output


def test_complete_new_run_binds_all_artifacts_and_refuses_overwrite(complete_run: Path) -> None:
    strong_ridge.validate_locked_run(complete_run)
    lock = json.loads((complete_run / "prediction-lock.json").read_bytes())
    summary = json.loads((complete_run / "summary.json").read_bytes())
    assert lock["study_id"] == strong_ridge.STUDY_ID
    assert lock["observed_fit_count"] == 75 and lock["observed_prediction_rows"] == 2070
    assert len(lock["artifact_hashes"]) == 159
    assert summary["ridge_alpha"] == 1000
    assert len(list((complete_run / "fits").glob("*.json"))) == 75
    assert len(list((complete_run / "fits").glob("*.npz"))) == 75
    before = file_sha256(complete_run / "prediction-lock.json")
    with pytest.raises(FileExistsError):
        compact._run_compact(None, complete_run, strong_ridge.PROTOCOL_PATH)  # type: ignore[arg-type]
    assert file_sha256(complete_run / "prediction-lock.json") == before


def test_wrong_identity_or_incomplete_lock_fails_before_annotation_access(
    complete_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_audit(*_args: object) -> None:
        pytest.fail("Annotation audit must not be reached for an invalid stronger-ridge lock")

    monkeypatch.setattr(strong_ridge, "audit_compact_run", forbidden_audit)
    path = complete_run / "prediction-lock.json"
    lock = json.loads(path.read_bytes())
    lock["artifact_hashes"].pop("fits/natural_task.fold-0.json")
    path.write_bytes(canonical_bytes(lock))
    with pytest.raises(ValueError, match="every expected fit"):
        strong_ridge.audit_strong_ridge_run(complete_run, complete_run / "absent-annotations")
    lock["study_id"] = compact.STUDY_ID
    path.write_bytes(canonical_bytes(lock))
    with pytest.raises(ValueError, match="identity"):
        strong_ridge.audit_strong_ridge_run(complete_run, complete_run / "absent-annotations")


def test_saved_parameters_use_original_standardised_input_coordinates(complete_run: Path) -> None:
    with np.load(complete_run / "fits/natural_task.fold-0.npz") as model:
        assert model["coefficients"].shape == (3328,)
        np.testing.assert_allclose(model["coefficients"], 0, atol=1e-14)
        assert float(model["intercept"]) == pytest.approx(0.5)


def test_strong_audit_identity_and_report_header(
    complete_run: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interval = {"estimate": 0.0, "percentile_95": [0.0, 0.0]}
    result = {
        "schema_version": "mit-study2-compact-aggregate-1",
        "study": "post_audit_exploratory_mit_compact_2026_10_21",
        "method": {},
        "utility": {},
        "utility_contrasts": {
            arm["arm_id"]: {"pooled": {"mae": interval, "srcc": interval}}
            for arm in comparison_arms()
        },
        "perceived_group_audit": {},
    }
    monkeypatch.setattr(strong_ridge, "audit_compact_run", lambda *_args: copy.deepcopy(result))
    actual = strong_ridge.audit_strong_ridge_run(complete_run, tmp_path / "annotations")
    assert actual["schema_version"] == "mit-strong-ridge-aggregate-1"
    assert actual["method"]["ridge_alpha"] == 1000
    output, report = tmp_path / "aggregate.json", tmp_path / "aggregate.md"
    strong_ridge.write_strong_ridge_audit(actual, output, report)
    assert "alpha-1000" in report.read_text()
    assert "fixed alpha-1 linear scorer" not in report.read_text()
    with pytest.raises(FileExistsError):
        strong_ridge.write_strong_ridge_audit(actual, output, report)
