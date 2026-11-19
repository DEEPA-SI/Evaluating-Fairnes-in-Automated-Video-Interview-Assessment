"""Aggregate-only audit of the locked compact exploratory MIT follow-up.

See the T-483 protocol for estimands. This module loads no feature or fitting code.
The prediction lock is checked before the phase-specific perceived annotations are read.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026
MINIMUM_GROUP_SIZE = 10
AXES = (
    "perceived_gender_presentation",
    "perceived_racialised_appearance",
    "perceived_skin_tone",
)
UNKNOWN_VALUES = {
    "",
    "unknown",
    "unobservable",
    "not_observable",
    "cannot_determine",
    "uncertain",
    "na",
    "n/a",
}
PRIMARY_ARMS = tuple(
    f"{source}_{exposure}_w1_lambda-0.02"
    for source in ("visual", "audio")
    for exposure in ("natural", "balanced")
)
ARM_IDS = (
    "natural_task",
    "visual_balanced_task",
    "audio_balanced_task",
    *(
        f"{source}_{exposure}_w1_lambda-{strength}"
        for source in ("visual", "audio")
        for exposure in ("natural", "balanced")
        for strength in ("0.002", "0.02", "0.2")
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase(value: str) -> str:
    phases = {"P": "pre", "PP": "post", "pre": "pre", "post": "post"}
    try:
        return phases[value]
    except KeyError as error:
        raise ValueError(f"unknown session phase: {value!r}") from error


def _participant(value: Any) -> str:
    return str(value)


def _session(row: dict[str, Any]) -> str:
    return str(row.get("session_id", row.get("canonical_session_id", "")))


def _matched_control(arm_id: str) -> str | None:
    if "_w1_lambda-" not in arm_id:
        return None
    source, exposure = arm_id.split("_")[:2]
    return "natural_task" if exposure == "natural" else f"{source}_balanced_task"


def load_locked_predictions(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Read complete predictions only after all files named by their lock match."""
    lock_path = run_dir / "prediction-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    bindings = lock.get("artifacts", lock.get("files"))
    if bindings is None and isinstance(lock.get("artifact_hashes"), dict):
        bindings = {
            filename: {"filename": filename, "sha256": digest}
            for filename, digest in lock["artifact_hashes"].items()
        }
    if not isinstance(bindings, dict):
        raise ValueError("prediction lock must bind artifacts/files by name")
    hashes = {"prediction_lock_sha256": _sha256(lock_path)}
    for filename, expected in lock.get("artifact_hashes", {}).items():
        path = (run_dir / filename).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError("locked artifact points outside the run directory")
        if _sha256(path) != expected:
            raise ValueError(f"locked artifact SHA-256 mismatch: {filename}")
    prediction_path: Path | None = None
    for name, entry in bindings.items():
        if not isinstance(entry, dict):
            raise ValueError(f"invalid lock binding: {name}")
        filename = entry.get("filename", entry.get("path"))
        expected = entry.get("file_sha256", entry.get("sha256"))
        if not isinstance(filename, str) or not isinstance(expected, str):
            raise ValueError(f"missing file/hash in lock binding: {name}")
        path = (run_dir / filename).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError("prediction lock points outside the run directory")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"prediction lock SHA-256 mismatch: {name}")
        hashes[f"{name}_sha256"] = actual
        if name == "predictions" or path.name == "predictions.json":
            prediction_path = path
    if prediction_path is None:
        raise ValueError("prediction lock does not bind predictions.json")
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("locked predictions must contain nonempty rows")
    _check_prediction_design(rows)
    return rows, hashes


def _check_prediction_design(rows: list[dict[str, Any]]) -> None:
    by_arm: dict[str, dict[str, tuple[Any, ...]]] = defaultdict(dict)
    for row in rows:
        arm = row["arm_id"]
        if arm not in ARM_IDS:
            raise ValueError(f"unknown compact arm: {arm}")
        session = _session(row)
        if not session or session in by_arm[arm]:
            raise ValueError("missing or duplicated session within prediction arm")
        target, prediction = float(row["target"]), float(row["prediction"])
        if not np.isfinite([target, prediction]).all():
            raise ValueError("non-finite locked target or prediction")
        if not 0 <= target <= 1:
            raise ValueError("target must use the frozen normalised scale")
        by_arm[arm][session] = (
            _participant(row["participant_id"]),
            str(row["source_session_id"]),
            _phase(row["phase"]),
            int(row["outer_fold"]),
            target,
        )
        expected_control = _matched_control(arm)
        if (expected_control is not None and expected_control != row.get("control_arm_id")) or (
            expected_control is None and row.get("control_arm_id") not in {None, arm}
        ):
            raise ValueError(f"incorrect matched control for {arm}")
    if set(by_arm) != set(ARM_IDS):
        raise ValueError("compact audit requires all fifteen locked arms")
    reference = by_arm["natural_task"]
    if len(reference) != 138:
        raise ValueError("compact audit requires 138 paired sessions")
    people = defaultdict(set)
    person_folds = defaultdict(set)
    for participant, _, phase, fold, _ in reference.values():
        people[participant].add(phase)
        person_folds[participant].add(fold)
    if len(people) != 69 or any(phases != {"pre", "post"} for phases in people.values()):
        raise ValueError("compact audit requires 69 complete participant pairs")
    if any(len(folds) != 1 for folds in person_folds.values()):
        raise ValueError("a participant crosses outer test folds")
    if any(value != reference for value in by_arm.values()):
        raise ValueError("arms disagree on session, source, fold or unchanged target")


def _load_annotations(
    release_dir: Path, reference_rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    manifest_path = release_dir / "source_annotation_release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["artifacts"]["session_release"]
    annotation_path = release_dir / entry["filename"]
    actual = _sha256(annotation_path)
    if actual != entry["file_sha256"]:
        raise ValueError("source annotation CSV does not match its accepted manifest")
    predictions = {str(row["source_session_id"]): row for row in reference_rows}
    result: dict[str, dict[str, str]] = {}
    with annotation_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            source = row["source_session_id"]
            if source in result or source not in predictions:
                raise ValueError("duplicated or unmatched source annotation row")
            pred = predictions[source]
            if (
                _participant(row["participant_id"]) != _participant(pred["participant_id"])
                or _phase(row["phase"]) != _phase(pred["phase"])
                or row["canonical_session_id"] != _session(pred)
            ):
                raise ValueError("source annotation identity/phase mapping disagrees")
            tone = row["perceived_skin_tone"].strip()
            if tone.lower() not in UNKNOWN_VALUES and tone not in {"1", "2", "3", "4", "5"}:
                raise ValueError("skin tone must retain the accepted ordinal 1–5 categories")
            result[source] = row
    if len(result) != entry["row_count"]:
        raise ValueError("source annotation row count disagrees with manifest")
    counts = Counter(_phase(row["phase"]) for row in result.values())
    if counts != {"post": 69, "pre": 57}:
        raise ValueError("accepted source annotation phase coverage must be 69 post/57 pre")
    return result, {
        "annotation_manifest_sha256": _sha256(manifest_path),
        "annotation_csv_sha256": actual,
    }


def utility_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Return utility and score-spread points without defining constant correlations."""
    residual = prediction - target
    finite_correlation = len(target) > 1 and np.ptp(target) > 0 and np.ptp(prediction) > 0
    srcc = np.corrcoef(rankdata(target), rankdata(prediction))[0, 1] if finite_correlation else None
    pearson = np.corrcoef(target, prediction)[0, 1] if finite_correlation else None
    return {
        "n_sessions": len(target),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "srcc": None if srcc is None else float(srcc),
        "pearson": None if pearson is None else float(pearson),
        "bias": float(residual.mean()),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
    }


def participant_bootstrap_weights(
    participants: list[str], replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED
) -> np.ndarray:
    """Return bootstrap counts per row, carrying every session of a sampled person."""
    unique, inverse = np.unique(np.asarray(participants), return_inverse=True)
    count = len(unique)
    draws = np.random.default_rng(seed).multinomial(
        count, np.full(count, 1.0 / count), size=replicates
    )
    return draws[:, inverse]


def _weighted_ranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(values, return_inverse=True)
    masses = np.zeros((len(weights), int(inverse.max()) + 1))
    for index, group in enumerate(inverse):
        masses[:, group] += weights[:, index]
    # Half-unit offset cancels in the correlation, including unequal tie masses.
    ranks = np.cumsum(masses, axis=1) - masses / 2
    return ranks[:, inverse]


def _weighted_correlation(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = weights.sum(axis=1)
    left_mean = np.sum(left * weights, axis=1) / total
    right_mean = np.sum(right * weights, axis=1) / total
    left_centered, right_centered = left - left_mean[:, None], right - right_mean[:, None]
    numerator = np.sum(weights * left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(weights * left_centered**2, axis=1) * np.sum(weights * right_centered**2, axis=1)
    )
    result = np.full(len(weights), np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def bootstrap_utilities(
    target: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray]:
    """Vectorise participant-weighted MAE and exact repeated-sample rank correlation."""
    return {
        "mae": np.sum(weights * np.abs(prediction - target), axis=1) / weights.sum(axis=1),
        "srcc": _weighted_correlation(
            _weighted_ranks(target, weights), _weighted_ranks(prediction, weights), weights
        ),
    }


def empirical_w1_bootstrap(
    prediction: np.ndarray,
    categories: np.ndarray,
    category_pair: tuple[str, str],
    weights: np.ndarray,
) -> np.ndarray:
    """Exact unequal-size empirical W1 for bootstrap counts, sorting scores only once."""
    left, right = category_pair
    keep = (categories == left) | (categories == right)
    scores, groups = prediction[keep], categories[keep]
    order = np.argsort(scores, kind="stable")
    scores, groups, counts = scores[order], groups[order], weights[:, keep][:, order]
    left_counts = counts * (groups == left)
    right_counts = counts * (groups == right)
    left_total, right_total = left_counts.sum(axis=1), right_counts.sum(axis=1)
    valid = (left_total > 0) & (right_total > 0)
    result = np.full(len(weights), np.nan)
    if scores.size < 2 or not valid.any():
        return result
    left_cdf = np.cumsum(left_counts[valid], axis=1) / left_total[valid, None]
    right_cdf = np.cumsum(right_counts[valid], axis=1) / right_total[valid, None]
    result[valid] = np.sum(np.abs(left_cdf[:, :-1] - right_cdf[:, :-1]) * np.diff(scores), axis=1)
    return result


def _disparity_draws(
    prediction: np.ndarray, labels: np.ndarray, supported: list[str], weights: np.ndarray
) -> np.ndarray:
    if len(supported) < 2:
        return np.full(len(weights), np.nan)
    pairwise = np.array(
        [
            empirical_w1_bootstrap(prediction, labels, pair, weights)
            for pair in itertools.combinations(supported, 2)
        ]
    )
    # A missing fixed category makes that draw undefined; never shrink the group universe.
    return np.max(pairwise, axis=0)


def _interval(draws: np.ndarray, point: float | None) -> dict[str, Any]:
    valid = draws[np.isfinite(draws)]
    interval = np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else [None, None]
    return {
        "estimate": point,
        "percentile_95": interval,
        "valid_draws": len(valid),
        "undefined_draws": len(draws) - len(valid),
        "interval_basis": "all_draws" if len(valid) == len(draws) else "valid_draws_only",
    }


def _point_difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _scope_rows(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    return [row for row in rows if scope == "pooled" or _phase(row["phase"]) == scope]


def _arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([float(row["target"]) for row in rows]),
        np.array([float(row["prediction"]) for row in rows]),
    )


def _utility_summaries(by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm, rows in by_arm.items():
        result[arm] = {}
        for scope in ("pooled", "pre", "post"):
            subset = _scope_rows(rows, scope)
            result[arm][scope] = utility_metrics(*_arrays(subset))
            result[arm][scope]["n_participants"] = len(
                {_participant(row["participant_id"]) for row in subset}
            )
    return result


def _utility_contrasts(
    by_arm: dict[str, list[dict[str, Any]]], utilities: dict[str, Any], replicates: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    cached: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for arm in by_arm:
        control = _matched_control(arm)
        if control is None:
            continue
        result[arm] = {"control_arm_id": control, "primary_contrast": arm in PRIMARY_ARMS}
        for scope in ("pooled", "pre", "post"):
            effects = {
                metric: _point_difference(
                    utilities[arm][scope][metric], utilities[control][scope][metric]
                )
                for metric in ("mae", "srcc")
            }
            if arm in PRIMARY_ARMS:
                for name in (arm, control):
                    if (name, scope) not in cached:
                        subset = _scope_rows(by_arm[name], scope)
                        weights = participant_bootstrap_weights(
                            [_participant(row["participant_id"]) for row in subset], replicates
                        )
                        cached[name, scope] = bootstrap_utilities(*_arrays(subset), weights)
                effects = {
                    metric: _interval(
                        cached[arm, scope][metric] - cached[control, scope][metric], point
                    )
                    for metric, point in effects.items()
                }
            result[arm][scope] = effects
    return result


def _proxy_summaries(by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm, rows in by_arm.items():
        if rows[0].get("proxy_source") in {None, "none", ""}:
            continue
        control = _matched_control(arm)
        controls = {} if control is None else {_session(row): row for row in by_arm[control]}
        summaries = []
        for fold in sorted({int(row["outer_fold"]) for row in rows}):
            fold_rows = [row for row in rows if int(row["outer_fold"]) == fold]
            for scope in ("pooled", "pre", "post"):
                subset = _scope_rows(fold_rows, scope)
                labels = np.array([str(row["proxy_group"]) for row in subset])
                groups = sorted(set(labels))
                scores = _arrays(subset)[1]
                point = None
                control_point = None
                if len(groups) == 2:
                    weights = np.ones((1, len(scores)), dtype=int)
                    pair = (groups[0], groups[1])
                    point = float(empirical_w1_bootstrap(scores, labels, pair, weights)[0])
                    if control is not None:
                        control_scores = np.array(
                            [controls[_session(row)]["prediction"] for row in subset]
                        )
                        control_point = float(
                            empirical_w1_bootstrap(control_scores, labels, pair, weights)[0]
                        )
                summaries.append(
                    {
                        "outer_fold": fold,
                        "scope": scope,
                        "group_session_counts": dict(Counter(labels.tolist())),
                        "proxy_w1": point,
                        "matched_control_proxy_w1": control_point,
                        "proxy_w1_reduction": _point_difference(control_point, point),
                    }
                )
        result[arm] = {"proxy_source": rows[0]["proxy_source"], "fold_local": summaries}
    return result


def _axis_summary(
    rows: list[dict[str, Any]], annotations: dict[str, dict[str, str]], axis: str
) -> tuple[dict[str, Any], np.ndarray, list[str]]:
    labels = np.array([annotations[str(row["source_session_id"])][axis].strip() for row in rows])
    counts = Counter(labels.tolist())
    supported = sorted(
        category
        for category, count in counts.items()
        if count >= MINIMUM_GROUP_SIZE and category.lower() not in UNKNOWN_VALUES
    )
    target, prediction = _arrays(rows)
    disparity = _disparity_draws(prediction, labels, supported, np.ones((1, len(rows)), dtype=int))[
        0
    ]
    groups = []
    for category in sorted(counts):
        selected = labels == category
        group: dict[str, Any] = {
            "category": category or "unknown",
            "n_participants": counts[category],
            "reportable": category in supported,
        }
        if category in supported:
            residual = prediction[selected] - target[selected]
            group.update(
                {
                    "mae": float(np.abs(residual).mean()),
                    "residual_bias": float(residual.mean()),
                    "target_mean": float(target[selected].mean()),
                    "prediction_mean": float(prediction[selected].mean()),
                }
            )
        groups.append(group)
    return (
        {
            "n_annotated_participants": len(rows),
            "unknown_count": sum(
                count for category, count in counts.items() if category.lower() in UNKNOWN_VALUES
            ),
            "all_category_counts": {
                category or "unknown": count for category, count in sorted(counts.items())
            },
            "supported_categories": supported,
            "groups": groups,
            "max_pairwise_w1": float(disparity) if np.isfinite(disparity) else None,
            "ordinal_scale": [1, 2, 3, 4, 5] if axis == "perceived_skin_tone" else None,
        },
        labels,
        supported,
    )


def evaluate_compact_audit(
    rows: list[dict[str, Any]],
    annotations: dict[str, dict[str, str]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Build aggregate utility, fold-local mechanism and phase-aligned paired audits."""
    by_arm = {
        arm: sorted([row for row in rows if row["arm_id"] == arm], key=_session) for arm in ARM_IDS
    }
    utilities = _utility_summaries(by_arm)
    audit: dict[str, Any] = {}
    for phase in ("post", "pre"):
        phase_rows = {
            arm: [
                row
                for row in values
                if _phase(row["phase"]) == phase and str(row["source_session_id"]) in annotations
            ]
            for arm, values in by_arm.items()
        }
        reference = phase_rows["natural_task"]
        weights = participant_bootstrap_weights(
            [_participant(row["participant_id"]) for row in reference], replicates
        )
        axis_results = {}
        for axis in AXES:
            points, labels, supported = {}, None, None
            for arm, subset in phase_rows.items():
                points[arm], labels, supported = _axis_summary(subset, annotations, axis)
            assert labels is not None and supported is not None
            cached: dict[str, np.ndarray] = {}
            contrasts = {}
            for arm in ARM_IDS:
                control = _matched_control(arm)
                if control is None:
                    continue
                point = _point_difference(
                    points[control]["max_pairwise_w1"], points[arm]["max_pairwise_w1"]
                )
                effect: Any = point
                if arm in PRIMARY_ARMS:
                    for name in (arm, control):
                        if name not in cached:
                            cached[name] = _disparity_draws(
                                _arrays(phase_rows[name])[1], labels, supported, weights
                            )
                    effect = _interval(cached[control] - cached[arm], point)
                contrasts[arm] = {
                    "control_arm_id": control,
                    "primary_contrast": arm in PRIMARY_ARMS and phase == "post",
                    "disparity_gain": effect,
                }
            axis_results[axis] = {"arms": points, "contrasts": contrasts}
        audit[phase] = {
            "role": "primary" if phase == "post" else "available_pre_session_sensitivity",
            "n_participants": len(reference),
            "axes": axis_results,
        }
    return {
        "schema_version": "mit-study2-compact-aggregate-1",
        "study": "post_audit_exploratory_mit_compact_2026_10_21",
        "status": "complete",
        "method": {
            "bootstrap_replicates": replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "participant_with_all_selected_sessions",
            "interval": "two_sided_percentile_95_conditional_on_fixed_fits",
            "minimum_reportable_phase_category_participants": MINIMUM_GROUP_SIZE,
            "disparity": "maximum_empirical_w1_over_fixed_supported_category_pairs",
            "disparity_gain_direction": "control_minus_arm_positive_is_reduction",
            "utility_change_direction": "arm_minus_control_negative_mae_positive_srcc_is_better",
            "skin_tone": "ordinal_1_to_5_categories_preserved_without_binarisation",
            "attributes": "single_observer_perceived_session_attributes_not_identity",
            "primary_contrasts": list(PRIMARY_ARMS),
            "grid": "all_strengths_reported_no_audit_selected_winner",
            "inference": "exploratory_no_multiplicity_adjustment_no_significance_claims",
            "proxy_identity": "fold_local_only_no_cross_fold_cluster_identity",
            "normalised_score_to_original_rating_difference": "multiply_by_6",
        },
        "coverage": {"n_arms": len(by_arm), "n_predictions": len(rows), "n_participants": 69},
        "utility": utilities,
        "utility_contrasts": _utility_contrasts(by_arm, utilities, replicates),
        "held_out_proxy_mechanism": _proxy_summaries(by_arm),
        "perceived_group_audit": audit,
    }


def _number(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{0.0 if round(value, 3) == 0 else value:.3f}"


def _effect_text(effect: dict[str, Any]) -> str:
    lower, upper = effect["percentile_95"]
    return f"{_number(effect['estimate'])} [{_number(lower)}, {_number(upper)}]"


def render_compact_report(result: dict[str, Any]) -> str:
    """Render readable aggregate findings without participant-level rows or long decimals."""
    lines = [
        "# Compact exploratory MIT follow-up",
        "",
        "The 75 deterministic fits compare a fixed alpha-1 linear scorer with proxy balancing "
        "and three full-training-fold W1 strengths. All strengths are reported; the perceived "
        "attributes did not select an arm. This follow-up is exploratory because the original "
        "MIT audit had already been inspected.",
        "",
        "Scores use the normalised rating scale. Multiply differences by six for the original "
        "1–7 rating scale. Perceived attributes come from one observer. Skin tone retains its "
        "five ordinal categories. Categories with fewer than ten phase-matched people remain "
        "in the counts and do not enter disparity estimates.",
        "",
        "## Prediction utility",
        "",
        "| Arm | Pooled MAE | Pooled SRCC | Post MAE | Post SRCC | Pre MAE | Pre SRCC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, scopes in result["utility"].items():
        values = [
            scopes[scope][metric]
            for scope in ("pooled", "post", "pre")
            for metric in ("mae", "srcc")
        ]
        lines.append(f"| {arm} | " + " | ".join(_number(value) for value in values) + " |")
    if "fit_status_counts" in result:
        counts = result["fit_status_counts"]
        lines += [
            "",
            "Solver status: "
            + "; ".join(f"{name}: {count}" for name, count in counts.items())
            + ".",
        ]
        nonconverged = [
            item
            for item in result.get("training_and_solver_diagnostics", [])
            if not item.get("optimizer_converged", False)
        ]
        if nonconverged:
            strengths = sorted({item["arm_id"].split("lambda-")[-1] for item in nonconverged})
            lines += [
                "",
                "Finite fits without reported solver convergence occur at strength(s) "
                + ", ".join(strengths)
                + ". Their predictions remain in the complete grid; they do not establish "
                "a converged or globally optimal strong-penalty solution.",
            ]
    if "training_and_solver_diagnostics" in result:
        lines += [
            "",
            "## Proxy mechanism at strength 0.02",
            "",
            "W1 values below are means of the five fold-specific distances. Training and "
            "held-out proxy groups use each fold's own fitted proxy; cluster numbers are "
            "never stitched across folds.",
            "",
            "| Arm | Training W1 | Matched training W1 | Held-out W1 | Matched held-out W1 |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm in PRIMARY_ARMS:
            source = arm.split("_")[0]
            fits = [
                item for item in result["training_and_solver_diagnostics"] if item["arm_id"] == arm
            ]
            held_out = [
                item
                for item in result["held_out_proxy_mechanism"][arm]["fold_local"]
                if item["scope"] == "pooled"
            ]
            train = float(np.mean([item["training_proxy_w1"][source] for item in fits]))
            control_train = float(
                np.mean([item["matched_control_training_proxy_w1"] for item in fits])
            )
            test = float(np.mean([item["proxy_w1"] for item in held_out]))
            control_test = float(np.mean([item["matched_control_proxy_w1"] for item in held_out]))
            lines.append(
                f"| {arm} | {_number(train)} | {_number(control_train)} "
                f"| {_number(test)} | {_number(control_test)} |"
            )
    lines += [
        "",
        "## Primary paired comparisons at strength 0.02",
        "",
        "Intervals use 10,000 participant bootstrap draws and are conditional on the fitted "
        "models. A positive disparity gain means lower disparity than the exact matched "
        "control. Negative MAE change and positive SRCC change mean better prediction. "
        "The intervals are exploratory; no significance or causal claim is made.",
        "",
        "| Arm | Pooled MAE change [95% interval] | Pooled SRCC change [95% interval] |",
        "|---|---:|---:|",
    ]
    for arm in PRIMARY_ARMS:
        comparison = result["utility_contrasts"][arm]["pooled"]
        lines.append(
            f"| {arm} | {_effect_text(comparison['mae'])} | {_effect_text(comparison['srcc'])} |"
        )
    for phase, summary in result["perceived_group_audit"].items():
        lines += [
            "",
            f"## {phase.capitalize()} perceived-group audit",
            "",
            f"Phase-matched annotation coverage: {summary['n_participants']} people.",
            "",
        ]
        for axis, axis_result in summary["axes"].items():
            counts = axis_result["arms"]["natural_task"]
            lines += [
                f"### {axis.replace('_', ' ')}",
                "",
                "All category counts: "
                + "; ".join(
                    f"{category}: {count}"
                    for category, count in counts["all_category_counts"].items()
                )
                + ".",
                "",
                "| Arm | Max pairwise W1 | Gain vs matched control [95% interval] |",
                "|---|---:|---:|",
            ]
            for arm in PRIMARY_ARMS:
                point = axis_result["arms"][arm]["max_pairwise_w1"]
                effect = axis_result["contrasts"][arm]["disparity_gain"]
                lines.append(f"| {arm} | {_number(point)} | {_effect_text(effect)} |")
                if effect["undefined_draws"]:
                    lines.append(
                        f"\n{arm}: {effect['undefined_draws']} draws were undefined; "
                        "the interval uses valid draws only.\n"
                    )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The aggregate JSON contains every strength, fold-local proxy mechanism, group "
        "error, sparse-category count and pre-session sensitivity. Maximum pairwise W1 "
        "describes prediction distributions, while group residuals describe prediction "
        "errors. Neither establishes hiring fairness, demographic identity or compliance. "
        "The pre-session sensitivity uses the 57 available annotations without filling "
        "the twelve missing rows. Participant bootstrap uncertainty does not estimate "
        "variability from retraining or an independent population.",
        "",
    ]
    return "\n".join(lines)


def audit_compact_run(run_dir: Path, annotation_release_dir: Path) -> dict[str, Any]:
    """Audit the accepted source only after the separate prediction publication lock."""
    rows, prediction_hashes = load_locked_predictions(run_dir)
    reference = [row for row in rows if row["arm_id"] == "natural_task"]
    annotations, annotation_hashes = _load_annotations(annotation_release_dir, reference)
    result = evaluate_compact_audit(rows, annotations)
    result["input_hashes"] = prediction_hashes | annotation_hashes
    lock = json.loads((run_dir / "prediction-lock.json").read_text(encoding="utf-8"))
    result["fit_status_counts"] = lock.get("fit_status_counts", {})
    result["all_optimizers_converged"] = lock.get("all_optimizers_converged")
    diagnostics = []
    receipts = {}
    for arm in ARM_IDS:
        for fold in range(5):
            path = run_dir / "fits" / f"{arm}.fold-{fold}.json"
            receipt = json.loads(path.read_text(encoding="utf-8"))
            receipts[arm, fold] = receipt
            keys = (
                "status",
                "optimizer_converged",
                "optimizer_status",
                "optimizer_message",
                "retained_initial",
                "iterations",
                "objective",
                "initial_objective",
                "initial_diagnostics",
                "final_diagnostics",
                "training_proxy_w1",
            )
            diagnostics.append(
                {
                    "arm_id": arm,
                    "outer_fold": fold,
                    **{key: receipt[key] for key in keys if key in receipt},
                }
            )
    for diagnostic in diagnostics:
        arm, fold = diagnostic["arm_id"], diagnostic["outer_fold"]
        control = _matched_control(arm)
        if control is not None:
            source = arm.split("_")[0]
            control_w1 = receipts[control, fold]["training_proxy_w1"][source]
            diagnostic["matched_control_training_proxy_w1"] = control_w1
            diagnostic["training_proxy_w1_reduction"] = (
                control_w1 - receipts[arm, fold]["training_proxy_w1"][source]
            )
    result["training_and_solver_diagnostics"] = diagnostics
    return result


def write_compact_audit(result: dict[str, Any], output: Path, report: Path) -> None:
    """Create distinct aggregate-only outputs, refusing to replace an earlier release."""
    if output == report or output.exists() or report.exists():
        raise FileExistsError("compact audit outputs must both be new, distinct paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with report.open("x", encoding="utf-8") as handle:
        handle.write(render_compact_report(result))
