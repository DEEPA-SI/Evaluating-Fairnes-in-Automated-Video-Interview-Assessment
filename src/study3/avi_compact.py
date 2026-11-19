"""Compact AVI intervention experiment using the accepted cached participant vectors.

The fit path has no audit-file argument. Reporting opens the separate attribute map
only after checking the complete immutable prediction lock. Mathematical fitting
and exact empirical W1 are reused from the working compact MIT implementation.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.study2.compact import (
    SOLVER_OPTIONS,
    RowSpaceDesign,
    canonical_bytes,
    comparison_arms,
    empirical_w1_gradient,
    file_sha256,
    fit_linear_arm,
    task_weights,
    utility_metrics,
    write_json,
)
from src.study2.compact_audit import (
    bootstrap_utilities,
    empirical_w1_bootstrap,
    participant_bootstrap_weights,
)

STUDY_ID = "avi-compact-reconciled-v1"
SPLITS = {"Training": 452, "Validation": 64, "Testing": 130}
ELIGIBLE = {"Training": 450, "Validation": 64, "Testing": 130}
ALIASES = {
    "train": "Training",
    "validation": "Validation",
    "test": "Testing",
    **{key: key for key in SPLITS},
}
PRIMARY_ARM = "visual_natural_w1_lambda-0.02"
DIMENSIONS = {"video": 1024, "audio": 1536}


def intake_binding(intake_release: Path, filename: str) -> tuple[Path, str]:
    """Resolve one separated intake artifact without opening its outcome/audit data."""
    if filename not in {"participant-outcomes.json", "audit-attributes.json"}:
        raise ValueError("Only the two accepted separated input artifacts are permitted")
    record_path = Path(__file__).resolve().parents[2] / "configs/avi_input_release.json"
    record = json.loads(record_path.read_bytes())
    binding = record["accepted_receipt_relative_to_project"]
    if not binding.startswith("validation/"):
        raise ValueError("Accepted intake receipt must be inside the validation directory")
    suffix = binding.removeprefix("validation/")
    relative = Path(suffix)
    if not suffix or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Accepted intake receipt path is invalid")
    return (
        intake_release / "validation" / relative / filename,
        record["output_file_sha256"][filename],
    )


@dataclass(frozen=True)
class AviInputs:
    """Only participant identities, official roles, primary targets and fixed features."""

    x: np.ndarray
    y: np.ndarray
    participant_ids: tuple[str, ...]
    splits: tuple[str, ...]
    provenance: dict[str, Any]


@dataclass
class AlphaDesign(RowSpaceDesign):
    """Represent alpha-ridge using the unchanged compact alpha-one fitting objective."""

    alpha: float = 1.0

    @classmethod
    def from_base(cls, base: RowSpaceDesign, alpha: float) -> AlphaDesign:
        """Scale non-intercept coordinates by sqrt(alpha), retaining training scope."""
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha must be finite and positive")
        train = base.train.copy()
        train[:, :-1] /= np.sqrt(alpha)
        return cls(base.scaler, base.basis, train, base.n, float(alpha))

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the same training-only coordinates and alpha scaling to held-out rows."""
        transformed = super().transform(x)
        transformed[:, :-1] /= np.sqrt(self.alpha)
        return transformed


def load_protocol(path: Path) -> dict[str, Any]:
    """Read the fixed comparison; reject changes to declared selection/audit settings."""
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "avi-compact-reconciled-protocol-1",
        "study_id": STUDY_ID,
        "status": "frozen_for_execution",
        "target": "mean_rating_hirea",
        "target_transform": "(raw-1)/4",
        "ridge_alpha_grid": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "strengths": [0.002, 0.02, 0.2],
        "official_participant_splits": SPLITS,
        "supervised_participant_splits": ELIGIBLE,
        "selection": "natural_task_validation_mae_then_smaller_alpha",
        "validation_refit": False,
        "expected_arm_fits": 15,
        "primary_arm": PRIMARY_ARM,
        "primary_control": "natural_task",
        "audit_axis": "self_reported_gender_code",
        "primary_categories": [1, 2],
        "minimum_group_size": 10,
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 2026,
        "solver": {"method": "L-BFGS-B", **SOLVER_OPTIONS},
    }
    if not isinstance(protocol, dict) or any(
        protocol.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Compact AVI protocol differs from its fixed comparison")
    return protocol


def load_inputs(
    release_root: Path,
    outcome_path: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_outcomes_sha256: str | None = None,
    require_official_counts: bool = True,
) -> AviInputs:
    """Load all feature bytes once, without accepting or opening an audit table."""
    manifest_path = release_root / "participant_feature_manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Feature manifest must be a regular file")
    manifest_raw = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
    if expected_manifest_sha256 is not None and manifest_hash != expected_manifest_sha256:
        raise ValueError("Accepted AVI feature manifest hash mismatch")
    manifest = json.loads(manifest_raw)
    required = {
        "schema_version": "avi-participant-video-audio-aggregation-1",
        "status": "complete_video_audio_only",
        "clips_per_participant": 6,
        "outcome_or_audit_fields_accessed": [],
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("Participant feature release schema/coverage differs")
    shards = manifest.get("participant_shards", [])
    if not shards or len({row["participant_id"] for row in shards}) != len(shards):
        raise ValueError("Feature roster is missing or duplicated")
    ledger = [{key: row[key] for key in ("participant_id", "split", "sha256")} for row in shards]
    if hashlib.sha256(canonical_bytes(ledger)).hexdigest() != manifest.get(
        "participant_shard_ledger_sha256"
    ):
        raise ValueError("Participant feature ledger hash mismatch")
    outcome_raw = outcome_path.read_bytes()
    if (
        expected_outcomes_sha256 is not None
        and hashlib.sha256(outcome_raw).hexdigest() != expected_outcomes_sha256
    ):
        raise ValueError("Accepted AVI outcomes hash mismatch")
    outcomes = json.loads(outcome_raw)
    if set(outcomes) != {row["participant_id"] for row in shards}:
        raise ValueError("Feature and outcome participant rosters differ")
    matrices, targets, participant_ids, splits = [], [], [], []
    for row in sorted(shards, key=lambda item: item["participant_id"]):
        participant_id, split = row["participant_id"], row["split"]
        if not re.fullmatch(r"[0-9a-f]{24}", participant_id) or split not in SPLITS:
            raise ValueError("Invalid participant identity or official role")
        source = outcomes[participant_id]
        if ALIASES.get(source.get("split")) != split:
            raise ValueError("Feature and outcome participant roles differ")
        target = source.get("outcomes", {}).get("mean_rating_hirea")
        if target is not None and (
            isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not np.isfinite(target)
            or not 1 <= target <= 5
        ):
            raise ValueError("Primary source target must be finite 1–5 or missing")
        shard_path = release_root / "shards" / f"{participant_id}.npz"
        if shard_path.is_symlink():
            raise ValueError("Participant shard must be a regular file")
        raw = shard_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Participant feature shard hash mismatch")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != {
                "video",
                "audio",
                "video_mask",
                "audio_mask",
                "text_available",
            }:
                raise ValueError("Participant shard fields differ")
            pieces = []
            for modality, width in DIMENSIONS.items():
                value, mask = archive[modality], archive[f"{modality}_mask"]
                if (
                    value.shape != (width,)
                    or value.dtype != np.float32
                    or not np.isfinite(value).all()
                    or mask.shape != (6,)
                    or mask.dtype != np.bool_
                    or not mask.all()
                ):
                    raise ValueError("Participant vector or six-clip coverage is invalid")
                pieces.append(value.astype(np.float64))
            text = archive["text_available"]
            if text.shape != () or text.dtype != np.bool_ or bool(text):
                raise ValueError("AVI text must be explicitly unavailable")
        matrices.append(np.concatenate(pieces))
        targets.append(np.nan if target is None else (float(target) - 1) / 4)
        participant_ids.append(participant_id)
        splits.append(split)
    counts = dict(Counter(splits))
    eligible = {
        role: int(
            sum(
                split == role and np.isfinite(target)
                for split, target in zip(splits, targets, strict=True)
            )
        )
        for role in SPLITS
    }
    if manifest.get("split_counts") != counts:
        raise ValueError("Feature split ledger differs from its roster")
    if require_official_counts and (counts != SPLITS or eligible != ELIGIBLE):
        raise ValueError("Official or supervised participant coverage differs")
    if any(
        split != "Training" and not np.isfinite(target)
        for split, target in zip(splits, targets, strict=True)
    ):
        raise ValueError("Held-out participant primary target is unavailable")
    return AviInputs(
        np.stack(matrices),
        np.asarray(targets),
        tuple(participant_ids),
        tuple(splits),
        {
            "feature_manifest_sha256": manifest_hash,
            "outcomes_sha256": hashlib.sha256(outcome_raw).hexdigest(),
            "feature_participants_by_split": counts,
            "supervised_participants_by_split": eligible,
            "target_transform": "(raw-1)/4",
            "audit_fields_accessed": [],
        },
    )


def role_indices(data: AviInputs) -> dict[str, np.ndarray]:
    """Return disjoint official-role row indices, excluding missing training targets."""
    return {
        role: np.asarray(
            [i for i, split in enumerate(data.splits) if split == role and np.isfinite(data.y[i])],
            dtype=int,
        )
        for role in SPLITS
    }


def fit_proxies(x: np.ndarray, train: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Fit both neutral proxies on eligible training participants only."""
    groups, receipts = {}, {}
    for source, subset in (("visual", slice(0, 1024)), ("audio", slice(1024, 2560))):
        matrix = x[:, subset]
        scaler = StandardScaler().fit(matrix[train])
        model = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=20,
            max_iter=300,
            tol=1e-4,
            algorithm="lloyd",
            random_state=0,
        )
        model.fit(scaler.transform(matrix[train]))
        order = sorted(
            range(2),
            key=lambda i: (
                hashlib.sha256(
                    np.ascontiguousarray(model.cluster_centers_[i], dtype="<f8").tobytes()
                ).digest(),
                tuple(model.cluster_centers_[i]),
            ),
        )
        permutation = np.empty(2, dtype=int)
        permutation[order] = np.arange(2)
        groups[source] = permutation[model.predict(scaler.transform(matrix))]
        if set(groups[source][train]) != {0, 1}:
            raise ValueError("Training proxy has an empty group")
        receipts[source] = {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "centres": model.cluster_centers_.tolist(),
            "raw_to_canonical": permutation.tolist(),
            "fit_scope": "eligible_training_participants_only",
        }
    return groups, receipts


def avi_utility(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Reuse compact utility definitions with AVI's four-point difference conversion."""
    result = utility_metrics(target, prediction)
    result["mae_raw_scale"] = result["mae"] * 4
    result["rmse_raw_scale"] = result["rmse"] * 4
    return result


def run_fit(data: AviInputs, output_dir: Path, protocol_path: Path) -> dict[str, Any]:
    """Execute all fixed arms and lock predictions with no audit-data capability."""
    protocol = load_protocol(protocol_path)
    roles = role_indices(data)
    if (
        data.x.shape != (646, 2560)
        or len(set(data.participant_ids)) != 646
        or dict(Counter(data.splits)) != SPLITS
        or {role: len(indices) for role, indices in roles.items()} != ELIGIBLE
    ):
        raise ValueError("Compact AVI fitting requires the complete official participant design")
    if output_dir.exists():
        raise FileExistsError("Preserve the existing run; use a new authorised namespace")
    output_dir.mkdir(parents=True)
    (output_dir / "fits").mkdir()
    train, validation, test = (roles[role] for role in SPLITS)
    with threadpool_limits(limits=4):
        base = RowSpaceDesign.fit(data.x[train])
        validation_scores, alpha_candidates = {}, []
        for alpha in protocol["ridge_alpha_grid"]:
            design = AlphaDesign.from_base(base, alpha)
            theta = design.exact_task_solution(data.y[train], task_weights(None, len(train)))
            prediction = design.transform(data.x[validation]) @ theta
            score = float(np.abs(prediction - data.y[validation]).mean())
            alpha_candidates.append({"alpha": float(alpha), "validation_mae": score})
            validation_scores[float(alpha)] = score
        alpha = min(validation_scores, key=lambda value: (validation_scores[value], value))
        design = AlphaDesign.from_base(base, alpha)
        groups, proxy_receipts = fit_proxies(data.x, train)
        manifest = {
            "schema_version": "avi-compact-manifest-1",
            "study_id": STUDY_ID,
            "interpretation": protocol["interpretation"],
            "protocol": protocol,
            "protocol_sha256": file_sha256(protocol_path),
            "implementation_sha256": file_sha256(Path(__file__)),
            "compact_implementation_sha256": file_sha256(
                Path(__file__).parents[1] / "study2/compact.py"
            ),
            "inputs": data.provenance,
            "selected_alpha": alpha,
            "alpha_selection": alpha_candidates,
            "alpha_selection_role": "Validation",
            "selection_scope": "natural_task_validation_mae_only",
            "training_mean_target": float(data.y[train].mean()),
            "training_participants": [data.participant_ids[i] for i in train],
            "validation_participants": [data.participant_ids[i] for i in validation],
            "test_participants": [data.participant_ids[i] for i in test],
            "arms": list(comparison_arms()),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit-learn": sklearn.__version__,
                "numerical_threads": 4,
            },
            "audit_fields_accessed": [],
        }
        try:
            manifest["code_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            manifest["code_commit"] = None
        write_json(output_dir / "manifest.json", manifest)
        write_json(output_dir / "proxies.json", proxy_receipts)
        prediction_rows, receipts, mechanism = [], [], []
        test_design, validation_design = (
            design.transform(data.x[indices]) for indices in (test, validation)
        )
        for arm in comparison_arms():
            source = arm["proxy_source"]
            arm_groups = None if source == "none" else groups[source][train]
            weights = task_weights(
                arm_groups if arm["exposure"] == "balanced" else None, len(train)
            )
            theta, receipt = fit_linear_arm(
                design, data.y[train], weights, arm_groups, arm["lambda"]
            )
            test_prediction = test_design @ theta
            if not np.isfinite(test_prediction).all():
                raise ValueError("Nonfinite test prediction; complete lock unavailable")
            receipt.update(arm)
            receipt["selected_alpha"] = alpha
            receipt["validation_utility"] = avi_utility(
                data.y[validation], validation_design @ theta
            )
            write_json(output_dir / "fits" / f"{arm['arm_id']}.json", receipt)
            with (output_dir / "fits" / f"{arm['arm_id']}.npz").open("xb") as handle:
                np.savez(
                    handle,
                    standardized_coefficients=base.basis @ theta[:-1] / np.sqrt(alpha),
                    intercept=theta[-1],
                    scaler_mean=base.scaler.mean_,
                    scaler_scale=base.scaler.scale_,
                )
            receipts.append(receipt)
            for index, value in zip(test, test_prediction, strict=True):
                prediction_rows.append(
                    {
                        **arm,
                        "participant_id": data.participant_ids[index],
                        "prediction": float(value),
                    }
                )
            for proxy_source in ("visual", "audio"):
                for role, indices, predictions in (
                    ("Training", train, design.train @ theta),
                    ("Testing", test, test_prediction),
                ):
                    labels = groups[proxy_source][indices]
                    counts = {
                        str(group): int(np.count_nonzero(labels == group)) for group in (0, 1)
                    }
                    w1 = (
                        empirical_w1_gradient(predictions, labels)[0]
                        if min(counts.values())
                        else None
                    )
                    mechanism.append(
                        {
                            "arm_id": arm["arm_id"],
                            "proxy_source": proxy_source,
                            "role": role,
                            "participant_counts": counts,
                            "proxy_w1": w1,
                        }
                    )
            print(f"{len(receipts)}/15 {arm['arm_id']}: {receipt['status']}", flush=True)
    expected_rows = len(test) * 15
    if len(receipts) != 15 or len(prediction_rows) != expected_rows:
        raise ValueError("Incomplete comparison")
    write_json(
        output_dir / "predictions.json",
        {
            "schema_version": "avi-compact-predictions-1",
            "study_id": STUDY_ID,
            "rows": prediction_rows,
        },
    )
    write_json(output_dir / "proxy-mechanism.json", {"rows": mechanism})
    counts = dict(Counter(receipt["status"] for receipt in receipts))
    lock = {
        "schema_version": "avi-compact-prediction-lock-1",
        "study_id": STUDY_ID,
        "state": "locked",
        "prediction_coverage_complete": True,
        "arm_count": 15,
        "prediction_count": expected_rows,
        "test_participant_count": len(test),
        "fit_status_counts": counts,
        "all_optimizers_converged": counts.get("converged") == 15,
        "audit_fields_accessed": [],
        "artifact_hashes": {
            str(path.relative_to(output_dir)): file_sha256(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        },
    }
    write_json(output_dir / "prediction-lock.json", lock)
    return lock


def _interval(values: np.ndarray, estimate: float | None) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    return {
        "estimate": estimate,
        "percentile_95": np.quantile(finite, [0.025, 0.975]).tolist()
        if len(finite)
        else [None, None],
        "valid_draws": len(finite),
        "undefined_draws": len(values) - len(finite),
    }


def read_locked_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict]:
    """Validate immutable artifact bindings before permitting post-lock reporting."""
    lock_path = run_dir / "prediction-lock.json"
    lock = json.loads(lock_path.read_bytes())
    if lock.get("state") != "locked" or lock.get("audit_fields_accessed") != []:
        raise ValueError("A pre-audit prediction lock is required")
    bindings = lock.get("artifact_hashes", {})
    if not {"predictions.json", "manifest.json", "proxy-mechanism.json"} <= set(bindings):
        raise ValueError("Prediction lock does not bind its required artifacts")
    arms = comparison_arms()
    receipt_files = {f"fits/{arm['arm_id']}.json" for arm in arms}
    bound_receipts = {
        filename
        for filename in bindings
        if Path(filename).parts[0] == "fits" and filename.endswith(".json")
    }
    if bound_receipts != receipt_files:
        raise ValueError("Prediction lock must bind the complete fifteen-arm fit receipts")
    report_files = {"manifest.json", "predictions.json", "proxy-mechanism.json"} | receipt_files
    verified_contents = {}
    for filename, expected in bindings.items():
        path = run_dir / filename
        if path.is_symlink() or not path.resolve().is_relative_to(run_dir.resolve()):
            raise ValueError("Unsafe prediction-lock artifact reference")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Prediction lock artifact hash mismatch")
        if filename in report_files:
            verified_contents[filename] = content
    manifest = json.loads(verified_contents["manifest.json"])
    rows = json.loads(verified_contents["predictions.json"])["rows"]
    lock["verified_proxy_mechanism"] = json.loads(verified_contents["proxy-mechanism.json"])["rows"]
    expected_ids = set(manifest["test_participants"])
    expected_arms = {arm["arm_id"] for arm in arms}
    if (
        len(expected_ids) != 130
        or lock.get("arm_count") != 15
        or lock.get("prediction_count") != 1950
        or lock.get("test_participant_count") != 130
        or {row["arm_id"] for row in rows} != expected_arms
        or len(rows) != 1950
    ):
        raise ValueError("Complete fifteen-arm prediction coverage is required")
    solver_fields = (
        "status",
        "optimizer_converged",
        "optimizer_status",
        "optimizer_message",
        "iterations",
        "function_evaluations",
        "retained_initial",
        "initial_objective",
        "objective",
        "global_optimum_claimed",
    )
    verified_receipts = {}
    for arm in arms:
        subset = [row for row in rows if row["arm_id"] == arm["arm_id"]]
        if (
            len(subset) != len(expected_ids)
            or {row["participant_id"] for row in subset} != expected_ids
            or any(row["control_arm_id"] != arm["control_arm_id"] for row in subset)
            or not np.isfinite([row["prediction"] for row in subset]).all()
        ):
            raise ValueError("Participant/arm prediction coverage differs")
        receipt = json.loads(verified_contents[f"fits/{arm['arm_id']}.json"])
        if (
            any(receipt.get(key) != value for key, value in arm.items())
            or receipt.get("selected_alpha") != manifest["selected_alpha"]
        ):
            raise ValueError("Fit receipt identity or selected alpha differs from the locked arm")
        if (
            not set(solver_fields) <= set(receipt)
            or receipt["status"] not in {"converged", "finite_with_termination"}
            or type(receipt["optimizer_converged"]) is not bool
            or receipt["optimizer_converged"] != (receipt["status"] == "converged")
            or type(receipt["retained_initial"]) is not bool
            or receipt["global_optimum_claimed"] is not False
            or not isinstance(receipt["optimizer_message"], str)
            or any(
                type(receipt[key]) is not int or receipt[key] < 0
                for key in ("optimizer_status", "iterations", "function_evaluations")
            )
            or any(
                isinstance(receipt[key], bool)
                or not isinstance(receipt[key], (int, float))
                or not np.isfinite(receipt[key])
                for key in ("initial_objective", "objective")
            )
        ):
            raise ValueError("Fit receipt solver disposition is missing or invalid")
        verified_receipts[arm["arm_id"]] = {key: receipt[key] for key in solver_fields}
    status_counts = dict(Counter(receipt["status"] for receipt in verified_receipts.values()))
    all_converged = status_counts.get("converged") == len(arms)
    if (
        status_counts != lock.get("fit_status_counts")
        or type(lock.get("all_optimizers_converged")) is not bool
        or all_converged != lock["all_optimizers_converged"]
    ):
        raise ValueError("Fit receipt solver counts disagree with the prediction lock")
    lock["verified_fit_receipts"] = verified_receipts
    lock["verified_fit_status_counts"] = status_counts
    lock["verified_all_optimizers_converged"] = all_converged
    return manifest, rows, lock


def report_run(
    run_dir: Path,
    outcome_path: Path,
    audit_path: Path,
    *,
    expected_outcomes_sha256: str | None = None,
    expected_audit_sha256: str | None = None,
) -> dict[str, Any]:
    """Join gender only after lock validation; compute all declared paired summaries."""
    manifest, rows, lock = read_locked_run(run_dir)
    outcome_raw = outcome_path.read_bytes()
    outcome_hash = hashlib.sha256(outcome_raw).hexdigest()
    if (
        outcome_hash != manifest["inputs"]["outcomes_sha256"]
        or expected_outcomes_sha256 is not None
        and outcome_hash != expected_outcomes_sha256
    ):
        raise ValueError("Reporting outcomes differ from the locked source")
    outcomes = json.loads(outcome_raw)
    # The separate source self-report map is first opened here, after lock validation.
    audit_raw = audit_path.read_bytes()
    audit_hash = hashlib.sha256(audit_raw).hexdigest()
    expected_audit = manifest["protocol"].get("audit_attributes_sha256")
    if (
        expected_audit is not None
        and audit_hash != expected_audit
        or expected_audit_sha256 is not None
        and audit_hash != expected_audit_sha256
    ):
        raise ValueError("Accepted AVI audit-attribute source hash mismatch")
    attributes = json.loads(audit_raw)
    participant_ids = sorted(manifest["test_participants"])
    target = np.asarray(
        [(outcomes[key]["outcomes"]["mean_rating_hirea"] - 1) / 4 for key in participant_ids]
    )
    labels = []
    for participant in participant_ids:
        value = attributes[participant].get("self_reported_gender_code")
        if value is not None and (
            isinstance(value, bool) or str(value) not in {"1", "2", "3", "4"}
        ):
            raise ValueError("Unknown source gender code; do not silently recode")
        labels.append("unknown" if value is None else str(value))
    labels_array = np.asarray(labels)
    counts = {code: labels.count(code) for code in ("1", "2", "3", "4", "unknown")}
    supported = counts["1"] >= 10 and counts["2"] >= 10
    weights = participant_bootstrap_weights(participant_ids, 10000, 2026)
    draws, points, result_arms = {}, {}, {}
    for arm in comparison_arms():
        subset = {
            row["participant_id"]: row["prediction"]
            for row in rows
            if row["arm_id"] == arm["arm_id"]
        }
        prediction = np.asarray([subset[key] for key in participant_ids])
        residual = prediction - target
        metrics = avi_utility(target, prediction)
        utility_draws = bootstrap_utilities(target, prediction, weights)
        disparity_draws = (
            empirical_w1_bootstrap(prediction, labels_array, ("1", "2"), weights)
            if supported
            else np.full(10000, np.nan)
        )
        point_w1 = (
            float(
                empirical_w1_bootstrap(
                    prediction, labels_array, ("1", "2"), np.ones((1, len(target)))
                )[0]
            )
            if supported
            else None
        )
        group_errors = {}
        for code in ("1", "2", "3", "4", "unknown"):
            keep = labels_array == code
            entry = {
                "n_participants": counts[code],
                "reportable": counts[code] >= 10 and code in {"1", "2", "3"},
            }
            if entry["reportable"]:
                entry.update(
                    mae=float(np.abs(residual[keep]).mean()),
                    mae_raw_scale=float(np.abs(residual[keep]).mean() * 4),
                    residual_bias=float(residual[keep].mean()),
                )
            group_errors[code] = entry
        if supported:
            left = weights * (labels_array == "1")
            right = weights * (labels_array == "2")
            left_total, right_total = left.sum(axis=1), right.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                signed_gap_draws = (left @ np.abs(residual)) / left_total - (
                    right @ np.abs(residual)
                ) / right_total
            signed_gap = float(
                np.abs(residual[labels_array == "1"]).mean()
                - np.abs(residual[labels_array == "2"]).mean()
            )
        else:
            signed_gap_draws, signed_gap = np.full(10000, np.nan), None
        draws[arm["arm_id"]] = {
            **utility_draws,
            "score_w1": disparity_draws,
            "signed_mae_gap": signed_gap_draws,
        }
        points[arm["arm_id"]] = {
            "mae": metrics["mae"],
            "srcc": metrics["srcc"],
            "score_w1": point_w1,
            "signed_mae_gap": signed_gap,
        }
        result_arms[arm["arm_id"]] = {
            **arm,
            "primary_contrast": arm["arm_id"] == PRIMARY_ARM,
            "solver": lock["verified_fit_receipts"][arm["arm_id"]],
            "utility": metrics,
            "group_errors": group_errors,
            "score_w1": _interval(disparity_draws, point_w1) if supported else None,
            "signed_mae_gap_code_1_minus_2": _interval(signed_gap_draws, signed_gap),
            "absolute_mae_gap_descriptive": None if signed_gap is None else abs(signed_gap),
        }
    for arm in comparison_arms():
        name, control = arm["arm_id"], arm["control_arm_id"]
        effects = {}
        for metric in ("mae", "srcc", "signed_mae_gap", "score_w1"):
            direction = -1 if metric == "score_w1" else 1
            point = (
                None
                if points[name][metric] is None or points[control][metric] is None
                else direction * (points[name][metric] - points[control][metric])
            )
            effects["disparity_gain" if metric == "score_w1" else f"delta_{metric}"] = _interval(
                direction * (draws[name][metric] - draws[control][metric]), point
            )
        result_arms[name]["matched_effects"] = effects
    baseline = np.full_like(target, manifest["training_mean_target"])
    return {
        "schema_version": "avi-compact-aggregate-1",
        "study_id": STUDY_ID,
        "status": "completed_fixed_comparison",
        "interpretation": manifest["interpretation"],
        "coverage": {
            "test_participants": len(participant_ids),
            "arm_count": 15,
            "prediction_rows": len(rows),
            "group_counts": counts,
        },
        "selection": {"alpha": manifest["selected_alpha"], "alpha_selection_role": "Validation"},
        "target_transform": "(raw-1)/4",
        "raw_scale_difference_factor": 4,
        "bootstrap": {
            "unit": "participant",
            "draws": 10000,
            "seed": 2026,
            "interval": "pointwise_percentile_95_conditional_on_fixed_fits",
            "same_draws_for_all_arms": True,
            "absolute_gap_is_descriptive_not_a_significance_test": True,
        },
        "training_mean_reference": avi_utility(target, baseline),
        "fit_status_counts": lock["verified_fit_status_counts"],
        "all_optimizers_converged": lock["verified_all_optimizers_converged"],
        "arms": result_arms,
        "proxy_mechanism": lock["verified_proxy_mechanism"],
        "input_hashes": {
            "prediction_lock": file_sha256(run_dir / "prediction-lock.json"),
            "outcomes": hashlib.sha256(outcome_raw).hexdigest(),
            "audit_attributes": hashlib.sha256(audit_raw).hexdigest(),
        },
    }


def render_report(result: dict[str, Any]) -> str:
    """Produce concise academic-readable numbers without disclosing participant rows."""

    def number(value: Any) -> str:
        return "undefined" if value is None else f"{0.0 if round(value, 3) == 0 else value:.3f}"

    def effect(value: dict, factor: float = 1.0) -> str:
        point = value["estimate"]
        low, high = value["percentile_95"]
        return (
            f"{number(None if point is None else factor * point)} "
            f"[{number(None if low is None else factor * low)}, "
            f"{number(None if high is None else factor * high)}]"
        )

    lines = [
        "# AVI compact fairness comparison",
        "",
        "This is an exploratory intervention follow-up on an independent interview corpus. "
        "The earlier AVI prediction/audit results had already been inspected. All fifteen "
        "settings are reported; no strength is selected from test or gender-audit results.",
        "",
        "Sampled full-frame visual and clip-audio features are pooled over six clips per "
        "participant; these are not MIT candidate-face crops. The official "
        "test contains 130 people. Target errors below use the original 1–5 rating scale; "
        "score W1 uses the normalised prediction scale. Source gender categories are "
        "self-reports, not MIT's perceived-appearance labels.",
        "",
        f"Selected ridge alpha: {result['selection']['alpha']:g} (validation MAE only).",
        f"Solver dispositions: {result['fit_status_counts']}.",
        "",
        "| Arm | Solver disposition | Test MAE | SRCC | Gender score W1 |",
        "|---|---|---:|---:|---:|",
    ]
    for arm, row in result["arms"].items():
        lines.append(
            f"| {arm} | {row['solver']['status']} | {number(row['utility']['mae_raw_scale'])} | "
            f"{number(row['utility']['srcc'])} | "
            f"{number(None if row['score_w1'] is None else row['score_w1']['estimate'])} |"
        )
    primary = result["arms"][PRIMARY_ARM]
    lines += [
        "",
        "## Primary paired comparison",
        "",
        f"Visual natural-exposure W1 at 0.02 versus its task-only control: MAE change "
        f"{effect(primary['matched_effects']['delta_mae'], 4)} rating points; SRCC change "
        f"{effect(primary['matched_effects']['delta_srcc'])}; disparity gain "
        f"{effect(primary['matched_effects']['disparity_gain'])}.",
        "",
        "Intervals use 10,000 paired participant bootstrap draws and are pointwise, "
        "conditional on these fitted models. They do not estimate retraining uncertainty "
        "or establish employment validity, compliance or universal mitigation.",
        "",
        "## Source groups and signed error gap",
        "",
        f"Test source-code counts: {result['coverage']['group_counts']}.",
        "",
        "| Arm | Signed MAE gap, code 1 minus code 2 [95% interval] |",
        "|---|---:|",
    ]
    for name in ("natural_task", PRIMARY_ARM):
        lines.append(
            f"| {name} | {effect(result['arms'][name]['signed_mae_gap_code_1_minus_2'], 4)} |"
        )
    lines += [
        "",
        "The absolute gap is descriptive; a positive interval for its absolute value "
        "is not evidence that the signed group difference excludes zero. Sparse and "
        "nonresponse source categories remain in counts without being merged.",
        "",
    ]
    if not result["all_optimizers_converged"]:
        lines += [
            "Some bounded solver fits terminated without reported convergence. Their "
            "finite predictions remain in the complete grid and are explicitly qualified; "
            "no global-optimum claim is made.",
            "",
        ]
    return "\n".join(lines)


def write_report(result: dict[str, Any], output: Path, report: Path) -> None:
    """Publish new aggregate and readable outputs without replacing an earlier release."""
    if output.exists() or report.exists() or output == report:
        raise FileExistsError("Report outputs must be new distinct files")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    with report.open("x", encoding="utf-8") as handle:
        handle.write(render_report(result))
