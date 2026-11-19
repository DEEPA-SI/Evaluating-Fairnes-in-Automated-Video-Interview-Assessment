"""Compact, post-audit MIT study using cached features and a fixed linear head.

This module deliberately imports no Stage-A trainer or encoder. Definitions and the
fixed comparison are described in docs/research-overview.md.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

STUDY_ID = "study2-compact-2026-10-21-v1"
STRENGTHS = (0.002, 0.02, 0.2)
DIMENSIONS = {"video": 1024, "audio": 1536, "text": 768}
SOLVER_OPTIONS = {"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40}


def canonical_bytes(payload: Any) -> bytes:
    """Encode finite JSON using the project's canonical convention."""
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def file_sha256(path: str | Path) -> str:
    """Hash an input or output without loading a large shard twice into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Publish a new file exclusively; a completed artifact is never overwritten."""
    with path.open("xb") as handle:
        handle.write(canonical_bytes(payload))


def _signed_json(path: Path, field: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    unsigned = dict(payload)
    signature = unsigned.pop(field)
    if hashlib.sha256(canonical_bytes(unsigned)).hexdigest() != signature:
        raise ValueError(f"Invalid payload hash: {path.name}")
    return payload, hashlib.sha256(raw).hexdigest()


def _child(root: Path, filename: str) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Artifact references must be plain filenames")
    path = root / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular input file: {filename}")
    return path


@dataclass(frozen=True)
class CachedInputs:
    """Ordered summaries, immutable source identities and unchanged source targets."""

    x: np.ndarray
    y: np.ndarray
    rows: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]


def load_cached_inputs(feature_dir: Path, stage_a_dir: Path) -> CachedInputs:
    """Read each accepted feature shard once, bound to eligibility and closure.

    The legacy predictions supply their unchanged target field only. Old model
    predictions are never used to train or choose the new scorer.
    """
    closure_path = feature_dir / "feature-extraction-closure.json"
    closure, closure_hash = _signed_json(closure_path, "closure_payload_sha256")
    eligibility_path = stage_a_dir / "complete-pair-eligibility.json"
    eligibility, eligibility_hash = _signed_json(eligibility_path, "manifest_payload_sha256")
    stage_closure_path = stage_a_dir / "stage-a-closure.json"
    stage_closure, stage_closure_hash = _signed_json(stage_closure_path, "closure_payload_sha256")
    lock_path = stage_a_dir / "erm.seed-1.prediction-lock.json"
    lock, lock_hash = _signed_json(lock_path, "manifest_payload_sha256")
    prediction_path = stage_a_dir / "erm.seed-1.predictions.json"
    prediction_hash = file_sha256(prediction_path)
    bundle = next(
        item
        for item in stage_closure["learned_bundles"]
        if item["configuration"] == "erm" and item["seed"] == 1
    )
    if (
        bundle["prediction_file_sha256"] != prediction_hash
        or bundle["prediction_lock_file_sha256"] != lock_hash
        or lock["predictions"]["sha256"] != prediction_hash
    ):
        raise ValueError("Sealed prediction source checksums disagree")
    if (
        eligibility["feature_extraction_closure_file_sha256"] != closure_hash
        or lock["feature_extraction_closure_file_sha256"] != closure_hash
        or lock["eligibility_manifest_sha256"] != eligibility["manifest_payload_sha256"]
        or stage_closure["context"]["eligibility_manifest_sha256"]
        != eligibility["manifest_payload_sha256"]
    ):
        raise ValueError("Feature, eligibility and sealed target releases disagree")
    if (
        eligibility["eligible_participant_count"] != 69
        or eligibility["eligible_session_count"] != 138
        or closure["failure_session_count"] != 0
    ):
        raise ValueError("Compact protocol requires all 69 complete participant pairs")
    old_rows = json.loads(prediction_path.read_bytes())["rows"]
    targets = {row["session_id"]: row for row in old_rows}
    records = {row["session_id"]: row for row in closure["records"]}
    if len(targets) != 138 or len(records) != 138:
        raise ValueError("Source release contains missing or duplicate session identities")
    matrices, outcomes, identities = [], [], []
    for participant in eligibility["eligible_participants"]:
        for session in participant["sessions"]:
            session_id = session["session_id"]
            record, target = records[session_id], targets[session_id]
            participant_id, fold = str(participant["participant_id"]), participant["fold"]
            if (
                record["participant_id"] != participant_id
                or record["fold"] != fold
                or target["participant_id"] != participant_id
                or target["fixed_fold"] != fold
                or target["session_phase"] != session["condition"]
                or record["condition"] != session["condition"]
                or target["source_session_id"] != session["source_session_id"]
                or record["source_session_id"] != session["source_session_id"]
                or record["shard_file_sha256"] != session["shard_sha256"]
                or record["metadata_file_sha256"] != session["sidecar_sha256"]
            ):
                raise ValueError(f"Canonical source join disagrees for {session_id}")
            metadata_path = _child(feature_dir, record["metadata_filename"])
            if file_sha256(metadata_path) != session["sidecar_sha256"]:
                raise ValueError(f"Feature sidecar checksum mismatch: {session_id}")
            shard_path = _child(feature_dir, record["shard_filename"])
            raw = shard_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != session["shard_sha256"]:
                raise ValueError(f"Feature shard checksum mismatch: {session_id}")
            pieces = []
            with np.load(io.BytesIO(raw), allow_pickle=False) as shard:
                for modality, dimension in DIMENSIONS.items():
                    value = shard[f"{modality}_mean"]
                    if (
                        value.shape != (dimension,)
                        or value.dtype != np.dtype("float32")
                        or not np.isfinite(value).all()
                        or hashlib.sha256(
                            np.ascontiguousarray(value, dtype="<f4").tobytes()
                        ).hexdigest()
                        != session["summary_sha256"][modality]
                    ):
                        raise ValueError(f"Feature summary mismatch: {session_id}/{modality}")
                    pieces.append(value.astype(np.float64))
            matrices.append(np.concatenate(pieces))
            outcomes.append(float(target["target"]))
            identities.append(
                {
                    "session_id": session_id,
                    "participant_id": participant_id,
                    "source_session_id": session["source_session_id"],
                    "phase": session["condition"],
                    "fold": fold,
                }
            )
    x, y = np.stack(matrices), np.asarray(outcomes, dtype=np.float64)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("Invalid unchanged source targets")
    return CachedInputs(
        x,
        y,
        tuple(identities),
        {
            "feature_closure_file_sha256": closure_hash,
            "eligibility_file_sha256": eligibility_hash,
            "stage_a_closure_file_sha256": stage_closure_hash,
            "target_prediction_file_sha256": prediction_hash,
            "target_prediction_lock_file_sha256": lock_hash,
            "target_manifest_sha256": lock["target_manifest_sha256"],
            "target_source": "checksum-bound sealed erm.seed-1.predictions.json target field",
            "target_transform": "(raw-1)/6",
            "annotation_fields_accessed": [],
        },
    )


def partition_indices(rows: tuple[dict[str, Any], ...], fold: int) -> dict[str, np.ndarray]:
    """Keep both sessions together in the five fixed train/validation/test roles."""
    if fold not in range(5):
        raise ValueError("Outer fold must be 0 through 4")
    participant_folds: dict[str, int] = {}
    for row in rows:
        previous = participant_folds.setdefault(row["participant_id"], row["fold"])
        if previous != row["fold"]:
            raise ValueError("A participant's sessions occupy different folds")
    validation = (fold + 1) % 5
    return {
        role: np.asarray([i for i, row in enumerate(rows) if condition(row["fold"])], dtype=int)
        for role, condition in (
            ("train", lambda f: f not in (fold, validation)),
            ("validation", lambda f: f == validation),
            ("test", lambda f: f == fold),
        )
    }


def empirical_w1_gradient(scores: np.ndarray, groups: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact unequal-sample empirical W1 and an analytic score subgradient.

    Quantile breakpoints integrate the two empirical inverse CDFs. Within-group
    exact ties receive their averaged derivative, independent of input row order.
    """
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups)
    if scores.ndim != 1 or groups.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("W1 requires matching finite score and group vectors")
    if set(np.unique(groups)) != {0, 1}:
        raise ValueError("W1 requires both binary proxy groups")
    positions = [np.flatnonzero(groups == group) for group in (0, 1)]
    order = [position[np.argsort(scores[position], kind="stable")] for position in positions]
    a, b = scores[order[0]], scores[order[1]]
    breaks = np.unique(
        np.concatenate((np.arange(len(a) + 1) / len(a), np.arange(len(b) + 1) / len(b)))
    )
    widths, midpoint = np.diff(breaks), (breaks[1:] + breaks[:-1]) / 2
    ia = np.minimum((midpoint * len(a)).astype(int), len(a) - 1)
    ib = np.minimum((midpoint * len(b)).astype(int), len(b) - 1)
    difference = a[ia] - b[ib]
    gradient = np.zeros_like(scores)
    np.add.at(gradient, order[0][ia], widths * np.sign(difference))
    np.add.at(gradient, order[1][ib], -widths * np.sign(difference))
    for indices, values in zip(order, (a, b), strict=True):
        starts = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1, len(values)]
        for start, end in zip(starts[:-1], starts[1:], strict=True):
            if end - start > 1:
                tied = indices[start:end]
                gradient[tied] = gradient[tied].mean()
    return float(np.dot(widths, np.abs(difference))), gradient


def task_weights(groups: np.ndarray | None, n: int) -> np.ndarray:
    """Natural or deterministic expected balanced exposure weights, summing to one."""
    if groups is None:
        return np.full(n, 1.0 / n)
    if groups.shape != (n,) or set(np.unique(groups)) != {0, 1}:
        raise ValueError("Balanced exposure needs both proxy groups")
    return np.asarray([1.0 / (2 * np.count_nonzero(groups == g)) for g in groups])


@dataclass
class RowSpaceDesign:
    """Training-only scaling and minimum-L2 row-space coordinates."""

    scaler: StandardScaler
    basis: np.ndarray
    train: np.ndarray
    n: int

    @classmethod
    def fit(cls, x: np.ndarray) -> RowSpaceDesign:
        """Fit the representation once per training partition, without outcomes."""
        scaler = StandardScaler().fit(np.asarray(x, dtype=np.float64))
        transformed = scaler.transform(x)
        _, singular, vh = np.linalg.svd(transformed, full_matrices=False)
        threshold = np.finfo(np.float64).eps * max(transformed.shape) * singular[0]
        basis = vh[singular > threshold].T
        z = transformed @ basis
        return cls(scaler, basis, np.column_stack((z, np.ones(len(z)))), len(z))

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the training-fitted transform without updating it."""
        return np.column_stack((self.scaler.transform(x) @ self.basis, np.ones(len(x))))

    def exact_task_solution(self, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Solve the alpha-1 weighted ridge objective with unpenalised intercept."""
        penalty = np.full(self.train.shape[1], 1.0 / self.n)
        penalty[-1] = 0
        return np.linalg.solve(
            self.train.T @ (weights[:, None] * self.train) + np.diag(penalty),
            self.train.T @ (weights * y),
        )


@dataclass
class AlphaRowSpaceDesign(RowSpaceDesign):
    """Represent alpha-ridge through the unchanged alpha-one fitting objective.

    This is the same non-intercept coordinate rescaling used by the AVI compact
    comparison. The original RowSpaceDesign path is retained for legacy MIT fits.
    """

    alpha: float = 1.0

    @classmethod
    def from_base(cls, base: RowSpaceDesign, alpha: float) -> AlphaRowSpaceDesign:
        """Rescale training coordinates without altering the intercept or scaler."""
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha must be finite and positive")
        train = base.train.copy()
        train[:, :-1] /= np.sqrt(alpha)
        return cls(base.scaler, base.basis, train, base.n, float(alpha))

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the unchanged training transform and matching alpha coordinates."""
        transformed = super().transform(x)
        transformed[:, :-1] /= np.sqrt(self.alpha)
        return transformed


def fit_linear_arm(
    design: RowSpaceDesign,
    y: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray | None,
    strength: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimise one fixed arm from its matched exact task-only solution."""
    if strength < 0 or (strength > 0 and groups is None):
        raise ValueError("Invalid W1 arm")
    initial = design.exact_task_solution(y, weights)

    def components(theta: np.ndarray) -> tuple[float, np.ndarray, dict[str, Any]]:
        prediction = design.train @ theta
        residual = prediction - y
        task = float(np.dot(weights, residual**2))
        regularisation = float(np.dot(theta[:-1], theta[:-1]) / design.n)
        task_gradient = design.train.T @ (2 * weights * residual)
        ridge_gradient = np.r_[2 * theta[:-1] / design.n, 0.0]
        w1, score_gradient = (
            empirical_w1_gradient(prediction, groups)
            if groups is not None
            else (0.0, np.zeros_like(prediction))
        )
        w1_gradient = design.train.T @ score_gradient
        gradient = task_gradient + ridge_gradient + strength * w1_gradient
        return (
            task + regularisation + strength * w1,
            gradient,
            {
                "task_mse": task,
                "regularisation": regularisation,
                "proxy_w1": w1,
                "task_gradient_l2": float(np.linalg.norm(task_gradient)),
                "regularisation_gradient_l2": float(np.linalg.norm(ridge_gradient)),
                "w1_gradient_l2": float(np.linalg.norm(w1_gradient)),
                "weighted_w1_gradient_l2": float(strength * np.linalg.norm(w1_gradient)),
                "total_gradient_l2": float(np.linalg.norm(gradient)),
            },
        )

    initial_value, _, initial_diagnostics = components(initial)
    history: list[dict[str, Any]] = []

    def callback(theta: np.ndarray) -> None:
        value, _, diagnostics = components(theta)
        history.append({"iteration": len(history) + 1, "objective": value, **diagnostics})

    started = time.perf_counter()
    result = minimize(
        lambda theta: components(theta)[:2],
        initial,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options=SOLVER_OPTIONS,
    )
    elapsed = time.perf_counter() - started
    finite = bool(np.isfinite(result.x).all() and np.isfinite(result.fun))
    retained_initial = not finite or float(result.fun) > initial_value
    theta = initial if retained_initial else result.x
    value, _, diagnostics = components(theta)
    return theta, {
        "status": "converged" if result.success and finite else "finite_with_termination",
        "optimizer_converged": bool(result.success and finite),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "elapsed_seconds": elapsed,
        "retained_initial": retained_initial,
        "initial_objective": initial_value,
        "objective": value,
        "initial_diagnostics": initial_diagnostics,
        "final_diagnostics": diagnostics,
        "history": history,
        "global_optimum_claimed": False,
    }


def fit_fold_proxies(data: CachedInputs, train: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Reuse the original train-only participant-mean scaler/KMeans policy."""
    participants = sorted({row["participant_id"] for row in data.rows}, key=int)
    train_ids = {data.rows[i]["participant_id"] for i in train}
    member_rows = [
        np.asarray([i for i, row in enumerate(data.rows) if row["participant_id"] == participant])
        for participant in participants
    ]
    if any(len(indices) != 2 for indices in member_rows):
        raise ValueError("Proxies require complete participant pairs")
    train_mask = np.asarray([participant in train_ids for participant in participants])
    assignments, manifests = {}, {}
    for source, subset in (("visual", slice(0, 1024)), ("audio", slice(1024, 2560))):
        matrix = np.stack([data.x[indices, subset].mean(axis=0) for indices in member_rows])
        scaler = StandardScaler().fit(matrix[train_mask])
        clusterer = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=20,
            max_iter=300,
            tol=1e-4,
            algorithm="lloyd",
            random_state=0,
        )
        clusterer.fit(scaler.transform(matrix[train_mask]))
        centres = clusterer.cluster_centers_
        order = sorted(
            range(2),
            key=lambda i: (
                hashlib.sha256(np.ascontiguousarray(centres[i], dtype="<f8").tobytes()).digest(),
                tuple(float(value) for value in centres[i]),
            ),
        )
        mapping = np.empty(2, dtype=int)
        mapping[order] = np.arange(2)
        groups = mapping[clusterer.predict(scaler.transform(matrix))]
        by_participant = dict(zip(participants, groups.tolist(), strict=True))
        assignments[source] = np.asarray(
            [by_participant[row["participant_id"]] for row in data.rows]
        )
        if set(np.unique(assignments[source][train])) != {0, 1}:
            raise ValueError("Training proxy has an empty group")
        manifests[source] = {
            "participant_groups": by_participant,
            "training_participants": sorted(train_ids, key=int),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "cluster_centres": centres.tolist(),
            "raw_to_canonical": mapping.tolist(),
            "settings": clusterer.get_params(),
            "fit_scope": "training_participants_only",
        }
    return assignments, manifests


def comparison_arms() -> tuple[dict[str, Any], ...]:
    """Return the complete fixed 15-arm comparison, including matched zero controls."""
    arms = [
        {
            "arm_id": "natural_task",
            "proxy_source": "none",
            "exposure": "natural",
            "lambda": 0.0,
            "control_arm_id": "natural_task",
        }
    ]
    for source in ("visual", "audio"):
        control = f"{source}_balanced_task"
        arms.append(
            {
                "arm_id": control,
                "proxy_source": source,
                "exposure": "balanced",
                "lambda": 0.0,
                "control_arm_id": control,
            }
        )
        for exposure in ("natural", "balanced"):
            for strength in STRENGTHS:
                arms.append(
                    {
                        "arm_id": f"{source}_{exposure}_w1_lambda-{strength:g}",
                        "proxy_source": source,
                        "exposure": exposure,
                        "lambda": strength,
                        "control_arm_id": "natural_task" if exposure == "natural" else control,
                    }
                )
    return tuple(arms)


def utility_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Return prediction diagnostics, preserving undefined correlations as null."""
    target, prediction = np.asarray(target), np.asarray(prediction)
    error = prediction - target
    correlations_defined = len(target) >= 2 and np.ptp(target) > 0 and np.ptp(prediction) > 0
    return {
        "n_sessions": len(target),
        "mae": float(np.abs(error).mean()),
        "mae_raw_scale": float(6 * np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "rmse_raw_scale": float(6 * np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "prediction_sd": float(np.std(prediction)),
        "target_sd": float(np.std(target)),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "srcc": float(spearmanr(target, prediction).statistic) if correlations_defined else None,
        "pearson": float(pearsonr(target, prediction).statistic) if correlations_defined else None,
        "correlation_undefined_reason": None
        if correlations_defined
        else "constant target/predictions or insufficient observations",
    }


def summarize_predictions(
    rows: list[dict[str, Any]], *, study_id: str = STUDY_ID
) -> dict[str, Any]:
    """Aggregate outcome utility without reading any annotation input."""
    results = []
    for arm in comparison_arms():
        arm_rows = [row for row in rows if row["arm_id"] == arm["arm_id"]]
        phases = {}
        for phase in ("pooled", "pre", "post"):
            selected = [row for row in arm_rows if phase == "pooled" or row["phase"] == phase]
            phases[phase] = utility_metrics(
                np.asarray([row["target"] for row in selected]),
                np.asarray([row["prediction"] for row in selected]),
            )
        results.append({**arm, "utility": phases})
    return {
        "study_id": study_id,
        "arms": results,
        "annotation_fields_accessed": [],
        "interpretation": "post-audit exploratory; all fixed strengths reported",
    }


def _run_compact(
    data: CachedInputs,
    output_dir: Path,
    protocol_path: Path,
    *,
    study_id: str = STUDY_ID,
    ridge_alpha: float = 1.0,
    extension: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Share the fixed fit matrix without changing the legacy public run identity."""
    if not np.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("Ridge alpha must be finite and positive")
    if extension is None and (study_id != STUDY_ID or ridge_alpha != 1.0):
        raise ValueError("Non-legacy fitting requires an explicitly bound extension identity")
    protocol_raw = protocol_path.read_bytes() if extension is not None else None
    if protocol_raw is not None and hashlib.sha256(protocol_raw).hexdigest() != extension.get(
        "protocol_file_sha256"
    ):
        raise ValueError("Frozen extension protocol changed after validation")
    output_dir.mkdir(parents=True, exist_ok=False)
    fit_dir = output_dir / "fits"
    fit_dir.mkdir()
    protocol_hash = file_sha256(protocol_path)
    if protocol_raw is not None:
        protocol_hash = hashlib.sha256(protocol_raw).hexdigest()
        with (output_dir / "frozen-protocol.yaml").open("xb") as handle:
            handle.write(protocol_raw)
    module_path = Path(__file__)
    manifest = {
        "study_id": study_id,
        "schema_version": "compact-study-manifest-v1"
        if extension is None
        else "mit-strong-ridge-manifest-1",
        "protocol_sha256": protocol_hash,
        "implementation_sha256": file_sha256(module_path),
        "inputs": data.provenance,
        "expected_fits": 75,
        "expected_prediction_rows": 2070,
        "participants": 69,
        "sessions": 138,
        "arms": list(comparison_arms()),
        "solver": {"method": "L-BFGS-B", **SOLVER_OPTIONS},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "post_audit_exploratory": True,
    }
    if extension is not None:
        manifest.update({"ridge_alpha": ridge_alpha, "extension": extension})
    try:
        manifest["code_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=module_path.parents[2], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["code_commit"] = None
    write_json(output_dir / "manifest.json", manifest)
    predictions, receipts, proxy_reports = [], [], []
    for fold in range(5):
        roles = partition_indices(data.rows, fold)
        train, validation, test = (roles[role] for role in ("train", "validation", "test"))
        base = RowSpaceDesign.fit(data.x[train])
        design = base if extension is None else AlphaRowSpaceDesign.from_base(base, ridge_alpha)
        test_design, validation_design = (
            design.transform(data.x[test]),
            design.transform(data.x[validation]),
        )
        groups, proxy_manifest = fit_fold_proxies(data, train)
        write_json(output_dir / f"proxies-fold-{fold}.json", proxy_manifest)
        for arm in comparison_arms():
            source = arm["proxy_source"]
            arm_groups = None if source == "none" else groups[source][train]
            weights = task_weights(
                arm_groups if arm["exposure"] == "balanced" else None, len(train)
            )
            theta, receipt = fit_linear_arm(
                design, data.y[train], weights, arm_groups, arm["lambda"]
            )
            test_prediction, validation_prediction = test_design @ theta, validation_design @ theta
            if not np.isfinite(test_prediction).all():
                raise ValueError("Nonfinite held-out prediction; no complete lock is published")
            fit_id = f"{arm['arm_id']}.fold-{fold}"
            receipt.update(
                {
                    **arm,
                    "fit_id": fit_id,
                    "outer_fold": fold,
                    "training_participants": sorted(
                        {data.rows[i]["participant_id"] for i in train}, key=int
                    ),
                    "validation_participants": sorted(
                        {data.rows[i]["participant_id"] for i in validation}, key=int
                    ),
                    "test_participants": sorted(
                        {data.rows[i]["participant_id"] for i in test}, key=int
                    ),
                    "training_row_count": len(train),
                    "rowspace_dimension": design.basis.shape[1],
                    "training_proxy_w1": {
                        proxy: empirical_w1_gradient(design.train @ theta, groups[proxy][train])[0]
                        for proxy in ("visual", "audio")
                    },
                    "validation_utility": utility_metrics(
                        data.y[validation], validation_prediction
                    ),
                }
            )
            if extension is not None:
                receipt.update(
                    {
                        "schema_version": "mit-strong-ridge-fit-receipt-1",
                        "study_id": study_id,
                        "ridge_alpha": ridge_alpha,
                        "protocol_sha256": protocol_hash,
                    }
                )
            write_json(fit_dir / f"{fit_id}.json", receipt)
            coefficients = design.basis @ theta[:-1]
            if extension is not None:
                coefficients = coefficients / np.sqrt(ridge_alpha)
            with (fit_dir / f"{fit_id}.npz").open("xb") as handle:
                np.savez(
                    handle,
                    coefficients=coefficients,
                    intercept=theta[-1],
                    scaler_mean=design.scaler.mean_,
                    scaler_scale=design.scaler.scale_,
                )
            receipts.append(receipt)
            for index, prediction in zip(test, test_prediction, strict=True):
                row = data.rows[index]
                predictions.append(
                    {
                        **arm,
                        "outer_fold": fold,
                        **{
                            key: row[key]
                            for key in (
                                "participant_id",
                                "session_id",
                                "source_session_id",
                                "phase",
                            )
                        },
                        "target": float(data.y[index]),
                        "prediction": float(prediction),
                        "proxy_group": None if source == "none" else int(groups[source][index]),
                    }
                )
            for proxy_source in ("visual", "audio"):
                held_groups = groups[proxy_source][test]
                for phase in ("pooled", "pre", "post"):
                    mask = np.asarray(
                        [phase == "pooled" or data.rows[i]["phase"] == phase for i in test]
                    )
                    counts = {str(g): int(np.count_nonzero(held_groups[mask] == g)) for g in (0, 1)}
                    valid = min(counts.values()) > 0
                    w1 = (
                        empirical_w1_gradient(test_prediction[mask], held_groups[mask])[0]
                        if valid
                        else None
                    )
                    proxy_reports.append(
                        {
                            "arm_id": arm["arm_id"],
                            "control_arm_id": arm["control_arm_id"],
                            "outer_fold": fold,
                            "proxy_source": proxy_source,
                            "phase": phase,
                            "session_counts": counts,
                            "w1": w1,
                            "undefined_reason": None if valid else "empty held-out proxy group",
                        }
                    )
            print(
                f"{len(receipts)}/75 {fit_id}: {receipt['status']} "
                f"({receipt['elapsed_seconds']:.2f}s)",
                flush=True,
            )
    if len(receipts) != 75 or len(predictions) != 2070:
        raise ValueError("Incomplete declared comparison; no complete prediction lock")
    write_json(
        output_dir / "predictions.json",
        {
            "schema_version": "compact-predictions-v1"
            if extension is None
            else "mit-strong-ridge-predictions-1",
            "study_id": study_id,
            "rows": predictions,
        },
    )
    write_json(
        output_dir / "fold-proxy-summary.json", {"study_id": study_id, "rows": proxy_reports}
    )
    statuses = dict(Counter(receipt["status"] for receipt in receipts))
    hashes = {
        str(path.relative_to(output_dir)): file_sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    lock = {
        "schema_version": "compact-prediction-lock-v1"
        if extension is None
        else "mit-strong-ridge-prediction-lock-1",
        "study_id": study_id,
        "prediction_coverage_complete": True,
        "expected_fit_count": 75,
        "observed_fit_count": len(receipts),
        "observed_prediction_rows": len(predictions),
        "fit_status_counts": statuses,
        "all_optimizers_converged": statuses.get("converged") == 75,
        "state": "predictions_locked"
        if statuses.get("converged") == 75
        else "predictions_locked_with_finite_optimizer_terminations",
        "files": {
            "predictions": {"filename": "predictions.json", "sha256": hashes["predictions.json"]}
        },
        "artifact_hashes": hashes,
        "annotation_fields_accessed": [],
    }
    if extension is not None:
        lock.update(
            {
                "ridge_alpha": ridge_alpha,
                "protocol_sha256": protocol_hash,
                "manifest_sha256": hashes["manifest.json"],
            }
        )
    write_json(output_dir / "prediction-lock.json", lock)
    summary = summarize_predictions(predictions, study_id=study_id)
    summary["fit_status_counts"] = statuses
    summary["prediction_lock_sha256"] = file_sha256(output_dir / "prediction-lock.json")
    if extension is not None:
        summary.update(
            {
                "schema_version": "mit-strong-ridge-summary-1",
                "ridge_alpha": ridge_alpha,
                "protocol_sha256": protocol_hash,
            }
        )
    write_json(output_dir / "summary.json", summary)
    return summary


def run_compact(data: CachedInputs, output_dir: Path, protocol_path: Path) -> dict[str, Any]:
    """Execute the unchanged alpha-one legacy comparison under its original identity."""
    return _run_compact(data, output_dir, protocol_path)
