"""Separately identified alpha-1000 MIT extension of the fixed compact comparison.

The fit entry point has no annotation argument. Its complete prediction lock is
verified before the separate audit entry point can open the accepted annotations.
Legacy alpha-one artifacts and their audit implementation remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.study2.compact import (
    SOLVER_OPTIONS,
    STRENGTHS,
    CachedInputs,
    _run_compact,
    comparison_arms,
    file_sha256,
)
from src.study2.compact_audit import (
    audit_compact_run,
    load_locked_predictions,
    render_compact_report,
)

ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "mit-strong-ridge-2026-11-19-v1"
RIDGE_ALPHA = 1000.0
DEVELOPMENT_SELECTION = ROOT / "results/study2_rq1_aggregate_v1.json"
PROTOCOL_PATH = ROOT / "configs/mit_strong_ridge_2026_11_19.yaml"


def _validate_protocol(protocol: Any) -> dict[str, Any]:
    expected = {
        "schema_version": "mit-strong-ridge-protocol-1",
        "study_id": STUDY_ID,
        "status": "frozen_for_execution",
        "ridge_alpha": RIDGE_ALPHA,
        "strengths": list(STRENGTHS),
        "participants": 69,
        "sessions": 138,
        "outer_folds": 5,
        "expected_arms": 15,
        "expected_fits": 75,
        "target_transform": "(raw-1)/6",
        "solver": {"method": "L-BFGS-B", **SOLVER_OPTIONS},
        "post_audit_exploratory": True,
        "annotation_fields_used_for_fitting": [],
    }
    if not isinstance(protocol, dict) or any(
        protocol.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Stronger-ridge protocol differs from its fixed comparison")
    evidence_hash = file_sha256(DEVELOPMENT_SELECTION)
    if protocol.get("development_selection_file_sha256") != evidence_hash:
        raise ValueError("Development-selected ridge evidence hash mismatch")
    evidence = json.loads(DEVELOPMENT_SELECTION.read_bytes())
    if evidence.get("selection", {}).get("chosen_c02_alpha_by_outer_fold") != [RIDGE_ALPHA] * 5:
        raise ValueError("Development evidence does not select alpha 1000 in every outer fold")
    return protocol


def load_protocol(path: Path) -> dict[str, Any]:
    """Validate the frozen comparison and its existing development-only alpha evidence."""
    return _validate_protocol(yaml.safe_load(path.read_bytes()))


def _check_inputs(data: CachedInputs) -> None:
    if (
        data.x.shape != (138, 3328)
        or data.y.shape != (138,)
        or len(data.rows) != 138
        or not np.isfinite(data.x).all()
        or not np.isfinite(data.y).all()
        or np.any((data.y < 0) | (data.y > 1))
    ):
        raise ValueError("Stronger-ridge extension requires the complete accepted MIT design")
    people: dict[str, list[dict[str, Any]]] = {}
    for row in data.rows:
        people.setdefault(row["participant_id"], []).append(row)
    if len({row["session_id"] for row in data.rows}) != 138 or len(people) != 69:
        raise ValueError("MIT participant/session coverage is missing or duplicated")
    for rows in people.values():
        if len(rows) != 2 or {row["phase"] for row in rows} != {"pre", "post"}:
            raise ValueError("MIT extension requires each complete pre/post participant pair")
        if len({row["fold"] for row in rows}) != 1:
            raise ValueError("A participant's sessions cross outer folds")
    folds = Counter(rows[0]["fold"] for rows in people.values())
    if folds != {0: 14, 1: 14, 2: 14, 3: 14, 4: 13}:
        raise ValueError("MIT extension requires the unchanged five participant folds")


def run_strong_ridge(
    data: CachedInputs, output_dir: Path, protocol_path: Path = PROTOCOL_PATH
) -> dict[str, Any]:
    """Run the new complete comparison without changing or reusing a prior run directory."""
    raw = protocol_path.read_bytes()
    protocol = _validate_protocol(yaml.safe_load(raw))
    _check_inputs(data)
    implementation = Path(__file__)
    extension = {
        "protocol": protocol,
        "protocol_file_sha256": hashlib.sha256(raw).hexdigest(),
        "alpha_selection_scope": "existing_rq1_development_selection_all_outer_folds",
        "development_selection_file_sha256": protocol["development_selection_file_sha256"],
        "implementation_hashes": {
            "src/study2/compact.py": file_sha256(implementation.with_name("compact.py")),
            "src/study2/strong_ridge.py": file_sha256(implementation),
            "scripts/run_mit_strong_ridge.py": file_sha256(
                ROOT / "scripts/run_mit_strong_ridge.py"
            ),
        },
        "coefficient_coordinates": "standardised_input_basis_backscaled_by_sqrt_alpha",
        "intercept_penalised": False,
    }
    return _run_compact(
        data,
        output_dir,
        protocol_path,
        study_id=STUDY_ID,
        ridge_alpha=RIDGE_ALPHA,
        extension=extension,
    )


def validate_locked_run(run_dir: Path) -> None:
    """Validate the new identity before using the unchanged compact numerical audit."""
    lock = json.loads((run_dir / "prediction-lock.json").read_bytes())
    manifest = json.loads((run_dir / "manifest.json").read_bytes())
    predictions = json.loads((run_dir / "predictions.json").read_bytes())
    for payload, schema in (
        (lock, "mit-strong-ridge-prediction-lock-1"),
        (manifest, "mit-strong-ridge-manifest-1"),
        (predictions, "mit-strong-ridge-predictions-1"),
    ):
        if payload.get("schema_version") != schema or payload.get("study_id") != STUDY_ID:
            raise ValueError("Locked stronger-ridge artifact identity differs")
    if lock.get("ridge_alpha") != RIDGE_ALPHA or manifest.get("ridge_alpha") != RIDGE_ALPHA:
        raise ValueError("Locked stronger-ridge alpha differs")
    if (
        lock.get("expected_fit_count") != 75
        or lock.get("observed_fit_count") != 75
        or lock.get("observed_prediction_rows") != 2070
        or lock.get("prediction_coverage_complete") is not True
        or lock.get("annotation_fields_accessed") != []
        or manifest.get("expected_fits") != 75
        or manifest.get("expected_prediction_rows") != 2070
    ):
        raise ValueError("Locked stronger-ridge coverage or annotation separation differs")
    protocol_path = run_dir / "frozen-protocol.yaml"
    protocol = load_protocol(protocol_path)
    digest = file_sha256(protocol_path)
    if (
        lock.get("protocol_sha256") != digest
        or manifest.get("protocol_sha256") != digest
        or manifest.get("extension", {}).get("protocol_file_sha256") != digest
        or manifest.get("extension", {}).get("protocol") != protocol
        or lock.get("manifest_sha256") != file_sha256(run_dir / "manifest.json")
    ):
        raise ValueError("Locked stronger-ridge protocol/manifest binding differs")
    arms = comparison_arms()
    required = {
        "manifest.json",
        "frozen-protocol.yaml",
        "predictions.json",
        "fold-proxy-summary.json",
    }
    required.update(f"proxies-fold-{fold}.json" for fold in range(5))
    required.update(
        f"fits/{arm['arm_id']}.fold-{fold}.{suffix}"
        for arm in arms
        for fold in range(5)
        for suffix in ("json", "npz")
    )
    if set(lock.get("artifact_hashes", {})) != required:
        raise ValueError("Stronger-ridge lock must bind every expected fit and proxy artifact")
    if lock.get("files", {}).get("predictions") != {
        "filename": "predictions.json",
        "sha256": lock["artifact_hashes"]["predictions.json"],
    }:
        raise ValueError(
            "Stronger-ridge prediction binding differs from its complete artifact lock"
        )
    if (
        manifest.get("arms") != list(arms)
        or manifest.get("solver") != {"method": "L-BFGS-B", **SOLVER_OPTIONS}
        or manifest.get("participants") != 69
        or manifest.get("sessions") != 138
    ):
        raise ValueError("Locked stronger-ridge comparison differs from its fixed design")
    # Check every bound artifact, then the complete legacy-shaped row design.
    load_locked_predictions(run_dir)
    statuses = Counter()
    for arm in arms:
        for fold in range(5):
            path = run_dir / "fits" / f"{arm['arm_id']}.fold-{fold}.json"
            receipt = json.loads(path.read_bytes())
            if (
                receipt.get("schema_version") != "mit-strong-ridge-fit-receipt-1"
                or receipt.get("study_id") != STUDY_ID
                or receipt.get("ridge_alpha") != RIDGE_ALPHA
                or receipt.get("protocol_sha256") != digest
                or receipt.get("arm_id") != arm["arm_id"]
                or receipt.get("outer_fold") != fold
            ):
                raise ValueError("Stronger-ridge fit receipt identity differs")
            statuses[receipt["status"]] += 1
    if dict(statuses) != lock.get("fit_status_counts"):
        raise ValueError("Stronger-ridge fit status counts disagree with the complete receipts")


def audit_strong_ridge_run(run_dir: Path, annotation_release_dir: Path) -> dict[str, Any]:
    """Audit separately only after the stronger-ridge identity and full lock validate."""
    validate_locked_run(run_dir)
    result = audit_compact_run(run_dir, annotation_release_dir)
    result.update(
        {
            "schema_version": "mit-strong-ridge-aggregate-1",
            "study": "post_audit_exploratory_mit_strong_ridge_2026_11_19",
            "study_id": STUDY_ID,
            "ridge_alpha": RIDGE_ALPHA,
        }
    )
    result["method"]["ridge_alpha"] = RIDGE_ALPHA
    result["method"]["alpha_selection_scope"] = "existing_rq1_development_selection"
    return result


def render_strong_ridge_report(result: dict[str, Any]) -> str:
    """Use the existing aggregate tables with an explicitly new, accurate method header."""
    if result.get("study_id") != STUDY_ID or result.get("ridge_alpha") != RIDGE_ALPHA:
        raise ValueError("Cannot render another study as the stronger-ridge extension")
    report = render_compact_report(result)
    report = report.replace(
        "# Compact exploratory MIT follow-up", "# Stronger-ridge exploratory MIT comparison", 1
    )
    return report.replace(
        "fixed alpha-1 linear scorer",
        "fixed alpha-1000 linear scorer selected by the existing development procedure",
        1,
    )


def write_strong_ridge_audit(result: dict[str, Any], output: Path, report: Path) -> None:
    """Publish distinct aggregate-only outputs without replacing an existing delivery."""
    if output.resolve() == report.resolve() or output.exists() or report.exists():
        raise FileExistsError("Stronger-ridge audit outputs must be new, distinct paths")
    rendered = render_strong_ridge_report(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with report.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
