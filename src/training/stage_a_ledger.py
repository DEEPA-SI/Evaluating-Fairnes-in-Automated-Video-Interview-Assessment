"""Fail-closed progress ledger and immutable closure gate for Stage A."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from src.data import CORPUS_ID, parse_session_id
from src.frozen_json import freeze_json
from src.secure_io import AnchoredDirectory, AnchoredFile

from .crossfit import (
    DIAGNOSTIC_FLOOR_CONFIGURATIONS,
    ELIGIBILITY_SCHEMA_VERSION,
    FIREWALLED_LEARNED_CONFIGURATIONS,
    PREDICTION_LOCK_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    SESSION_MAPPING_PROVENANCE_FIELDS,
    TARGET_SCHEMA_VERSION,
    TARGET_TRANSFORM,
    EligibilityResult,
    PredictionBundle,
    RunProvenance,
    TargetSet,
    _parse_fold_manifest,
    canonical_json_bytes,
)
from .proxy_training import FROZEN_FINAL_SEEDS, FROZEN_OUTER_ROTATIONS

STAGE_A_LEDGER_SCHEMA_VERSION = "fair-avi-mit-stage-a-ledger-1"
STAGE_A_CLOSURE_SCHEMA_VERSION = "fair-avi-mit-stage-a-closure-2"
STAGE_A_CELL_COUNT = 105
STAGE_A_LEDGER_FILENAME = "stage-a-ledger.json"
STAGE_A_CLOSURE_FILENAME = "stage-a-closure.json"
STAGE_A_RECOVERY_CONTRACT = {
    "publication_unit": "configuration_seed_five_fold_bundle",
    "incomplete_bundle_recovery": "recompute_all_five_folds",
    "partial_bundle_cells_count_as_completed": False,
}

_LOCK_IDENTITY_FIELDS = (
    "schema_version",
    "corpus",
    "configuration",
    "seed",
    "execution_mode",
    "eligibility_manifest_sha256",
    "fold_manifest_sha256",
    "target_manifest_sha256",
    "session_mapping",
    "feature_contract_sha256",
    "preprocessing_sha256",
    "feature_run_contract_sha256",
    "feature_config_sha256",
    "feature_extraction_closure_file_sha256",
    "feature_extraction_closure_payload_sha256",
    "config_sha256",
    "code_commit_sha",
    "code_source_tree_sha256",
    "model_config_sha256",
    "artifacts",
)
_CONTEXT_FIELDS = {
    "session_mapping",
    "eligibility_manifest_sha256",
    "fold_manifest_sha256",
    "target_manifest_sha256",
    "feature_contract_sha256",
    "preprocessing_sha256",
    "feature_run_contract_sha256",
    "feature_config_sha256",
    "feature_extraction_closure_file_sha256",
    "feature_extraction_closure_payload_sha256",
    "config_sha256",
    "code_commit_sha",
    "code_source_tree_sha256",
    "source_date_epoch",
    "eligible_session_roster_sha256",
    "eligible_participant_count",
    "expected_oof_row_count",
}


@dataclass(frozen=True, slots=True)
class StageAExecutionContext:
    """Shared immutable context required by every Stage-A bundle."""

    payload: Mapping[str, Any]
    eligible_sessions: tuple[tuple[str, str, int], ...]
    eligible_participant_count: int
    _initial_payload_sha256: str = dataclass_field(init=False, repr=False)
    _initial_roster_sha256: str = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        frozen_payload = freeze_json(self.payload)
        frozen_sessions = tuple(tuple(row) for row in self.eligible_sessions)
        object.__setattr__(self, "payload", frozen_payload)
        object.__setattr__(self, "eligible_sessions", frozen_sessions)
        _validate_context_payload(self.payload)
        if not self.eligible_sessions or len(self.eligible_sessions) != 2 * (
            self.eligible_participant_count
        ):
            raise ValueError("Stage-A context must contain exactly two sessions per participant")
        if (
            self.payload["eligible_participant_count"] != self.eligible_participant_count
            or self.payload["expected_oof_row_count"] != len(self.eligible_sessions)
        ):
            raise ValueError("Stage-A context eligible counts drifted")
        sessions: set[str] = set()
        sources: set[str] = set()
        for session_id, source_session_id, fold in self.eligible_sessions:
            if (
                not isinstance(session_id, str)
                or not isinstance(source_session_id, str)
                or session_id in sessions
                or source_session_id in sources
                or isinstance(fold, bool)
                or fold not in FROZEN_OUTER_ROTATIONS
            ):
                raise ValueError("Stage-A eligible-session context is invalid")
            sessions.add(session_id)
            sources.add(source_session_id)
        roster_sha256 = _eligible_roster_sha256(self.eligible_sessions)
        if self.payload["eligible_session_roster_sha256"] != roster_sha256:
            raise ValueError("Stage-A eligible-session roster checksum drifted")
        object.__setattr__(self, "_initial_payload_sha256", _payload_sha256(self.payload))
        object.__setattr__(self, "_initial_roster_sha256", roster_sha256)

    @property
    def sha256(self) -> str:
        """Return the canonical shared-context payload checksum."""
        return _payload_sha256(self.payload)

    def verify_integrity(self) -> None:
        """Fail if retained context state changed after its initial validation."""
        _validate_context_payload(self.payload)
        if (
            self.sha256 != self._initial_payload_sha256
            or _eligible_roster_sha256(self.eligible_sessions)
            != self._initial_roster_sha256
            or self.payload["eligible_session_roster_sha256"]
            != self._initial_roster_sha256
            or self.payload["eligible_participant_count"]
            != self.eligible_participant_count
            or self.payload["expected_oof_row_count"] != len(self.eligible_sessions)
        ):
            raise RuntimeError("Stage-A execution context changed after validation")


def build_stage_a_execution_context(
    eligibility: EligibilityResult,
    targets: TargetSet,
    provenance: RunProvenance,
) -> StageAExecutionContext:
    """Bind mapping, support, feature, configuration and code provenance once."""
    if not eligibility.production_ready or not targets.production_ready:
        raise ValueError("Stage-A closure context requires research-ready support artefacts")
    if eligibility.manifest.get("session_mapping") != targets.manifest.get("session_mapping"):
        raise ValueError("Stage-A mapping context disagrees between eligibility and targets")
    eligible_sessions = tuple(
        (
            row.identity.session_id,
            row.identity.source_session_id,
            row.identity.fold,
        )
        for row in eligibility.features
    )
    if any(not isinstance(source, str) for _, source, _ in eligible_sessions):
        raise ValueError("Stage-A research context lacks source-session identities")
    typed_sessions = tuple(
        (session, source, fold)
        for session, source, fold in eligible_sessions
        if isinstance(source, str)
    )
    roster = [
        {"session_id": session, "source_session_id": source, "fixed_fold": fold}
        for session, source, fold in typed_sessions
    ]
    payload = {
        "session_mapping": eligibility.manifest["session_mapping"],
        "eligibility_manifest_sha256": eligibility.manifest_sha256,
        "fold_manifest_sha256": eligibility.fold_manifest_sha256,
        "target_manifest_sha256": targets.manifest_sha256,
        "feature_contract_sha256": eligibility.feature_contract_sha256,
        "preprocessing_sha256": eligibility.preprocessing_sha256,
        "feature_run_contract_sha256": eligibility.manifest["feature_run_contract_sha256"],
        "feature_config_sha256": eligibility.feature_config_sha256,
        "feature_extraction_closure_file_sha256": eligibility.manifest[
            "feature_extraction_closure_file_sha256"
        ],
        "feature_extraction_closure_payload_sha256": eligibility.manifest[
            "feature_extraction_closure_payload_sha256"
        ],
        "config_sha256": provenance.config_sha256,
        "code_commit_sha": provenance.code_commit_sha,
        "code_source_tree_sha256": provenance.code_source_tree_sha256,
        "source_date_epoch": provenance.source_date_epoch,
        "eligible_session_roster_sha256": _payload_sha256({"eligible_sessions": roster}),
        "eligible_participant_count": len(eligibility.participant_ids),
        "expected_oof_row_count": len(typed_sessions),
    }
    return StageAExecutionContext(
        payload=payload,
        eligible_sessions=typed_sessions,
        eligible_participant_count=len(eligibility.participant_ids),
    )


class StageARunLedger:
    """Rebuild progress from immutable bundles and publish a truthful 105-cell ledger."""

    def __init__(
        self,
        output_dir: str | Path | AnchoredDirectory,
        *,
        context: StageAExecutionContext,
        learned_configurations: Sequence[str],
        diagnostic_floor_configurations: Sequence[str],
        requested_seeds: Sequence[int],
    ) -> None:
        if isinstance(output_dir, AnchoredDirectory):
            output_dir.verify_path_identity()
            self._output_anchor = output_dir
            self._owns_output_anchor = False
        else:
            self._output_anchor = _open_stage_a_directory(output_dir, create=True)
            self._owns_output_anchor = True
        self.output_dir = self._output_anchor.path
        self.context = context
        self.context.verify_integrity()
        self._initial_context_payload_sha256 = context.sha256
        self._initial_context_roster_sha256 = _eligible_roster_sha256(
            context.eligible_sessions
        )
        self.learned_configurations = _unique_names(
            learned_configurations, "learned configurations"
        )
        self.diagnostic_floor_configurations = _unique_names(
            diagnostic_floor_configurations, "diagnostic-floor configurations"
        )
        if set(self.learned_configurations) & set(self.diagnostic_floor_configurations):
            raise ValueError("learned and diagnostic-floor configuration registries overlap")
        if self.learned_configurations != FIREWALLED_LEARNED_CONFIGURATIONS:
            raise ValueError("Stage-A learned configuration registry drifted")
        if self.diagnostic_floor_configurations != DIAGNOSTIC_FLOOR_CONFIGURATIONS:
            raise ValueError("Stage-A diagnostic-floor registry drifted")
        if tuple(requested_seeds) not in (
            (1,),
            (11,),
            (21,),
            (1, 11),
            (1, 21),
            (11, 21),
            tuple(FROZEN_FINAL_SEEDS),
        ):
            raise ValueError("Stage-A requested seeds must be a canonical frozen subset")
        self.requested_seeds = tuple(requested_seeds)
        expected_cells = (
            len(self.learned_configurations)
            * len(FROZEN_OUTER_ROTATIONS)
            * len(FROZEN_FINAL_SEEDS)
        )
        if expected_cells != STAGE_A_CELL_COUNT:
            raise ValueError("Stage-A learned registry must contain exactly 105 cells")
        self._cells = {
            _cell_key(configuration, fold, seed): _pending_cell(configuration, fold, seed)
            for configuration in self.learned_configurations
            for fold in FROZEN_OUTER_ROTATIONS
            for seed in FROZEN_FINAL_SEEDS
        }
        self._learned_bundles: dict[tuple[str, int], dict[str, Any]] = {}
        self._diagnostics = {
            configuration: {
                "configuration": configuration,
                "seed": None,
                "status": "pending",
                "bundle": None,
                "failure": None,
            }
            for configuration in self.diagnostic_floor_configurations
        }
        self._scan_existing()
        self.write_progress()

    def close(self) -> None:
        """Close an internally owned Stage-A directory capability."""
        if self._owns_output_anchor:
            self._output_anchor.close()

    def __enter__(self) -> StageARunLedger:
        self._require_initial_context()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_owns_output_anchor", False):
            self._output_anchor.close()

    def has_completed_bundle(self, configuration: str, seed: int | None) -> bool:
        """Return whether a fully checksum-validated bundle can be resumed unchanged."""
        self._require_initial_context()
        if seed is None:
            return self._diagnostics[configuration]["status"] == "completed"
        return all(
            self._cells[_cell_key(configuration, fold, seed)]["status"] == "completed"
            for fold in FROZEN_OUTER_ROTATIONS
        )

    def record_written_bundle(self, bundle: PredictionBundle) -> dict[str, Any]:
        """Re-read a newly published bundle and mark only verified cells completed."""
        self._require_initial_context()
        record = self._validate_disk_bundle(bundle.configuration, bundle.seed)
        self._require_initial_context()
        self._record_completed(record)
        self.write_progress()
        return _json_copy(record)

    def record_failure(
        self,
        configuration: str,
        seed: int | None,
        error: BaseException,
        *,
        code: str = "execution_failed",
    ) -> None:
        """Record a structured failure without changing any completed cell."""
        self._require_initial_context()
        failure = freeze_json(
            {
                "code": code,
                "exception_type": type(error).__name__,
                "message": str(error)[:1000],
            }
        )
        if seed is None:
            row = self._diagnostics[configuration]
            if row["status"] != "completed":
                row.update(status="failed", bundle=None, failure=failure)
        else:
            for fold in FROZEN_OUTER_ROTATIONS:
                row = self._cells[_cell_key(configuration, fold, seed)]
                if row["status"] != "completed":
                    row.update(status="failed", bundle=None, failure=failure)
        self.write_progress()

    def bundle_record(self, configuration: str, seed: int | None) -> dict[str, Any] | None:
        """Return one resumed/new bundle record for the run report."""
        self._require_initial_context()
        if seed is None:
            bundle = self._diagnostics[configuration]["bundle"]
            return _json_copy(bundle) if isinstance(bundle, Mapping) else None
        bundle = self._learned_bundles.get((configuration, seed))
        return _json_copy(bundle) if isinstance(bundle, Mapping) else None

    def write_progress(self) -> dict[str, Any]:
        """Atomically replace the reconstructible progress ledger."""
        self._require_initial_context()
        payload = self._ledger_payload()
        self._require_initial_context()
        self._output_anchor.replace_bytes(
            STAGE_A_LEDGER_FILENAME, canonical_json_bytes(payload), mode=0o600
        )
        self._require_initial_context()
        return _json_copy(payload)

    def finalize_if_complete(self) -> dict[str, Any] | None:
        """Publish the immutable closure only after all 105 cells and floors validate."""
        self._require_initial_context()
        if self.requested_seeds != tuple(FROZEN_FINAL_SEEDS):
            self.write_progress()
            return None
        if any(row["status"] != "completed" for row in self._cells.values()) or any(
            row["status"] != "completed" for row in self._diagnostics.values()
        ):
            self.write_progress()
            return None
        self._revalidate_complete_registry()
        self._require_initial_context()
        unsigned = {
            "schema_version": STAGE_A_CLOSURE_SCHEMA_VERSION,
            "state": "closed",
            "execution_mode": "research",
            "firewall_stage": "A",
            "protected_annotation_fields_accessed": [],
            "expected_learned_configuration_count": len(self.learned_configurations),
            "expected_learned_bundle_count": len(self.learned_configurations)
            * len(FROZEN_FINAL_SEEDS),
            "completed_learned_bundle_count": len(self.learned_configurations)
            * len(FROZEN_FINAL_SEEDS),
            "expected_learned_cell_count": STAGE_A_CELL_COUNT,
            "completed_learned_cell_count": STAGE_A_CELL_COUNT,
            "learned_configurations": list(self.learned_configurations),
            "frozen_seeds": list(FROZEN_FINAL_SEEDS),
            "frozen_outer_folds": list(FROZEN_OUTER_ROTATIONS),
            "recovery_contract": STAGE_A_RECOVERY_CONTRACT,
            "context": _json_copy(self.context.payload),
            "context_payload_sha256": self.context.sha256,
            "learned_bundles": [
                dict(self._learned_bundles[(configuration, seed)])
                for configuration in self.learned_configurations
                for seed in FROZEN_FINAL_SEEDS
            ],
            "diagnostic_floor_configurations": list(
                self.diagnostic_floor_configurations
            ),
            "expected_diagnostic_floor_count": len(
                self.diagnostic_floor_configurations
            ),
            "completed_diagnostic_floor_count": len(
                self.diagnostic_floor_configurations
            ),
            "diagnostic_floors": [
                dict(self._diagnostics[name]["bundle"])
                for name in self.diagnostic_floor_configurations
            ],
            "cells": [dict(row) for row in self._cells.values()],
        }
        self._require_initial_context()
        payload_sha256 = _payload_sha256(unsigned)
        closure = {**unsigned, "closure_payload_sha256": payload_sha256}
        self._require_initial_context()
        closure_file = self._output_anchor.publish_bytes(
            STAGE_A_CLOSURE_FILENAME,
            canonical_json_bytes(closure),
            mode=0o600,
        )
        validated = require_stage_b_release(
            self._output_anchor,
            closure_file_sha256=closure_file.sha256,
            closure_payload_sha256=payload_sha256,
        )
        self._require_initial_context()
        self.write_progress()
        return validated

    def report(self) -> dict[str, Any]:
        """Return the current canonical ledger payload."""
        self._require_initial_context()
        return _json_copy(self._ledger_payload())

    def _scan_existing(self) -> None:
        self._require_initial_context()
        for configuration in self.diagnostic_floor_configurations:
            try:
                record = self._validate_disk_bundle(configuration, None, absent_ok=True)
            except Exception as error:
                self.record_failure(
                    configuration, None, error, code="existing_bundle_invalid"
                )
            else:
                if record is not None:
                    self._record_completed(record)
        for configuration in self.learned_configurations:
            for seed in FROZEN_FINAL_SEEDS:
                try:
                    record = self._validate_disk_bundle(configuration, seed, absent_ok=True)
                except Exception as error:
                    self.record_failure(
                        configuration, seed, error, code="existing_bundle_invalid"
                    )
                else:
                    if record is not None:
                        self._record_completed(record)
        self._require_initial_context()

    def _record_completed(self, record: Mapping[str, Any]) -> None:
        self._require_initial_context()
        configuration = record["configuration"]
        seed = record["seed"]
        if seed is None:
            self._diagnostics[configuration].update(
                status="completed", bundle=freeze_json(record), failure=None
            )
            return
        fit_by_fold = {
            row["outer_fold"]: row for row in record["fit_artifacts"]
        }
        row_counts = record["oof_rows_per_fold"]
        self._learned_bundles[(configuration, seed)] = freeze_json(record)
        for fold in FROZEN_OUTER_ROTATIONS:
            cell_bundle = {
                "prediction_lock_filename": record["prediction_lock_filename"],
                "prediction_lock_file_sha256": record["prediction_lock_file_sha256"],
                "prediction_lock_payload_sha256": record[
                    "prediction_lock_payload_sha256"
                ],
                "prediction_lock_id": record["prediction_lock_id"],
                "prediction_filename": record["prediction_filename"],
                "prediction_file_sha256": record["prediction_file_sha256"],
                "fit_artifact": fit_by_fold[fold],
                "oof_row_count": row_counts[str(fold)],
            }
            self._cells[_cell_key(configuration, fold, seed)].update(
                status="completed", bundle=freeze_json(cell_bundle), failure=None
            )

    def _require_initial_context(self) -> None:
        self._output_anchor.verify_path_identity()
        self.context.verify_integrity()
        if (
            self.context.sha256 != self._initial_context_payload_sha256
            or _eligible_roster_sha256(self.context.eligible_sessions)
            != self._initial_context_roster_sha256
        ):
            raise RuntimeError("Stage-A ledger execution context changed after scanning")

    def _revalidate_complete_registry(self) -> None:
        self._require_initial_context()
        validated = [
            self._validate_disk_bundle(configuration, None)
            for configuration in self.diagnostic_floor_configurations
        ]
        validated.extend(
            self._validate_disk_bundle(configuration, seed)
            for configuration in self.learned_configurations
            for seed in FROZEN_FINAL_SEEDS
        )
        if any(record is None for record in validated):
            raise RuntimeError("Stage-A complete registry unexpectedly lost a bundle")
        self._require_initial_context()
        for record in validated:
            if record is not None:
                self._record_completed(record)
        self._require_initial_context()

    def _validate_disk_bundle(
        self,
        configuration: str,
        seed: int | None,
        *,
        absent_ok: bool = False,
    ) -> dict[str, Any] | None:
        self._require_initial_context()
        stem = configuration if seed is None else f"{configuration}.seed-{seed}"
        lock_filename = f"{stem}.prediction-lock.json"
        prediction_filename = f"{stem}.predictions.json"
        names = self._output_anchor.list_names()
        related = tuple(
            name
            for name in names
            if name.startswith(f"{stem}.")
            and name not in {STAGE_A_LEDGER_FILENAME, STAGE_A_CLOSURE_FILENAME}
        )
        if lock_filename not in names:
            if related:
                raise ValueError(f"{stem} has orphaned artefacts without a prediction lock")
            if absent_ok:
                self._require_initial_context()
                return None
            raise ValueError(f"{stem} prediction lock is unavailable")
        manifest, lock_file = _read_canonical_object(
            self._output_anchor, lock_filename, "prediction lock"
        )
        observed_files = {lock_filename: lock_file}
        _validate_signed_payload(manifest, "manifest_payload_sha256", "prediction lock")
        if (
            manifest.get("schema_version") != PREDICTION_LOCK_SCHEMA_VERSION
            or manifest.get("configuration") != configuration
            or manifest.get("seed") != seed
            or manifest.get("execution_mode") != "research"
            or manifest.get("state") != "locked"
            or manifest.get("firewall_stage") != "A"
            or manifest.get("protected_annotation_fields_accessed") != []
        ):
            raise ValueError(f"{stem} prediction lock identity drifted")
        self._validate_bundle_context(manifest, stem)
        if any(field not in manifest for field in _LOCK_IDENTITY_FIELDS):
            raise ValueError(f"{stem} prediction lock identity fields are incomplete")
        lock_identity = {field: manifest[field] for field in _LOCK_IDENTITY_FIELDS}
        if _payload_sha256(lock_identity) != manifest.get("prediction_lock_id"):
            raise ValueError(f"{stem} prediction lock ID checksum failed")

        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"{stem} prediction lock has no artefact inventory")
        artifacts_by_name: dict[str, Mapping[str, Any]] = {}
        for row in artifacts:
            if (
                not isinstance(row, Mapping)
                or set(row) != {"filename", "sha256", "bytes"}
                or not isinstance(row.get("filename"), str)
                or Path(row["filename"]).name != row["filename"]
                or row["filename"] in artifacts_by_name
            ):
                raise ValueError(f"{stem} artefact inventory is invalid")
            observed_files[row["filename"]] = _validate_file_record(
                self._output_anchor, row["filename"], row, f"{stem} artefact"
            )
            artifacts_by_name[row["filename"]] = row
        expected_related = {
            lock_filename,
            prediction_filename,
            *(
                filename
                for filename in artifacts_by_name
                if filename.startswith(f"{stem}.")
            ),
        }
        if set(related) != expected_related:
            raise ValueError(f"{stem} contains untracked or missing bundle artefacts")
        self._validate_support_artifacts(artifacts_by_name, stem)

        prediction_record = manifest.get("predictions")
        if (
            not isinstance(prediction_record, Mapping)
            or prediction_record.get("filename") != prediction_filename
        ):
            raise ValueError(f"{stem} prediction file record drifted")
        prediction_file = _validate_file_record(
            self._output_anchor,
            prediction_filename,
            prediction_record,
            f"{stem} predictions",
        )
        observed_files[prediction_filename] = prediction_file
        prediction_payload = _canonical_object_from_bytes(
            prediction_file.content, "predictions"
        )
        rows = prediction_payload.get("rows")
        expected_rows = 2 * self.context.eligible_participant_count
        if (
            prediction_payload.get("schema_version") != PREDICTION_SCHEMA_VERSION
            or prediction_payload.get("prediction_lock_id")
            != manifest.get("prediction_lock_id")
            or not isinstance(rows, list)
            or len(rows) != expected_rows
            or prediction_record.get("row_count") != expected_rows
            or manifest.get("expected_oof_session_count") != expected_rows
            or manifest.get("observed_oof_session_count") != expected_rows
        ):
            raise ValueError(f"{stem} must contain exactly 2E OOF rows")
        fold_counts = self._validate_oof_rows(rows, manifest, configuration, seed, stem)

        rotations = manifest.get("rotations")
        if not isinstance(rotations, list) or len(rotations) != len(FROZEN_OUTER_ROTATIONS):
            raise ValueError(f"{stem} must contain five ordered rotations")
        fit_artifacts: list[dict[str, Any]] = []
        for fold, rotation in zip(FROZEN_OUTER_ROTATIONS, rotations, strict=True):
            fit = rotation.get("fit_artifact") if isinstance(rotation, Mapping) else None
            if (
                not isinstance(rotation, Mapping)
                or rotation.get("outer_rotation") != fold
                or not isinstance(fit, Mapping)
                or set(fit) != {"filename", "sha256"}
                or artifacts_by_name.get(fit.get("filename"), {}).get("sha256")
                != fit.get("sha256")
            ):
                raise ValueError(f"{stem} rotation fit provenance drifted")
            fit_artifacts.append(
                {
                    "outer_fold": fold,
                    "filename": fit["filename"],
                    "sha256": fit["sha256"],
                }
            )
        result = {
            "configuration": configuration,
            "seed": seed,
            "prediction_lock_filename": lock_filename,
            "prediction_lock_file_sha256": lock_file.sha256,
            "prediction_lock_payload_sha256": manifest["manifest_payload_sha256"],
            "prediction_lock_id": manifest["prediction_lock_id"],
            "prediction_filename": prediction_filename,
            "prediction_file_sha256": prediction_record["sha256"],
            "oof_row_count": expected_rows,
            "oof_rows_per_fold": {str(fold): fold_counts[fold] for fold in FROZEN_OUTER_ROTATIONS},
            "fit_artifacts": fit_artifacts,
        }
        for filename, observed in observed_files.items():
            self._output_anchor.verify_file(
                filename, observed, expected_mode=0o600
            )
        self._require_initial_context()
        return result

    def _validate_support_artifacts(
        self, artifacts_by_name: Mapping[str, Mapping[str, Any]], stem: str
    ) -> None:
        required = {
            "complete-pair-eligibility.json",
            "participant-folds.csv",
            "target-manifest.json",
        }
        if not required.issubset(artifacts_by_name):
            raise ValueError(f"{stem} lacks required Stage-A support artefacts")
        _validate_stage_a_support_files(self._output_anchor, self.context.payload)

    def _validate_bundle_context(self, manifest: Mapping[str, Any], stem: str) -> None:
        expected = self.context.payload
        fields = (
            "session_mapping",
            "eligibility_manifest_sha256",
            "fold_manifest_sha256",
            "target_manifest_sha256",
            "feature_contract_sha256",
            "preprocessing_sha256",
            "feature_run_contract_sha256",
            "feature_config_sha256",
            "feature_extraction_closure_file_sha256",
            "feature_extraction_closure_payload_sha256",
            "config_sha256",
            "code_commit_sha",
            "code_source_tree_sha256",
            "source_date_epoch",
        )
        if any(manifest.get(field) != expected[field] for field in fields):
            raise ValueError(f"{stem} does not match the shared Stage-A context")

    def _validate_oof_rows(
        self,
        rows: Sequence[Any],
        manifest: Mapping[str, Any],
        configuration: str,
        seed: int | None,
        stem: str,
    ) -> dict[int, int]:
        expected = {
            session_id: (source_session_id, fold)
            for session_id, source_session_id, fold in self.context.eligible_sessions
        }
        seen: set[str] = set()
        fold_counts = {fold: 0 for fold in FROZEN_OUTER_ROTATIONS}
        context = self.context.payload
        for row in rows:
            session_id = row.get("session_id") if isinstance(row, Mapping) else None
            identity = expected.get(session_id)
            if (
                not isinstance(row, Mapping)
                or identity is None
                or session_id in seen
                or row.get("source_session_id") != identity[0]
                or row.get("fixed_fold") != identity[1]
                or row.get("outer_rotation") != identity[1]
                or row.get("rotation_role") != "test"
                or row.get("configuration") != configuration
                or row.get("seed") != seed
                or row.get("prediction_lock_id") != manifest.get("prediction_lock_id")
                or row.get("session_mapping") != context["session_mapping"]
                or row.get("eligibility_manifest_sha256")
                != context["eligibility_manifest_sha256"]
                or row.get("target_manifest_sha256") != context["target_manifest_sha256"]
                or row.get("feature_contract_sha256") != context["feature_contract_sha256"]
                or row.get("preprocessing_sha256") != context["preprocessing_sha256"]
            ):
                raise ValueError(f"{stem} OOF row coverage or shared context drifted")
            seen.add(session_id)
            fold_counts[identity[1]] += 1
        if seen != set(expected):
            raise ValueError(f"{stem} has duplicate or missing OOF session rows")
        return fold_counts

    def _ledger_payload(self) -> dict[str, Any]:
        self._require_initial_context()
        cells = [_json_copy(row) for row in self._cells.values()]
        diagnostics = [
            _json_copy(self._diagnostics[name])
            for name in self.diagnostic_floor_configurations
        ]
        completed = sum(row["status"] == "completed" for row in cells)
        failed = sum(row["status"] == "failed" for row in cells)
        pending = STAGE_A_CELL_COUNT - completed - failed
        diagnostic_completed = sum(row["status"] == "completed" for row in diagnostics)
        diagnostic_failed = sum(row["status"] == "failed" for row in diagnostics)
        unsigned = {
            "schema_version": STAGE_A_LEDGER_SCHEMA_VERSION,
            "status": "partial",
            "execution_mode": "research",
            "firewall_stage": "A",
            "protected_annotation_fields_accessed": [],
            "expected_learned_cell_count": STAGE_A_CELL_COUNT,
            "expected_learned_bundle_count": len(self.learned_configurations)
            * len(FROZEN_FINAL_SEEDS),
            "completed_learned_bundle_count": completed // len(FROZEN_OUTER_ROTATIONS),
            "failed_learned_bundle_count": failed // len(FROZEN_OUTER_ROTATIONS),
            "pending_learned_bundle_count": pending // len(FROZEN_OUTER_ROTATIONS),
            "completed_learned_cell_count": completed,
            "failed_learned_cell_count": failed,
            "pending_learned_cell_count": pending,
            "requested_seeds": list(self.requested_seeds),
            "learned_configurations": list(self.learned_configurations),
            "frozen_seeds": list(FROZEN_FINAL_SEEDS),
            "frozen_outer_folds": list(FROZEN_OUTER_ROTATIONS),
            "recovery_contract": STAGE_A_RECOVERY_CONTRACT,
            "context": _json_copy(self.context.payload),
            "context_payload_sha256": self.context.sha256,
            "diagnostic_floor_expected_count": len(self.diagnostic_floor_configurations),
            "diagnostic_floor_completed_count": diagnostic_completed,
            "diagnostic_floor_failed_count": diagnostic_failed,
            "diagnostic_floor_pending_count": len(diagnostics)
            - diagnostic_completed
            - diagnostic_failed,
            "diagnostic_floors": diagnostics,
            "cells": cells,
        }
        if (
            self.requested_seeds == tuple(FROZEN_FINAL_SEEDS)
            and completed == STAGE_A_CELL_COUNT
            and diagnostic_completed == len(self.diagnostic_floor_configurations)
        ):
            unsigned["status"] = "complete"
        result = {**unsigned, "ledger_payload_sha256": _payload_sha256(unsigned)}
        self._require_initial_context()
        return result


def require_stage_b_release(
    output_dir: str | Path | AnchoredDirectory,
    *,
    closure_file_sha256: str,
    closure_payload_sha256: str,
) -> dict[str, Any]:
    """Mechanically open Stage B only for an exact immutable Stage-A closure."""
    if isinstance(output_dir, AnchoredDirectory):
        output_dir.verify_path_identity()
        return _require_stage_b_release_anchored(
            output_dir,
            closure_file_sha256=closure_file_sha256,
            closure_payload_sha256=closure_payload_sha256,
        )
    with _open_stage_a_directory(output_dir, create=False) as output:
        return _require_stage_b_release_anchored(
            output,
            closure_file_sha256=closure_file_sha256,
            closure_payload_sha256=closure_payload_sha256,
        )


def _require_stage_b_release_anchored(
    output: AnchoredDirectory,
    *,
    closure_file_sha256: str,
    closure_payload_sha256: str,
) -> dict[str, Any]:
    try:
        closure, closure_file = _read_canonical_object(
            output, STAGE_A_CLOSURE_FILENAME, "Stage-A closure"
        )
    except ValueError as error:
        raise ValueError("Stage B requires the immutable Stage-A closure file") from error
    if closure_file.sha256 != closure_file_sha256:
        raise ValueError("Stage-A closure file checksum failed")
    _validate_signed_payload(closure, "closure_payload_sha256", "Stage-A closure")
    if (
        closure.get("schema_version") != STAGE_A_CLOSURE_SCHEMA_VERSION
        or closure.get("state") != "closed"
        or closure.get("execution_mode") != "research"
        or closure.get("firewall_stage") != "A"
        or closure.get("protected_annotation_fields_accessed") != []
        or closure.get("expected_learned_configuration_count") != 7
        or closure.get("expected_learned_bundle_count") != 21
        or closure.get("completed_learned_bundle_count") != 21
        or closure.get("expected_learned_cell_count") != STAGE_A_CELL_COUNT
        or closure.get("completed_learned_cell_count") != STAGE_A_CELL_COUNT
        or closure.get("recovery_contract") != STAGE_A_RECOVERY_CONTRACT
        or closure.get("closure_payload_sha256") != closure_payload_sha256
    ):
        raise ValueError("Stage-A closure does not authorize Stage B")
    cells = closure.get("cells")
    if (
        not isinstance(cells, list)
        or len(cells) != STAGE_A_CELL_COUNT
        or any(not isinstance(row, Mapping) or row.get("status") != "completed" for row in cells)
    ):
        raise ValueError("Stage-A closure does not contain 105 completed cells")
    diagnostics = closure.get("diagnostic_floors")
    diagnostic_configurations = closure.get("diagnostic_floor_configurations")
    expected_configurations = closure.get("learned_configurations")
    expected_seeds = closure.get("frozen_seeds")
    expected_folds = closure.get("frozen_outer_folds")
    if (
        expected_configurations != list(FIREWALLED_LEARNED_CONFIGURATIONS)
        or expected_seeds != list(FROZEN_FINAL_SEEDS)
        or expected_folds != list(FROZEN_OUTER_ROTATIONS)
    ):
        raise ValueError("Stage-A closure registry drifted")
    observed_cell_keys = {
        (row.get("configuration"), row.get("outer_fold"), row.get("seed"))
        for row in cells
    }
    expected_cell_keys = {
        (configuration, fold, seed)
        for configuration in expected_configurations
        for fold in FROZEN_OUTER_ROTATIONS
        for seed in FROZEN_FINAL_SEEDS
    }
    if observed_cell_keys != expected_cell_keys:
        raise ValueError("Stage-A closure cell registry is incomplete or duplicated")
    context = closure.get("context")
    if (
        not isinstance(context, Mapping)
        or closure.get("context_payload_sha256") != _payload_sha256(context)
    ):
        raise ValueError("Stage-A closure shared context checksum failed")
    _validate_context_payload(context)
    expected_sessions = _validate_stage_a_support_files(output, context)
    learned_bundles = closure.get("learned_bundles")
    if (
        not isinstance(learned_bundles, list)
        or len(learned_bundles) != 21
        or {
            (row.get("configuration"), row.get("seed"))
            for row in learned_bundles
            if isinstance(row, Mapping)
        }
        != {
            (configuration, seed)
            for configuration in expected_configurations
            for seed in FROZEN_FINAL_SEEDS
        }
        or any(
            not isinstance(row, Mapping)
            or row.get("oof_row_count") != context.get("expected_oof_row_count")
            or set(row.get("oof_rows_per_fold", {}))
            != {str(fold) for fold in FROZEN_OUTER_ROTATIONS}
            or len(row.get("fit_artifacts", [])) != 5
            or not isinstance(row.get("prediction_lock_filename"), str)
            or not isinstance(row.get("prediction_lock_file_sha256"), str)
            or not isinstance(row.get("prediction_lock_payload_sha256"), str)
            or not isinstance(row.get("prediction_lock_id"), str)
            or not isinstance(row.get("prediction_filename"), str)
            or not isinstance(row.get("prediction_file_sha256"), str)
            for row in learned_bundles
        )
    ):
        raise ValueError("Stage-A closure learned-bundle registry is invalid")
    learned_by_key = {
        (row["configuration"], row["seed"]): row for row in learned_bundles
    }
    for bundle in learned_bundles:
        _validate_closure_bundle_files(output, bundle, context, expected_sessions)
    for cell in cells:
        learned = learned_by_key[(cell["configuration"], cell["seed"])]
        cell_bundle = cell.get("bundle")
        fold = cell["outer_fold"]
        fit_by_fold = {
            row["outer_fold"]: row for row in learned["fit_artifacts"]
        }
        if (
            not isinstance(cell_bundle, Mapping)
            or cell_bundle.get("prediction_lock_filename")
            != learned["prediction_lock_filename"]
            or cell_bundle.get("prediction_lock_file_sha256")
            != learned["prediction_lock_file_sha256"]
            or cell_bundle.get("prediction_lock_payload_sha256")
            != learned["prediction_lock_payload_sha256"]
            or cell_bundle.get("prediction_lock_id") != learned["prediction_lock_id"]
            or cell_bundle.get("prediction_filename") != learned["prediction_filename"]
            or cell_bundle.get("prediction_file_sha256")
            != learned["prediction_file_sha256"]
            or cell_bundle.get("fit_artifact") != fit_by_fold.get(fold)
            or cell_bundle.get("oof_row_count")
            != learned["oof_rows_per_fold"].get(str(fold))
        ):
            raise ValueError("Stage-A closure cell-to-bundle binding is invalid")
    if (
        not isinstance(diagnostics, list)
        or not diagnostics
        or diagnostic_configurations != list(DIAGNOSTIC_FLOOR_CONFIGURATIONS)
        or closure.get("expected_diagnostic_floor_count")
        != len(diagnostic_configurations)
        or closure.get("completed_diagnostic_floor_count")
        != len(diagnostic_configurations)
        or len(diagnostics) != len(diagnostic_configurations)
        or any(
            not isinstance(row, Mapping)
            or row.get("seed") is not None
            or not isinstance(row.get("configuration"), str)
            for row in diagnostics
        )
        or len({row["configuration"] for row in diagnostics}) != len(diagnostics)
        or {row["configuration"] for row in diagnostics}
        != set(diagnostic_configurations)
    ):
        raise ValueError("Stage-A closure lacks validated diagnostic-floor provenance")
    for bundle in diagnostics:
        _validate_closure_bundle_files(output, bundle, context, expected_sessions)
    output.verify_file(
        STAGE_A_CLOSURE_FILENAME, closure_file, expected_mode=0o600
    )
    output.verify_path_identity()
    return closure


def _validate_closure_bundle_files(
    output: AnchoredDirectory,
    bundle: Mapping[str, Any],
    context: Mapping[str, Any],
    expected_sessions: Mapping[str, tuple[str, int]],
) -> None:
    configuration = bundle.get("configuration")
    seed = bundle.get("seed")
    lock_filename = bundle.get("prediction_lock_filename")
    prediction_filename = bundle.get("prediction_filename")
    if (
        not isinstance(configuration, str)
        or not isinstance(lock_filename, str)
        or Path(lock_filename).name != lock_filename
        or not isinstance(prediction_filename, str)
        or Path(prediction_filename).name != prediction_filename
    ):
        raise ValueError("Stage-A closure bundle filenames are unsafe")
    lock_file = output.read_bytes(lock_filename, expected_mode=0o600)
    prediction_file = output.read_bytes(prediction_filename, expected_mode=0o600)
    observed_files = {
        lock_filename: lock_file,
        prediction_filename: prediction_file,
    }
    if (
        lock_file.sha256 != bundle.get("prediction_lock_file_sha256")
        or prediction_file.sha256 != bundle.get("prediction_file_sha256")
    ):
        raise ValueError("Stage-A closure bundle file checksum failed")
    lock = _canonical_object_from_bytes(
        lock_file.content, "Stage-A closure prediction lock"
    )
    prediction = _canonical_object_from_bytes(
        prediction_file.content, "Stage-A closure predictions"
    )
    _validate_signed_payload(lock, "manifest_payload_sha256", "prediction lock")
    prediction_record = lock.get("predictions")
    context_fields = (
        "session_mapping",
        "eligibility_manifest_sha256",
        "fold_manifest_sha256",
        "target_manifest_sha256",
        "feature_contract_sha256",
        "preprocessing_sha256",
        "feature_run_contract_sha256",
        "feature_config_sha256",
        "feature_extraction_closure_file_sha256",
        "feature_extraction_closure_payload_sha256",
        "config_sha256",
        "code_commit_sha",
        "code_source_tree_sha256",
        "source_date_epoch",
    )
    if (
        lock.get("schema_version") != PREDICTION_LOCK_SCHEMA_VERSION
        or lock.get("configuration") != configuration
        or lock.get("seed") != seed
        or lock.get("execution_mode") != "research"
        or lock.get("state") != "locked"
        or lock.get("firewall_stage") != "A"
        or lock.get("protected_annotation_fields_accessed") != []
        or lock.get("manifest_payload_sha256")
        != bundle.get("prediction_lock_payload_sha256")
        or lock.get("prediction_lock_id") != bundle.get("prediction_lock_id")
        or not isinstance(prediction_record, Mapping)
        or prediction_record.get("filename") != prediction_filename
        or prediction_record.get("sha256")
        != bundle.get("prediction_file_sha256")
        or prediction_record.get("row_count") != bundle.get("oof_row_count")
        or any(field not in context for field in context_fields)
        or any(
            lock.get(field) != context[field]
            for field in context_fields
        )
    ):
        raise ValueError("Stage-A closure bundle lock or context binding failed")
    if any(field not in lock for field in _LOCK_IDENTITY_FIELDS):
        raise ValueError("Stage-A closure prediction-lock identity is incomplete")
    if _payload_sha256({field: lock[field] for field in _LOCK_IDENTITY_FIELDS}) != lock.get(
        "prediction_lock_id"
    ):
        raise ValueError("Stage-A closure prediction-lock identity checksum failed")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Stage-A closure prediction lock has no artefact inventory")
    artifacts_by_name: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        filename = artifact.get("filename") if isinstance(artifact, Mapping) else None
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {"filename", "sha256", "bytes"}
            or not isinstance(filename, str)
            or Path(filename).name != filename
            or filename in artifacts_by_name
        ):
            raise ValueError("Stage-A closure artefact inventory is invalid")
        observed_files[filename] = _validate_file_record(
            output, filename, artifact, "Stage-A closure artefact"
        )
        artifacts_by_name[filename] = artifact
    for required in (
        "complete-pair-eligibility.json",
        "participant-folds.csv",
        "target-manifest.json",
    ):
        if required not in artifacts_by_name:
            raise ValueError("Stage-A closure bundle lacks support evidence")
    rows = prediction.get("rows")
    if (
        prediction.get("schema_version") != PREDICTION_SCHEMA_VERSION
        or
        prediction.get("prediction_lock_id") != bundle.get("prediction_lock_id")
        or not isinstance(rows, list)
        or len(rows) != bundle.get("oof_row_count")
    ):
        raise ValueError("Stage-A closure prediction rows failed validation")
    _validate_release_oof_rows(
        rows,
        lock=lock,
        configuration=configuration,
        seed=seed,
        context=context,
        expected_sessions=expected_sessions,
    )
    fits = bundle.get("fit_artifacts")
    if not isinstance(fits, list) or len(fits) != len(FROZEN_OUTER_ROTATIONS):
        raise ValueError("Stage-A closure fit-artifact registry is incomplete")
    for fold, fit in zip(FROZEN_OUTER_ROTATIONS, fits, strict=True):
        filename = fit.get("filename") if isinstance(fit, Mapping) else None
        if (
            not isinstance(fit, Mapping)
            or fit.get("outer_fold") != fold
            or not isinstance(filename, str)
            or Path(filename).name != filename
        ):
            raise ValueError("Stage-A closure fit-artifact binding failed")
        fit_file = output.read_bytes(filename, expected_mode=0o600)
        if fit_file.sha256 != fit.get("sha256"):
            raise ValueError("Stage-A closure fit-artifact binding failed")
        observed_files[filename] = fit_file
    for filename, observed in observed_files.items():
        output.verify_file(filename, observed, expected_mode=0o600)


def _validate_stage_a_support_files(
    output: AnchoredDirectory, context: Mapping[str, Any]
) -> dict[str, tuple[str, int]]:
    """Reopen and semantically bind the exact eligibility, folds and targets."""
    eligibility, eligibility_file = _read_canonical_object(
        output, "complete-pair-eligibility.json", "eligibility manifest"
    )
    target, target_file = _read_canonical_object(
        output, "target-manifest.json", "target manifest"
    )
    fold_file = output.read_bytes("participant-folds.csv", expected_mode=0o600)
    fold_bytes = fold_file.content
    _validate_signed_payload(
        eligibility, "manifest_payload_sha256", "eligibility manifest"
    )
    _validate_signed_payload(target, "manifest_payload_sha256", "target manifest")
    fold_by_participant = _parse_fold_manifest(fold_bytes)

    if (
        eligibility.get("schema_version") != ELIGIBILITY_SCHEMA_VERSION
        or eligibility.get("corpus") != CORPUS_ID
        or eligibility.get("construction_mode") != "full_corpus"
        or eligibility.get("outcome_fields_accessed") != []
        or eligibility.get("protected_annotation_fields_accessed") != []
        or eligibility.get("manifest_payload_sha256")
        != context["eligibility_manifest_sha256"]
        or eligibility.get("fold_manifest_sha256") != context["fold_manifest_sha256"]
        or hashlib.sha256(fold_bytes).hexdigest() != context["fold_manifest_sha256"]
        or eligibility.get("session_mapping") != context["session_mapping"]
        or eligibility.get("feature_contract_sha256")
        != context["feature_contract_sha256"]
        or eligibility.get("preprocessing_sha256") != context["preprocessing_sha256"]
        or eligibility.get("feature_run_contract_sha256")
        != context["feature_run_contract_sha256"]
        or eligibility.get("feature_config_sha256") != context["feature_config_sha256"]
        or eligibility.get("feature_extraction_closure_file_sha256")
        != context["feature_extraction_closure_file_sha256"]
        or eligibility.get("feature_extraction_closure_payload_sha256")
        != context["feature_extraction_closure_payload_sha256"]
    ):
        raise ValueError("Stage-A eligibility/fold support disagrees with closure context")

    input_rows = eligibility.get("input_sessions")
    if not isinstance(input_rows, list) or not input_rows:
        raise ValueError("Stage-A eligibility lacks its canonical input roster")
    input_by_session: dict[str, tuple[str, str, int]] = {}
    source_sessions: set[str] = set()
    for row in input_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "session_id",
            "source_session_id",
            "participant_id",
            "condition",
            "fold",
        }:
            raise ValueError("Stage-A eligibility input-session schema drifted")
        session_id = row["session_id"]
        source_session_id = row["source_session_id"]
        if not isinstance(session_id, str) or not isinstance(source_session_id, str):
            raise ValueError("Stage-A eligibility session identities are invalid")
        canonical, participant_id, condition = parse_session_id(session_id)
        source_canonical, _, source_condition = parse_session_id(source_session_id)
        fold = row["fold"]
        if (
            canonical != session_id
            or source_canonical != source_session_id
            or source_condition != condition
            or row["participant_id"] != participant_id
            or row["condition"] != condition
            or fold_by_participant.get(participant_id) != fold
            or session_id in input_by_session
            or source_session_id in source_sessions
        ):
            raise ValueError("Stage-A eligibility input roster is not a phase-preserving bijection")
        input_by_session[session_id] = (source_session_id, participant_id, fold)
        source_sessions.add(source_session_id)
    if (
        eligibility.get("input_participant_count") != len(fold_by_participant)
        or eligibility.get("input_session_count") != len(input_by_session)
        or len(input_by_session) != 2 * len(fold_by_participant)
    ):
        raise ValueError("Stage-A eligibility input-roster counts drifted")

    participant_rows = eligibility.get("eligible_participants")
    if not isinstance(participant_rows, list) or not participant_rows:
        raise ValueError("Stage-A eligibility has no complete participant pairs")
    eligible_sessions: dict[str, tuple[str, int]] = {}
    eligible_roster: list[tuple[str, str, int]] = []
    eligible_participants: set[str] = set()
    observed_fold_counts = {str(fold): 0 for fold in FROZEN_OUTER_ROTATIONS}
    for participant in participant_rows:
        if not isinstance(participant, Mapping):
            raise ValueError("Stage-A eligible-participant row is invalid")
        participant_id = participant.get("participant_id")
        fold = participant.get("fold")
        sessions = participant.get("sessions")
        if (
            not isinstance(participant_id, str)
            or participant_id in eligible_participants
            or fold_by_participant.get(participant_id) != fold
            or not isinstance(sessions, list)
            or len(sessions) != 2
        ):
            raise ValueError("Stage-A eligible participant/fold pairing drifted")
        eligible_participants.add(participant_id)
        observed_fold_counts[str(fold)] += 1
        observed_conditions: set[str] = set()
        for session in sessions:
            if not isinstance(session, Mapping):
                raise ValueError("Stage-A eligible session row is invalid")
            session_id = session.get("session_id")
            source_session_id = session.get("source_session_id")
            condition = session.get("condition")
            identity = input_by_session.get(session_id)
            if (
                not isinstance(session_id, str)
                or not isinstance(source_session_id, str)
                or identity != (source_session_id, participant_id, fold)
                or condition not in {"pre", "post"}
                or condition in observed_conditions
                or session_id in eligible_sessions
            ):
                raise ValueError("Stage-A eligible session identities drifted")
            observed_conditions.add(condition)
            eligible_sessions[session_id] = (source_session_id, fold)
            eligible_roster.append((session_id, source_session_id, fold))
        if observed_conditions != {"pre", "post"}:
            raise ValueError("Stage-A eligibility does not contain complete pre/post pairs")
    if (
        eligibility.get("eligible_participant_count") != len(eligible_participants)
        or eligibility.get("eligible_session_count") != len(eligible_sessions)
        or eligibility.get("eligible_participants_per_fold") != observed_fold_counts
        or len(eligible_sessions) != 2 * len(eligible_participants)
        or len(eligible_participants) != context["eligible_participant_count"]
        or len(eligible_sessions) != context["expected_oof_row_count"]
        or _eligible_roster_sha256(tuple(eligible_roster))
        != context["eligible_session_roster_sha256"]
    ):
        raise ValueError("Stage-A eligible roster or counts disagree with closure context")
    exclusions = eligibility.get("exclusions")
    if not isinstance(exclusions, list):
        raise ValueError("Stage-A eligibility exclusion registry is invalid")
    excluded_participants = {
        row.get("participant_id")
        for row in exclusions
        if isinstance(row, Mapping) and isinstance(row.get("participant_id"), str)
    }
    if (
        len(excluded_participants) != len(exclusions)
        or excluded_participants & eligible_participants
        or excluded_participants | eligible_participants != set(fold_by_participant)
    ):
        raise ValueError("Stage-A eligibility participant disposition is incomplete")

    target_rows = target.get("rows")
    eligibility_lock = target.get("eligibility_lock")
    source = target.get("source")
    if (
        target.get("schema_version") != TARGET_SCHEMA_VERSION
        or target.get("corpus") != CORPUS_ID
        or target.get("construction_mode") != "verified_mit_aggregate_rating_table"
        or target.get("construct") != "crowd_perceived_hireability_not_actual_hiring"
        or target.get("raw_scale") != [1.0, 7.0]
        or target.get("transform") != TARGET_TRANSFORM
        or target.get("protected_annotation_fields_accessed") != []
        or target.get("session_mapping") != context["session_mapping"]
        or target.get("manifest_payload_sha256") != context["target_manifest_sha256"]
        or not isinstance(source, Mapping)
        or source.get("reference") != "Labels/turker_scores_full_interview.csv"
        or source.get("worker") != "AGGR"
        or source.get("field") != "RecommendHiring"
        or not _is_sha256(source.get("sha256"))
        or not isinstance(eligibility_lock, Mapping)
        or eligibility_lock.get("eligibility_manifest_sha256")
        != context["eligibility_manifest_sha256"]
        or eligibility_lock.get("fold_file_sha256") != context["fold_manifest_sha256"]
        or not isinstance(target_rows, list)
        or len(target_rows) != len(eligible_sessions)
    ):
        raise ValueError("Stage-A target support disagrees with closure context")
    observed_target_roster: list[tuple[str, str, int]] = []
    seen_targets: set[str] = set()
    for row in target_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "session_id",
            "source_session_id",
            "recommend_hiring_raw",
            "target",
        }:
            raise ValueError("Stage-A target row schema drifted")
        session_id = row["session_id"]
        identity = eligible_sessions.get(session_id)
        raw = row["recommend_hiring_raw"]
        normalised = row["target"]
        if (
            not isinstance(session_id, str)
            or identity is None
            or session_id in seen_targets
            or row["source_session_id"] != identity[0]
            or isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or isinstance(normalised, bool)
            or not isinstance(normalised, (int, float))
            or not math.isfinite(float(raw))
            or not 1.0 <= float(raw) <= 7.0
            or not math.isclose(
                float(normalised), (float(raw) - 1.0) / 6.0, rel_tol=0.0, abs_tol=0.0
            )
        ):
            raise ValueError("Stage-A target identity or normalisation drifted")
        seen_targets.add(session_id)
        observed_target_roster.append((session_id, identity[0], identity[1]))
    if (
        seen_targets != set(eligible_sessions)
        or _eligible_roster_sha256(tuple(observed_target_roster))
        != context["eligible_session_roster_sha256"]
    ):
        raise ValueError("Stage-A target roster disagrees with eligibility")
    output.verify_file(
        "complete-pair-eligibility.json", eligibility_file, expected_mode=0o600
    )
    output.verify_file("participant-folds.csv", fold_file, expected_mode=0o600)
    output.verify_file("target-manifest.json", target_file, expected_mode=0o600)
    return eligible_sessions


def _validate_release_oof_rows(
    rows: Sequence[Any],
    *,
    lock: Mapping[str, Any],
    configuration: str,
    seed: int | None,
    context: Mapping[str, Any],
    expected_sessions: Mapping[str, tuple[str, int]],
) -> None:
    seen: set[str] = set()
    for row in rows:
        session_id = row.get("session_id") if isinstance(row, Mapping) else None
        identity = expected_sessions.get(session_id)
        if (
            not isinstance(row, Mapping)
            or identity is None
            or session_id in seen
            or row.get("source_session_id") != identity[0]
            or row.get("fixed_fold") != identity[1]
            or row.get("outer_rotation") != identity[1]
            or row.get("rotation_role") != "test"
            or row.get("configuration") != configuration
            or row.get("seed") != seed
            or row.get("prediction_lock_id") != lock.get("prediction_lock_id")
            or row.get("session_mapping") != context["session_mapping"]
            or row.get("eligibility_manifest_sha256")
            != context["eligibility_manifest_sha256"]
            or row.get("target_manifest_sha256") != context["target_manifest_sha256"]
            or row.get("feature_contract_sha256") != context["feature_contract_sha256"]
            or row.get("preprocessing_sha256") != context["preprocessing_sha256"]
        ):
            raise ValueError("Stage-A closure OOF row context or coverage drifted")
        seen.add(session_id)
    if seen != set(expected_sessions):
        raise ValueError("Stage-A closure OOF rows are duplicated or incomplete")


def _pending_cell(configuration: str, fold: int, seed: int) -> dict[str, Any]:
    return {
        "configuration": configuration,
        "outer_fold": fold,
        "seed": seed,
        "status": "pending",
        "bundle": None,
        "failure": None,
    }


def _validate_context_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _CONTEXT_FIELDS:
        raise ValueError("Stage-A shared context schema drifted")
    mapping = payload["session_mapping"]
    if (
        not isinstance(mapping, Mapping)
        or set(mapping) != set(SESSION_MAPPING_PROVENANCE_FIELDS)
        or not isinstance(mapping.get("schema_version"), str)
        or not mapping["schema_version"]
        or any(
            not _is_sha256(mapping.get(field))
            for field in SESSION_MAPPING_PROVENANCE_FIELDS
            if field != "schema_version"
        )
    ):
        raise ValueError("Stage-A shared mapping provenance drifted")
    sha256_fields = {
        "eligibility_manifest_sha256",
        "fold_manifest_sha256",
        "target_manifest_sha256",
        "feature_contract_sha256",
        "preprocessing_sha256",
        "feature_run_contract_sha256",
        "feature_config_sha256",
        "feature_extraction_closure_file_sha256",
        "feature_extraction_closure_payload_sha256",
        "config_sha256",
        "code_source_tree_sha256",
        "eligible_session_roster_sha256",
    }
    if any(not _is_sha256(payload[field]) for field in sha256_fields):
        raise ValueError("Stage-A shared context contains an invalid SHA-256")
    commit = payload["code_commit_sha"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("Stage-A shared context commit hash is invalid")
    for field in ("source_date_epoch", "eligible_participant_count", "expected_oof_row_count"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Stage-A shared context field {field!r} is invalid")
    if (
        payload["eligible_participant_count"] <= 0
        or payload["expected_oof_row_count"]
        != 2 * payload["eligible_participant_count"]
    ):
        raise ValueError("Stage-A shared context eligible counts drifted")


def _eligible_roster_sha256(
    eligible_sessions: Sequence[tuple[str, str, int]],
) -> str:
    return _payload_sha256(
        {
            "eligible_sessions": [
                {
                    "session_id": session_id,
                    "source_session_id": source_session_id,
                    "fixed_fold": fold,
                }
                for session_id, source_session_id, fold in eligible_sessions
            ]
        }
    )


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _open_stage_a_directory(
    output_dir: str | Path, *, create: bool
) -> AnchoredDirectory:
    if not create:
        return AnchoredDirectory.open(output_dir, directory_mode=0o700)
    try:
        return AnchoredDirectory.open(output_dir, directory_mode=0o700)
    except ValueError:
        return AnchoredDirectory.open(
            output_dir, create=True, directory_mode=0o700
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _cell_key(configuration: str, fold: int, seed: int) -> tuple[str, int, int]:
    return configuration, fold, seed


def _unique_names(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if (
        not result
        or any(not isinstance(value, str) or not value for value in result)
        or len(result) != len(set(result))
    ):
        raise ValueError(f"{label} must be non-empty unique strings")
    return result


def _validate_file_record(
    output: AnchoredDirectory,
    filename: str,
    record: Mapping[str, Any],
    label: str,
) -> AnchoredFile:
    try:
        artifact = output.read_bytes(filename, expected_mode=0o600)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"{label} file is unavailable or unsafe") from error
    if (
        record.get("bytes") not in (None, artifact.size)
        or record.get("sha256") != artifact.sha256
    ):
        raise ValueError(f"{label} checksum or size failed")
    return artifact


def _read_canonical_object(
    output: AnchoredDirectory, filename: str, label: str
) -> tuple[dict[str, Any], AnchoredFile]:
    try:
        artifact = output.read_bytes(filename, expected_mode=0o600)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"{label} file is unavailable or unsafe") from error
    return _canonical_object_from_bytes(artifact.content, label), artifact


def _canonical_object_from_bytes(raw: bytes, label: str) -> dict[str, Any]:

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON cannot be read") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise ValueError(f"{label} JSON is not canonical")
    return payload


def _validate_signed_payload(payload: Mapping[str, Any], field: str, label: str) -> None:
    observed = payload.get(field)
    unsigned = {key: value for key, value in payload.items() if key != field}
    if observed != _payload_sha256(unsigned):
        raise ValueError(f"{label} payload checksum failed")


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
