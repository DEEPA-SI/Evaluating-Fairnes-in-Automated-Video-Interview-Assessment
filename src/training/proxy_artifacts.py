"""Fold-scoped, outcome-blind participant proxy artefacts and validation."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.data import CORPUS_ID, EXPECTED_PARTICIPANTS, canonical_participant_id, parse_session_id
from src.features.pseudo_attributes import (
    RANDOM_PROXY_SALT,
    BalancedParticipantSampler,
    FittedBinaryProxy,
    _canonical_cluster_mapping,
    estimate_audio_session_pseudo_attr,
    estimate_face_pseudo_attr,
    random_proxy_groups,
)
from src.training.crossfit import (
    ArtifactPayload,
    EligibilityResult,
    FeatureSummary,
    RunProvenance,
    _parse_fold_manifest,
    _validate_mapping_provenance,
    canonical_json_bytes,
    deterministic_npz_bytes,
    validate_eligibility_result,
)

ProxySource = Literal["visual", "interview_audio", "random"]
RotationRole = Literal["train", "validation", "test"]

PROXY_ARTIFACT_SCHEMA_VERSION = "fair-avi-mit-fold-proxy-artifact-2"
PROXY_FIT_SCHEMA_VERSION = "fair-avi-mit-fold-proxy-fit-1"
_LEARNED_SOURCES: dict[str, tuple[str, int]] = {
    "visual": ("video", 1024),
    "interview_audio": ("audio", 1536),
}
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "corpus",
    "construction_mode",
    "proxy_source",
    "outer_rotation",
    "join_unit",
    "fit_scope",
    "aggregation",
    "held_out_assignment",
    "cluster_identity_claim",
    "outcome_fields_accessed",
    "protected_annotation_fields_accessed",
    "session_mapping",
    "eligibility_manifest_sha256",
    "fold_manifest_sha256",
    "feature_contract_sha256",
    "preprocessing_sha256",
    "feature_run_contract_sha256",
    "feature_config_sha256",
    "input_modality",
    "input_participants_sha256",
    "assignments_sha256",
    "participant_count",
    "fit_participant_count",
    "roles",
    "participants",
    "estimator",
    "fit_artifact",
    "provenance",
    "runtime",
    "manifest_payload_sha256",
}
@dataclass(frozen=True, slots=True)
class SessionProxyInput:
    """One checksum-bound session summary consumed by a participant proxy."""

    session_id: str
    source_session_id: str | None
    condition: str
    sidecar_sha256: str
    shard_sha256: str
    extraction_fingerprint: str
    summary_sha256: str


@dataclass(frozen=True, slots=True)
class ParticipantRepresentation:
    """Arithmetic mean of the verified pre/post session summaries for one participant."""

    participant_id: str
    fixed_fold: int
    rotation_role: RotationRole
    sessions: tuple[SessionProxyInput, SessionProxyInput]
    vector: np.ndarray
    vector_sha256: str


class ProxyArtifact:
    """Non-copyable wrapper created only after contextual proxy validation."""

    __slots__ = (
        "_source",
        "_outer_rotation",
        "_manifest",
        "_manifest_bytes",
        "_manifest_sha256",
        "_fit_artifact",
        "_eligibility",
        "_provenance",
        "_global_participant_ids",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("ProxyArtifact must be created by a validated build or load operation")

    @classmethod
    def _from_validated(
        cls,
        *,
        source: ProxySource,
        outer_rotation: int,
        manifest: Mapping[str, Any],
        manifest_bytes: bytes,
        manifest_sha256: str,
        fit_artifact: ArtifactPayload,
        eligibility: EligibilityResult,
        provenance: RunProvenance,
        global_participant_ids: Sequence[str] | np.ndarray | None,
    ) -> ProxyArtifact:
        artifact = object.__new__(cls)
        object.__setattr__(artifact, "_source", source)
        object.__setattr__(artifact, "_outer_rotation", outer_rotation)
        object.__setattr__(artifact, "_manifest", manifest)
        object.__setattr__(artifact, "_manifest_bytes", manifest_bytes)
        object.__setattr__(artifact, "_manifest_sha256", manifest_sha256)
        object.__setattr__(artifact, "_fit_artifact", fit_artifact)
        object.__setattr__(artifact, "_eligibility", eligibility)
        object.__setattr__(artifact, "_provenance", provenance)
        bound_global_ids = (
            None
            if global_participant_ids is None
            else tuple(_canonical_unique_ids(global_participant_ids))
        )
        object.__setattr__(artifact, "_global_participant_ids", bound_global_ids)
        _validate_proxy_artifact_structure(artifact)
        return artifact

    def _validate_bound_context(self) -> None:
        _contextually_validated_proxy_artifact(
            self,
            object.__getattribute__(self, "_eligibility"),
            provenance=object.__getattribute__(self, "_provenance"),
            global_participant_ids=object.__getattribute__(self, "_global_participant_ids"),
        )

    @property
    def source(self) -> ProxySource:
        return self._source

    @property
    def outer_rotation(self) -> int:
        return self._outer_rotation

    @property
    def manifest(self) -> Mapping[str, Any]:
        self._validate_bound_context()
        return _loads_unique_json(object.__getattribute__(self, "_manifest_bytes"))

    @property
    def manifest_bytes(self) -> bytes:
        self._validate_bound_context()
        return object.__getattribute__(self, "_manifest_bytes")

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @property
    def fit_artifact(self) -> ArtifactPayload:
        self._validate_bound_context()
        return object.__getattribute__(self, "_fit_artifact")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("ProxyArtifact is immutable; rebuild it from validated context")

    def __delattr__(self, _name: str) -> None:
        raise TypeError("ProxyArtifact is immutable; rebuild it from validated context")

    def __copy__(self) -> ProxyArtifact:
        raise TypeError("ProxyArtifact may not be copied; rebuild it from validated context")

    def __deepcopy__(self, _memo: dict[int, Any]) -> ProxyArtifact:
        raise TypeError("ProxyArtifact may not be copied; rebuild it from validated context")

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("ProxyArtifact may not be serialized; rebuild it from validated context")

    def participant_groups(self, role: RotationRole | None = None) -> dict[str, int]:
        """Return participant assignments, optionally restricted to one frozen rotation role."""
        self._validate_bound_context()
        if role is not None and role not in {"train", "validation", "test"}:
            raise ValueError("role must be train, validation or test")
        manifest = object.__getattribute__(self, "_manifest")
        return {
            str(row["participant_id"]): int(row["proxy_group"])
            for row in manifest["participants"]
            if role is None or row["rotation_role"] == role
        }

    def session_groups(self, role: RotationRole | None = None) -> dict[str, int]:
        """Return the same participant assignment for both sessions in each selected role."""
        self._validate_bound_context()
        if role is not None and role not in {"train", "validation", "test"}:
            raise ValueError("role must be train, validation or test")
        manifest = object.__getattribute__(self, "_manifest")
        groups: dict[str, int] = {}
        for row in manifest["participants"]:
            if role is not None and row["rotation_role"] != role:
                continue
            for session in row["session_inputs"]:
                groups[str(session["session_id"])] = int(row["proxy_group"])
        return groups

    def paired_training_schedule(
        self, *, seed: int, batch_count: int
    ) -> tuple[dict[str, Any], ...]:
        """Build checksum-bound-role 8+8 batches with both sessions per participant draw."""
        self._validate_bound_context()
        manifest = object.__getattribute__(self, "_manifest")
        rows = [row for row in manifest["participants"] if row["rotation_role"] == "train"]
        participant_ids = np.asarray([row["participant_id"] for row in rows], dtype=object)
        groups = np.asarray([row["proxy_group"] for row in rows], dtype=np.int8)
        pairs = {
            row["participant_id"]: tuple(
                session["session_id"] for session in row["session_inputs"]
            )
            for row in rows
        }
        return BalancedParticipantSampler(participant_ids, groups, seed=seed).paired_schedule(
            pairs, batch_count=batch_count
        )


@dataclass(frozen=True, slots=True)
class _UnvalidatedProxyArtifact:
    """Structurally parsed bytes that cannot expose assignments or be written."""

    source: ProxySource
    outer_rotation: int
    manifest: Mapping[str, Any]
    manifest_bytes: bytes
    manifest_sha256: str
    fit_artifact: ArtifactPayload


@dataclass(frozen=True, slots=True)
class _ProxyArtifactView:
    """Internal raw view used while contextual validation is already in progress."""

    source: ProxySource
    outer_rotation: int
    manifest: Mapping[str, Any]
    manifest_bytes: bytes
    manifest_sha256: str
    fit_artifact: ArtifactPayload


def _artifact_view(
    artifact: ProxyArtifact | _UnvalidatedProxyArtifact | _ProxyArtifactView,
) -> _ProxyArtifactView:
    if isinstance(artifact, _ProxyArtifactView):
        return artifact
    if isinstance(artifact, ProxyArtifact):
        return _ProxyArtifactView(
            source=object.__getattribute__(artifact, "_source"),
            outer_rotation=object.__getattribute__(artifact, "_outer_rotation"),
            manifest=object.__getattribute__(artifact, "_manifest"),
            manifest_bytes=object.__getattribute__(artifact, "_manifest_bytes"),
            manifest_sha256=object.__getattribute__(artifact, "_manifest_sha256"),
            fit_artifact=object.__getattribute__(artifact, "_fit_artifact"),
        )
    return _ProxyArtifactView(
        source=artifact.source,
        outer_rotation=artifact.outer_rotation,
        manifest=artifact.manifest,
        manifest_bytes=artifact.manifest_bytes,
        manifest_sha256=artifact.manifest_sha256,
        fit_artifact=artifact.fit_artifact,
    )


def participant_representations(
    features: Sequence[FeatureSummary],
    *,
    source: Literal["visual", "interview_audio"],
    outer_rotation: int,
) -> tuple[ParticipantRepresentation, ...]:
    """Aggregate verified pre/post session means into fold-role participant vectors."""
    _validate_outer_rotation(outer_rotation)
    if source not in _LEARNED_SOURCES:
        raise ValueError("source must be visual or interview_audio")
    if not isinstance(features, Sequence) or not features:
        raise ValueError("features must contain complete participant pairs")
    modality, dimension = _LEARNED_SOURCES[source]
    grouped: dict[str, list[FeatureSummary]] = {}
    seen_sessions: set[str] = set()
    for feature in features:
        if not isinstance(feature, FeatureSummary):
            raise TypeError("features must contain FeatureSummary rows")
        identity = feature.identity
        session_id, participant_id, condition = parse_session_id(identity.session_id)
        if (
            identity.session_id != session_id
            or identity.participant_id != participant_id
            or identity.condition != condition
            or canonical_participant_id(participant_id) != participant_id
        ):
            raise ValueError("feature session identity is not canonical")
        if identity.session_id in seen_sessions:
            raise ValueError("feature rows contain a duplicate session")
        if isinstance(identity.fold, bool) or identity.fold not in range(5):
            raise ValueError("feature fold must be an integer from 0 to 4")
        summary = np.asarray(getattr(feature, modality))
        if (
            summary.dtype != np.float32
            or summary.shape != (dimension,)
            or not np.isfinite(summary).all()
        ):
            raise ValueError(
                f"{source} session summaries must be finite float32 vectors of length {dimension}"
            )
        seen_sessions.add(identity.session_id)
        grouped.setdefault(participant_id, []).append(feature)

    representations: list[ParticipantRepresentation] = []
    for participant_id in sorted(grouped, key=int):
        pair = sorted(grouped[participant_id], key=lambda row: row.identity.condition == "post")
        if len(pair) != 2 or {row.identity.condition for row in pair} != {"pre", "post"}:
            raise ValueError(
                f"participant {participant_id} does not have one complete feature pair"
            )
        if len({row.identity.fold for row in pair}) != 1:
            raise ValueError(f"participant {participant_id} sessions do not share a fold")
        fold = pair[0].identity.fold
        role = _rotation_role(fold, outer_rotation)
        matrix = np.stack(
            [np.asarray(getattr(row, modality), dtype=np.float64) for row in pair],
            axis=0,
        )
        vector = np.ascontiguousarray(matrix.mean(axis=0, dtype=np.float64), dtype="<f8")
        vector.setflags(write=False)
        sessions = tuple(
            SessionProxyInput(
                session_id=row.identity.session_id,
                source_session_id=row.identity.source_session_id,
                condition=row.identity.condition,
                sidecar_sha256=_required_hex(row.sidecar_sha256, "sidecar_sha256"),
                shard_sha256=_required_hex(row.shard_sha256, "shard_sha256"),
                extraction_fingerprint=_required_hex(
                    row.extraction_fingerprint, "extraction_fingerprint"
                ),
                summary_sha256=_summary_sha256(getattr(row, modality)),
            )
            for row in pair
        )
        representations.append(
            ParticipantRepresentation(
                participant_id=participant_id,
                fixed_fold=fold,
                rotation_role=role,
                sessions=(sessions[0], sessions[1]),
                vector=vector,
                vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
            )
        )
    return tuple(representations)


def global_random_proxy_mapping(participant_ids: Sequence[str] | np.ndarray) -> dict[str, int]:
    """Create the one 34/35 assignment before any eligibility or fold-role subsetting."""
    identifiers = tuple(sorted(_canonical_unique_ids(participant_ids), key=int))
    groups = random_proxy_groups(np.asarray(identifiers, dtype=object))
    return {
        participant_id: int(group)
        for participant_id, group in zip(identifiers, groups, strict=True)
    }


def subset_random_proxy_groups(
    global_participant_ids: Sequence[str] | np.ndarray,
    eligible_participant_ids: Sequence[str] | np.ndarray,
) -> np.ndarray:
    """Subset the already-ranked global random map without recalculating its 34/35 split."""
    mapping = global_random_proxy_mapping(global_participant_ids)
    eligible = _canonical_unique_ids(eligible_participant_ids)
    if not set(eligible).issubset(mapping):
        raise ValueError("eligible participants must be a subset of the global random roster")
    return np.asarray([mapping[participant_id] for participant_id in eligible], dtype=np.int8)


def build_fold_proxy_artifact(
    eligibility: EligibilityResult,
    *,
    source: Literal["visual", "interview_audio"],
    outer_rotation: int,
    provenance: RunProvenance,
) -> ProxyArtifact:
    """Fit one train-only learned proxy and assign all frozen rotation roles."""
    validate_eligibility_result(eligibility)
    _validate_outer_rotation(outer_rotation)
    if source not in _LEARNED_SOURCES:
        raise ValueError("source must be visual or interview_audio")
    if not isinstance(provenance, RunProvenance):
        raise TypeError("provenance must be RunProvenance")
    if (
        eligibility.production_ready
        and provenance.config_sha256 != eligibility.feature_config_sha256
    ):
        raise ValueError("proxy provenance config does not match frozen feature configuration")

    representations = participant_representations(
        eligibility.features,
        source=source,
        outer_rotation=outer_rotation,
    )
    _require_all_roles(representations)
    training = tuple(row for row in representations if row.rotation_role == "train")
    training_ids = np.asarray([row.participant_id for row in training], dtype=object)
    training_matrix = np.stack([row.vector for row in training], axis=0)
    fitter = (
        estimate_face_pseudo_attr
        if source == "visual"
        else estimate_audio_session_pseudo_attr
    )
    fitted = fitter(training_matrix, training_ids)
    all_matrix = np.stack([row.vector for row in representations], axis=0)
    groups = fitted.transform(all_matrix)
    fit_artifact = _learned_fit_artifact(
        source,
        outer_rotation,
        training_ids,
        representations,
        groups,
        fitted,
    )
    return _assemble_artifact(
        eligibility,
        source=source,
        outer_rotation=outer_rotation,
        provenance=provenance,
        participant_rows=_learned_participant_rows(representations, groups),
        fit_participant_count=len(training),
        estimator=fitted.manifest(),
        fit_artifact=fit_artifact,
        global_participant_ids=None,
    )


def build_random_proxy_artifact(
    eligibility: EligibilityResult,
    *,
    outer_rotation: int,
    provenance: RunProvenance,
    global_participant_ids: Sequence[str] | np.ndarray | None = None,
) -> ProxyArtifact:
    """Bind the frozen global random assignment to one fold's eligible role manifest."""
    validate_eligibility_result(eligibility)
    _validate_outer_rotation(outer_rotation)
    if not isinstance(provenance, RunProvenance):
        raise TypeError("provenance must be RunProvenance")
    fold_roster = tuple(_parse_fold_manifest(eligibility.fold_manifest_bytes))
    supplied_global = (
        fold_roster
        if global_participant_ids is None
        else tuple(sorted(_canonical_unique_ids(global_participant_ids), key=int))
    )
    if len(supplied_global) != EXPECTED_PARTICIPANTS:
        raise ValueError("the global random proxy requires the complete 69-participant roster")
    if eligibility.production_ready and set(supplied_global) != set(fold_roster):
        raise ValueError("global random roster does not match the immutable fold manifest")
    mapping = global_random_proxy_mapping(supplied_global)
    participants = _identity_participants(eligibility.features, outer_rotation)
    if not {row[0] for row in participants}.issubset(mapping):
        raise ValueError("eligible participants are absent from the global random roster")
    groups = np.asarray([mapping[row[0]] for row in participants], dtype=np.int8)
    fit_artifact = _random_fit_artifact(
        outer_rotation,
        supplied_global,
        mapping,
        participants,
        groups,
    )
    estimator = {
        "algorithm": "global_sha256_rank",
        "salt": RANDOM_PROXY_SALT,
        "separator": "NUL",
        "group_0_count": 34,
        "group_1_count": 35,
        "eligibility_subsetting": "apply_global_map_without_reranking",
        "fold_role_or_seed_dependency": "forbidden",
    }
    return _assemble_artifact(
        eligibility,
        source="random",
        outer_rotation=outer_rotation,
        provenance=provenance,
        participant_rows=_random_participant_rows(participants, groups),
        fit_participant_count=EXPECTED_PARTICIPANTS,
        estimator=estimator,
        fit_artifact=fit_artifact,
        global_participant_ids=supplied_global,
    )


def load_proxy_artifact(
    manifest_bytes: bytes,
    fit_bytes: bytes,
    *,
    eligibility: EligibilityResult,
    provenance: RunProvenance,
    global_participant_ids: Sequence[str] | np.ndarray | None = None,
) -> ProxyArtifact:
    """Load bytes only after full contextual reconstruction validates every assignment."""
    if not isinstance(manifest_bytes, bytes) or not isinstance(fit_bytes, bytes):
        raise TypeError("proxy artifact inputs must be bytes")
    manifest = _loads_unique_json(manifest_bytes)
    fit = manifest.get("fit_artifact")
    if not isinstance(fit, Mapping) or not isinstance(fit.get("filename"), str):
        raise ValueError("proxy fit artifact reference is invalid")
    candidate = _UnvalidatedProxyArtifact(
        source=manifest.get("proxy_source"),
        outer_rotation=manifest.get("outer_rotation"),
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest.get("manifest_payload_sha256"),
        fit_artifact=ArtifactPayload(str(fit["filename"]), fit_bytes),
    )
    _validate_proxy_artifact_structure(candidate)
    return _contextually_validated_proxy_artifact(
        candidate,
        eligibility,
        provenance=provenance,
        global_participant_ids=global_participant_ids,
    )


def validate_proxy_artifact(
    artifact: ProxyArtifact,
    eligibility: EligibilityResult,
    *,
    provenance: RunProvenance,
    global_participant_ids: Sequence[str] | np.ndarray | None = None,
) -> ProxyArtifact:
    """Rebuild an artefact against frozen inputs and return the validated instance."""
    if not isinstance(artifact, ProxyArtifact):
        raise TypeError("artifact must be a ProxyArtifact")
    _validate_proxy_artifact_structure(artifact)
    artifact._validate_bound_context()
    _contextually_validated_proxy_artifact(
        artifact,
        eligibility,
        provenance=provenance,
        global_participant_ids=global_participant_ids,
    )
    return artifact


def write_proxy_artifact(
    output_dir: str | Path,
    artifact: ProxyArtifact,
    *,
    eligibility: EligibilityResult,
    provenance: RunProvenance,
    global_participant_ids: Sequence[str] | np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Context-validate, then write inside one resolved output directory."""
    if not isinstance(artifact, ProxyArtifact):
        raise TypeError("artifact must be a ProxyArtifact")
    validate_proxy_artifact(
        artifact,
        eligibility,
        provenance=provenance,
        global_participant_ids=global_participant_ids,
    )
    view = _artifact_view(artifact)
    root = Path(output_dir)
    if root.is_symlink():
        raise ValueError("proxy output directory may not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    manifest_path = _resolved_output_child(
        root, f"{view.source}.fold-{view.outer_rotation}.proxy.json"
    )
    fit_path = _resolved_output_child(root, view.fit_artifact.filename)
    for path, content in (
        (manifest_path, view.manifest_bytes),
        (fit_path, view.fit_artifact.content),
    ):
        if path.is_symlink():
            raise ValueError("proxy output path may not be a symlink")
        if path.exists() and (not path.is_file() or path.read_bytes() != content):
            raise FileExistsError(
                f"refusing to replace a different proxy artifact: {path.name}"
            )
    _write_immutable(fit_path, view.fit_artifact.content)
    _write_immutable(manifest_path, view.manifest_bytes)
    return manifest_path, fit_path


def _contextually_validated_proxy_artifact(
    artifact: ProxyArtifact | _UnvalidatedProxyArtifact | _ProxyArtifactView,
    eligibility: EligibilityResult,
    *,
    provenance: RunProvenance,
    global_participant_ids: Sequence[str] | np.ndarray | None,
) -> ProxyArtifact:
    """Reconstruct frozen state before loaded bytes can become a public artefact."""
    validate_eligibility_result(eligibility)
    if not isinstance(provenance, RunProvenance):
        raise TypeError("provenance must be RunProvenance")
    artifact = _artifact_view(artifact)
    _validate_proxy_artifact_structure(artifact)
    if artifact.source != "random":
        _validate_stored_learned_assignments(artifact, eligibility)
        expected_artifact = build_fold_proxy_artifact(
            eligibility,
            source=artifact.source,
            outer_rotation=artifact.outer_rotation,
            provenance=provenance,
        )
    else:
        expected_artifact = build_random_proxy_artifact(
            eligibility,
            outer_rotation=artifact.outer_rotation,
            provenance=provenance,
            global_participant_ids=global_participant_ids,
        )
    expected = _artifact_view(expected_artifact)
    if (
        artifact.manifest_bytes != expected.manifest_bytes
        or artifact.fit_artifact.content != expected.fit_artifact.content
    ):
        raise ValueError("proxy artifact does not match the frozen inputs, roles or provenance")
    return expected_artifact


def _assemble_artifact(
    eligibility: EligibilityResult,
    *,
    source: ProxySource,
    outer_rotation: int,
    provenance: RunProvenance,
    participant_rows: list[dict[str, Any]],
    fit_participant_count: int,
    estimator: Mapping[str, Any],
    fit_artifact: ArtifactPayload,
    global_participant_ids: Sequence[str] | np.ndarray | None,
) -> ProxyArtifact:
    input_rows = [
        {key: value for key, value in row.items() if key != "proxy_group"}
        for row in participant_rows
    ]
    assignment_rows = [
        {"participant_id": row["participant_id"], "proxy_group": row["proxy_group"]}
        for row in participant_rows
    ]
    roles = {
        role: [row["participant_id"] for row in participant_rows if row["rotation_role"] == role]
        for role in ("train", "validation", "test")
    }
    modality = _LEARNED_SOURCES.get(source)
    unsigned = {
        "schema_version": PROXY_ARTIFACT_SCHEMA_VERSION,
        "corpus": CORPUS_ID,
        "construction_mode": eligibility.manifest["construction_mode"],
        "proxy_source": source,
        "outer_rotation": outer_rotation,
        "join_unit": "participant_with_both_sessions",
        "fit_scope": (
            "training_participants_only_per_outer_rotation"
            if source != "random"
            else "global_all_69_participants_once"
        ),
        "aggregation": (
            "mean_valid_vectors_within_session_then_arithmetic_mean_pre_post"
            if source != "random"
            else "not_applicable_identity_hash_control"
        ),
        "held_out_assignment": (
            "unchanged_training_scaler_and_nearest_frozen_centre"
            if source != "random"
            else "unchanged_global_assignment_without_reranking"
        ),
        "cluster_identity_claim": "forbidden",
        "outcome_fields_accessed": [],
        "protected_annotation_fields_accessed": [],
        "session_mapping": eligibility.manifest["session_mapping"],
        "eligibility_manifest_sha256": eligibility.manifest_sha256,
        "fold_manifest_sha256": eligibility.fold_manifest_sha256,
        "feature_contract_sha256": eligibility.feature_contract_sha256,
        "preprocessing_sha256": eligibility.preprocessing_sha256,
        "feature_run_contract_sha256": eligibility.manifest["feature_run_contract_sha256"],
        "feature_config_sha256": eligibility.feature_config_sha256,
        "input_modality": (
            {"name": modality[0], "dimension": modality[1]}
            if modality is not None
            else {"name": "canonical_participant_id", "dimension": None}
        ),
        "input_participants_sha256": _payload_sha256({"participants": input_rows}),
        "assignments_sha256": _payload_sha256({"assignments": assignment_rows}),
        "participant_count": len(participant_rows),
        "fit_participant_count": fit_participant_count,
        "roles": roles,
        "participants": participant_rows,
        "estimator": dict(estimator),
        "fit_artifact": {
            "filename": fit_artifact.filename,
            "sha256": fit_artifact.sha256,
            "bytes": len(fit_artifact.content),
        },
        "provenance": {
            "config_sha256": provenance.config_sha256,
            "code_commit_sha": provenance.code_commit_sha,
            "code_source_tree_sha256": provenance.code_source_tree_sha256,
            "source_date_epoch": provenance.source_date_epoch,
        },
        "runtime": {
            "numpy": np.__version__,
            "scikit_learn": _package_version("scikit-learn"),
        },
    }
    manifest_sha256 = _payload_sha256(unsigned)
    manifest = {**unsigned, "manifest_payload_sha256": manifest_sha256}
    return ProxyArtifact._from_validated(
        source=source,
        outer_rotation=outer_rotation,
        manifest=manifest,
        manifest_bytes=canonical_json_bytes(manifest),
        manifest_sha256=manifest_sha256,
        fit_artifact=fit_artifact,
        eligibility=eligibility,
        provenance=provenance,
        global_participant_ids=global_participant_ids,
    )


def _learned_participant_rows(
    representations: Sequence[ParticipantRepresentation], groups: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {
            "participant_id": row.participant_id,
            "fixed_fold": row.fixed_fold,
            "rotation_role": row.rotation_role,
            "session_inputs": [
                {
                    "session_id": session.session_id,
                    "source_session_id": session.source_session_id,
                    "condition": session.condition,
                    "sidecar_sha256": session.sidecar_sha256,
                    "shard_sha256": session.shard_sha256,
                    "extraction_fingerprint": session.extraction_fingerprint,
                    "summary_sha256": session.summary_sha256,
                }
                for session in row.sessions
            ],
            "participant_representation_sha256": row.vector_sha256,
            "proxy_group": int(group),
        }
        for row, group in zip(representations, groups, strict=True)
    ]


def _random_participant_rows(
    participants: Sequence[
        tuple[str, int, RotationRole, tuple[tuple[str, str | None], tuple[str, str | None]]]
    ],
    groups: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "participant_id": participant_id,
            "fixed_fold": fold,
            "rotation_role": role,
            "session_inputs": [
                {
                    "session_id": session_ids[0][0],
                    "source_session_id": session_ids[0][1],
                    "condition": "pre",
                },
                {
                    "session_id": session_ids[1][0],
                    "source_session_id": session_ids[1][1],
                    "condition": "post",
                },
            ],
            "participant_representation_sha256": None,
            "proxy_group": int(group),
        }
        for (participant_id, fold, role, session_ids), group in zip(
            participants, groups, strict=True
        )
    ]


def _learned_fit_artifact(
    source: str,
    outer_rotation: int,
    training_ids: np.ndarray,
    representations: Sequence[ParticipantRepresentation],
    groups: np.ndarray,
    fitted: FittedBinaryProxy,
) -> ArtifactPayload:
    raw_centres = np.asarray(fitted.clusterer.cluster_centers_, dtype="<f8")
    canonical_centres = raw_centres[np.argsort(np.asarray(fitted.raw_to_canonical))]
    arrays = {
        "schema_version_utf8": np.frombuffer(
            PROXY_FIT_SCHEMA_VERSION.encode("utf-8"), dtype=np.uint8
        ),
        "training_participant_ids": _ascii_array(training_ids.tolist()),
        "assigned_participant_ids": _ascii_array([row.participant_id for row in representations]),
        "assigned_groups": np.asarray(groups, dtype=np.int8),
        "scaler_mean": np.asarray(fitted.scaler.mean_, dtype="<f8"),
        "scaler_scale": np.asarray(fitted.scaler.scale_, dtype="<f8"),
        "scaler_var": np.asarray(fitted.scaler.var_, dtype="<f8"),
        "scaler_n_samples_seen": np.asarray(fitted.scaler.n_samples_seen_, dtype="<i8"),
        "cluster_centres_raw": raw_centres,
        "cluster_centres_canonical": np.asarray(canonical_centres, dtype="<f8"),
        "raw_to_canonical": np.asarray(fitted.raw_to_canonical, dtype=np.int8),
        "kmeans_inertia": np.asarray(fitted.clusterer.inertia_, dtype="<f8"),
        "kmeans_n_iter": np.asarray(fitted.clusterer.n_iter_, dtype="<i8"),
    }
    return ArtifactPayload(
        _proxy_fit_filename(source, outer_rotation),
        deterministic_npz_bytes(arrays),
    )


def _random_fit_artifact(
    outer_rotation: int,
    global_ids: Sequence[str],
    mapping: Mapping[str, int],
    participants: Sequence[tuple[str, int, RotationRole, tuple[str, str]]],
    groups: np.ndarray,
) -> ArtifactPayload:
    arrays = {
        "schema_version_utf8": np.frombuffer(
            PROXY_FIT_SCHEMA_VERSION.encode("utf-8"), dtype=np.uint8
        ),
        "global_participant_ids": _ascii_array(global_ids),
        "global_groups": np.asarray([mapping[value] for value in global_ids], dtype=np.int8),
        "assigned_participant_ids": _ascii_array([row[0] for row in participants]),
        "assigned_groups": np.asarray(groups, dtype=np.int8),
    }
    return ArtifactPayload(
        _proxy_fit_filename("random", outer_rotation),
        deterministic_npz_bytes(arrays),
    )


def _identity_participants(
    features: Sequence[FeatureSummary], outer_rotation: int
) -> tuple[
    tuple[str, int, RotationRole, tuple[tuple[str, str | None], tuple[str, str | None]]], ...
]:
    grouped: dict[str, list[FeatureSummary]] = {}
    for feature in features:
        if not isinstance(feature, FeatureSummary):
            raise TypeError("features must contain FeatureSummary rows")
        grouped.setdefault(feature.identity.participant_id, []).append(feature)
    rows = []
    for participant_id in sorted(grouped, key=int):
        pair = sorted(grouped[participant_id], key=lambda row: row.identity.condition == "post")
        if len(pair) != 2 or {row.identity.condition for row in pair} != {"pre", "post"}:
            raise ValueError(
                f"participant {participant_id} does not have one complete feature pair"
            )
        if len({row.identity.fold for row in pair}) != 1:
            raise ValueError(f"participant {participant_id} sessions do not share a fold")
        rows.append(
            (
                participant_id,
                pair[0].identity.fold,
                _rotation_role(pair[0].identity.fold, outer_rotation),
                (
                    (pair[0].identity.session_id, pair[0].identity.source_session_id),
                    (pair[1].identity.session_id, pair[1].identity.source_session_id),
                ),
            )
        )
    result = tuple(rows)
    if not result:
        raise ValueError("eligible participant roster is empty")
    if {row[2] for row in result} != {"train", "validation", "test"}:
        raise ValueError("every proxy rotation requires non-empty train, validation and test roles")
    return result


def _validate_proxy_artifact_structure(
    artifact: ProxyArtifact | _UnvalidatedProxyArtifact | _ProxyArtifactView,
) -> None:
    artifact = _artifact_view(artifact)
    if artifact.source not in {"visual", "interview_audio", "random"}:
        raise ValueError("proxy source is invalid")
    _validate_outer_rotation(artifact.outer_rotation)
    if not isinstance(artifact.manifest, Mapping):
        raise TypeError("proxy manifest must be a mapping")
    if not isinstance(artifact.manifest_bytes, bytes):
        raise TypeError("proxy manifest bytes must be bytes")
    if not isinstance(artifact.fit_artifact, ArtifactPayload):
        raise TypeError("proxy fit artifact is invalid")
    parsed = _loads_unique_json(artifact.manifest_bytes)
    if parsed != artifact.manifest or canonical_json_bytes(parsed) != artifact.manifest_bytes:
        raise ValueError("proxy manifest is not canonical or disagrees with its object")
    if set(parsed) != _TOP_LEVEL_FIELDS:
        raise ValueError("proxy manifest fields are incomplete or unexpected")
    unsigned = dict(parsed)
    signed_sha256 = unsigned.pop("manifest_payload_sha256")
    if (
        not isinstance(signed_sha256, str)
        or signed_sha256 != artifact.manifest_sha256
        or signed_sha256 != _payload_sha256(unsigned)
    ):
        raise ValueError("proxy manifest payload checksum failed")
    if (
        parsed["schema_version"] != PROXY_ARTIFACT_SCHEMA_VERSION
        or parsed["corpus"] != CORPUS_ID
        or parsed["construction_mode"] not in {"full_corpus", "synthetic_contract_test"}
        or parsed["proxy_source"] != artifact.source
        or parsed["outer_rotation"] != artifact.outer_rotation
        or parsed["join_unit"] != "participant_with_both_sessions"
        or parsed["cluster_identity_claim"] != "forbidden"
        or parsed["outcome_fields_accessed"] != []
        or parsed["protected_annotation_fields_accessed"] != []
    ):
        raise ValueError("proxy manifest identity or label firewall is invalid")
    production = parsed["construction_mode"] == "full_corpus"
    _validate_mapping_provenance(parsed["session_mapping"], required=production)
    expected_learned = artifact.source != "random"
    expected_fit_scope = (
        "training_participants_only_per_outer_rotation"
        if expected_learned
        else "global_all_69_participants_once"
    )
    expected_aggregation = (
        "mean_valid_vectors_within_session_then_arithmetic_mean_pre_post"
        if expected_learned
        else "not_applicable_identity_hash_control"
    )
    expected_assignment = (
        "unchanged_training_scaler_and_nearest_frozen_centre"
        if expected_learned
        else "unchanged_global_assignment_without_reranking"
    )
    expected_modality = (
        {
            "name": _LEARNED_SOURCES[artifact.source][0],
            "dimension": _LEARNED_SOURCES[artifact.source][1],
        }
        if expected_learned
        else {"name": "canonical_participant_id", "dimension": None}
    )
    if (
        parsed["fit_scope"] != expected_fit_scope
        or parsed["aggregation"] != expected_aggregation
        or parsed["held_out_assignment"] != expected_assignment
        or parsed["input_modality"] != expected_modality
    ):
        raise ValueError("proxy fit or input contract is invalid")
    for name in (
        "eligibility_manifest_sha256",
        "fold_manifest_sha256",
        "feature_contract_sha256",
        "preprocessing_sha256",
        "feature_run_contract_sha256",
        "feature_config_sha256",
        "input_participants_sha256",
        "assignments_sha256",
    ):
        _required_hex(parsed[name], name)
    provenance = parsed["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "config_sha256",
        "code_commit_sha",
        "code_source_tree_sha256",
        "source_date_epoch",
    }:
        raise ValueError("proxy provenance is invalid")
    _required_hex(provenance["config_sha256"], "config_sha256")
    _required_hex(provenance["code_source_tree_sha256"], "code_source_tree_sha256")
    _required_hex(provenance["code_commit_sha"], "code_commit_sha", length=40)
    if (
        isinstance(provenance["source_date_epoch"], bool)
        or not isinstance(provenance["source_date_epoch"], int)
        or provenance["source_date_epoch"] < 0
    ):
        raise ValueError("proxy source_date_epoch is invalid")
    runtime = parsed["runtime"]
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != {"numpy", "scikit_learn"}
        or not isinstance(runtime["numpy"], str)
        or (
            runtime["scikit_learn"] is not None
            and not isinstance(runtime["scikit_learn"], str)
        )
    ):
        raise ValueError("proxy runtime provenance is invalid")

    participants = parsed["participants"]
    roles = parsed["roles"]
    if not isinstance(participants, list) or not participants or not isinstance(roles, Mapping):
        raise ValueError("proxy participant rows or roles are invalid")
    if set(roles) != {"train", "validation", "test"}:
        raise ValueError("proxy role fields are invalid")
    _validate_participant_rows(
        participants,
        artifact.source,
        artifact.outer_rotation,
        roles,
        production=production,
    )
    if artifact.source != "random" and {
        row["proxy_group"] for row in participants if row["rotation_role"] == "train"
    } != {0, 1}:
        raise ValueError("learned proxy training assignments must contain both groups")
    if parsed["participant_count"] != len(participants):
        raise ValueError("proxy participant count disagrees with its rows")
    if (
        isinstance(parsed["fit_participant_count"], bool)
        or not isinstance(parsed["fit_participant_count"], int)
        or parsed["fit_participant_count"] <= 0
    ):
        raise ValueError("proxy fit participant count is invalid")
    input_rows = [
        {key: value for key, value in row.items() if key != "proxy_group"}
        for row in participants
    ]
    assignments = [
        {"participant_id": row["participant_id"], "proxy_group": row["proxy_group"]}
        for row in participants
    ]
    if parsed["input_participants_sha256"] != _payload_sha256({"participants": input_rows}):
        raise ValueError("proxy input participant checksum failed")
    if parsed["assignments_sha256"] != _payload_sha256({"assignments": assignments}):
        raise ValueError("proxy assignment checksum failed")

    fit_reference = parsed["fit_artifact"]
    expected_fit_filename = _proxy_fit_filename(artifact.source, artifact.outer_rotation)
    if (
        not isinstance(fit_reference, Mapping)
        or set(fit_reference) != {"filename", "sha256", "bytes"}
        or fit_reference["filename"] != expected_fit_filename
        or artifact.fit_artifact.filename != expected_fit_filename
        or fit_reference["filename"] != artifact.fit_artifact.filename
        or fit_reference["sha256"] != artifact.fit_artifact.sha256
        or fit_reference["bytes"] != len(artifact.fit_artifact.content)
    ):
        raise ValueError("proxy fit artifact filename, checksum or reference is not canonical")
    arrays = _load_deterministic_npz(artifact.fit_artifact.content)
    _validate_fit_arrays(parsed, arrays)


def _validate_participant_rows(
    participants: list[Any],
    source: str,
    outer_rotation: int,
    roles: Mapping[str, Any],
    *,
    production: bool,
) -> None:
    expected_fields = {
        "participant_id",
        "fixed_fold",
        "rotation_role",
        "session_inputs",
        "participant_representation_sha256",
        "proxy_group",
    }
    seen: set[str] = set()
    seen_sources: set[str] = set()
    observed_roles = {role: [] for role in ("train", "validation", "test")}
    for row in participants:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError("proxy participant row fields are invalid")
        participant_id = row["participant_id"]
        if (
            not isinstance(participant_id, str)
            or canonical_participant_id(participant_id) != participant_id
            or participant_id in seen
        ):
            raise ValueError("proxy participant ID is invalid or duplicated")
        fold = row["fixed_fold"]
        if isinstance(fold, bool) or fold not in range(5):
            raise ValueError("proxy participant fold is invalid")
        role = row["rotation_role"]
        if role != _rotation_role(fold, outer_rotation):
            raise ValueError("proxy participant role leaks across the frozen fold rotation")
        if isinstance(row["proxy_group"], bool) or row["proxy_group"] not in {0, 1}:
            raise ValueError("proxy group must be binary")
        session_inputs = row["session_inputs"]
        if not isinstance(session_inputs, list) or len(session_inputs) != 2:
            raise ValueError("proxy participant does not contain one session pair")
        conditions: set[str] = set()
        for session in session_inputs:
            expected_session_fields = (
                {"session_id", "source_session_id", "condition"}
                if source == "random"
                else {
                    "session_id",
                    "source_session_id",
                    "condition",
                    "sidecar_sha256",
                    "shard_sha256",
                    "extraction_fingerprint",
                    "summary_sha256",
                }
            )
            if not isinstance(session, Mapping) or set(session) != expected_session_fields:
                raise ValueError("proxy session input fields are invalid")
            session_id, parsed_participant, condition = parse_session_id(session["session_id"])
            source_session_id = session["source_session_id"]
            if (
                session_id != session["session_id"]
                or parsed_participant != participant_id
                or session["condition"] != condition
                or condition in conditions
            ):
                raise ValueError("proxy session pair identity is invalid or duplicated")
            if production:
                if not isinstance(source_session_id, str):
                    raise ValueError("production proxy input lacks its source-session ID")
                parsed_source, _, source_condition = parse_session_id(source_session_id)
                if (
                    parsed_source != source_session_id
                    or source_condition != condition
                    or source_session_id in seen_sources
                ):
                    raise ValueError("proxy source-session phase provenance is invalid")
                seen_sources.add(source_session_id)
            elif source_session_id is not None:
                raise ValueError("synthetic proxy input may not claim a source-session join")
            conditions.add(condition)
            if source != "random":
                for name in (
                    "sidecar_sha256",
                    "shard_sha256",
                    "extraction_fingerprint",
                    "summary_sha256",
                ):
                    _required_hex(session[name], name)
        if conditions != {"pre", "post"}:
            raise ValueError("proxy participant does not contain one pre/post pair")
        representation_sha = row["participant_representation_sha256"]
        if source == "random":
            if representation_sha is not None:
                raise ValueError("random proxy may not record a feature representation")
        else:
            _required_hex(representation_sha, "participant_representation_sha256")
        seen.add(participant_id)
        observed_roles[role].append(participant_id)
    expected_order = sorted(seen, key=int)
    if [row["participant_id"] for row in participants] != expected_order:
        raise ValueError("proxy participants are not in numeric canonical order")
    if any(not observed_roles[role] for role in observed_roles):
        raise ValueError("every proxy rotation requires non-empty train, validation and test roles")
    if dict(roles) != observed_roles:
        raise ValueError("proxy role membership disagrees with participant rows")


def _validate_fit_arrays(manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    schema = arrays.get("schema_version_utf8")
    if (
        schema is None
        or schema.dtype != np.uint8
        or schema.tobytes().decode("utf-8") != PROXY_FIT_SCHEMA_VERSION
    ):
        raise ValueError("proxy fit schema is invalid")
    assigned_ids = _decode_ascii_array(arrays.get("assigned_participant_ids"))
    assigned_groups = arrays.get("assigned_groups")
    manifest_ids = [row["participant_id"] for row in manifest["participants"]]
    manifest_groups = [row["proxy_group"] for row in manifest["participants"]]
    if (
        assigned_ids != manifest_ids
        or assigned_groups is None
        or assigned_groups.dtype != np.int8
        or assigned_groups.ndim != 1
        or assigned_groups.tolist() != manifest_groups
    ):
        raise ValueError("proxy fit assignments disagree with the manifest")
    if manifest["proxy_source"] == "random":
        expected_keys = {
            "schema_version_utf8",
            "global_participant_ids",
            "global_groups",
            "assigned_participant_ids",
            "assigned_groups",
        }
        if set(arrays) != expected_keys:
            raise ValueError("random proxy fit arrays are incomplete or unexpected")
        global_ids = _decode_ascii_array(arrays["global_participant_ids"])
        global_groups = arrays["global_groups"]
        if (
            len(global_ids) != EXPECTED_PARTICIPANTS
            or global_groups.dtype != np.int8
            or global_groups.shape != (EXPECTED_PARTICIPANTS,)
            or global_groups.tolist()
            != random_proxy_groups(np.asarray(global_ids, dtype=object)).tolist()
            or np.bincount(global_groups, minlength=2).tolist() != [34, 35]
        ):
            raise ValueError("global random proxy state is invalid or reranked")
        if manifest["fit_participant_count"] != EXPECTED_PARTICIPANTS:
            raise ValueError("random proxy fit participant count is invalid")
        expected_estimator = {
            "algorithm": "global_sha256_rank",
            "salt": RANDOM_PROXY_SALT,
            "separator": "NUL",
            "group_0_count": 34,
            "group_1_count": 35,
            "eligibility_subsetting": "apply_global_map_without_reranking",
            "fold_role_or_seed_dependency": "forbidden",
        }
        if manifest["estimator"] != expected_estimator:
            raise ValueError("random proxy estimator contract is invalid")
        mapping = dict(zip(global_ids, global_groups.tolist(), strict=True))
        if any(
            mapping.get(pid) != group
            for pid, group in zip(assigned_ids, manifest_groups, strict=True)
        ):
            raise ValueError("eligible random groups were reranked after subsetting")
        return

    expected_keys = {
        "schema_version_utf8",
        "training_participant_ids",
        "assigned_participant_ids",
        "assigned_groups",
        "scaler_mean",
        "scaler_scale",
        "scaler_var",
        "scaler_n_samples_seen",
        "cluster_centres_raw",
        "cluster_centres_canonical",
        "raw_to_canonical",
        "kmeans_inertia",
        "kmeans_n_iter",
    }
    if set(arrays) != expected_keys:
        raise ValueError("learned proxy fit arrays are incomplete or unexpected")
    dimension = _LEARNED_SOURCES[manifest["proxy_source"]][1]
    training_ids = _decode_ascii_array(arrays["training_participant_ids"])
    if training_ids != manifest["roles"]["train"]:
        raise ValueError("proxy scaler/KMeans training membership is invalid")
    if manifest["fit_participant_count"] != len(training_ids):
        raise ValueError("learned proxy fit participant count is invalid")
    for name in ("scaler_mean", "scaler_scale", "scaler_var"):
        value = arrays[name]
        if value.dtype != np.float64 or value.shape != (dimension,) or not np.isfinite(value).all():
            raise ValueError(f"proxy {name} is nonfinite or dimensionally invalid")
    if not np.all(arrays["scaler_scale"] > 0) or not np.all(arrays["scaler_var"] >= 0):
        raise ValueError("proxy scaler state is invalid")
    n_samples = arrays["scaler_n_samples_seen"]
    if n_samples.dtype != np.int64 or n_samples.ndim != 0 or int(n_samples) != len(training_ids):
        raise ValueError("proxy scaler fit count is invalid")
    raw_centres = arrays["cluster_centres_raw"]
    canonical_centres = arrays["cluster_centres_canonical"]
    mapping_array = arrays["raw_to_canonical"]
    if (
        raw_centres.dtype != np.float64
        or raw_centres.shape != (2, dimension)
        or not np.isfinite(raw_centres).all()
        or canonical_centres.dtype != np.float64
        or canonical_centres.shape != (2, dimension)
        or not np.isfinite(canonical_centres).all()
        or mapping_array.dtype != np.int8
        or mapping_array.shape != (2,)
    ):
        raise ValueError("proxy cluster centre state is invalid")
    mapping = _canonical_cluster_mapping(raw_centres)
    if mapping_array.tolist() != list(mapping):
        raise ValueError("proxy canonical cluster ordering is invalid")
    expected_canonical = raw_centres[np.argsort(mapping_array)]
    if not np.array_equal(canonical_centres, expected_canonical):
        raise ValueError("proxy canonical centres disagree with raw centres")
    if (
        arrays["kmeans_inertia"].dtype != np.float64
        or arrays["kmeans_inertia"].ndim != 0
        or not np.isfinite(arrays["kmeans_inertia"])
        or arrays["kmeans_n_iter"].dtype != np.int64
        or arrays["kmeans_n_iter"].ndim != 0
        or int(arrays["kmeans_n_iter"]) <= 0
    ):
        raise ValueError("proxy KMeans diagnostics are invalid")
    scaler = type("StoredScaler", (), {})()
    scaler.mean_ = arrays["scaler_mean"]
    scaler.scale_ = arrays["scaler_scale"]
    clusterer = type("StoredKMeans", (), {})()
    clusterer.cluster_centers_ = raw_centres
    expected_manifest = FittedBinaryProxy(scaler, clusterer, mapping).manifest()
    if manifest["estimator"] != expected_manifest:
        raise ValueError("proxy estimator manifest disagrees with fit arrays")


def _validate_stored_learned_assignments(
    artifact: ProxyArtifact | _UnvalidatedProxyArtifact | _ProxyArtifactView,
    eligibility: EligibilityResult,
) -> None:
    """Reapply stored scaler/centres to frozen participant representations."""
    artifact = _artifact_view(artifact)
    if artifact.source not in _LEARNED_SOURCES:
        raise ValueError("stored learned-assignment validation requires a learned proxy")
    arrays = _load_deterministic_npz(artifact.fit_artifact.content)
    representations = participant_representations(
        eligibility.features,
        source=artifact.source,
        outer_rotation=artifact.outer_rotation,
    )
    participant_ids = [row.participant_id for row in representations]
    if _decode_ascii_array(arrays.get("assigned_participant_ids")) != participant_ids:
        raise ValueError("stored proxy assignments do not cover the frozen participant inputs")
    manifest_rows = artifact.manifest["participants"]
    if [row["participant_id"] for row in manifest_rows] != participant_ids or [
        row["participant_representation_sha256"] for row in manifest_rows
    ] != [row.vector_sha256 for row in representations]:
        raise ValueError("stored proxy assignments do not match participant representations")

    matrix = np.stack([row.vector for row in representations], axis=0)
    scaled = (matrix - arrays["scaler_mean"]) / arrays["scaler_scale"]
    raw_centres = arrays["cluster_centres_raw"]
    squared_distances = np.sum(
        np.square(scaled[:, np.newaxis, :] - raw_centres[np.newaxis, :, :]),
        axis=2,
        dtype=np.float64,
    )
    raw_groups = np.argmin(squared_distances, axis=1)
    recomputed_groups = arrays["raw_to_canonical"][raw_groups].astype(np.int8, copy=False)
    assigned_groups = arrays["assigned_groups"]
    if not np.array_equal(recomputed_groups, assigned_groups):
        raise ValueError(
            "stored proxy assignments disagree with the stored scaler, centres and inputs"
        )


def _load_deterministic_npz(content: bytes) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, EOFError) as error:
        raise ValueError("proxy fit artifact is not a safe NPZ") from error
    if deterministic_npz_bytes(arrays) != content:
        raise ValueError("proxy fit artifact is not byte-canonical")
    return arrays


def _proxy_fit_filename(source: ProxySource, outer_rotation: int) -> str:
    """Return the only accepted basename for one source/rotation fit state."""
    return f"{source}.fold-{outer_rotation}.proxy-fit.npz"


def _resolved_output_child(root: Path, filename: str) -> Path:
    """Resolve one canonical basename and prove it remains directly inside ``root``."""
    name = Path(filename)
    if name.is_absolute() or name.name != filename or len(name.parts) != 1:
        raise ValueError("proxy output filename must be one canonical basename")
    lexical_path = root / filename
    if lexical_path.is_symlink():
        raise ValueError("proxy output path may not be a symlink")
    resolved_path = lexical_path.resolve(strict=False)
    if resolved_path.parent != root:
        raise ValueError("proxy output path must remain inside the resolved output directory")
    return resolved_path


def _rotation_role(fixed_fold: int, outer_rotation: int) -> RotationRole:
    if fixed_fold == outer_rotation:
        return "test"
    if fixed_fold == (outer_rotation + 1) % 5:
        return "validation"
    return "train"


def _require_all_roles(rows: Sequence[ParticipantRepresentation]) -> None:
    if {row.rotation_role for row in rows} != {"train", "validation", "test"}:
        raise ValueError("every proxy rotation requires non-empty train, validation and test roles")


def _validate_outer_rotation(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in range(5):
        raise ValueError("outer_rotation must be an integer from 0 to 4")


def _canonical_unique_ids(values: Sequence[str] | np.ndarray) -> tuple[str, ...]:
    if isinstance(values, np.ndarray):
        supplied = values.tolist()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        supplied = list(values)
    else:
        raise TypeError("participant IDs must be a one-dimensional sequence")
    if not supplied:
        raise ValueError("participant IDs must be non-empty")
    identifiers = []
    for value in supplied:
        if not isinstance(value, str):
            raise TypeError("participant IDs must be canonical strings")
        participant_id = canonical_participant_id(value)
        if participant_id != value:
            raise ValueError("participant IDs must already be canonical")
        identifiers.append(participant_id)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("participant IDs must be unique")
    return tuple(identifiers)


def _summary_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("feature summary digest requires a finite float32 vector")
    canonical = np.ascontiguousarray(array.astype("<f4", copy=False))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _ascii_array(values: Sequence[str]) -> np.ndarray:
    if not values:
        raise ValueError("ASCII array values must be non-empty")
    encoded = [value.encode("ascii") for value in values]
    width = max(len(value) for value in encoded)
    return np.asarray(encoded, dtype=f"S{width}")


def _decode_ascii_array(value: np.ndarray | None) -> list[str]:
    if value is None or value.ndim != 1 or value.dtype.kind != "S":
        raise ValueError("proxy participant ID array is invalid")
    try:
        result = [item.decode("ascii") for item in value.tolist()]
    except UnicodeError as error:
        raise ValueError("proxy participant ID array is not ASCII") from error
    canonical = _ascii_array(result)
    if canonical.dtype != value.dtype or canonical.tobytes() != value.tobytes():
        raise ValueError("proxy participant ID array is not canonical")
    _canonical_unique_ids(result)
    return result


def _required_hex(value: Any, name: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return value


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _loads_unique_json(raw: bytes) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("proxy manifest is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("proxy manifest must be a JSON object")
    return payload


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _write_immutable(path: Path, content: bytes) -> None:
    if path.is_symlink():
        raise ValueError("proxy output path may not be a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(f"refusing to replace a different proxy artifact: {path.name}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
