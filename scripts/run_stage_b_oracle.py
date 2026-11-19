#!/usr/bin/env python3
"""Run the all-fifteen-or-zero protected-annotation Stage-B oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    load_mit_interview_dataset,
    load_mit_interview_roster,
    load_verified_session_mapping,
)
from src.eval.stage_b_oracle import (  # noqa: E402
    STAGE_B_EXECUTION_OWNED_SOURCE_PATHS,
    prepare_oracle_annotation_binding,
    prepare_oracle_training_releases,
)
from src.eval.stage_c import (  # noqa: E402
    load_annotation_release,
    load_stage_a_release,
    validate_exact_binary_oracle_eligibility,
)
from src.features import load_verified_feature_extraction_closure  # noqa: E402
from src.features.extraction import code_provenance  # noqa: E402
from src.secure_io import AnchoredDirectory  # noqa: E402
from src.training import (  # noqa: E402
    RunProvenance,
    SessionIdentity,
    StageBOracleEvidence,
    build_complete_pair_eligibility,
    build_target_set,
    load_eligibility_lock,
    run_conditional_stage_b_oracle,
    session_identities,
)

REPORT_SCHEMA_VERSION = "fair-avi-mit-stage-b-run-report-1"
ORACLE_FIELDS = (
    "perceived_gender_presentation",
    "perceived_racialised_appearance",
    "perceived_skin_tone",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--session-mapping", type=Path, required=True)
    parser.add_argument("--identity-audit", type=Path, required=True)
    parser.add_argument("--face-crops", type=Path, required=True)
    parser.add_argument("--codebook", type=Path)
    parser.add_argument("--consensus", type=Path)
    parser.add_argument("--annotation-audit", type=Path)
    parser.add_argument("--rater-annotations", type=Path)
    parser.add_argument("--stimulus-manifest", type=Path)
    parser.add_argument(
        "--source-annotation-release",
        type=Path,
        help="accepted single-rater source_annotation_release_manifest.json",
    )
    parser.add_argument("--oracle-field", choices=ORACLE_FIELDS, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/restricted/stage-b/oracle"),
        help="restricted immutable Stage-B oracle directory",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require all model, feature and evidence artefacts to be local",
    )
    parser.add_argument(
        "--synthetic-contract-test",
        action="store_true",
        help="reserved for isolated injected contract fixtures; never research evidence",
    )
    parser.add_argument(
        "--synthetic-smoke-epoch-cap",
        type=int,
        help="1--200 epoch cap; valid only with --synthetic-contract-test",
    )
    return parser.parse_args(argv)


def _regular_file_sha256(path: Path, label: str) -> str:
    source = path if path.is_absolute() else Path.cwd() / path
    try:
        with AnchoredDirectory.open(source.parent, directory_mode=None) as parent:
            return parent.read_bytes(source.name, expected_mode=None).sha256
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"{label} must be a stable regular file") from error


def _validated_output_dir(args: argparse.Namespace) -> Path:
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output = output.absolute()
    if args.synthetic_contract_test:
        return output
    restricted_root = (ROOT / "data" / "restricted").absolute()
    if output == restricted_root or not output.is_relative_to(restricted_root):
        raise ValueError("research Stage-B artifacts must remain beneath data/restricted")
    return output


def _run_provenance(args: argparse.Namespace, release: Any) -> RunProvenance:
    context = release.closure.get("context")
    if not isinstance(context, dict):
        raise ValueError("Stage-A closure lacks its shared execution context")
    config_sha256 = _regular_file_sha256(args.config, "configuration")
    code = code_provenance(ROOT, STAGE_B_EXECUTION_OWNED_SOURCE_PATHS)
    if config_sha256 != context.get("config_sha256"):
        raise ValueError("current configuration does not match the Stage-A closure")
    if not args.synthetic_contract_test and code.dirty:
        raise ValueError("research Stage B requires clean committed Stage-B sources")
    return RunProvenance(
        config_sha256=config_sha256,
        code_commit_sha=code.commit_sha,
        code_source_tree_sha256=code.source_tree_sha256,
        source_date_epoch=context["source_date_epoch"],
    )


def _outcome_blind_roster(
    dataset_root: Path, *, session_mapping: Any
) -> tuple[SessionIdentity, ...]:
    session_mapping.verify_current_files()
    return tuple(
        SessionIdentity(
            session_id=row.session_id,
            source_session_id=row.source_session_id,
            participant_id=row.participant_id,
            condition=row.condition,
            fold=row.fold,
        )
        for row in load_mit_interview_roster(dataset_root, session_mapping=session_mapping)
    )


def _build_stage_b_inputs(args: argparse.Namespace, release: Any) -> tuple[Any, Any]:
    context = release.closure["context"]
    session_mapping = load_verified_session_mapping(
        args.session_mapping,
        identity_join_audit_report_path=args.identity_audit,
        face_crop_manifest_path=args.face_crops,
    )
    if session_mapping.provenance != context["session_mapping"]:
        raise ValueError("current session mapping does not match the Stage-A closure")

    # Preserve the Stage-A outcome firewall: lock features/folds before reopening outcomes.
    roster = _outcome_blind_roster(args.dataset_root, session_mapping=session_mapping)
    with load_verified_feature_extraction_closure(
        args.feature_dir,
        closure_file_sha256=context["feature_extraction_closure_file_sha256"],
        closure_payload_sha256=context["feature_extraction_closure_payload_sha256"],
        session_mapping=session_mapping,
    ) as feature_closure:
        eligibility = build_complete_pair_eligibility(
            roster,
            None,
            feature_closure=feature_closure,
            session_mapping=session_mapping,
            require_full_corpus=True,
        )
        feature_closure.verify_current_files()
    expected_eligibility = {
        "eligibility_manifest_sha256": eligibility.manifest_sha256,
        "fold_manifest_sha256": eligibility.fold_manifest_sha256,
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
        "eligible_participant_count": len(eligibility.participant_ids),
        "expected_oof_row_count": len(eligibility.features),
    }
    if any(context.get(name) != value for name, value in expected_eligibility.items()):
        raise ValueError("current feature eligibility does not match the Stage-A closure")
    eligibility_lock = load_eligibility_lock(release.root, eligibility)

    session_mapping.verify_current_files()
    dataset = load_mit_interview_dataset(args.dataset_root, session_mapping=session_mapping)
    if session_identities(dataset) != roster:
        raise ValueError("post-lock dataset roster disagrees with feature eligibility")
    targets = build_target_set(
        dataset,
        args.dataset_root,
        eligibility=eligibility,
        eligibility_lock=eligibility_lock,
        session_mapping=session_mapping,
    )
    if targets.manifest_sha256 != context["target_manifest_sha256"]:
        raise ValueError("current target set does not match the Stage-A closure")
    session_mapping.verify_current_files()
    return eligibility, targets


def _load_stage_b_evidence(args: argparse.Namespace) -> StageBOracleEvidence:
    """Reload every exact Stage-B source and issue least-authority capabilities."""
    release = load_stage_a_release(args.stage_a_dir)
    execution_mode = release.closure.get("execution_mode")
    expected_mode = "synthetic_contract_test" if args.synthetic_contract_test else "research"
    if execution_mode != expected_mode:
        raise ValueError("Stage-B mode does not match the Stage-A closure")
    provenance = _run_provenance(args, release)
    annotations = load_annotation_release(
        release=release,
        codebook_path=args.codebook,
        consensus_path=args.consensus,
        annotation_audit_path=args.annotation_audit,
        rater_annotations_path=args.rater_annotations,
        stimulus_manifest_path=args.stimulus_manifest,
        source_release_manifest_path=getattr(args, "source_annotation_release", None),
        allow_synthetic_codebook=args.synthetic_contract_test,
    )
    if getattr(args, "source_annotation_release", None) is not None:
        designated_field = getattr(annotations, "designated_stage_b_field", None)
        if not isinstance(designated_field, str) or args.oracle_field != designated_field:
            raise ValueError(
                "accepted source release Stage B requires its designated exploratory field"
            )
    oracle_eligibility = validate_exact_binary_oracle_eligibility(
        release, annotations, field=args.oracle_field
    )

    eligibility = targets = None
    training_releases = ()
    if oracle_eligibility["status"] == "eligible_all_15":
        annotation_binding, training_releases = prepare_oracle_training_releases(
            release=release,
            annotations=annotations,
            eligibility=oracle_eligibility,
        )
        eligibility, targets = _build_stage_b_inputs(args, release)
    else:
        annotation_binding = prepare_oracle_annotation_binding(
            release=release,
            annotations=annotations,
            eligibility=oracle_eligibility,
        )
    return StageBOracleEvidence(
        eligibility=eligibility,
        targets=targets,
        release=release,
        annotation_binding=annotation_binding,
        training_releases=training_releases,
        provenance=provenance,
    )


def run_stage_b_workflow(args: argparse.Namespace) -> Any:
    """Validate every release input before emitting the all-fifteen-or-zero result."""
    if args.local_files_only is not True:
        raise ValueError("Stage B requires --local-files-only")
    output_dir = _validated_output_dir(args)
    if args.synthetic_contract_test:
        if (
            isinstance(args.synthetic_smoke_epoch_cap, bool)
            or not isinstance(args.synthetic_smoke_epoch_cap, int)
            or not 1 <= args.synthetic_smoke_epoch_cap <= 200
        ):
            raise ValueError("synthetic Stage B requires an epoch cap from 1 to 200")
    elif args.synthetic_smoke_epoch_cap is not None:
        raise ValueError("research Stage B cannot use a synthetic epoch cap")

    evidence = _load_stage_b_evidence(args)
    return run_conditional_stage_b_oracle(
        evidence,
        output_dir=output_dir,
        evidence_reloader=(
            None if args.synthetic_contract_test else lambda _boundary: _load_stage_b_evidence(args)
        ),
        synthetic_smoke_epoch_cap=args.synthetic_smoke_epoch_cap,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        disposition = run_stage_b_workflow(args)
    except Exception as error:
        print(f"Stage-B oracle failed: {error}", file=sys.stderr)
        return 2
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "execution_mode": (
            "synthetic_contract_test" if args.synthetic_contract_test else "research"
        ),
        **disposition.public_summary,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
