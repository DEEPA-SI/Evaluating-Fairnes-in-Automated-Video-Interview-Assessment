#!/usr/bin/env python3
"""Run the one-way Stage-C audit over immutable Stage-A predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.stage_c import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    StageCSourcePaths,
    evaluate_stage_c,
    evaluate_stage_c_from_paths,
    load_annotation_release,
    load_stage_a_release,
    write_stage_c_outputs,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-dir", type=Path, required=True)
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
    parser.add_argument(
        "--oracle-field",
        choices=(
            "perceived_gender_presentation",
            "perceived_racialised_appearance",
            "perceived_skin_tone",
        ),
        required=True,
    )
    parser.add_argument(
        "--stage-b-oracle-disposition",
        type=Path,
        help="immutable stage-b-oracle-lock.json (all 15 fits locked or all omitted)",
    )
    parser.add_argument("--modality-mask-predictions", type=Path)
    parser.add_argument(
        "--restricted-join-output",
        type=Path,
        default=Path("data/restricted/stage-c/restricted-join.json"),
    )
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        default=Path("runs/stage-c/aggregate.json"),
    )
    parser.add_argument(
        "--claim-ledger-output",
        type=Path,
        default=Path("runs/stage-c/claim-ledger.json"),
    )
    parser.add_argument(
        "--no-rerun-output",
        type=Path,
        default=Path("runs/stage-c/no-rerun.json"),
    )
    parser.add_argument(
        "--synthetic-contract-test",
        action="store_true",
        help="allow a synthetic codebook and a reduced bootstrap; never research evidence",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args(argv)


def _resolve_output(path: Path) -> Path:
    supplied = path.expanduser()
    if any(part in {".", ".."} for part in supplied.parts):
        raise ValueError("Stage-C output paths may not contain dot components")
    return supplied if supplied.is_absolute() else ROOT / supplied


def _validated_output_scopes(args: argparse.Namespace) -> dict[str, Path]:
    outputs = {
        "restricted_join": _resolve_output(args.restricted_join_output),
        "aggregate": _resolve_output(args.aggregate_output),
        "claim_ledger": _resolve_output(args.claim_ledger_output),
        "no_rerun": _resolve_output(args.no_rerun_output),
    }
    if args.synthetic_contract_test:
        return outputs
    restricted_root = ROOT / "data" / "restricted"
    restricted = outputs["restricted_join"]
    if restricted == restricted_root or not restricted.is_relative_to(restricted_root):
        raise ValueError("research restricted join must be beneath data/restricted")
    for name in ("aggregate", "claim_ledger", "no_rerun"):
        public = outputs[name]
        if public == restricted_root or public.is_relative_to(restricted_root):
            raise ValueError("aggregate Stage-C outputs may not be stored beneath data/restricted")
    return outputs


def _validated_research_sources(args: argparse.Namespace) -> StageCSourcePaths:
    raw = {
        "stage_a_dir": args.stage_a_dir,
        "codebook": args.codebook,
        "consensus": args.consensus,
        "annotation_audit": args.annotation_audit,
        "rater_annotations": args.rater_annotations,
        "stimulus_manifest": args.stimulus_manifest,
        "source_annotation_release": args.source_annotation_release,
        "stage_b_oracle_disposition": args.stage_b_oracle_disposition,
        "modality_mask_predictions": args.modality_mask_predictions,
    }
    resolved = {name: None if path is None else _resolve_output(path) for name, path in raw.items()}
    restricted_root = ROOT / "data" / "restricted"
    for name in (
        "codebook",
        "consensus",
        "annotation_audit",
        "rater_annotations",
        "stimulus_manifest",
        "source_annotation_release",
        "stage_b_oracle_disposition",
        "modality_mask_predictions",
    ):
        path = resolved[name]
        if path is None:
            continue
        if path == restricted_root or not path.is_relative_to(restricted_root):
            raise ValueError(f"research Stage-C {name} must remain beneath data/restricted")
        _reject_existing_symlink_components(path, name)
    return StageCSourcePaths(**resolved)


def _reject_existing_symlink_components(path: Path, label: str) -> None:
    """Reject visible source-path aliases early; the evaluator repeats this via dirfds."""
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"research Stage-C {label} may not contain symlink components")
        if not current.exists():
            break


def _report_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output_paths = _validated_output_scopes(args)
        if args.synthetic_contract_test:
            if args.source_annotation_release is not None or any(
                path is None
                for path in (
                    args.codebook,
                    args.consensus,
                    args.annotation_audit,
                    args.rater_annotations,
                    args.stimulus_manifest,
                )
            ):
                raise ValueError("synthetic Stage C requires all five legacy annotation paths")
            release = load_stage_a_release(args.stage_a_dir)
            annotations = load_annotation_release(
                release=release,
                codebook_path=args.codebook,
                consensus_path=args.consensus,
                annotation_audit_path=args.annotation_audit,
                rater_annotations_path=args.rater_annotations,
                stimulus_manifest_path=args.stimulus_manifest,
                allow_synthetic_codebook=True,
            )
            result = evaluate_stage_c(
                release,
                annotations,
                oracle_field=args.oracle_field,
                stage_b_oracle_path=args.stage_b_oracle_disposition,
                modality_mask_path=args.modality_mask_predictions,
                bootstrap_replicates=args.bootstrap_replicates,
                synthetic_contract_test=True,
            )
        else:
            if args.stage_b_oracle_disposition is None:
                raise ValueError("research Stage C requires the immutable Stage-B lock or omission")
            if args.modality_mask_predictions is None:
                raise ValueError("research Stage C requires locked modality-mask predictions")
            result = evaluate_stage_c_from_paths(
                source_paths=_validated_research_sources(args),
                oracle_field=args.oracle_field,
                bootstrap_replicates=args.bootstrap_replicates,
            )
        aggregate = result.aggregate
        outputs = write_stage_c_outputs(
            restricted_join_path=output_paths["restricted_join"],
            aggregate_path=output_paths["aggregate"],
            claim_ledger_path=output_paths["claim_ledger"],
            no_rerun_path=output_paths["no_rerun"],
            result=result,
        )
    except Exception as error:
        print(f"Stage-C evaluation failed: {error}", file=sys.stderr)
        return 2
    report = {
        "status": "closed",
        "execution_mode": aggregate["execution_mode"],
        "stage_a_closure_file_sha256": aggregate["stage_a"]["closure_file_sha256"],
        "aggregate_payload_sha256": aggregate["aggregate_payload_sha256"],
        "bootstrap_replicates": aggregate["participant_cluster_bootstrap"]["resamples"],
        "outputs": {name: _report_path(path) for name, path in outputs.items()},
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
