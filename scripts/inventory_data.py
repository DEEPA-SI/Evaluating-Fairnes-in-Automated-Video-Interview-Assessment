"""Validate and inventory the locally supplied MIT Interview corpus."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.data.inventory import build_inventory, canonical_json, write_inventory


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="../../datasets/mit-interview",
        help="restricted corpus root; the path is never written to inventory output",
    )
    parser.add_argument(
        "--output",
        default="data/interim/mit_interview_inventory.json",
        help="generated canonical aggregate inventory",
    )
    parser.add_argument(
        "--session-mapping",
        default="src/data/mit_source_session_map.json",
        help="approved canonical-to-source session mapping artefact",
    )
    parser.add_argument(
        "--identity-audit",
        default="data/restricted/mit_identity_join_audit.json",
        help="exact identity-join audit report bound by the session mapping",
    )
    parser.add_argument(
        "--face-crops",
        default="data/restricted/face_crops.json",
        help="exact face-crop manifest bound by the session mapping",
    )
    return parser


def _load_mapping_context(
    session_mapping_path: Path,
    *,
    identity_join_audit_report_path: Path,
    face_crop_manifest_path: Path,
) -> Any:
    """Load the shared fail-closed mapping context without widening inventory imports."""
    from src.data.mit_session_mapping import load_verified_session_mapping

    return load_verified_session_mapping(
        session_mapping_path,
        identity_join_audit_report_path=identity_join_audit_report_path,
        face_crop_manifest_path=face_crop_manifest_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation and emit identical canonical JSON to disk and stdout."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        session_mapping = _load_mapping_context(
            Path(arguments.session_mapping),
            identity_join_audit_report_path=Path(arguments.identity_audit),
            face_crop_manifest_path=Path(arguments.face_crops),
        )
        inventory = build_inventory(
            Path(arguments.dataset_root), session_mapping=session_mapping
        )
        write_inventory(inventory, Path(arguments.output))
    except (OSError, ValueError) as error:
        parser.exit(1, f"data inventory failed: {error}\n")
    sys.stdout.write(canonical_json(inventory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
