#!/usr/bin/env python3
"""Fit the reconciled compact AVI comparison from cached participant features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.study3.avi_compact import intake_binding, load_inputs, load_protocol, run_fit  # noqa: E402


def main() -> None:
    """Fit without any audit-path argument or protected-attribute access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument(
        "--intake-release",
        type=Path,
        help="Pinned source-release folder containing its validation receipts",
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/avi_compact_reconciled_v1.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "runs/study3/avi-compact-reconciled-v1"
    )
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    outcomes = args.outcomes
    if outcomes is None:
        if args.intake_release is None:
            parser.error("Set --intake-release or --outcomes")
        outcomes, expected = intake_binding(args.intake_release, "participant-outcomes.json")
        if expected != protocol["outcomes_sha256"]:
            parser.error("Accepted intake and protocol outcome hashes disagree")
    data = load_inputs(
        args.release_root,
        outcomes,
        expected_manifest_sha256=protocol["feature_manifest_sha256"],
        expected_outcomes_sha256=protocol["outcomes_sha256"],
    )
    lock = run_fit(data, args.output_dir, args.protocol)
    print(
        json.dumps(
            {
                key: lock[key]
                for key in ("state", "arm_count", "prediction_count", "fit_status_counts")
            }
        )
    )


if __name__ == "__main__":
    main()
