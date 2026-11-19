#!/usr/bin/env python3
"""Audit the locked compact AVI predictions and publish aggregate-only reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.study3.avi_compact import intake_binding, report_run, write_report  # noqa: E402


def main() -> None:
    """Open the audit source only after validating the prediction publication lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--audit-attributes", type=Path)
    parser.add_argument("--intake-release", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    outcomes, attributes = args.outcomes, args.audit_attributes
    expected_outcomes = expected_audit = None
    if outcomes is None or attributes is None:
        if args.intake_release is None:
            parser.error("Set --intake-release or both --outcomes and --audit-attributes")
        if outcomes is None:
            outcomes, expected_outcomes = intake_binding(
                args.intake_release, "participant-outcomes.json"
            )
        if attributes is None:
            attributes, expected_audit = intake_binding(
                args.intake_release, "audit-attributes.json"
            )
    result = report_run(
        args.run_dir,
        outcomes,
        attributes,
        expected_outcomes_sha256=expected_outcomes,
        expected_audit_sha256=expected_audit,
    )
    write_report(result, args.output, args.report)
    print(json.dumps({"status": result["status"], "coverage": result["coverage"]}))


if __name__ == "__main__":
    main()
