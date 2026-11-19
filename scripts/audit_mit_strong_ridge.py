#!/usr/bin/env python3
"""Audit the separately locked MIT stronger-ridge predictions against source annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.study2.strong_ridge import (  # noqa: E402
    audit_strong_ridge_run,
    write_strong_ridge_audit,
)


def main(argv: list[str] | None = None) -> int:
    """Write a new aggregate without reading annotations before lock validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotation-release-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = args.report or args.output.with_suffix(".md")
    if args.output.exists() or report.exists() or args.output.resolve() == report.resolve():
        parser.error("Audit outputs must be new, distinct paths")
    result = audit_strong_ridge_run(args.run_dir, args.annotation_release_dir)
    write_strong_ridge_audit(result, args.output, report)
    print(json.dumps({"status": "complete", "study_id": result["study_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
