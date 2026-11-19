#!/usr/bin/env python3
"""Audit locked compact Study-2 predictions using the accepted source annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.study2.compact_audit import audit_compact_run, write_compact_audit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Produce a new aggregate JSON and readable report without individual-level rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotation-release-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = args.report or args.output.with_suffix(".md")
    result = audit_compact_run(args.run_dir, args.annotation_release_dir)
    write_compact_audit(result, args.output, report)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "report": str(report),
                "coverage": result["coverage"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
