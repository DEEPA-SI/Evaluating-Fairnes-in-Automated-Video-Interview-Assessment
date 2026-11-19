#!/usr/bin/env python3
"""Fit the separately identified 75-fit MIT alpha-1000 comparison from cached inputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "4")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threadpoolctl import threadpool_limits  # noqa: E402

from src.study2.compact import load_cached_inputs  # noqa: E402
from src.study2.strong_ridge import PROTOCOL_PATH, load_protocol, run_strong_ridge  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Validate the frozen protocol before opening cached summaries; never open annotations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "features/mit-interview")
    parser.add_argument("--stage-a-dir", type=Path, default=ROOT / "runs/stage-a/baselines")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Output directory already exists; preserve the previous attempt")
    load_protocol(args.protocol)
    with threadpool_limits(limits=4):
        data = load_cached_inputs(args.feature_dir, args.stage_a_dir)
        run_strong_ridge(data, args.output_dir, args.protocol)
    print(f"Stronger-ridge predictions locked at {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
