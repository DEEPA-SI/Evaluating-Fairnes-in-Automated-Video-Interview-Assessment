#!/usr/bin/env python3
"""Run the authorised 75-fit cached-feature MIT follow-up."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Set numerical thread caps before importing NumPy/scikit-learn.
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

from src.study2.compact import load_cached_inputs, run_compact  # noqa: E402


def main() -> None:
    """Load accepted cached summaries once, then execute the fixed comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "features/mit-interview")
    parser.add_argument("--stage-a-dir", type=Path, default=ROOT / "runs/stage-a/baselines")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/study2/compact-2026-10-21")
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/mit_compact_alpha1.yaml"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists; preserve the previous attempt")
    print("Loading checksum-bound cached features and unchanged source targets...", flush=True)
    with threadpool_limits(limits=4):
        data = load_cached_inputs(args.feature_dir, args.stage_a_dir)
        print("Loaded 138 sessions; beginning the 75 fixed linear fits.", flush=True)
        run_compact(data, args.output_dir, args.protocol)
    print(f"Prediction lock and annotation-free summary written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
