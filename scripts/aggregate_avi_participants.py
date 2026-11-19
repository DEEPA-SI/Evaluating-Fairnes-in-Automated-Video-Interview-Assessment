#!/usr/bin/env python3
"""Build the target-free, participant-level AVI video/audio feature release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.extract_avi_embeddings import EXPECTED_MANIFEST_SHA256, load_manifest  # noqa: E402
from src.features.avi_participant_aggregation import aggregate_release  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "artifacts/avi/avi_extraction_manifest.csv"
    )
    parser.add_argument("--video-root", type=Path, default=Path("build/avi-extraction/video"))
    parser.add_argument("--audio-root", type=Path, default=Path("build/avi-extraction/audio"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("build/avi-extraction/participants/v1")
    )
    args = parser.parse_args()
    payload = aggregate_release(
        load_manifest(args.manifest),
        video_root=args.video_root,
        audio_root=args.audio_root,
        output_root=args.output_root,
        manifest_sha256=EXPECTED_MANIFEST_SHA256,
    )
    keys = (
        "schema_version",
        "status",
        "participant_count",
        "split_counts",
        "text",
        "participant_shard_ledger_sha256",
        "outcome_or_audit_fields_accessed",
    )
    print(json.dumps({key: payload[key] for key in keys}, sort_keys=True))


if __name__ == "__main__":
    main()
