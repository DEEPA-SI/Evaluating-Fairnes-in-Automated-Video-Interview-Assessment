"""Record or verify the reviewed public release without inspecting private source trees."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("results/publication_manifest.json")
PUBLISHED_DIRS = {"src", "scripts", "tests", "configs", "results", "data", "docs", ".github"}
ROOT_PATTERNS = (
    "README*",
    "LICENSE*",
    "NOTICE*",
    "CITATION.cff",
    "pyproject.toml",
    "requirements*.lock",
    "requirements*.txt",
    "environment.y*ml",
    "Makefile",
    ".gitignore",
    ".gitattributes",
)
IGNORED = {
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".DS_Store",
    ".coverage",
    "htmlcov",
    "coverage.xml",
}
PRIVATE_DIRS = {
    "runs",
    "embeddings",
    "coordination",
    ".claude",
    ".codex",
    ".agents",
    "raw",
    "raw_data",
    "restricted",
}
PRIVATE_SUFFIXES = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wav",
    ".mp3",
    ".flac",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".xlsx",
    ".xls",
    ".vtt",
    ".srt",
}
PRIVATE_FILENAMES = {
    "predictions.json",
    "prediction-lock.json",
    "face_crops.json",
    "source_session_annotations.csv",
    "feature-extraction-closure.json",
}
TEXT_RISKS = {
    "Hugging Face credential": re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}"),
    "GitHub credential": re.compile(
        r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"
    ),
    "private absolute path": re.compile(
        r"/(?:Users|home)/[A-Za-z0-9_.-]+/|[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+\\"
    ),
}


def _ignored(path: Path) -> bool:
    return any(part in IGNORED or part.endswith(".egg-info") for part in path.parts) or (
        path.suffix in {".pyc", ".pyo"}
    )


def _check_individual_records(payload: Any, relative: str) -> None:
    if isinstance(payload, dict):
        keys = set(payload)
        if keys & {"participant_id", "session_id", "canonical_session_id"} and keys & {
            "prediction",
            "predictions",
            "target",
            "transcript",
            "annotations",
            "attributes",
        }:
            raise ValueError(
                f"Individual-level research record is not a public aggregate: {relative}"
            )
        for value in payload.values():
            _check_individual_records(value, relative)
    elif isinstance(payload, list):
        for value in payload:
            _check_individual_records(value, relative)


def _check_content(path: Path, relative: Path, raw: bytes) -> None:
    if (
        set(relative.parts) & PRIVATE_DIRS
        or path.suffix.lower() in PRIVATE_SUFFIXES
        or (path.name in PRIVATE_FILENAMES)
    ):
        raise ValueError(
            f"Private/raw research artifact cannot be published: {relative.as_posix()}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return
    for kind, pattern in TEXT_RISKS.items():
        if pattern.search(text):
            # Never echo a matched credential or private path into logs.
            raise ValueError(f"Potential {kind} in public file: {relative.as_posix()}")
    if relative.parts[0] == "results" and path.suffix.lower() == ".json":
        _check_individual_records(json.loads(text), relative.as_posix())


def inventory(root: Path = ROOT) -> list[dict[str, Any]]:
    """List reviewed public files and reject private content or filesystem escapes."""
    root = Path(root).resolve()
    files: list[Path] = []
    for item in root.iterdir():
        relative = item.relative_to(root)
        if _ignored(relative):
            continue
        if item.is_symlink():
            raise ValueError(f"Public release contains a symlink: {relative.as_posix()}")
        if item.is_dir() and item.name in PUBLISHED_DIRS:
            candidates = item.rglob("*")
        elif item.is_file() and any(fnmatch.fnmatchcase(item.name, p) for p in ROOT_PATTERNS):
            candidates = (item,)
        else:
            raise ValueError(
                f"Unreviewed root item is outside the public release: {relative.as_posix()}"
            )
        for path in candidates:
            relative = path.relative_to(root)
            if _ignored(relative) or relative == MANIFEST:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Public file is a symlink or escape: {relative.as_posix()}")
            if path.is_dir():
                if set(relative.parts) & PRIVATE_DIRS:
                    raise ValueError(
                        f"Private research directory cannot be published: {relative.as_posix()}"
                    )
                continue
            if not path.is_file():
                raise ValueError(f"Public artifact is not a regular file: {relative.as_posix()}")
            files.append(path)
    records = []
    for path in sorted(files):
        relative = path.relative_to(root)
        raw = path.read_bytes()
        _check_content(path, relative, raw)
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return records


def write_manifest(root: Path = ROOT) -> dict[str, Any]:
    """Intentionally record the current reviewed release; normal users run --check."""
    root = Path(root).resolve()
    path = root / MANIFEST
    if path.is_symlink():
        raise ValueError("Publication manifest must not be a symlink")
    result = {"schema_version": "public-research-release-manifest-1", "files": inventory(root)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def check_manifest(root: Path = ROOT) -> dict[str, Any]:
    """Verify all content, file names, byte sizes and hashes against the recorded release."""
    root = Path(root).resolve()
    path = root / MANIFEST
    if path.is_symlink():
        raise ValueError("Publication manifest must not be a symlink")
    manifest = json.loads(path.read_bytes())
    if manifest.get("schema_version") != "public-research-release-manifest-1":
        raise ValueError("Unsupported public publication manifest")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("Publication manifest must contain a file inventory")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Publication manifest file records must be objects")
        name = record.get("path")
        if not isinstance(name, str) or not name or "\\" in name:
            raise ValueError("Manifest file paths must be relative POSIX paths")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError("Manifest file path escapes the public release")
    if records != inventory(root):
        raise ValueError("Public release differs from the recorded file inventory, sizes or hashes")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Check an existing release, or explicitly record an intentional maintainer update."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--write", action="store_true", help="Record an intentional reviewed release"
    )
    modes.add_argument("--check", action="store_true", help="Verify the current release (default)")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        result = write_manifest(args.root) if args.write else check_manifest(args.root)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Publication verification failed: {error}\n")
    print(f"Publication {'recorded' if args.write else 'verified'}: {len(result['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
