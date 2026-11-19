"""Public release integrity and metadata guards, independent of private checkouts."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from scripts.publication_manifest import MANIFEST, ROOT, check_manifest, inventory, write_manifest


def _fixture(root: Path) -> Path:
    (root / "src").mkdir()
    (root / "src/example.py").write_text('"""A public numerical method."""\n')
    (root / "README.md").write_text("# Public research\n")
    return root


def test_manifest_is_deterministic_and_detects_tampering(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    first = write_manifest(root)
    original = (root / MANIFEST).read_bytes()
    assert write_manifest(root) == first
    assert (root / MANIFEST).read_bytes() == original
    assert check_manifest(root) == first
    assert all(set(row) == {"path", "size_bytes", "sha256"} for row in first["files"])
    assert not any(row["path"] == MANIFEST.as_posix() for row in first["files"])
    (root / "src/example.py").write_text("altered after publication\n")
    with pytest.raises(ValueError, match="differs"):
        check_manifest(root)


def test_manifest_detects_added_and_removed_published_files(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    write_manifest(root)
    extra = root / "src/extra.py"
    extra.write_text("# an unrecorded file\n")
    with pytest.raises(ValueError, match="differs"):
        check_manifest(root)
    extra.unlink()
    (root / "src/example.py").unlink()
    with pytest.raises(ValueError, match="differs"):
        check_manifest(root)


def test_manifest_ignores_local_build_and_cache_artifacts(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    first = write_manifest(root)
    for folder in ("build", ".venv", ".git", "src/__pycache__", ".pytest_cache"):
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        (path / "local-only.pyc").write_bytes(b"not published")
    assert check_manifest(root) == first


def test_symlink_and_manifest_escape_are_rejected(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    root = _fixture(release)
    outside = tmp_path / "outside.txt"
    outside.write_text("not a public input")
    link = root / "src/linked.txt"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        inventory(root)
    link.unlink()
    payload = write_manifest(root)
    payload["files"][0]["path"] = "../outside.txt"
    (root / MANIFEST).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="escapes"):
        check_manifest(root)


@pytest.mark.parametrize("name", ["runs", "features", "coordination"])
def test_private_research_roots_are_not_public_exports(tmp_path: Path, name: str) -> None:
    root = _fixture(tmp_path)
    (root / name).mkdir()
    with pytest.raises(ValueError, match="outside the public release"):
        inventory(root)


def test_feature_source_package_is_not_confused_with_feature_data(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    (root / "src/features").mkdir()
    (root / "src/features/example.py").write_text("# feature extraction implementation\n")
    assert "src/features/example.py" in {row["path"] for row in inventory(root)}


@pytest.mark.parametrize("suffix", [".npz", ".mp4", ".wav", ".xlsx", ".pt"])
def test_raw_media_features_and_individual_sources_are_rejected(
    tmp_path: Path, suffix: str
) -> None:
    root = _fixture(tmp_path)
    (root / "data").mkdir()
    (root / "data" / f"private-input{suffix}").write_bytes(b"not an aggregate")
    with pytest.raises(ValueError, match="Private/raw"):
        inventory(root)


@pytest.mark.parametrize("kind", ["hf", "gh", "private_path"])
def test_real_looking_secrets_and_private_paths_fail_without_echoing_them(
    tmp_path: Path, kind: str
) -> None:
    root = _fixture(tmp_path)
    value = {
        "hf": "hf_" + "X" * 32,
        "gh": "ghp_" + "Y" * 36,
        "private_path": "/" + "Users" + "/" + "private-researcher" + "/" + "dataset",
    }[kind]
    (root / "README.md").write_text(value)
    with pytest.raises(ValueError, match="Potential") as error:
        inventory(root)
    assert value not in str(error.value)


def test_individual_prediction_records_are_not_accepted_as_aggregates(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    (root / "results").mkdir()
    (root / "results/example.json").write_text(
        json.dumps({"rows": [{"participant_id": "synthetic", "prediction": 0.5}]})
    )
    with pytest.raises(ValueError, match="Individual-level"):
        inventory(root)


def test_actual_package_contains_only_reviewed_public_artifacts() -> None:
    records = inventory(ROOT)
    names = {row["path"] for row in records}
    assert {"README.md", "LICENSE", "NOTICE.md", "CITATION.cff", "pyproject.toml"} <= names
    assert not any(name.startswith(("runs/", "features/", "coordination/")) for name in names)


def test_public_metadata_and_citation_identify_the_actual_thesis() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    cff = (ROOT / "CITATION.cff").read_text()
    title = "Evaluating Fairness in Automated Video Interview Assessment"
    assert (ROOT / "README.md").read_text().startswith(f"# {title}")
    assert "cff-version: 1.2.0" in cff
    assert f'title: "{title}"' in cff
    assert 'family-names: "Singh"' in cff and 'given-names: "Deepa"' in cff
    assert project["authors"] == [{"name": "Deepa Singh"}]
    assert project["urls"]["Repository"] in cff
    assert "preferred-citation:" in cff and "type: thesis" in cff


def test_local_document_links_resolve_within_the_public_checkout() -> None:
    documents = [
        ROOT / "README.md",
        *sorted((ROOT / "docs").glob("*.md")),
        ROOT / "data/README.md",
        ROOT / "results/README.md",
    ]
    for document in documents:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            if urlparse(target).scheme or target.startswith("#"):
                continue
            target = unquote(target.split("#", 1)[0].strip("<>"))
            path = (document.parent / target).resolve()
            assert path.is_relative_to(ROOT), (document.name, target)
            assert path.exists(), (document.name, target)
