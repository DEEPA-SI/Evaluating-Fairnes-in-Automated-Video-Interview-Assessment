"""Portable public-aggregate replay checks; no raw data, models or external paths."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.publication import (
    LABELS,
    PRIMARY_STRENGTH_ARMS,
    ROOT,
    SETTINGS,
    SOURCE_FILES,
    complete_grid,
    load_evidence,
    matched_contrasts,
    reproduce,
)


@pytest.fixture(scope="module")
def evidence() -> dict:
    return load_evidence()[0]


def test_all_three_complete_grids_and_raw_rating_units_are_retained(
    evidence: dict,
) -> None:
    rows = complete_grid(evidence)
    assert len(rows) == 45
    assert {r["setting"] for r in rows} == set(SETTINGS)
    for setting in SETTINGS:
        actual = [r for r in rows if r["setting"] == setting]
        assert len(actual) == 15
        assert {r["strength"] for r in actual} == {0, 0.002, 0.02, 0.2}
        for row in actual:
            if setting == "avi":
                source = evidence[setting]["arms"][row["arm_id"]]["utility"]
                assert row["mae_rating_points"] == source["mae_raw_scale"]
                assert row["converged_fits"] is None
                assert row["finite_termination_fits"] is None
                assert row["solver_status_scope"] == "per_arm_status_unavailable"
                assert row["audit_participants"] == 129
            else:
                source = evidence[setting]["utility"][row["arm_id"]]["pooled"]
                assert row["mae_rating_points"] == 6 * source["mae"]
                assert row["converged_fits"] + row["finite_termination_fits"] == 5
                assert row["audit_participants"] == 69
            assert row["srcc"] == source["srcc"]
    strong = next(
        r for r in rows if r["setting"] == "mit_alpha1000" and r["arm_id"] == "natural_task"
    )
    assert strong["mae_normalised"] == pytest.approx(0.08124787100183284)
    assert strong["srcc"] == pytest.approx(0.4945637780043393)


def test_saved_intervals_keep_every_mit_axis_and_phase_with_correct_scale(
    evidence: dict,
) -> None:
    rows = matched_contrasts(evidence)
    assert len(rows) == 112
    for setting in ("mit_alpha1", "mit_alpha1000"):
        disparity = [r for r in rows if r["setting"] == setting and r["metric"] == "disparity_gain"]
        assert len(disparity) == 24
        assert {r["scope"] for r in disparity} == {"pre", "post"}
        assert {r["axis"] for r in disparity} == {
            "perceived_gender_presentation",
            "perceived_racialised_appearance",
            "perceived_skin_tone",
        }
        for row in disparity:
            source = evidence[setting]["perceived_group_audit"][row["scope"]]["axes"][row["axis"]][
                "contrasts"
            ][row["arm_id"]]["disparity_gain"]
            assert [row["estimate"], row["lower_95"], row["upper_95"]] == [
                source["estimate"],
                *source["percentile_95"],
            ]
            assert row["direction"] == "control_minus_arm"
    avi_mae = next(
        r
        for r in rows
        if r["setting"] == "avi"
        and r["metric"] == "delta_mae"
        and r["arm_id"] == PRIMARY_STRENGTH_ARMS[0]
    )
    source = evidence["avi"]["arms"][avi_mae["arm_id"]]["matched_effects"]["delta_mae"]
    assert avi_mae["estimate"] == 4 * source["estimate"]
    assert avi_mae["lower_95"] == 4 * source["percentile_95"][0]
    assert avi_mae["upper_95"] == 4 * source["percentile_95"][1]
    assert avi_mae["direction"] == "arm_minus_control"


def test_near_zero_endpoints_retain_opposite_signs_and_no_new_inference(
    evidence: dict,
) -> None:
    rows = [
        r
        for r in matched_contrasts(evidence)
        if r["setting"] == "mit_alpha1000"
        and r["scope"] == "post"
        and r["axis"] == "perceived_gender_presentation"
    ]
    by_arm = {r["arm_id"]: r for r in rows}
    assert by_arm["visual_natural_w1_lambda-0.02"]["lower_95"] < 0
    assert by_arm["visual_balanced_w1_lambda-0.02"]["lower_95"] > 0
    assert all(r["inference_scope"].startswith("existing_pointwise") for r in rows)


def test_replay_is_deterministic_and_source_preserving(tmp_path: Path) -> None:
    sources = {
        filename: (ROOT / "results" / filename).read_bytes() for filename in SOURCE_FILES.values()
    }
    first, second = tmp_path / "first", tmp_path / "second"
    receipt = reproduce(first)
    again = reproduce(second)
    assert receipt == again
    assert receipt["complete_grid_rows"] == 45 and receipt["matched_contrast_rows"] == 112
    assert receipt["source_sha256"] == {
        filename: hashlib.sha256(raw).hexdigest() for filename, raw in sources.items()
    }
    assert set(receipt["output_sha256"]) == {
        "complete_grid.csv",
        "matched_contrasts.csv",
        "contribution_overview.md",
        *(
            f"{name}.{extension}"
            for name in ("baseline_quality", "lambda_tradeoffs", "primary_gender_gains")
            for extension in ("svg", "png")
        ),
    }
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
        if path.suffix == ".png":
            assert path.read_bytes().startswith(b"\x89PNG")
        elif path.suffix == ".svg":
            assert "<svg" in path.read_text()
    with (first / "complete_grid.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 45
    assert sources == {
        filename: (ROOT / "results" / filename).read_bytes() for filename in SOURCE_FILES.values()
    }
    report = (first / "contribution_overview.md").read_text()
    assert all(label in report for label in LABELS.values())
    assert "does not" in report and "rerun" in report
    assert "individual-arm or primary-arm convergence" in report
    saved = json.loads((first / "source_receipt.json").read_bytes())
    assert saved == receipt


def test_existing_output_is_refused_without_overwriting_anything(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        reproduce(output, source_dir=tmp_path / "absent")
    assert marker.read_text() == "preserve"
