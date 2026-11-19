"""Replay public aggregate tables and figures without fitting or bootstrap execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = {
    "mit_alpha1": "study2-compact-2026-10-21-aggregate.json",
    "mit_alpha1000": "mit-strong-ridge-2026-11-19-v1.json",
    "avi": "avi-compact-reconciled-v1.json",
    "neural": "stage_a_secondary_utility.json",
    "association": "proxy-association-correction-2026-11-06.json",
}
SETTINGS = ("mit_alpha1", "mit_alpha1000", "avi")
LABELS = {
    "mit_alpha1": "MIT: ridge alpha 1",
    "mit_alpha1000": "MIT: ridge alpha 1000",
    "avi": "AVI: validation-selected ridge",
}
STRENGTHS = (0.0, 0.002, 0.02, 0.2)
PRIMARY_STRENGTH_ARMS = tuple(
    f"{proxy}_{exposure}_w1_lambda-0.02"
    for proxy in ("visual", "audio")
    for exposure in ("natural", "balanced")
)


def load_evidence(source_dir: Path = ROOT / "results") -> tuple[dict, dict[str, str]]:
    """Read the five published aggregates once and bind their exact public bytes."""
    evidence, hashes = {}, {}
    for name, filename in SOURCE_FILES.items():
        raw = (source_dir / filename).read_bytes()
        evidence[name] = json.loads(raw)
        hashes[filename] = hashlib.sha256(raw).hexdigest()
    for setting in SETTINGS:
        arms = evidence[setting]["arms" if setting == "avi" else "utility"]
        if len(arms) != 15 or not set(PRIMARY_STRENGTH_ARMS).issubset(arms):
            raise ValueError(f"Published {setting} evidence must retain all fifteen arms")
    return evidence, hashes


def _metadata(arm: str) -> tuple[str, str, float]:
    if arm == "natural_task":
        return "none", "natural", 0.0
    proxy, exposure = arm.split("_")[:2]
    return proxy, exposure, float(arm.split("lambda-")[1]) if "lambda-" in arm else 0.0


def _control(arm: str) -> str:
    proxy, exposure, _ = _metadata(arm)
    return "natural_task" if exposure == "natural" else f"{proxy}_balanced_task"


def _proxy_w1(data: dict, setting: str, arm: str, proxy: str) -> float:
    if setting == "avi":
        return next(
            row["proxy_w1"]
            for row in data["proxy_mechanism"]
            if row["arm_id"] == arm and row["proxy_source"] == proxy and row["role"] == "Testing"
        )
    _, exposure, strength = _metadata(arm)
    key = arm if strength else f"{proxy}_{exposure}_w1_lambda-0.002"
    field = "proxy_w1" if strength else "matched_control_proxy_w1"
    return fmean(
        row[field]
        for row in data["held_out_proxy_mechanism"][key]["fold_local"]
        if row["scope"] == "pooled"
    )


def complete_grid(evidence: dict) -> list[dict[str, Any]]:
    """Return all 45 arms with explicit units, rosters and available solver evidence."""
    rows = []
    for setting in SETTINGS:
        data = evidence[setting]
        arms = data["arms"] if setting == "avi" else data["utility"]
        for arm in sorted(arms):
            proxy, exposure, strength = _metadata(arm)
            if setting == "avi":
                source = data["arms"][arm]
                utility = source["utility"]
                distance = source["score_w1"]["estimate"]
                gain = source["matched_effects"]["disparity_gain"]["estimate"]
                alpha, factor, scope = data["selection"]["alpha"], 4, "official_test"
                primary = source["primary_contrast"]
                audit_n = sum(
                    g["n_participants"] for g in source["group_errors"].values() if g["reportable"]
                )
                solver = source.get("solver")
                converged = int(solver["optimizer_converged"]) if solver else None
                terminated = int(not solver["optimizer_converged"]) if solver else None
                status_scope = "per_arm_receipt" if solver else "per_arm_status_unavailable"
                fit_count = 1
            else:
                utility = data["utility"][arm]["pooled"]
                gender = data["perceived_group_audit"]["post"]["axes"][
                    "perceived_gender_presentation"
                ]
                distance = gender["arms"][arm]["max_pairwise_w1"]
                gain = 0.0
                if strength:
                    effect = gender["contrasts"][arm]["disparity_gain"]
                    gain = effect["estimate"] if isinstance(effect, dict) else effect
                alpha = data.get("ridge_alpha", 1.0)
                factor, scope = 6, "pooled_out_of_fold_sessions"
                primary = arm in data["method"]["primary_contrasts"]
                audit_n = data["perceived_group_audit"]["post"]["n_participants"]
                fits = [f for f in data["training_and_solver_diagnostics"] if f["arm_id"] == arm]
                fit_count = len(fits)
                converged = sum(f["optimizer_converged"] for f in fits)
                terminated = fit_count - converged
                status_scope = "five_outer_fold_receipts"
            numeric = (utility["mae"], utility["srcc"], distance)
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(f"Nonfinite published utility/disparity in {setting}/{arm}")
            rows.append(
                {
                    "setting": setting,
                    "arm_id": arm,
                    "matched_control": _control(arm),
                    "proxy_source": proxy,
                    "exposure": exposure,
                    "strength": strength,
                    "ridge_alpha": alpha,
                    "design_primary_arm": primary,
                    "utility_scope": scope,
                    "utility_sessions": utility["n_sessions"],
                    "audit_participants": audit_n,
                    "mae_normalised": utility["mae"],
                    "mae_rating_points": factor * utility["mae"],
                    "rmse_rating_points": factor * utility["rmse"],
                    "srcc": utility["srcc"],
                    "gender_score_w1_normalised": distance,
                    "gender_gain_vs_matched_control": gain,
                    "held_out_proxy_w1": _proxy_w1(data, setting, arm, proxy)
                    if proxy != "none"
                    else None,
                    "fit_count": fit_count,
                    "converged_fits": converged,
                    "finite_termination_fits": terminated,
                    "solver_status_scope": status_scope,
                }
            )
    return rows


def _effect_row(
    setting: str,
    arm: str,
    scope: str,
    axis: str,
    metric: str,
    effect: dict,
    *,
    factor: float = 1,
    primary: bool = False,
    unit: str = "dimensionless",
) -> dict[str, Any]:
    lower, upper = effect["percentile_95"]
    return {
        "setting": setting,
        "arm_id": arm,
        "matched_control": _control(arm),
        "scope": scope,
        "axis": axis,
        "metric": metric,
        "design_primary_contrast": primary,
        "estimate": effect["estimate"] * factor,
        "lower_95": lower * factor,
        "upper_95": upper * factor,
        "unit": unit,
        "direction": "control_minus_arm" if metric == "disparity_gain" else "arm_minus_control",
        "valid_draws": effect.get("valid_draws"),
        "undefined_draws": effect.get("undefined_draws"),
        "inference_scope": "existing_pointwise_percentiles_conditional_on_fixed_predictions",
    }


def matched_contrasts(evidence: dict) -> list[dict[str, Any]]:
    """Copy saved strength-0.02 contrasts/CIs, converting only documented rating units."""
    rows = []
    for setting in SETTINGS:
        data = evidence[setting]
        for arm in PRIMARY_STRENGTH_ARMS:
            if setting == "avi":
                source = data["arms"][arm]
                effects, primary = source["matched_effects"], source["primary_contrast"]
                for key, metric, factor, unit, axis in (
                    (
                        "delta_mae",
                        "delta_mae",
                        4,
                        "1_to_5_rating_points",
                        "all_participants",
                    ),
                    (
                        "delta_srcc",
                        "delta_srcc",
                        1,
                        "dimensionless",
                        "all_participants",
                    ),
                    (
                        "disparity_gain",
                        "disparity_gain",
                        1,
                        "normalised_score",
                        "self_reported_gender",
                    ),
                    (
                        "delta_signed_mae_gap",
                        "delta_signed_mae_gap",
                        4,
                        "1_to_5_rating_points",
                        "self_reported_gender",
                    ),
                ):
                    rows.append(
                        _effect_row(
                            setting,
                            arm,
                            "official_test",
                            axis,
                            metric,
                            effects[key],
                            factor=factor,
                            primary=primary,
                            unit=unit,
                        )
                    )
                continue
            for scope in ("pooled", "post", "pre"):
                contrast = data["utility_contrasts"][arm]
                for metric, factor, unit in (
                    ("mae", 6, "1_to_7_rating_points"),
                    ("srcc", 1, "dimensionless"),
                ):
                    rows.append(
                        _effect_row(
                            setting,
                            arm,
                            scope,
                            "all_participants",
                            f"delta_{metric}",
                            contrast[scope][metric],
                            factor=factor,
                            primary=contrast["primary_contrast"],
                            unit=unit,
                        )
                    )
            for phase in ("post", "pre"):
                for axis, values in data["perceived_group_audit"][phase]["axes"].items():
                    contrast = values["contrasts"][arm]
                    rows.append(
                        _effect_row(
                            setting,
                            arm,
                            phase,
                            axis,
                            "disparity_gain",
                            contrast["disparity_gain"],
                            primary=contrast["primary_contrast"],
                            unit="normalised_score",
                        )
                    )
    return rows


def _number(value: float) -> str:
    return f"{value:.2g}" if value and abs(value) < 0.0005 else f"{value:.3f}"


def contribution_overview(evidence: dict, grid: list[dict], contrasts: list[dict]) -> str:
    """Summarise the measured contributions without converting alignment into mitigation."""
    lines = [
        "# Aggregate research overview",
        "",
        (
            "This release examines prediction, learned-group score alignment and independently "
            "measured group outcomes as separate questions. Matched task-only controls distinguish "
            "the added Wasserstein penalty from balanced exposure. Every fixed strength remains "
            "in the complete comparison; no best arm is selected from the audit."
        ),
        "",
        "## Predictive references",
        "",
        "| Setting | MAE in original rating points | Spearman correlation | Evaluation |",
        "|---|---:|---:|---|",
    ]
    for row in grid:
        if row["arm_id"] == "natural_task":
            lines.append(
                f"| {LABELS[row['setting']]} | {_number(row['mae_rating_points'])} | "
                f"{_number(row['srcc'])} | {row['utility_scope'].replace('_', ' ')} |"
            )
    lines += [
        "",
        (
            "MIT errors use a 1–7 rating scale and AVI errors a 1–5 scale; they are "
            "not pooled across corpora. MIT's pooled correlations are not the separately "
            "reported mean-fold correlation of the development-selected procedure."
        ),
        "",
    ]
    association = next(
        item
        for item in evidence["association"]["sources"]["visual"]["aligned_oof_associations"]
        if item["field"] == "perceived_gender_presentation"
    )
    lines += [
        "## What the learned groups represent",
        "",
        (
            f"The visual proxy partition agrees exactly with one observer's perceived "
            f"gender-presentation grouping in the corrected MIT comparison "
            f"(NMI {_number(association['nmi'])}; "
            f"adjusted Rand index {_number(association['ari'])}). "
            "Here, excluding annotations during fitting still leaves associated information "
            "in the representations. These measurements describe perceived presentation, "
            "not verified demographic identity."
        ),
        "",
        "## Primary-strength group-score comparisons",
        "",
        "| Setting and arm | Gender score-disparity gain | Existing 95% interval |",
        "|---|---:|---:|",
    ]
    for row in contrasts:
        if (
            row["metric"] == "disparity_gain"
            and row["scope"] in {"post", "official_test"}
            and row["axis"] in {"perceived_gender_presentation", "self_reported_gender"}
        ):
            lines.append(
                f"| {LABELS[row['setting']]}: {row['arm_id'].replace('_', ' ')} | "
                f"{_number(row['estimate'])} | [{_number(row['lower_95'])}, "
                f"{_number(row['upper_95'])}] |"
            )
    lines += [
        "",
        (
            "Positive gain denotes a smaller score-distribution distance, not necessarily "
            "more accurate prediction for every group. The CSV retains every available MIT "
            "axis and pre-session sensitivity, plus AVI's separately sourced gender audit. "
            "AVI companions are marked secondary unless its source explicitly designates "
            "the contrast primary."
        ),
        "",
        "## Reproduction scope",
        "",
        (
            "This command replays tables and figures from published aggregates. It does not "
            "extract features, train models, reconstruct participant predictions or rerun "
            "participant-bootstrap inference. Intervals are copied from the saved paired "
            "pointwise analyses, conditional on the fitted predictions. MIT attributes come "
            "from one observer; AVI uses self-reported gender. AVI's aggregate solver counts "
            "do not identify individual-arm or primary-arm convergence."
        ),
        "",
    ]
    return "\n".join(lines)


def _figures(evidence: dict, grid: list[dict], contrasts: list[dict]) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "interview-fairness-publication",
            "svg.fonttype": "none",
        }
    )
    figures = {}
    references = [
        (LABELS[r["setting"]], r["mae_rating_points"], r["srcc"])
        for r in grid
        if r["setting"] != "avi" and r["arm_id"] == "natural_task"
    ]
    neural = next(
        row
        for row in evidence["neural"]["pooled_utility"]["learned_three_seed_summaries"]
        if row["configuration"] == "erm"
    )
    references.insert(
        0,
        (
            f"Neural ERM (mean of {neural['fixed_seed_count']} seeds)",
            neural["metrics"]["mae"]["original_1_7"]["mean"],
            neural["metrics"]["srcc"]["mean"],
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True)
    y = np.arange(len(references))
    for ax, index, title in zip(
        axes,
        (1, 2),
        (
            "MAE: original 1–7 rating points (lower is better)",
            "Spearman correlation (higher is better)",
        ),
        strict=True,
    ):
        ax.barh(y, [r[index] for r in references], color=["#738392", "#D88929", "#127B75"])
        ax.set_yticks(y, [r[0] for r in references])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("MIT predictive references: the same participant-grouped task")
    figures["baseline_quality"] = fig
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    rows = {(r["setting"], r["arm_id"]): r for r in grid}
    for row_index, setting in enumerate(SETTINGS):
        for proxy, color in (("visual", "#127B75"), ("audio", "#D88929")):
            for exposure, style in (("natural", "-"), ("balanced", "--")):
                arms = ["natural_task" if exposure == "natural" else f"{proxy}_balanced_task"] + [
                    f"{proxy}_{exposure}_w1_lambda-{v:g}" for v in STRENGTHS[1:]
                ]
                values = [
                    [_proxy_w1(evidence[setting], setting, arm, proxy) for arm in arms],
                    [rows[setting, arm]["mae_rating_points"] for arm in arms],
                    [rows[setting, arm]["gender_score_w1_normalised"] for arm in arms],
                ]
                for col, points in enumerate(values):
                    axes[row_index, col].plot(
                        STRENGTHS,
                        points,
                        marker="o",
                        markersize=4,
                        linestyle=style,
                        color=color,
                        label=f"{proxy}, {exposure}",
                    )
                    for strength, arm, point in zip(STRENGTHS, arms, points, strict=True):
                        terminated = rows[setting, arm]["finite_termination_fits"]
                        if terminated is not None and terminated > 0:
                            axes[row_index, col].scatter(
                                strength, point, marker="x", color=color, s=60, zorder=4
                            )
        for col, heading in enumerate(
            (
                "Held-out proxy W1",
                "MAE in original rating points",
                "Audited gender score W1",
            )
        ):
            ax = axes[row_index, col]
            ax.set_xscale("symlog", linthresh=0.002)
            ax.set_xticks(STRENGTHS, ["0", ".002", ".02", ".2"])
            ax.set_title(f"{LABELS[setting]}\n{heading}", fontsize=10)
            ax.set_xlabel("W1 coefficient (discrete grid)")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Complete fixed-strength comparisons: mechanism, prediction and audit remain distinct\n"
        "Crosses: one or more finite solver terminations; "
        "AVI individual-arm statuses are unavailable"
    )
    figures["lambda_tradeoffs"] = fig
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, setting in zip(axes, SETTINGS, strict=True):
        points = [
            row
            for row in contrasts
            if row["setting"] == setting
            and row["metric"] == "disparity_gain"
            and row["scope"] in {"post", "official_test"}
            and row["axis"] in {"perceived_gender_presentation", "self_reported_gender"}
        ]
        for index, point in enumerate(points):
            color = "#127B75" if point["arm_id"].startswith("visual") else "#D88929"
            ax.plot(
                [point["lower_95"], point["upper_95"]],
                [index, index],
                color=color,
                linewidth=2,
            )
            ax.scatter(
                [point["estimate"]],
                [index],
                color=color,
                marker="o" if point["design_primary_contrast"] else "s",
                zorder=3,
            )
        ax.axvline(0, color="#54616D", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(points)), [" ".join(p["arm_id"].split("_")[:2]) for p in points])
        ax.invert_yaxis()
        ax.set_title(LABELS[setting])
        ax.grid(axis="x", alpha=0.2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlabel("Control minus penalty: normalised score W1")
    fig.suptitle(
        "Gender score-disparity gains at coefficient 0.02\n"
        "Pointwise 95% participant intervals; circles primary, squares secondary"
    )
    figures["primary_gender_gains"] = fig
    return figures


def reproduce(output_dir: Path, source_dir: Path = ROOT / "results") -> dict[str, Any]:
    """Write a fresh, deterministic aggregate-only publication package."""
    output_dir, source_dir = Path(output_dir), Path(source_dir)
    if output_dir.exists():
        raise FileExistsError("Preserve the existing reproduction output; choose a new directory")
    evidence, hashes = load_evidence(source_dir)
    grid, contrasts = complete_grid(evidence), matched_contrasts(evidence)
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, rows in (
        ("complete_grid.csv", grid),
        ("matched_contrasts.csv", contrasts),
    ):
        with (output_dir / filename).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "contribution_overview.md").write_text(
        contribution_overview(evidence, grid, contrasts), encoding="utf-8"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, figure in _figures(evidence, grid, contrasts).items():
        figure.savefig(
            output_dir / f"{name}.svg",
            metadata={"Date": None, "Creator": "Aggregate publication replay"},
        )
        figure.savefig(
            output_dir / f"{name}.png",
            dpi=160,
            metadata={"Software": "Aggregate publication replay"},
        )
        plt.close(figure)
    receipt = {
        "schema_version": "public-aggregate-replay-1",
        "scope": "aggregate_report_and_figure_replay_not_fitting_or_bootstrap_inference",
        "source_sha256": hashes,
        "complete_grid_rows": len(grid),
        "matched_contrast_rows": len(contrasts),
        "matplotlib_version": matplotlib.__version__,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
    }
    (output_dir / "source_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Replay published aggregate evidence without features, network access or TeX."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = reproduce(args.output_dir, args.source_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "grid_rows": receipt["complete_grid_rows"],
                "matched_contrast_rows": receipt["matched_contrast_rows"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
