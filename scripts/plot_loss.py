"""Plot training losses and periodic template reconstruction, including resumed runs."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

COLORS = ("#2563eb", "#d97706", "#16a34a", "#9333ea")
MODES = (("", "Conserved", "-"), ("_no_composition", "Unconstrained", "--"))


def find_runs(path):
    """Accept a CSV, logs directory, run directory, or collection of runs."""
    if path.is_file():
        return [path]
    if (path / "metrics.csv").is_file():
        return [path / "metrics.csv"]
    if (path / "logs/metrics.csv").is_file():
        return [path]
    runs = {csv.parent.parent for csv in path.glob("**/logs/metrics.csv")}
    return sorted(
        run
        for run in runs
        if not (run.name.startswith("resume_") and run.parent in runs)
    )


def reconstruction_values(run_dir):
    """Read the equivalent-template evaluation results without recomputing matches."""
    rows = []
    for path in sorted(run_dir.glob("reconstruction/epoch_*/summary.json")):
        summary = json.loads(path.read_text())
        row = {"epoch": summary["epoch"]}
        for prefix, result in (("", summary), ("joint_", summary.get("joint"))):
            if result is None:
                continue
            for suffix, _, _ in MODES:
                values = result if not suffix else result.get("no_composition")
                if values is None:
                    continue
                for name, value in {
                    "gwa_top1": values["match_rate_top1"],
                    f"gwa_top{values['top_k']}": values["match_rate"],
                    "composition_accuracy": values["composition_accuracy"],
                }.items():
                    row[f"val/{prefix}{name}{suffix}"] = value
        rows.append(row)
    return rows


def load_metrics(source):
    if source.is_file():
        paths = [source]
        run_dir = (
            source.parent.parent if source.parent.name == "logs" else source.parent
        )
    else:
        run_dir = source
        paths = [run_dir / "logs/metrics.csv"]
        paths.extend(sorted(run_dir.glob("resume_*/logs/metrics.csv")))

    frames = [pd.read_csv(path) for path in paths]
    # Summaries also cover completed evaluations not yet flushed to the CSV logger.
    rows = []
    summary_dirs = [run_dir]
    summary_dirs.extend(path.parent.parent for path in paths[1:])
    for directory in summary_dirs:
        rows.extend(reconstruction_values(directory))
    if rows:
        frames.append(pd.DataFrame(rows))
    metrics = pd.concat(frames, ignore_index=True, sort=False)
    # Lightning writes train, validation, and reconstruction on separate sparse rows.
    # Keep the last non-null value per metric, preferring resumed logs and summaries.
    metrics = metrics.groupby("epoch", as_index=False).last()
    return run_dir, metrics, paths


def logged_values(metrics, x_column, loss_column):
    values = metrics.loc[metrics[loss_column].notna(), [x_column, loss_column]]
    return values.groupby(x_column, as_index=False).last()


def train_validation_series(components):
    return [
        (column, f"{split} {label}".strip(), COLORS[index], style)
        for index, (name, label) in enumerate(components)
        for column, split, style in (
            (f"train/{name}_epoch", "Train", "-"),
            (f"val/{name}", "Validation", "--"),
        )
    ]


def gwa_series(metrics, prefix=""):
    top_ks = sorted(
        {
            int(match[1])
            for column in metrics.columns
            if (
                match := re.fullmatch(
                    rf"val/{prefix}gwa_top(\d+)(?:_no_composition)?", column
                )
            )
        }
    )
    return [
        (
            f"val/{prefix}gwa_top{top_k}{suffix}",
            f"GWA@{top_k} / {mode}",
            COLORS[index % len(COLORS)],
            style,
        )
        for index, top_k in enumerate(top_ks)
        for suffix, mode, style in MODES
    ]


def composition_series(prefix=""):
    return [
        (f"val/{prefix}composition_accuracy{suffix}", mode, COLORS[index], style)
        for index, (suffix, mode, style) in enumerate(MODES)
    ]


def plot_metrics(metrics, title, output):
    # Only include panels and curves that have recorded values.
    panels = [
        (
            "Loss components",
            False,
            False,
            train_validation_series(
                [("zero_df_loss", "Zero-DOF"), ("inf_df_loss", "Inf-DOF")]
            ),
        ),
        ("Total loss", False, False, train_validation_series([("loss", "")])),
        (
            "Joint task losses",
            False,
            False,
            train_validation_series(
                [("flow_loss", "Flow"), ("sg_loss", "Space group")]
            ),
        ),
        (
            "Space group accuracy",
            True,
            False,
            train_validation_series([("spg_top1", "Top-1"), ("spg_top5", "Top-5")]),
        ),
        ("Template reconstruction (true SG)", True, True, gwa_series(metrics)),
        ("Composition accuracy (true SG)", True, True, composition_series()),
        (
            "Template reconstruction (predicted SG)",
            True,
            True,
            gwa_series(metrics, "joint_"),
        ),
        (
            "Composition accuracy (predicted SG)",
            True,
            True,
            composition_series("joint_"),
        ),
    ]
    available = []
    for panel_title, percentage, sparse, series in panels:
        series = [
            item
            for item in series
            if item[0] in metrics and metrics[item[0]].notna().any()
        ]
        if series:
            available.append((panel_title, percentage, sparse, series))
    if not available:
        raise ValueError("No loss or reconstruction metrics found")

    columns = min(2, len(available))
    rows = (len(available) + columns - 1) // columns
    with plt.rc_context(
        {"font.size": 10, "legend.fontsize": 8, "lines.linewidth": 1.4}
    ):
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(6.4 * columns, 3.8 * rows),
            squeeze=False,
            sharex=True,
            layout="constrained",
        )
        figure.suptitle(title)
        for axis, (panel_title, percentage, sparse, series) in zip(
            axes.flat, available
        ):
            loss_ranges = []
            for column, label, color, style in series:
                values = logged_values(metrics, "epoch", column)
                if not percentage:
                    loss_ranges.append(
                        (
                            values[column].min(),
                            values[column].quantile(0.95, interpolation="higher"),
                        )
                    )
                axis.plot(
                    values["epoch"] + 1,
                    values[column] * (100 if percentage else 1),
                    label=label,
                    color=color,
                    linestyle=style,
                    marker="o" if sparse else None,
                    markersize=4,
                )
            axis.set_title(panel_title)
            axis.set_xlabel("Completed epochs")
            axis.set_ylabel("Rate" if percentage else "Loss")
            if percentage:
                axis.set_ylim(0, 102)
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            else:
                # Keep each curve's main range visible without early spikes
                # compressing the rest of training into a nearly flat line.
                lower = min(low for low, _ in loss_ranges)
                upper = max(high for _, high in loss_ranges)
                padding = 0.05 * (upper - lower or abs(upper) or 1)
                axis.set_ylim(lower - padding, upper + padding)
            axis.grid(alpha=0.25)
            axis.legend()
        for axis in list(axes.flat)[len(available) :]:
            axis.set_visible(False)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Run directory, logs directory, metrics.csv, or outputs root",
    )
    parser.add_argument(
        "--output", type=Path, help="Output image path (single run only)"
    )
    args = parser.parse_args()

    sources = find_runs(args.path)
    if not sources:
        parser.error(f"No logs/metrics.csv found under {args.path}")
    if args.output and len(sources) != 1:
        parser.error("--output requires a single run or metrics.csv")
    for source in sources:
        run_dir, metrics, paths = load_metrics(source)
        output = args.output or paths[0].with_name("loss.png")
        plot_metrics(metrics, run_dir.name, output)
        print(f"{output} ({len(paths)} log files, {len(metrics)} epochs)")


if __name__ == "__main__":
    main()
