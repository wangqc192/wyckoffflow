"""Plot loss components and total loss from Lightning CSV logs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def logged_values(metrics, x_column, loss_column):
    values = metrics.loc[metrics[loss_column].notna(), [x_column, loss_column]]
    return values.groupby(x_column, as_index=False).last()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Run directory or metrics.csv")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    args = parser.parse_args()

    metrics_path = args.path / "logs/metrics.csv" if args.path.is_dir() else args.path
    metrics = pd.read_csv(metrics_path)
    epoch_loss = (
        "train/loss_epoch" in metrics and metrics["train/loss_epoch"].notna().any()
    )
    x_column = "epoch" if epoch_loss else "step"
    train_suffix = "epoch" if epoch_loss else "step"

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    components_axis, total_axis = axes

    component_styles = {
        "zero_df_loss": ("Zero-DOF loss", "#2563eb"),
        "inf_df_loss": ("Inf-DOF loss", "#f59e0b"),
    }
    for component, (label, color) in component_styles.items():
        train_column = f"train/{component}_{train_suffix}"
        if train_column in metrics and metrics[train_column].notna().any():
            train = logged_values(metrics, x_column, train_column)
            components_axis.plot(
                train[x_column],
                train[train_column],
                label=f"Train {label}",
                color=color,
            )

        validation_column = f"val/{component}"
        if validation_column in metrics and metrics[validation_column].notna().any():
            validation = logged_values(metrics, x_column, validation_column)
            components_axis.plot(
                validation[x_column],
                validation[validation_column],
                label=f"Validation {label}",
                color=color,
                linestyle="--",
            )

    train_column = f"train/loss_{train_suffix}"
    train = logged_values(metrics, x_column, train_column)
    total_axis.plot(
        train[x_column],
        train[train_column],
        label="Train",
        color="#2563eb",
    )
    if "val/loss" in metrics and metrics["val/loss"].notna().any():
        validation = logged_values(metrics, x_column, "val/loss")
        total_axis.plot(
            validation[x_column],
            validation["val/loss"],
            label="Validation",
            color="#dc2626",
        )

    for axis, title in zip(axes, ("Loss components", "Total loss")):
        axis.set_title(title)
        axis.set_xlabel(x_column.title())
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()

    output = args.output or metrics_path.with_name("loss.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
