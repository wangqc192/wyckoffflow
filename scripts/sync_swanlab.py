"""Import or follow Lightning CSV logs without restarting a training process."""

import argparse
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import swanlab
from omegaconf import OmegaConf

from scripts.plot_loss import reconstruction_values


def read_history(run_dir):
    paths = [run_dir / "logs/metrics.csv"]
    paths.extend(sorted(run_dir.glob("resume_*/logs/metrics.csv")))
    frames = []
    for path in paths:
        # A running CSVLogger can be appending a row or rewriting the header.
        content = path.read_text().rsplit("\n", 1)[0]
        if "\n" in content:
            frames.append(pd.read_csv(io.StringIO(content)))
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True)
    history = history.dropna(subset=["step"])
    completed = history.dropna(subset=["train/loss_epoch"])
    epoch_steps = completed.groupby("epoch")["step"].max()
    summaries = []
    for path in paths:
        for row in reconstruction_values(path.parent.parent):
            if row["epoch"] in epoch_steps:
                summaries.append({**row, "step": epoch_steps[row["epoch"]]})
    if summaries:
        history = pd.concat([history, pd.DataFrame(summaries)], ignore_index=True)
    # Train, validation and reconstruction occupy different rows at the same step.
    # Resumed runs and equivalent-template summaries take precedence per metric.
    return history.groupby("step").last().sort_index()


def log_pending(run, history, last_steps):
    count = 0
    for step, row in history.iterrows():
        step = int(step)
        metrics = {
            key: float(value)
            for key, value in row.items()
            if math.isfinite(value) and step > last_steps.get(key, -1)
        }
        if metrics:
            run.log(metrics, step=step)
            last_steps.update({key: step for key in metrics})
            count += len(metrics)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--name")
    parser.add_argument(
        "--project", default=os.environ.get("SWANLAB_PROJECT", "wyckoffflow-pl")
    )
    parser.add_argument("--workspace", default=os.environ.get("SWANLAB_WORKSPACE"))
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument(
        "--stop-file", type=Path, help="Training launcher's exit_code file"
    )
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    log_dir = directory / "swanlab"
    log_dir.mkdir(exist_ok=True)
    target = f"{args.workspace}/{args.project}/{args.mode}"
    token = hashlib.sha256(target.encode()).hexdigest()[:12]
    state_path = log_dir / f"csv_sync_{token}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    # Offline runs have no server-side resume; each import gets a complete history.
    last_steps = state.get("last_steps", {}) if args.mode == "online" else {}
    config = OmegaConf.to_container(
        OmegaConf.load(directory / "hparams.yaml"), resolve=True
    )
    config["csv_source"] = str(directory)
    config["metric_step"] = "Lightning global step (same as logs/metrics.csv)"
    with swanlab.init(
        project=args.project,
        workspace=args.workspace,
        name=args.name or directory.name,
        mode=args.mode,
        id=state.get("id") if args.mode == "online" else None,
        resume="allow",
        config=config,
        log_dir=str(log_dir),
        tags=["csv-import"],
        settings=swanlab.Settings(
            interactive=False,
            # This process reads logs; its resource usage is not training usage.
            probe=swanlab.Settings.Probe(
                monitor=False, hardware=False, requirements=False, git=False
            ),
        ),
    ) as run:
        url = run.url if args.mode == "online" else None
        print(f"SwanLab: {url or run.dir}", flush=True)
        while True:
            count = log_pending(run, read_history(directory), last_steps)
            state.update(id=run.id, url=url, last_steps=last_steps)
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(state_path)
            if count:
                print(
                    f"Synced {count} metric values; latest step {max(last_steps.values())}",
                    flush=True,
                )
            if args.stop_file and args.stop_file.exists():
                exit_code = int(args.stop_file.read_text().strip())
                if exit_code:
                    raise RuntimeError(f"Training exited with code {exit_code}")
                break
            if not args.watch:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
