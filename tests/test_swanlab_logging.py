from types import SimpleNamespace

import pandas as pd
import pytorch_lightning as pl
import swanlab
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, TensorDataset

from models.common.utils import PROJECT_ROOT
from models.run import build_loggers
from scripts.sync_swanlab import log_pending, read_history


class LoggingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        loss = (self.weight * batch[0]).square().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("val/gwa_top1", 0.75)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_lightning_logs_to_csv_and_swanlab(monkeypatch, tmp_path):
    calls, finished, init_kwargs = [], [], {}
    run = SimpleNamespace(
        id="test-run",
        url="https://example.test/run",
        config={},
        log=lambda metrics, step: calls.append((step, metrics)),
        finish=lambda **kwargs: finished.append(kwargs),
    )

    def init(**kwargs):
        init_kwargs.update(kwargs)
        return run

    monkeypatch.setattr(swanlab, "init", init)
    with initialize_config_dir(str(PROJECT_ROOT / "conf"), version_base="1.3"):
        config = compose(
            config_name="default", overrides=["model_label=test", "run_name=test"]
        )
    config.logging.logger.save_dir = str(tmp_path / "logs")
    config.logging.swanlab.save_dir = str(tmp_path)
    config.data.root_path = str(PROJECT_ROOT / "data/mp20")
    loggers = build_loggers(config)
    loader = DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=loggers,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    trainer.fit(LoggingModel(), loader, loader)

    csv = pd.read_csv(tmp_path / "logs/metrics.csv")
    observed = pd.DataFrame([{**values, "step": step} for step, values in calls])
    pd.testing.assert_frame_equal(
        csv.groupby("step").last().sort_index(axis=1),
        observed.groupby("step").last().sort_index(axis=1),
        check_dtype=False,
    )
    assert init_kwargs["config"]["seed"] == 42
    assert init_kwargs["config"]["model"]["_target_"] == config.model._target_
    assert finished == [{"state": "success"}]
    assert (tmp_path / "swanlab/run.json").exists()


def test_csv_sync_preserves_sparse_rows_resume_and_late_metrics(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    path = logs / "metrics.csv"
    header = "epoch,step,train/loss_step,train/loss_epoch,val/loss\n"
    # Ignore a row while CSVLogger is still writing it.
    path.write_text(header + "0,9,0.8,,\n0,19,,,0.5\n0,19,,0.7,\n1,29,0.")
    resumed = tmp_path / "resume_01/logs"
    resumed.mkdir(parents=True)
    (resumed / "metrics.csv").write_text(header + "0,19,,,0.4\n")
    calls = []
    run = SimpleNamespace(log=lambda data, step: calls.append((step, data)))
    last_steps = {}
    history = read_history(tmp_path)
    log_pending(run, history, last_steps)
    assert calls == [
        (9, {"epoch": 0.0, "train/loss_step": 0.8}),
        (19, {"epoch": 0.0, "train/loss_epoch": 0.7, "val/loss": 0.4}),
    ]
    assert log_pending(run, history, last_steps) == 0

    # A reconstruction metric can arrive later at an already imported global step.
    summary = tmp_path / "reconstruction/epoch_0000/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        '{"epoch":0,"top_k":20,"match_rate":0.9,'
        '"match_rate_top1":0.8,"composition_accuracy":1.0}'
    )
    assert log_pending(run, read_history(tmp_path), last_steps) == 3
    assert calls[-1] == (
        19,
        {"val/gwa_top1": 0.8, "val/gwa_top20": 0.9, "val/composition_accuracy": 1.0},
    )
