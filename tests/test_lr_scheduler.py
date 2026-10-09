import hydra
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from models.common.utils import PROJECT_ROOT
from models.pl_models.base import OptimizedLightningModule
from models.pl_models.flow import DiscreteFlowModule

SCHEDULER_CONFIG = {
    "scheduler": {
        "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "mode": "min",
        "factor": 0.5,
        "patience": 1,
        "threshold": 1e-4,
        "min_lr": 2.5e-5,
    },
    "monitor": "val/loss",
    "interval": "epoch",
    "frequency": 1,
    "strict": True,
}


class PlateauModel(OptimizedLightningModule):
    def __init__(self):
        super().__init__(
            {"_target_": "torch.optim.AdamW", "lr": 1e-4},
            "test",
            SCHEDULER_CONFIG,
        )
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.observed_lrs = []

    def training_step(self, batch, batch_idx):
        self.observed_lrs.append(self.optimizers().param_groups[0]["lr"])
        return (self.weight * batch[0]).square().mean()

    def validation_step(self, batch, batch_idx):
        self.log("val/loss", 1.0)


def test_plateau_uses_validation_metric_logs_lr_and_resumes_state(tmp_path):
    loader = DataLoader(TensorDataset(torch.ones(1, 1)))

    def trainer(epochs, name):
        return pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=epochs,
            logger=CSVLogger(tmp_path, name=name),
            callbacks=[LearningRateMonitor(logging_interval="epoch")],
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
        )

    model = PlateauModel()
    first = trainer(2, "first")
    first.fit(model, loader, loader)
    scheduler = first.lr_scheduler_configs[0].scheduler
    assert scheduler.num_bad_epochs == 1
    checkpoint = tmp_path / "resume.ckpt"
    first.save_checkpoint(checkpoint)

    resumed = PlateauModel()
    second = trainer(8, "resumed")
    second.fit(resumed, loader, loader, ckpt_path=checkpoint)
    # The first resumed validation exceeds patience; the next epoch uses half LR.
    assert resumed.observed_lrs == pytest.approx(
        [1e-4, 5e-5, 5e-5, 2.5e-5, 2.5e-5, 2.5e-5]
    )
    logged = pd.read_csv(second.logger.experiment.metrics_file_path)
    assert logged["lr-AdamW"].dropna().tolist() == pytest.approx(resumed.observed_lrs)


def test_flow_checkpoint_preserves_scheduler_config(tmp_path):
    with initialize_config_dir(str(PROJECT_ROOT / "conf"), version_base="1.3"):
        config = compose(
            config_name="default",
            overrides=[
                "model/decoder=crystal_gnn",
                "model.decoder.hidden_dim=16",
                "model.decoder.element_dim=8",
                "model.decoder.num_heads=2",
                "model.decoder.num_gnn_layers=1",
            ],
        )
    config.model.scheduler_config = SCHEDULER_CONFIG
    model = hydra.utils.instantiate(
        config.model, optimizer_config=config.optim, _recursive_=False
    )
    path = tmp_path / "flow.ckpt"
    torch.save(
        {
            "pytorch-lightning_version": pl.__version__,
            "hyper_parameters": dict(model.hparams),
            "state_dict": model.state_dict(),
        },
        path,
    )
    loaded = DiscreteFlowModule.load_from_checkpoint(path)
    assert loaded.scheduler_config == SCHEDULER_CONFIG
    assert isinstance(
        loaded.configure_optimizers()["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )
