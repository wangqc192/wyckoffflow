"""Shared Lightning model setup."""

from collections.abc import Mapping
from typing import Any

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf


def resolve_config(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        return dict(OmegaConf.to_container(config, resolve=True))
    return dict(config)


class OptimizedLightningModule(pl.LightningModule):
    """Lightning module configured with a Hydra optimizer."""

    def __init__(self, optimizer_config, task, scheduler_config=None):
        super().__init__()
        self.optimizer_config = resolve_config(optimizer_config)
        self.scheduler_config = (
            resolve_config(scheduler_config) if scheduler_config is not None else None
        )
        self.task = task
        self.save_hyperparameters(
            {
                "task": task,
                "optimizer_config": self.optimizer_config,
            }
        )

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.optimizer_config,
            params=self.parameters(),
        )
        if self.scheduler_config is None:
            return optimizer
        config = dict(self.scheduler_config)
        config["scheduler"] = hydra.utils.instantiate(
            config["scheduler"], optimizer=optimizer
        )
        return {"optimizer": optimizer, "lr_scheduler": config}
