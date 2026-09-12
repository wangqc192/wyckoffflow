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

    def __init__(self, model_config, optimizer_config, task):
        super().__init__()
        self.model_config = resolve_config(model_config)
        self.optimizer_config = resolve_config(optimizer_config)
        self.task = task
        self.save_hyperparameters(
            {
                "task": task,
                "model_config": self.model_config,
                "optimizer_config": self.optimizer_config,
            }
        )

    def configure_optimizers(self):
        return hydra.utils.instantiate(
            self.optimizer_config,
            params=self.parameters(),
        )
