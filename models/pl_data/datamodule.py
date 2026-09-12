"""Lightning data module for the Wyckoff ``CrystalDataset``."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from models.common.utils import PROJECT_ROOT


def worker_init_fn(_worker_id: int) -> None:
    """Give each DataLoader worker an independent, reproducible RNG state."""

    uint64_seed = torch.initial_seed()
    seed_sequence = np.random.SeedSequence([uint64_seed])
    np.random.seed(seed_sequence.generate_state(4))
    random.seed(uint64_seed)


class CrystDataModule(pl.LightningDataModule):
    """Create train, validation, and test loaders in the PODGen style.

    ``datasets`` should contain ``train``, ``val`` and ``test`` entries.  The
    latter two may each be one dataset configuration or a list of configurations.
    ``num_workers`` and ``batch_size`` are mappings with ``train``, ``val`` and
    ``test`` keys, matching the reference configuration layout.
    """

    def __init__(
        self,
        datasets: Mapping[str, Any],
        num_workers: Mapping[str, int],
        batch_size: Mapping[str, int],
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory

        self.train_dataset: Dataset | None = None
        self.val_datasets: list[Dataset] | None = None
        self.test_datasets: list[Dataset] | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]
        if stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=self.pin_memory,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                pin_memory=self.pin_memory,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"datasets={self.datasets!r}, "
            f"num_workers={self.num_workers!r}, "
            f"batch_size={self.batch_size!r})"
        )


# A descriptive alias is convenient for callers that use the dataset's name.
CrystalDataModule = CrystDataModule


__all__ = ["CrystDataModule", "CrystalDataModule", "worker_init_fn"]


@hydra.main(
    config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3"
)
def main(cfg: DictConfig):
    print(cfg)
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        print(batch)
        break


if __name__ == "__main__":
    main()
