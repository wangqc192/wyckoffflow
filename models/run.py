"""Hydra entry point for Lightning training."""

import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from models.common.utils import PROJECT_ROOT

log = logging.getLogger(__name__)


def build_callbacks(config: DictConfig) -> list[ModelCheckpoint]:
    checkpoint = config.train.checkpoint
    directory = (
        Path(config.resume_from).resolve().parent
        if config.resume_from
        else Path(HydraConfig.get().runtime.output_dir) / "checkpoints"
    )
    return [
        ModelCheckpoint(
            dirpath=directory,
            filename="best",
            monitor=checkpoint.monitor,
            mode=checkpoint.mode,
            save_top_k=checkpoint.save_top_k,
            save_last=checkpoint.save_last,
            enable_version_counter=False,
        ),
        ModelCheckpoint(
            dirpath=directory,
            filename="epoch_{epoch:04d}",
            every_n_epochs=checkpoint.every_n_epochs,
            save_top_k=-1,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
        ),
    ]


def run(config: DictConfig) -> None:
    pl.seed_everything(config.seed, workers=True)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    OmegaConf.save(config, output_dir / "hparams.yaml", resolve=True)

    log.info("Instantiating %s", config.data.datamodule._target_)
    datamodule = hydra.utils.instantiate(
        config.data.datamodule,
        _recursive_=False,
    )
    log.info("Instantiating %s", config.model._target_)
    model = hydra.utils.instantiate(
        config.model,
        optimizer_config=config.optim,
        _recursive_=False,
    )
    logger = (
        hydra.utils.instantiate(config.logging.logger)
        if config.logging.logger
        else False
    )
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        logger=logger,
        callbacks=build_callbacks(config),
        **config.train.trainer,
    )

    log.info("Starting training")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.resume_from,
    )


@hydra.main(
    config_path=str(PROJECT_ROOT / "conf"),
    config_name="default",
    version_base="1.3",
)
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    main()
