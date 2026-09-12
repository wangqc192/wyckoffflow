"""Model checkpoint loading for inference."""

from pathlib import Path

import hydra
from omegaconf import OmegaConf


def load_model(model_path, load_data=False, testing=True, device="cpu"):
    model_path = Path(model_path)
    if model_path.is_file():
        checkpoint_path = model_path
        model_path = (
            model_path.parent.parent
            if model_path.parent.name == "checkpoints"
            else model_path.parent
        )
    else:
        checkpoint_dir = model_path / "checkpoints"
        checkpoint_path = checkpoint_dir / "last.ckpt"
        if not checkpoint_path.exists():
            checkpoint_path = sorted(checkpoint_dir.glob("*.ckpt"))[-1]

    config = OmegaConf.load(model_path / "hparams.yaml")
    model = hydra.utils.instantiate(
        config.model,
        optimizer_config=config.optim,
        _recursive_=False,
    )
    model = type(model).load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=True,
    )
    model = model.to(device).eval()

    loaders = None
    if load_data:
        datamodule = hydra.utils.instantiate(
            config.data.datamodule,
            _recursive_=False,
        )
        if testing:
            datamodule.setup("test")
            loaders = datamodule.test_dataloader()[0]
        else:
            datamodule.setup("fit")
            loaders = (
                datamodule.train_dataloader(),
                datamodule.val_dataloader()[0],
            )
    return model, loaders, config
