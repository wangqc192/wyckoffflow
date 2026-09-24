"""Model checkpoint loading for inference."""

from pathlib import Path

import hydra
from omegaconf import OmegaConf


def load_model(model_path, load_data=False, testing=True, device="cpu"):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
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
            checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in: {checkpoint_dir}")
            checkpoint_path = checkpoints[-1]

    config = OmegaConf.load(model_path / "hparams.yaml")
    model_class = hydra.utils.get_class(config.model._target_)
    model = model_class.load_from_checkpoint(
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
