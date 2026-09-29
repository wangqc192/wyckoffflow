import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from models.common.checkpoint import load_model
from models.pl_models.flow import DiscreteFlowModule
from models.pl_models.space_group import SpaceGroupModule

OPTIMIZER_CONFIG = {"_target_": "torch.optim.AdamW", "lr": 2e-4}


def save_checkpoint(model, path):
    torch.save(
        {
            "pytorch-lightning_version": pl.__version__,
            "hyper_parameters": dict(model.hparams),
            "state_dict": model.state_dict(),
        },
        path,
    )


def save_run(model, path):
    checkpoint_dir = path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "last.ckpt"
    save_checkpoint(model, checkpoint_path)
    model_config = {
        "_target_": f"{type(model).__module__}.{type(model).__name__}",
        **{
            key: value
            for key, value in model.hparams.items()
            if key not in {"task", "optimizer_config"}
        },
    }
    OmegaConf.save(
        {"model": model_config, "optim": OPTIMIZER_CONFIG},
        path / "hparams.yaml",
    )
    return checkpoint_path


@pytest.mark.parametrize("stale_run_config", [False, True])
def test_load_flow_model(tmp_path, stale_run_config):
    config = {
        "num_elements": 118,
        "max_num_atoms": 8,
        "flow_source": "zeros",
        "zero_df_loss_weight": 2.0,
        "inf_df_loss_weight": 3.0,
        "conditional_composition": True,
    }
    decoder = {
        "_target_": "models.pl_models.gnn.WyckoffGNN",
        "composition_encoder_dim": 4,
        "composition_film": False,
        "num_gnn_layers": 1,
        "hidden_dim": 8,
        "dof_pos_sg_emb_size": 4,
        "gnn_activation": "SiLU",
        "no_multiplicity_encoding": False,
        "binary_dof_encoding": False,
        "no_softmax": False,
        "mlp_hidden_layers": 2,
        "mlp_activation": "SiLU",
    }
    model = DiscreteFlowModule(
        optimizer_config=OPTIMIZER_CONFIG,
        decoder=decoder,
        validation_seed=7,
        **config,
    )
    path = tmp_path / "flow"
    checkpoint_path = save_run(model, path)
    if stale_run_config:
        saved_config = OmegaConf.load(path / "hparams.yaml")
        saved_config.model.count_conserving = False
        saved_config.model.decoder.hidden_dim = 16
        OmegaConf.save(saved_config, path / "hparams.yaml")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint["hyper_parameters"]["count_conserving"] = False
        torch.save(checkpoint, checkpoint_path)

    loaded, loaders, config = load_model(path)

    assert isinstance(loaded, DiscreteFlowModule)
    assert loaded.validation_seed == 7
    assert "flow_steps" not in config.model
    assert "flow_steps" not in loaded.hparams
    assert loaded.zero_df_loss_weight == config.model.zero_df_loss_weight == 2.0
    assert loaded.inf_df_loss_weight == config.model.inf_df_loss_weight == 3.0
    assert "model_config" not in loaded.hparams
    assert "count_conserving" not in loaded.hparams
    assert not loaded.training
    assert loaders is None
    assert config.model._target_ == "models.pl_models.flow.DiscreteFlowModule"
    assert loaded.decoder_config == decoder
    assert config.model.decoder.hidden_dim == (16 if stale_run_config else 8)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_load_space_group_model(tmp_path):
    config = {
        "num_elements": 118,
        "max_num_atoms": 8,
        "hidden_dim": 8,
        "composition_encoder_dim": 4,
        "compatibility": False,
        "mlp_hidden_layers": 2,
        "mlp_activation": "SiLU",
    }
    model = SpaceGroupModule(optimizer_config=OPTIMIZER_CONFIG, **config)
    path = tmp_path / "space_group"
    save_run(model, path)

    loaded, loaders, config = load_model(path)

    assert isinstance(loaded, SpaceGroupModule)
    assert not loaded.training
    assert loaders is None
    assert config.model._target_ == "models.pl_models.space_group.SpaceGroupModule"


def test_load_model_from_checkpoint_path(tmp_path):
    config = {
        "num_elements": 118,
        "max_num_atoms": 8,
        "hidden_dim": 8,
        "composition_encoder_dim": 4,
        "compatibility": False,
        "mlp_hidden_layers": 2,
        "mlp_activation": "SiLU",
    }
    model = SpaceGroupModule(optimizer_config=OPTIMIZER_CONFIG, **config)
    run_path = tmp_path / "run"
    checkpoint_path = save_run(model, run_path)

    loaded, loaders, config = load_model(checkpoint_path)

    assert isinstance(loaded, SpaceGroupModule)
    assert loaders is None
    assert config.model._target_ == "models.pl_models.space_group.SpaceGroupModule"


def test_missing_explicit_checkpoint_does_not_fall_back_to_another(tmp_path):
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "last.ckpt").touch()
    missing = checkpoint_dir / "best_epoch_0021.ckpt"

    with pytest.raises(FileNotFoundError, match="Model path does not exist") as error:
        load_model(missing)

    assert str(missing) in str(error.value)


@pytest.mark.parametrize("create_directory", [False, True])
def test_missing_run_or_empty_checkpoint_directory(tmp_path, create_directory):
    run_path = tmp_path / "run"
    if create_directory:
        (run_path / "checkpoints").mkdir(parents=True)

    message = (
        "No checkpoints found" if create_directory else "Model path does not exist"
    )
    with pytest.raises(FileNotFoundError, match=message) as error:
        load_model(run_path)

    assert str(run_path) in str(error.value)
