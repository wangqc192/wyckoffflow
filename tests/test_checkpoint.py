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
        "model_config": model.model_config,
    }
    if isinstance(model, DiscreteFlowModule):
        model_config["validation_seed"] = model.validation_seed
    OmegaConf.save(
        {"model": model_config, "optim": OPTIMIZER_CONFIG},
        path / "hparams.yaml",
    )
    return checkpoint_path


def test_load_flow_model(tmp_path):
    config = {
        "num_elements": 118,
        "max_num_atoms": 8,
        "flow_source": "zeros",
        "flow_steps": 2,
        "conditional_composition": True,
        "composition_encoder_dim": 4,
        "composition_film": False,
        "count_conserving": False,
        "num_gnn_layers": 1,
        "hidden_dim": 8,
        "dof_pos_sg_emb_size": 4,
        "gnn_activation": "SiLU",
        "no_multiplicity_encoding": False,
        "binary_dof_encoding": False,
        "no_softmax": False,
        "mlp_hidden_layers": 1,
        "mlp_activation": "SiLU",
    }
    model = DiscreteFlowModule(config, OPTIMIZER_CONFIG, validation_seed=7)
    path = tmp_path / "flow"
    save_run(model, path)

    loaded, loaders, config = load_model(path)

    assert isinstance(loaded, DiscreteFlowModule)
    assert loaded.validation_seed == 7
    assert not loaded.training
    assert loaders is None
    assert config.model._target_ == "models.pl_models.flow.DiscreteFlowModule"


def test_load_space_group_model(tmp_path):
    config = {
        "num_elements": 118,
        "max_num_atoms": 8,
        "hidden_dim": 8,
        "composition_encoder_dim": 4,
        "compatibility": False,
        "mlp_hidden_layers": 1,
        "mlp_activation": "SiLU",
    }
    model = SpaceGroupModule(config, OPTIMIZER_CONFIG)
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
        "mlp_hidden_layers": 1,
        "mlp_activation": "SiLU",
    }
    model = SpaceGroupModule(config, OPTIMIZER_CONFIG)
    run_path = tmp_path / "run"
    checkpoint_path = save_run(model, run_path)

    loaded, loaders, config = load_model(checkpoint_path)

    assert isinstance(loaded, SpaceGroupModule)
    assert loaders is None
    assert config.model._target_ == "models.pl_models.space_group.SpaceGroupModule"
