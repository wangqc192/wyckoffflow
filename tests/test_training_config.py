from hydra import compose, initialize_config_dir

from models.common.utils import PROJECT_ROOT


def compose_config(*overrides):
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "conf"),
        version_base="1.3",
    ):
        return compose(config_name="default", overrides=list(overrides))


def test_default_training_config_uses_local_flow_model():
    config = compose_config()

    assert config.model._target_ == "models.pl_models.flow.DiscreteFlowModule"
    assert config.data.datamodule._target_ == (
        "models.pl_data.datamodule.CrystDataModule"
    )
    assert config.optim._target_ == "torch.optim.AdamW"


def test_space_group_experiment_overrides_model_and_epochs():
    config = compose_config("experiment=space_group")

    assert config.model._target_ == ("models.pl_models.space_group.SpaceGroupModule")
    assert config.model.model_config.composition_encoder_dim == 256
    assert config.model.model_config.compatibility is True
    assert config.train.trainer.max_epochs == 200


def test_mini_data_group_overrides_only_the_data_path():
    config = compose_config("data=mini")

    assert config.data.dataset_name == "mp20"


def test_mp20_data_group_uses_large_training_batches():
    config = compose_config()

    assert dict(config.data.datamodule.batch_size) == {
        "train": 256,
        "val": 256,
        "test": 256,
    }
    assert config.data.datamodule._target_ == (
        "models.pl_data.datamodule.CrystDataModule"
    )
