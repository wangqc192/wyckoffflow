import hydra
import torch
from hydra import compose, initialize_config_dir

from models.common.composition import formula_to_counts
from models.common.utils import PROJECT_ROOT
from models.pl_models.crystal_gnn import CrystalGNN


class ScaledCrystalGNN(CrystalGNN):
    def __init__(self, logit_scale, **kwargs):
        super().__init__(**kwargs)
        self.logit_scale = logit_scale

    def forward(self, data, time):
        zero_logits, inf_logits = super().forward(data, time)
        return zero_logits * self.logit_scale, inf_logits * self.logit_scale


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
    assert config.model.decoder._target_ == "models.pl_models.crystal_gnn.CrystalGNN"
    assert config.model.label_smoothing == 0.0
    assert "model_config" not in config.model
    assert "flow_steps" not in config.model
    assert "hidden_dim" not in config.model
    assert "composition_encoder_dim" not in config.model


def test_decoder_overrides_support_forward_and_backward():
    config = compose_config(
        "model.decoder.num_gnn_layers=2",
        "model.decoder.hidden_dim=8",
        "model.decoder.element_dim=4",
        "model.decoder.num_heads=2",
        "model.max_num_atoms=8",
        "model.zero_df_loss_weight=2.0",
        "model.inf_df_loss_weight=3.0",
    )
    model = hydra.utils.instantiate(
        config.model, optimizer_config=config.optim, _recursive_=False
    )
    assert len(model.decoder.layers) == 2
    assert model.decoder.hidden_dim == 8
    assert model.decoder.element_embedding.embedding_dim == 4
    assert model.max_num_atoms == 8
    assert model.zero_df_loss_weight == 2.0
    assert model.inf_df_loss_weight == 3.0
    assert model.label_smoothing == 0.0
    assert "model_config" not in model.hparams
    assert "flow_steps" not in model.hparams
    assert isinstance(model.configure_optimizers(), torch.optim.AdamW)

    batch = model._build_source_from_compositions(
        formula_to_counts("Ga4Te4", model.num_elements).unsqueeze(0),
        fixed_space_group=194,
    )
    loss = model(batch)["loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert model.decoder.element_head[-1].weight.grad is not None
    assert model.decoder.count_head[-1].weight.grad is not None
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_decoder_config_group_can_select_another_network(tmp_path):
    decoder_dir = tmp_path / "model" / "decoder"
    decoder_dir.mkdir(parents=True)
    (decoder_dir / "scaled.yaml").write_text(
        "defaults:\n  - crystal_gnn\n  - _self_\n"
        f"_target_: {__name__}.ScaledCrystalGNN\n"
        "logit_scale: 2.0\n"
        "num_gnn_layers: 1\n"
        "hidden_dim: 8\n"
        "element_dim: 4\n"
        "num_heads: 2\n",
        encoding="utf-8",
    )
    config = compose_config(
        f"hydra.searchpath=[file://{tmp_path}]",
        "model/decoder=scaled",
        "model.max_num_atoms=8",
    )
    model = hydra.utils.instantiate(
        config.model, optimizer_config=config.optim, _recursive_=False
    )
    assert isinstance(model.decoder, ScaledCrystalGNN)
    assert model.decoder.logit_scale == 2.0

    batch = model._build_source_from_compositions(
        formula_to_counts("Ga4Te4", model.num_elements).unsqueeze(0),
        fixed_space_group=194,
    )
    model.eval()
    time = torch.zeros(1)
    zero_logits, inf_logits = model.decoder(batch, time)
    base_zero, base_inf = CrystalGNN.forward(model.decoder, batch, time)
    torch.testing.assert_close(zero_logits, 2 * base_zero)
    torch.testing.assert_close(inf_logits, 2 * base_inf)


def test_space_group_experiment_overrides_model_and_epochs():
    config = compose_config("experiment=space_group")

    assert config.model._target_ == ("models.pl_models.space_group.SpaceGroupModule")
    assert "model_config" not in config.model
    assert config.model.composition_encoder_dim == 256
    assert config.model.compatibility is True
    assert config.train.trainer.max_epochs == 200


def test_standalone_space_group_dropout_config_controls_predictions():
    config = compose_config(
        "experiment=space_group",
        "model.hidden_dim=16",
        "model.composition_encoder_dim=8",
        "model.dropout=0.5",
    )
    model = hydra.utils.instantiate(
        config.model, optimizer_config=config.optim, _recursive_=False
    )
    composition = formula_to_counts("Li2O2", model.num_elements)[None]
    assert model.hparams.dropout == 0.5
    assert not torch.equal(model.predictor(composition), model.predictor(composition))
    model.eval()
    torch.testing.assert_close(
        model.predictor(composition), model.predictor(composition), rtol=0, atol=0
    )


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
