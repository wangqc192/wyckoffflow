from types import SimpleNamespace

import hydra
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import models.common.reconstruction as reconstruction
from models.common.checkpoint import load_model
from models.common.composition import formula_to_counts
from models.common.utils import PROJECT_ROOT
from models.pl_models.count_conserving import DecodingResult, decode_composition_logits
from models.pl_models.crystal_gnn import CrystalGNN, occupation_counts
from models.pl_models.model_utils import create_wyckoff_graph, get_degrees_of_freedom
from models.run import build_callbacks
from models.sampling import predict_space_group_conditions
from scripts import sample_wy


def joint_config(*overrides):
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "conf"), version_base="1.3"
    ):
        return compose(
            config_name="default",
            overrides=[
                "experiment=joint",
                "data.num_elements=10",
                "model.max_num_atoms=8",
                "model.decoder.hidden_dim=16",
                "model.decoder.element_dim=8",
                "model.decoder.num_heads=2",
                "model.decoder.num_gnn_layers=1",
                "model.decoder.dropout=0.0",
                *overrides,
            ],
        )


def make_model():
    config = joint_config()
    return hydra.utils.instantiate(
        config.model, optimizer_config=config.optim, _recursive_=False
    )


def make_graph(space_group):
    degrees = get_degrees_of_freedom(space_group)
    fixed = torch.zeros(int((degrees == 0).sum()), dtype=torch.long)
    variable = torch.zeros(int((degrees != 0).sum()), 10, dtype=torch.long)
    # The general orbit has multiplicity 1 in SG 1 and 2 in SG 2.
    variable[-1, 2] = 2 // space_group
    variable[-1, 7] = 2 // space_group
    graph = create_wyckoff_graph(space_group, fixed, variable)
    graph.composition = formula_to_counts("Li2O2", 10)[None]
    return graph


@pytest.mark.parametrize("task", ["space_group", "flow"])
def test_each_task_updates_the_single_shared_encoder(task):
    model = make_model()
    batch = Batch.from_data_list([make_graph(1), make_graph(2)])
    features = model.encode_composition(batch.composition)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        if task == "space_group":
            loss = F.cross_entropy(
                model.space_group_logits(batch.composition, features), batch.space_group
            )
        else:
            loss = model.flow_loss(batch, features)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert model.decoder.element_embedding is None
    for parameter in model.composition_encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0
    embeddings = [
        name for name in model.state_dict() if name.endswith("element_embedding.weight")
    ]
    assert embeddings == ["composition_encoder.element_embedding.weight"]


@pytest.mark.parametrize("scale", [0.0, 0.1, 1.0])
def test_flow_gradient_scaling_preserves_values_and_other_parameter_gradients(scale):
    reference = make_model().eval()
    config = joint_config(f"model.flow_encoder_grad_scale={scale}")
    candidate = hydra.utils.instantiate(
        config.model, optimizer_config=config.optim, _recursive_=False
    ).eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    batch = Batch.from_data_list([make_graph(1), make_graph(2)])
    reference_parameters = list(reference.named_parameters())
    torch.manual_seed(321)
    expected_outputs = reference(batch)
    sg_gradients = torch.autograd.grad(
        expected_outputs["sg_loss"],
        [p for _, p in reference_parameters],
        allow_unused=True,
        retain_graph=True,
    )
    flow_gradients = torch.autograd.grad(
        expected_outputs["flow_loss"],
        [p for _, p in reference_parameters],
        allow_unused=True,
    )
    torch.manual_seed(321)
    outputs = candidate(batch)
    for key in expected_outputs:
        torch.testing.assert_close(outputs[key], expected_outputs[key], rtol=0, atol=0)
    outputs["loss"].backward()
    candidate_parameters = dict(candidate.named_parameters())
    for (name, parameter), sg_grad, flow_grad in zip(
        reference_parameters, sg_gradients, flow_gradients
    ):
        expected = torch.zeros_like(parameter)
        if sg_grad is not None:
            expected += reference.sg_loss_weight * sg_grad
        if flow_grad is not None:
            encoder_scale = scale if name.startswith("composition_encoder.") else 1
            expected += reference.flow_loss_weight * encoder_scale * flow_grad
        actual = candidate_parameters[name].grad
        if actual is None:
            assert not expected.count_nonzero()
        else:
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-7)
    assert candidate.composition_encoder.element_embedding.weight.grad.abs().sum() > 0
    assert candidate.decoder.empty_head[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("scale", [None, 0.0, 0.1, 1.0])
def test_flow_encoder_gradient_scale_checkpoint_roundtrip(tmp_path, scale):
    config = joint_config()
    if scale is not None:
        config.model.flow_encoder_grad_scale = scale
    model = hydra.utils.instantiate(
        config.model, optimizer_config=config.optim, _recursive_=False
    ).eval()
    hyperparameters = dict(model.hparams)
    if scale is None:
        hyperparameters.pop("flow_encoder_grad_scale")
    path = tmp_path / "model.ckpt"
    torch.save(
        {
            "pytorch-lightning_version": pl.__version__,
            "hyper_parameters": hyperparameters,
            "state_dict": model.state_dict(),
        },
        path,
    )
    loaded = type(model).load_from_checkpoint(path).eval()
    assert loaded.flow_encoder_grad_scale == (1.0 if scale is None else scale)
    batch = Batch.from_data_list([make_graph(1), make_graph(2)])
    torch.testing.assert_close(
        loaded.predict_space_groups(batch.composition),
        model.predict_space_groups(batch.composition),
        rtol=0,
        atol=0,
    )


def test_sg_regularization_trains_and_dropout_turns_off_for_prediction():
    torch.manual_seed(42)
    model = make_model()
    batch = Batch.from_data_list([make_graph(1), make_graph(2)])
    optimizer = model.configure_optimizers()
    # The compatibility output starts at zero; its hidden layers receive
    # gradients after the first update of that output layer.
    for _ in range(2):
        optimizer.zero_grad()
        loss = F.cross_entropy(
            model.space_group_logits(batch.composition), batch.space_group
        )
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)

    for head in (model.sg_head.mlp, model.sg_head.compatibility_mlp):
        norms = [layer for layer in head if isinstance(layer, torch.nn.LayerNorm)]
        assert norms
        for norm in norms:
            for parameter in norm.parameters():
                assert torch.isfinite(parameter.grad).all()
                assert parameter.grad.count_nonzero() > 0
        dropout = [layer for layer in head if isinstance(layer, torch.nn.Dropout)]
        assert dropout and all(layer.p == 0.1 for layer in dropout)
        assert isinstance(head[-1], torch.nn.Linear)

    for encoder in (model.composition_encoder, model.sg_head.compatibility_encoder):
        assert not any(
            isinstance(layer, torch.nn.LayerNorm)
            or (isinstance(layer, torch.nn.Dropout) and layer.p > 0)
            for layer in encoder.modules()
        )

    before = model.predict_space_groups(batch.composition)
    after = model.predict_space_groups(batch.composition)
    assert not torch.equal(before, after)
    model.eval()
    before = model.predict_space_groups(batch.composition)
    after = model.predict_space_groups(batch.composition)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(
        before[:1], model.predict_space_groups(batch.composition[:1])
    )


def test_joint_weight_decay_covers_all_branches_and_exempts_vectors():
    config = joint_config(
        "optim.weight_decay=0.01", "model.decay_matrix_weights_only=true"
    )
    model = hydra.utils.instantiate(
        config.model, optimizer_config=config.optim, _recursive_=False
    )
    optimizer = model.configure_optimizers()
    parameters = dict(model.named_parameters())
    grouped = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert len(grouped) == len(parameters)
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in parameters.values()
    }
    expected_decay = {
        "sg_head.mlp.0.weight": 0.01,
        "sg_head.mlp.0.bias": 0.0,
        "sg_head.mlp.1.weight": 0.0,
        "sg_head.mlp.1.bias": 0.0,
        "sg_head.compatibility_mlp.0.weight": 0.01,
        "sg_head.compatibility_encoder.token_encoder.0.weight": 0.01,
        "sg_head.compatibility_encoder.multiplicity_embedding.weight": 0.01,
        "composition_encoder.token_encoder.0.weight": 0.01,
        "composition_encoder.element_embedding.weight": 0.01,
        "decoder.empty_head.0.weight": 0.01,
        "decoder.empty_head.0.bias": 0.0,
        "decoder.input_norm.weight": 0.0,
    }
    before = {name: parameters[name].detach().clone() for name in expected_decay}
    for parameter in parameters.values():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for name, weight_decay in expected_decay.items():
        torch.testing.assert_close(
            parameters[name],
            before[name] * (1 - config.optim.lr * weight_decay),
            rtol=0,
            atol=0,
        )
    restored_optimizer = model.configure_optimizers()
    restored_optimizer.load_state_dict(optimizer.state_dict())
    assert [group["weight_decay"] for group in restored_optimizer.param_groups] == [
        0.01,
        0.0,
    ]


def test_space_group_prediction_ignores_template_and_masks_infeasible_groups():
    model = make_model().eval()
    original = make_graph(1)
    changed = make_graph(2)
    changed.x_0_dof.fill_(8)
    changed.x_inf_dof.fill_(3)
    with torch.no_grad():
        before = model.predict_space_groups(original.composition)
        after = model.predict_space_groups(changed.composition)
        original_loss = model(Batch.from_data_list([original]))["sg_loss"]
        changed_loss = model(Batch.from_data_list([changed]))["sg_loss"]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(original_loss, -before[0, 1].log())
    torch.testing.assert_close(changed_loss, -before[0, 2].log())
    assert before[0, 0] == 0
    single_atom = formula_to_counts("Li", 10)[None]
    probabilities = model.predict_space_groups(single_atom)
    assert probabilities[0, 225] == 0  # F-centred group cannot fit one atom.
    assert probabilities[0, 1] > 0
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1))
    records = predict_space_group_conditions(model, single_atom, [17], 230)
    assert all(int(record.target_index) == 17 for record in records)
    assert all(probabilities[0, record.space_group] > 0 for record in records)


def test_flow_uses_dynamic_occupations_and_caches_static_features():
    model = make_model().eval()
    batch = Batch.from_data_list([make_graph(2)])
    empty = batch.clone()
    empty.x_0_dof.zero_()
    empty.x_inf_dof.zero_()
    with torch.no_grad():
        features = model.encode_composition(batch.composition)
        full_logits = model.decode(batch, torch.tensor([0.5]), features)
        empty_logits = model.decode(empty, torch.tensor([0.5]), features)
    assert not torch.allclose(full_logits[1], empty_logits[1])
    assert not torch.allclose(
        model.decoder._encode_budget(
            batch.composition[:, 1:], torch.zeros(1, 10), features
        )[1],
        model.decoder._encode_budget(
            batch.composition[:, 1:], batch.composition[:, 1:], features
        )[1],
    )
    calls = []
    handle = model.composition_encoder.register_forward_hook(
        lambda *args: calls.append(1)
    )
    conditions = predict_space_group_conditions(model, batch.composition, [23], 2)
    calls.clear()
    data, zero, variable = model.sample_logits(
        Batch.from_data_list(conditions), flow_steps=3
    )
    handle.remove()
    assert calls == [1]
    decoded = decode_composition_logits(
        data, zero, variable, 8, stochastic=False, cpu_workers=1
    )
    assert len(decoded.samples) == 2
    for sample in decoded.samples:
        assert int(sample.target_index) == 23
        _, allocated = occupation_counts(Batch.from_data_list([sample]), 10)
        torch.testing.assert_close(allocated, batch.composition[:, 1:])


@pytest.mark.parametrize("legacy_config", [False, True])
@pytest.mark.parametrize("decay_matrix_weights_only", [False, True])
def test_joint_checkpoint_roundtrip_and_weighted_loss(
    tmp_path, legacy_config, decay_matrix_weights_only
):
    config = joint_config("model.sg_loss_weight=0.3", "model.flow_loss_weight=2.0")
    config.model.decay_matrix_weights_only = decay_matrix_weights_only
    if legacy_config:
        del config.model.sg_head.layer_norm
        del config.model.sg_head.dropout
    model = hydra.utils.instantiate(
        config.model, optimizer_config=config.optim, _recursive_=False
    ).eval()
    batch = Batch.from_data_list([make_graph(1), make_graph(2)])
    metrics = model(batch)
    torch.testing.assert_close(
        metrics["loss"], 0.3 * metrics["sg_loss"] + 2 * metrics["flow_loss"]
    )
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    OmegaConf.save(config, tmp_path / "hparams.yaml")
    torch.save(
        {
            "pytorch-lightning_version": pl.__version__,
            "hyper_parameters": dict(model.hparams),
            "state_dict": model.state_dict(),
        },
        directory / "last.ckpt",
    )
    loaded, _, _ = load_model(tmp_path)
    assert loaded.decay_matrix_weights_only == decay_matrix_weights_only
    with torch.no_grad():
        torch.testing.assert_close(
            model.predict_space_groups(batch.composition),
            loaded.predict_space_groups(batch.composition),
            rtol=0,
            atol=0,
        )
        for instance in (model, loaded):
            torch.manual_seed(7)
            outputs = instance(batch)
            if instance is model:
                expected = outputs
            else:
                for key in expected:
                    torch.testing.assert_close(
                        expected[key], outputs[key], rtol=0, atol=0
                    )


def test_joint_reconstruction_uses_predicted_groups_and_fixed_total_budget(
    monkeypatch, tmp_path
):
    graph = make_graph(1)
    graph.aflow_label = "AB_aP4_1_2a_2a:Li-O"
    seen = []

    def probabilities(compositions):
        result = torch.zeros(len(compositions), 231)
        result[:, 2] = 0.7
        result[:, 1] = 0.3
        result[1:, 2] = 1.0
        result[1:, 1] = 0.0
        return result

    def sample_logits(batch, flow_steps):
        seen.append((batch.space_group.tolist(), batch.target_index.tolist()))
        return batch, None, None

    def decode(data, *args, **kwargs):
        samples = []
        for condition in data.to_data_list():
            sample = make_graph(int(condition.space_group))
            sample.target_index = condition.target_index
            samples.append(sample)
        return DecodingResult(samples, [])

    monkeypatch.setattr(reconstruction, "decode_composition_logits", decode)
    monkeypatch.setattr(reconstruction, "decode_independent_logits", decode)
    model = SimpleNamespace(
        predict_space_groups=probabilities, sample_logits=sample_logits, max_num_atoms=8
    )
    summary = reconstruction.evaluate_reconstruction(
        model,
        [graph, graph.clone()],
        tmp_path,
        num_samples=5,
        flow_steps=2,
        batch_size=10,
        cpu_workers=1,
        predicted_space_groups=2,
    )
    assert seen == [([2, 2, 2, 1, 1] + [2] * 5, [0] * 5 + [1] * 5)]
    assert summary["requested_samples"] == summary["generated_samples"] == 10
    assert summary["match_rate_top1"] == 0
    assert summary["match_rate"] == 0.5
    assert pd.read_csv(tmp_path / "details.csv").generated_count.tolist() == [5, 5]


def test_joint_cli_predicts_conditions_from_one_checkpoint(monkeypatch, tmp_path):
    model = make_model().eval()
    monkeypatch.setattr(
        sample_wy, "load_model", lambda *args, **kwargs: (model, None, None)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = sample_wy.build_parser().parse_args(
        [
            "--model_path",
            str(tmp_path),
            "--formula",
            "Li2O2",
            "--space-group-top-k",
            "2",
            "--num-samples",
            "2",
            "--flow_steps",
            "2",
            "--cpu-workers",
            "1",
            "--save_path",
            str(tmp_path / "samples"),
        ]
    )
    sample_wy.main(args)
    result = torch.load(tmp_path / "samples.pt", weights_only=False)
    assert len(result["generated_samples"]) == 4
    assert {int(s.target_index) for s in result["generated_samples"]} == {0}
    assert len({int(s.space_group) for s in result["generated_samples"]}) == 2


def test_cli_can_use_a_separate_chemical_space_group_model(monkeypatch, tmp_path):
    model = make_model().eval()
    monkeypatch.setattr(sample_wy, "load_model", lambda *a, **kw: (model, None, None))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []

    def predict(composition):
        calls.append(composition.clone())
        probability = torch.zeros(len(composition), 231)
        probability[:, 1] = 0.6
        probability[:, 2] = 0.4
        return probability

    monkeypatch.setattr(
        sample_wy,
        "load_chemical_space_group_model",
        lambda *a: SimpleNamespace(predict_space_groups=predict),
    )
    args = sample_wy.build_parser().parse_args(
        [
            "--model_path",
            str(tmp_path),
            "--formula",
            "Li2O2",
            "--space-group-top-k",
            "2",
            "--space-group-model-path",
            "chemical.pt",
            "--num-samples",
            "1",
            "--flow_steps",
            "2",
            "--cpu-workers",
            "1",
            "--save_path",
            str(tmp_path / "chemical_samples"),
        ]
    )
    sample_wy.main(args)
    result = torch.load(tmp_path / "chemical_samples.pt", weights_only=False)
    assert len(calls) == 1
    assert {int(s.space_group) for s in result["generated_samples"]} == {1, 2}


def test_lightning_training_saves_joint_metric_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "models.run.HydraConfig.get",
        lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path))),
    )
    config = joint_config(
        "train.reconstruction.every_n_epochs=1",
        "train.reconstruction.num_samples=2",
        "train.reconstruction.predicted_space_groups=2",
        "train.reconstruction.flow_steps=2",
        "train.reconstruction.batch_size=2",
        "train.reconstruction.cpu_workers=1",
    )
    graph = make_graph(1)
    graph.aflow_label = "AB_aP4_1_2a_2a:Li-O"

    class DataModule(pl.LightningDataModule):
        def setup(self, stage):
            self.val_datasets = [[graph]]

        def train_dataloader(self):
            return DataLoader([graph], batch_size=1)

        def val_dataloader(self):
            return DataLoader([graph], batch_size=1)

    model = make_model()
    assert isinstance(model.decoder, CrystalGNN)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        callbacks=build_callbacks(config),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=DataModule())
    assert "val/spg_top5" in trainer.callback_metrics
    assert "val/joint_gwa_top2" in trainer.callback_metrics
    assert (tmp_path / "checkpoints/best_joint_gwa_epoch_0000.ckpt").exists()
    assert (tmp_path / "reconstruction/epoch_0000/joint/summary.json").exists()
