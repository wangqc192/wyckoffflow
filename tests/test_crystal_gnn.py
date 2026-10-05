import hydra
import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from torch_geometric.data import Batch

from models.common.composition import formula_to_counts
from models.common.utils import PROJECT_ROOT
from models.pl_models.count_conserving import decode_composition_logits
from models.pl_models.crystal_gnn import CrystalGNN, occupation_counts
from models.pl_models.flow import DiscreteFlowModule
from models.pl_models.model_utils import (
    create_wyckoff_graph,
    get_degrees_of_freedom,
)
from models.sampling import SamplingData


def make_graph(space_group, formula="Li4O4", occupied=False):
    degrees = get_degrees_of_freedom(space_group)
    zero = torch.zeros(int((degrees == 0).sum()), dtype=torch.long)
    variable = torch.zeros(int((degrees != 0).sum()), 10, dtype=torch.long)
    if occupied:
        zero[:1] = 3
        variable[-1, 7] = 2
    graph = create_wyckoff_graph(space_group, zero, variable)
    graph.composition = formula_to_counts(formula, 10).unsqueeze(0)
    return graph


def make_decoder(**kwargs):
    torch.manual_seed(42)
    return CrystalGNN(
        num_elements=10,
        max_num_atoms=8,
        hidden_dim=32,
        element_dim=16,
        num_gnn_layers=2,
        num_heads=4,
        dropout=0.0,
        **kwargs,
    )


def test_atom_accounting_uses_multiplicity_and_current_branch_state():
    graph = make_graph(2, formula="Li2O4")
    # SG 2 has multiplicity-1 fixed positions and one multiplicity-2 general orbit.
    graph.x_0_dof[0] = 3
    graph.x_inf_dof[0, 7] = 3
    assert graph.x.count_nonzero() == 0  # Intentionally stale convenience matrix.
    batch = Batch.from_data_list([graph])
    counts, allocated = occupation_counts(batch, 10)
    assert counts[:, 2].sum() == 1
    assert counts[:, 7].sum() == 3
    torch.testing.assert_close(allocated, formula_to_counts("LiO6", 10)[None, 1:])
    output = make_decoder()(batch, torch.tensor([0.5]))
    assert all(torch.isfinite(logits).all() for logits in output)


@pytest.mark.parametrize("use_symmetry_features", [False, True])
def test_all_space_groups_and_empty_fixed_branch(use_symmetry_features):
    graphs = [make_graph(group) for group in range(1, 231)]
    batch = Batch.from_data_list(graphs)
    decoder = make_decoder(use_symmetry_features=use_symmetry_features).eval()
    with torch.no_grad():
        zero, variable = decoder(batch, torch.linspace(0, 1, 230))
    assert zero.shape == (int(batch.zero_dof.sum()), 11)
    assert variable.shape == (int((~batch.zero_dof).sum()), 10, 9)
    assert torch.isfinite(zero).all() and torch.isfinite(variable).all()
    with torch.no_grad():
        zero, variable = decoder(Batch.from_data_list([graphs[0]]), torch.zeros(1))
    assert zero.shape == (0, 11)
    assert variable.shape == (1, 10, 9)


def test_graph_batching_does_not_mix_compositions_or_messages():
    decoder = make_decoder().eval()
    graph = make_graph(194, occupied=True)
    other = make_graph(1, formula="Be3F2", occupied=True)
    with torch.no_grad():
        alone = decoder(Batch.from_data_list([graph]), torch.tensor([0.3]))
        together = decoder(
            Batch.from_data_list([other, graph]), torch.tensor([0.8, 0.3])
        )
    for branch, offset in zip(alone, (int(other.num_0_dof), int(other.num_inf_dof))):
        index = 0 if branch.ndim == 2 else 1
        torch.testing.assert_close(branch, together[index][offset:])


def test_node_permutation_preserves_predictions():
    decoder = make_decoder().eval()
    graph = make_graph(194, occupied=True)
    permutation = torch.randperm(graph.num_nodes)
    inverse = torch.argsort(permutation)
    permuted = graph.clone()
    for name in (
        "x",
        "zero_dof",
        "degrees_of_freedom",
        "multiplicities",
        "wyckoff_pos_idx",
    ):
        setattr(permuted, name, getattr(graph, name)[permutation])
    permuted.edge_index = inverse[graph.edge_index]
    permuted.x_0_dof = permuted.x[permuted.zero_dof, 0].long()
    permuted.x_inf_dof = permuted.x[~permuted.zero_dof, 1:].long()
    time = torch.tensor([0.4])
    with torch.no_grad():
        original = decoder(Batch.from_data_list([graph]), time)
        reordered = decoder(Batch.from_data_list([permuted]), time)
    for mask, before, after in zip(
        (graph.zero_dof, ~graph.zero_dof), original, reordered
    ):
        full = before.new_zeros((graph.num_nodes, *before.shape[1:]))
        full[mask] = before
        torch.testing.assert_close(full[permutation][mask[permutation]], after)


def test_time_endpoints_and_composition_amount_affect_predictions():
    decoder = make_decoder().eval()
    small = Batch.from_data_list([make_graph(1, formula="Li2O2")])
    large = Batch.from_data_list([make_graph(1, formula="Li4O4")])
    with torch.no_grad():
        at_zero = decoder(small, torch.zeros(1))[1]
        at_one = decoder(small, torch.ones(1))[1]
        doubled = decoder(large, torch.zeros(1))[1]
    assert not torch.allclose(at_zero[:, 7], at_one[:, 7])
    assert not torch.allclose(at_zero[:, 7], doubled[:, 7])


@pytest.mark.parametrize("external_composition", [False, True])
def test_global_composition_encoder_ablation_preserves_initialization_and_only_removes_branch(
    external_composition,
):
    from models.pl_models.composition_encoder import CrystalCompositionEncoder

    baseline = make_decoder(external_composition=external_composition).eval()
    baseline_rng = torch.random.get_rng_state()
    ablated = make_decoder(
        external_composition=external_composition, use_global_composition_encoder=False
    ).eval()
    assert torch.equal(torch.random.get_rng_state(), baseline_rng)
    assert ablated.global_composition_encoder is None
    for name, value in ablated.state_dict().items():
        torch.testing.assert_close(value, baseline.state_dict()[name], rtol=0, atol=0)

    batch = Batch.from_data_list([make_graph(2, occupied=True), make_graph(194)])
    features = (
        CrystalCompositionEncoder(10, 16, 32)(batch.composition)
        if external_composition
        else None
    )
    time = torch.tensor([0.2, 0.8])
    original = baseline(batch, time, features)
    hook = baseline.global_composition_encoder.register_forward_hook(
        lambda module, args, output: torch.zeros_like(output)
    )
    removed = baseline(batch, time, features)
    hook.remove()
    actual = ablated(batch, time, features)
    for expected, result in zip(removed, actual):
        torch.testing.assert_close(expected, result, rtol=0, atol=0)
    assert any(
        not torch.allclose(before, after) for before, after in zip(original, actual)
    )
    sum(logits.square().mean() for logits in actual).backward()
    gradients = [parameter.grad for parameter in ablated.budget_encoder.parameters()]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients)
    assert any(grad.count_nonzero() for grad in gradients)


def make_flow(*overrides):
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "conf"), version_base="1.3"
    ):
        config = compose(
            config_name="default",
            overrides=[
                "model/decoder=crystal_gnn",
                "data.num_elements=10",
                "model.max_num_atoms=8",
                "model.decoder.hidden_dim=32",
                "model.decoder.element_dim=16",
                "model.decoder.num_heads=4",
                "model.decoder.num_gnn_layers=2",
                "model.decoder.dropout=0.0",
                *overrides,
            ],
        )
    return hydra.utils.instantiate(
        config.model, optimizer_config=config.optim, _recursive_=False
    )


@pytest.mark.parametrize(
    "loss_type,smoothing", [("cross_entropy", 0.05), ("brier", 0.0)]
)
def test_flow_loss_backward_with_positive_counts_and_mixed_precision(
    loss_type, smoothing
):
    model = make_flow(
        f"model.loss_type={loss_type}", f"model.label_smoothing={smoothing}"
    )
    batch = Batch.from_data_list(
        [make_graph(1, occupied=True), make_graph(2, occupied=True)]
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(batch)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    for module in (
        model.decoder.count_head,
        model.decoder.element_head,
        model.decoder.budget_encoder,
        model.decoder.symmetry_encoder,
        model.decoder.layers[0].attention.qkv,
    ):
        gradients = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(g.count_nonzero() > 0 for g in gradients)


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("unmasked", [False, True])
def test_flow_sampling_and_exact_composition_decoding(greedy, unmasked):
    model = make_flow(
        f"model.mask_loss_by_composition={not unmasked}",
        f"model.decoder.predict_all_elements={unmasked}",
    ).eval()
    composition = formula_to_counts("Li2O4", 10)
    conditions = Batch.from_data_list(
        [
            SamplingData(formula=composition[None], space_group=torch.tensor(sg))
            for sg in (1, 2)
        ]
    )
    data, zero, variable = model.sample_logits(conditions, flow_steps=3, greedy=greedy)
    absent = composition == 0
    absent[0] = False
    assert torch.isneginf(zero[:, absent]).all()
    assert torch.isneginf(variable[:, absent[1:], 1:]).all()
    decoded = decode_composition_logits(
        data, zero, variable, 8, stochastic=False, cpu_workers=1
    )
    assert len(decoded.samples) == 2
    for sample in decoded.samples:
        _, allocated = occupation_counts(Batch.from_data_list([sample]), 10)
        torch.testing.assert_close(allocated[0], composition[1:])


@pytest.mark.parametrize(
    "unmasked,legacy", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize(
    "loss_type,smoothing", [("cross_entropy", 0.05), ("brier", 0.0)]
)
def test_new_decoder_checkpoint_restores_outputs(
    tmp_path, unmasked, legacy, loss_type, smoothing
):
    model = make_flow(
        f"model.mask_loss_by_composition={not unmasked}",
        f"model.decoder.predict_all_elements={unmasked}",
        f"model.loss_type={loss_type}",
        f"model.label_smoothing={smoothing}",
    ).eval()
    batch = Batch.from_data_list([make_graph(194, occupied=True)])
    path = tmp_path / "crystal.ckpt"
    hyperparameters = dict(model.hparams)
    state_dict = model.state_dict()
    if legacy:
        hyperparameters.pop("mask_loss_by_composition")
        hyperparameters["normalize_loss_by_variables"] = True
        hyperparameters["decoder"] = dict(hyperparameters["decoder"])
        hyperparameters["decoder"].pop("predict_all_elements")
        if loss_type == "cross_entropy":
            hyperparameters.pop("loss_type")
        # Older blocks stored the attention projections directly on each layer.
        state_dict = {
            key.replace(".attention.", ".").replace(
                ".adaLN_modulation.1.", ".condition_affine."
            ): value
            for key, value in state_dict.items()
        }
    torch.save(
        {
            "pytorch-lightning_version": pl.__version__,
            "hyper_parameters": hyperparameters,
            "state_dict": state_dict,
        },
        path,
    )
    loaded = DiscreteFlowModule.load_from_checkpoint(path, weights_only=False).eval()
    assert loaded.mask_loss_by_composition == (not unmasked)
    assert loaded.decoder.predict_all_elements == unmasked
    assert loaded.loss_type == loss_type
    with torch.no_grad():
        before = model.decoder(batch, torch.tensor([0.4]))
        after = loaded.decoder(batch, torch.tensor([0.4]))
    for expected, actual in zip(before, after):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_ablation_options_support_backward():
    decoder = make_decoder(
        use_symmetry_features=False,
        use_edge_bias=False,
        use_composition_residual=False,
    )
    batch = Batch.from_data_list([make_graph(194, occupied=True)])
    zero, variable = decoder(batch, torch.tensor([0.5]))
    (zero.square().sum() + variable.square().sum()).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None
    )


def test_residual_scale_ablation_preserves_shared_initialization_and_checkpoint(
    tmp_path,
):
    torch.manual_seed(42)
    reference = make_flow().eval()
    reference_rng = torch.get_rng_state()
    torch.manual_seed(42)
    ablated = make_flow("model.decoder.use_residual_scale=false").eval()
    assert torch.equal(torch.get_rng_state(), reference_rng)
    reference_state = reference.state_dict()
    ablated_state = ablated.state_dict()
    assert not any(key.endswith("residual_scale") for key in ablated_state)
    for key, value in ablated_state.items():
        torch.testing.assert_close(value, reference_state[key], rtol=0, atol=0)
    with torch.no_grad():
        for block in reference.decoder.layers:
            block.residual_scale.fill_(1)
    batch = Batch.from_data_list([make_graph(2, occupied=True)])
    time = torch.tensor([0.5])
    expected = reference.decoder(batch, time)
    actual = ablated.decoder(batch, time)
    for before, after in zip(expected, actual):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    sum(logits.square().sum() for logits in actual).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in ablated.parameters() if p.grad is not None
    )
    path = tmp_path / "no_residual_scale.ckpt"
    torch.save(
        {
            "pytorch-lightning_version": pl.__version__,
            "hyper_parameters": dict(ablated.hparams),
            "state_dict": ablated_state,
        },
        path,
    )
    restored = DiscreteFlowModule.load_from_checkpoint(path, weights_only=False).eval()
    assert all(block.residual_scale is None for block in restored.decoder.layers)
    for before, after in zip(actual, restored.decoder(batch, time)):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_all_element_heads_preserve_present_predictions():
    batch = Batch.from_data_list([make_graph(1), make_graph(2, occupied=True)])
    restricted = make_decoder().eval()
    full = make_decoder(predict_all_elements=True).eval()
    time = torch.tensor([0.2, 0.7])
    with torch.no_grad():
        zero, inf = restricted(batch, time)
        full_zero, full_inf = full(batch, time)
    torch.testing.assert_close(full_zero[:, [0, 3, 8]], zero[:, [0, 3, 8]])
    torch.testing.assert_close(full_inf[:, [2, 7]], inf[:, [2, 7]])


def test_absent_element_predictions_have_finite_nonzero_gradients():
    graph = make_graph(2, occupied=True)
    # Include a noisy occupation of H, absent from the target Li/O composition.
    graph.x_inf_dof[0, 0] = 1
    decoder = make_decoder(predict_all_elements=True)
    batch = Batch.from_data_list([graph])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        zero, inf = decoder(batch, torch.tensor([0.5]))
        loss = F.cross_entropy(
            zero[:, [0, 1]], torch.zeros(zero.shape[0], dtype=torch.long)
        ) + F.cross_entropy(inf[:, 0], torch.zeros(inf.shape[0], dtype=torch.long))
    loss.backward()
    assert torch.isfinite(loss)
    for module in (decoder.element_head, decoder.count_head, decoder.pair_encoder):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(g.count_nonzero() for g in gradients)
    assert decoder.element_embedding.weight.grad[1].count_nonzero() > 0
    assert all(
        torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None
    )


def test_unmasked_loss_counts_all_elements_and_averages_per_crystal(monkeypatch):
    model = make_flow(
        "model.mask_loss_by_composition=false",
        "model.decoder.predict_all_elements=true",
    )
    graphs = [make_graph(1, occupied=True), make_graph(2, occupied=True)]
    batch = Batch.from_data_list(graphs)
    zero = torch.randn(batch.x_0_dof.numel(), 11, requires_grad=True)
    inf = torch.randn(*batch.x_inf_dof.shape, 9, requires_grad=True)
    monkeypatch.setattr(model, "decode", lambda *_args: (zero, inf))
    loss = model(batch)["loss"]
    per_graph = []
    nz_offset = ni_offset = 0
    for graph in graphs:
        nz, ni = int(graph.num_0_dof), int(graph.num_inf_dof)
        total = F.cross_entropy(
            zero[nz_offset : nz_offset + nz],
            graph.x_0_dof,
            reduction="sum",
            label_smoothing=model.label_smoothing,
        ) + F.cross_entropy(
            inf[ni_offset : ni_offset + ni].reshape(-1, 9),
            graph.x_inf_dof.flatten(),
            reduction="sum",
            label_smoothing=model.label_smoothing,
        )
        per_graph.append(total)
        nz_offset += nz
        ni_offset += ni
    torch.testing.assert_close(loss, torch.stack(per_graph).mean())
    loss.backward()
    # H is absent: its count-zero targets must now contribute to training.
    assert (inf.grad[:, 0, 0] < 0).all()
    assert torch.isfinite(inf.grad[:, 0, 1:]).all()
    assert inf.grad[:, 0, 1:].abs().sum() > 0


def test_unmasked_loss_rejects_placeholder_element_outputs():
    with pytest.raises(
        hydra.errors.InstantiationException, match="predict_all_elements"
    ):
        make_flow("model.mask_loss_by_composition=false")


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("loss_type", ["cross_entropy", "brier"])
def test_graph_loss_reduction_and_composition_mask(monkeypatch, masked, loss_type):
    model = make_flow(
        f"model.loss_type={loss_type}",
        "model.label_smoothing=0.0",
        "model.zero_df_loss_weight=2.0",
        "model.inf_df_loss_weight=3.0",
        f"model.mask_loss_by_composition={masked}",
        f"model.decoder.predict_all_elements={not masked}",
    )
    graphs = [make_graph(1, occupied=True), make_graph(2, occupied=True)]
    batch = Batch.from_data_list(graphs)
    zero = torch.randn(batch.x_0_dof.numel(), 11, requires_grad=True)
    inf = torch.randn(*batch.x_inf_dof.shape, 9, requires_grad=True)
    monkeypatch.setattr(model, "decode", lambda *_args: (zero, inf))
    losses = model(batch)
    zero_values, inf_values = [], []
    nz_offset = ni_offset = 0
    for graph in graphs:
        nz, ni = int(graph.num_0_dof), int(graph.num_inf_dof)
        elements = torch.tensor([2, 7]) if masked else torch.arange(10)
        zlogits = zero[nz_offset : nz_offset + nz]
        if masked:
            zlogits = zlogits.masked_fill(
                ~torch.isin(torch.arange(11), torch.tensor([0, 3, 8])), -torch.inf
            )
        ilogits = inf[ni_offset : ni_offset + ni, elements]
        ztargets = graph.x_0_dof
        itargets = graph.x_inf_dof[:, elements]
        if loss_type == "brier":
            zloss = (zlogits.softmax(-1) - F.one_hot(ztargets, 11)).square().sum()
            iloss = (ilogits.softmax(-1) - F.one_hot(itargets, 9)).square().sum()
        else:
            zloss = F.cross_entropy(zlogits, ztargets, reduction="sum")
            iloss = F.cross_entropy(
                ilogits.reshape(-1, 9), itargets.flatten(), reduction="sum"
            )
        zero_values.append(zloss)
        inf_values.append(iloss)
        nz_offset += nz
        ni_offset += ni
    expected_zero, expected_inf = (
        torch.stack(zero_values).mean(),
        torch.stack(inf_values).mean(),
    )
    torch.testing.assert_close(losses["zero_df_loss"], expected_zero)
    torch.testing.assert_close(losses["inf_df_loss"], expected_inf)
    torch.testing.assert_close(losses["loss"], 2 * expected_zero + 3 * expected_inf)
    losses["loss"].backward()
    assert torch.isfinite(zero.grad).all() and torch.isfinite(inf.grad).all()
    if masked:
        assert zero.grad[:, 1].count_nonzero() == 0
        assert inf.grad[:, 0].count_nonzero() == 0
    else:
        assert (inf.grad[:, 0, 0] < 0).all()


def test_brier_requires_hard_targets():
    with pytest.raises(
        hydra.errors.InstantiationException, match="label_smoothing=0.0"
    ):
        make_flow("model.loss_type=brier", "model.label_smoothing=0.05")
