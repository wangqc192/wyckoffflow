import hydra
import pytest
import pytorch_lightning as pl
import torch
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


def make_flow():
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
            ],
        )
    return hydra.utils.instantiate(
        config.model, optimizer_config=config.optim, _recursive_=False
    )


def test_flow_loss_backward_with_positive_counts_and_mixed_precision():
    model = make_flow()
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
        model.decoder.layers[0].qkv,
    ):
        gradients = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(g.count_nonzero() > 0 for g in gradients)


@pytest.mark.parametrize("greedy", [False, True])
def test_flow_sampling_and_exact_composition_decoding(greedy):
    model = make_flow().eval()
    composition = formula_to_counts("Li2O4", 10)
    conditions = Batch.from_data_list(
        [
            SamplingData(formula=composition[None], space_group=torch.tensor(sg))
            for sg in (1, 2)
        ]
    )
    data, zero, variable = model.sample_logits(conditions, flow_steps=3, greedy=greedy)
    decoded = decode_composition_logits(
        data, zero, variable, 8, stochastic=False, cpu_workers=1
    )
    assert len(decoded.samples) == 2
    for sample in decoded.samples:
        _, allocated = occupation_counts(Batch.from_data_list([sample]), 10)
        torch.testing.assert_close(allocated[0], composition[1:])


def test_new_decoder_checkpoint_restores_outputs(tmp_path):
    model = make_flow().eval()
    batch = Batch.from_data_list([make_graph(194, occupied=True)])
    path = tmp_path / "crystal.ckpt"
    torch.save(
        {
            "pytorch-lightning_version": pl.__version__,
            "hyper_parameters": dict(model.hparams),
            "state_dict": model.state_dict(),
        },
        path,
    )
    loaded = DiscreteFlowModule.load_from_checkpoint(path, weights_only=False).eval()
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
