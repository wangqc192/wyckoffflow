import os

import torch
from torch.utils.data import ConcatDataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from wyckoff_generation.common.composition import decode_composition

import scripts.sample_wy as sample_wy
from models.common.composition import formula_to_counts
from models.pl_models.count_conserving import _graph_offsets, cpu_dp_worker_count
from models.pl_models.flow import DiscreteFlowModule
from models.pl_models.model_utils import create_x_matrix
from scripts.sample_wy import (
    SampleDataset,
    flow_logits,
    repair_logits_file,
    repair_saved_logits,
)

OPTIMIZER_CONFIG = {"_target_": "torch.optim.AdamW", "lr": 2e-4}


MODEL_CONFIG = {
    "num_elements": 118,
    "max_num_atoms": 8,
    "flow_source": "zeros",
    "flow_steps": 1,
    "conditional_composition": True,
    "composition_encoder_dim": 4,
    "composition_film": False,
    "count_conserving": True,
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


def test_sample_uses_final_logits_for_exact_composition():
    model = DiscreteFlowModule(MODEL_CONFIG, OPTIMIZER_CONFIG).eval()
    batch = Batch.from_data_list(
        [
            Data(
                formula=formula_to_counts("Ga4Te4", 118).unsqueeze(0),
                num_evals=torch.tensor(2),
                space_group=torch.tensor(194),
            )
        ]
    )

    sampled = model.sample(batch, count_conserving=True, flow_steps=1)

    assert sampled.num_graphs == 2
    assert sampled.x_0_dof.device.type == "cpu"
    assert sampled.x_inf_dof.device.type == "cpu"
    assert sampled.x_0_dof.dtype == torch.long
    decoded = decode_composition(sampled, 118).round().long()
    assert torch.equal(decoded, sampled.composition.round().long())


def test_saved_logits_from_all_batches_are_repaired_together():
    model = DiscreteFlowModule(MODEL_CONFIG, OPTIMIZER_CONFIG).eval()
    dataset = ConcatDataset(
        [
            SampleDataset("Ga4Te4", 2, space_group=194, target_index=0),
            SampleDataset("Ga4Te4", 2, space_group=194, target_index=1),
        ]
    )
    loader = DataLoader(dataset, batch_size=1)

    saved_logits = flow_logits(loader, model, flow_steps=1)

    assert len(saved_logits) == 2
    assert all(zero.device.type == "cpu" for _, zero, _ in saved_logits)
    generated = repair_saved_logits(
        saved_logits, model.max_num_atoms, cpu_workers=2, cpu_task_size=1
    )

    assert len(generated) == 4
    decoded = decode_composition(Batch.from_data_list(generated), 118).round().long()
    targets = Batch.from_data_list(generated).composition.round().long()
    assert torch.equal(decoded, targets)
    for sample in generated:
        expected_x = create_x_matrix(sample.x_inf_dof, sample.x_0_dof, sample.zero_dof)
        assert torch.equal(sample.x, expected_x)


def test_impossible_graph_is_skipped_without_retrying_batch(monkeypatch, capsys):
    model = DiscreteFlowModule(MODEL_CONFIG, OPTIMIZER_CONFIG).eval()
    dataset = ConcatDataset(
        [
            SampleDataset("Ga4Te4", 2, space_group=194, target_index=0),
            SampleDataset("O40F4Na12P8Ti8", 2, space_group=43, target_index=1),
        ]
    )
    saved_logits = flow_logits(DataLoader(dataset, batch_size=2), model, flow_steps=1)
    original = sample_wy.sample_batch_to_compositions
    call_count = 0

    def counted_sample_batch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sample_wy, "sample_batch_to_compositions", counted_sample_batch)
    generated = repair_saved_logits(
        saved_logits, model.max_num_atoms, cpu_workers=1, cpu_task_size=1
    )

    assert call_count == 1
    assert len(generated) == 2
    assert all(int(sample.target_index) == 0 for sample in generated)
    output = capsys.readouterr().out
    assert "Skipping formula=O40F4Na12P8Ti8, space_group=43" in output
    assert "(2 samples)" in output


def test_repair_logits_file_skips_model_and_writes_samples(tmp_path):
    model = DiscreteFlowModule(MODEL_CONFIG, OPTIMIZER_CONFIG).eval()
    dataset = ConcatDataset(
        [SampleDataset("Ga4Te4", 1, space_group=194, target_index=0)]
    )
    saved_logits = flow_logits(DataLoader(dataset, batch_size=1), model, flow_steps=1)
    logits_path = tmp_path / "sample.logits.pt"
    output_path = tmp_path / "sample.pt"
    torch.save(
        {"time": 1.25, "logit_batches": saved_logits, "args": {"num_evals": 1}},
        logits_path,
    )

    generated, cpu_dp_time = repair_logits_file(logits_path, output_path, cpu_workers=1)

    assert len(generated) == 1
    assert cpu_dp_time >= 0
    output = torch.load(output_path, map_location="cpu", weights_only=False)
    assert output["gpu_flow_time"] == 1.25
    assert output["reused_logits_path"] == str(logits_path)
    assert output["args"] == {"num_evals": 1}
    sample = output["generated_samples"][0]
    expected_x = create_x_matrix(sample.x_inf_dof, sample.x_0_dof, sample.zero_dof)
    assert torch.equal(sample.x, expected_x)


def test_graph_offsets_are_linear_prefix_sums():
    assignment = torch.tensor([0, 0, 1, 3, 3])

    offsets = _graph_offsets(assignment, num_graphs=4)

    assert offsets.tolist() == [0, 2, 3, 3, 5]


def test_cpu_worker_count_uses_requested_52_workers():
    assert cpu_dp_worker_count(52, 180_920) == min(52, os.cpu_count())


def test_top_n_returns_distinct_candidates_and_expands_small_beam():
    from models.pl_models.count_conserving import _decode_single_candidates

    zero_logits = torch.zeros((1, 3))
    inf_logits = torch.zeros((1, 2, 3))
    target = torch.tensor([0, 1, 1])

    candidates = _decode_single_candidates(
        zero_logits,
        inf_logits,
        (1,),
        (1,),
        target,
        max_variable_count=2,
        beam_size=1,
        candidate_count=5,
    )

    signatures = {
        (tuple(zero.tolist()), tuple(inf.flatten().tolist()))
        for zero, inf in candidates
    }
    assert len(candidates) == 3
    assert len(signatures) == 3


def test_top_n_does_not_deduplicate_different_target_groups():
    import numpy as np

    from models.pl_models.count_conserving import _candidate_data_list
    from scripts.sample_wy import SamplingData

    graph = SamplingData(
        x=torch.tensor([[0.0, 1.0, 0.0]]),
        x_0_dof=torch.tensor([], dtype=torch.long),
        x_inf_dof=torch.tensor([[1, 0]], dtype=torch.long),
        zero_dof=torch.tensor([False]),
        target_index=torch.tensor(10),
        space_group=torch.tensor(1),
    )
    graph_other = graph.clone()
    graph_other.target_index = torch.tensor(20)
    batch = Batch.from_data_list([graph, graph_other])
    decoded_zero = np.empty(0, dtype=np.int64)
    decoded_inf = np.array([[1, 0]], dtype=np.int64)

    samples = _candidate_data_list(
        batch,
        {0: [(decoded_zero, decoded_inf)], 1: [(decoded_zero, decoded_inf)]},
    )

    assert len(samples) == 2
    assert [int(sample.target_index) for sample in samples] == [10, 20]
