import os

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from wyckoff_generation.common.composition import decode_composition

import models.sampling as sampling
import scripts.sample_wy as sample_wy
from models.common.composition import formula_to_counts
from models.pl_models.count_conserving import _graph_offsets, cpu_dp_worker_count
from models.pl_models.flow import DiscreteFlowModule
from models.pl_models.model_utils import (
    create_wyckoff_graph,
    create_x_matrix,
    get_degrees_of_freedom,
)
from models.sampling import (
    collect_flow_logits,
    decode_logit_batches,
    make_condition,
    sample_batch,
)
from scripts.sample_wy import decode_logits_file

OPTIMIZER_CONFIG = {"_target_": "torch.optim.AdamW", "lr": 2e-4}


MODEL_CONFIG = {
    "num_elements": 118,
    "max_num_atoms": 8,
    "flow_source": "zeros",
    "conditional_composition": True,
}

DECODER_CONFIG = {
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


def test_loss_weights_control_flow_loss():
    degrees = get_degrees_of_freedom(194)
    graph = create_wyckoff_graph(
        194,
        torch.zeros(int((degrees == 0).sum()), dtype=torch.long),
        torch.zeros(int((degrees != 0).sum()), 118, dtype=torch.long),
    )
    graph.composition = formula_to_counts("Ga4Te4", 118).unsqueeze(0)
    batch = Batch.from_data_list([graph])

    base_config = {
        **MODEL_CONFIG,
        "zero_df_loss_weight": 1.0,
        "inf_df_loss_weight": 1.0,
    }
    weighted_config = {
        **MODEL_CONFIG,
        "zero_df_loss_weight": 2.0,
        "inf_df_loss_weight": 3.0,
    }
    base_model = DiscreteFlowModule(
        optimizer_config=OPTIMIZER_CONFIG, decoder=DECODER_CONFIG, **base_config
    ).eval()
    weighted_model = DiscreteFlowModule(
        optimizer_config=OPTIMIZER_CONFIG, decoder=DECODER_CONFIG, **weighted_config
    ).eval()
    weighted_model.load_state_dict(base_model.state_dict())

    torch.manual_seed(7)
    base_losses = base_model(batch)
    torch.manual_seed(7)
    weighted_losses = weighted_model(batch)

    assert torch.allclose(weighted_losses["zero_df_loss"], base_losses["zero_df_loss"])
    assert torch.allclose(weighted_losses["inf_df_loss"], base_losses["inf_df_loss"])
    assert torch.allclose(
        weighted_losses["loss"],
        2 * base_losses["zero_df_loss"] + 3 * base_losses["inf_df_loss"],
    )


def test_sample_uses_final_logits_for_exact_composition():
    model = DiscreteFlowModule(
        optimizer_config=OPTIMIZER_CONFIG, decoder=DECODER_CONFIG, **MODEL_CONFIG
    ).eval()
    batch = Batch.from_data_list(
        [
            Data(
                formula=formula_to_counts("Ga4Te4", 118).unsqueeze(0),
                space_group=torch.tensor(194),
            )
        ]
    )

    result = sample_batch(model, batch, num_samples=2, flow_steps=1)
    sampled = Batch.from_data_list(result.samples)

    assert sampled.num_graphs == 2
    assert sampled.x_0_dof.device.type == "cpu"
    assert sampled.x_inf_dof.device.type == "cpu"
    assert sampled.x_0_dof.dtype == torch.long
    decoded = decode_composition(sampled, 118).round().long()
    assert torch.equal(decoded, sampled.composition.round().long())


def test_saved_logits_share_a_cpu_pool_across_batches():
    model = DiscreteFlowModule(
        optimizer_config=OPTIMIZER_CONFIG, decoder=DECODER_CONFIG, **MODEL_CONFIG
    ).eval()
    dataset = [
        make_condition("Ga4Te4", 194, 118, target_index=0),
        make_condition("Ga4Te4", 194, 118, target_index=1),
    ]
    loader = DataLoader(dataset, batch_size=1)

    saved_logits = collect_flow_logits(loader, model, flow_steps=1, num_samples=2)

    assert len(saved_logits) == 2
    assert all(zero.device.type == "cpu" for _, zero, _ in saved_logits)
    result = decode_logit_batches(
        saved_logits, model.max_num_atoms, cpu_workers=2, cpu_task_size=1
    )

    generated = result.samples
    assert len(generated) == 4
    decoded = decode_composition(Batch.from_data_list(generated), 118).round().long()
    targets = Batch.from_data_list(generated).composition.round().long()
    assert torch.equal(decoded, targets)
    for sample in generated:
        expected_x = create_x_matrix(sample.x_inf_dof, sample.x_0_dof, sample.zero_dof)
        assert torch.equal(sample.x, expected_x)


def test_impossible_graph_is_skipped_without_retrying_batch(monkeypatch, capsys):
    model = DiscreteFlowModule(
        optimizer_config=OPTIMIZER_CONFIG, decoder=DECODER_CONFIG, **MODEL_CONFIG
    ).eval()
    dataset = [
        make_condition("Ga4Te4", 194, 118, target_index=0),
        make_condition("O40F4Na12P8Ti8", 43, 118, target_index=1),
    ]
    saved_logits = collect_flow_logits(
        DataLoader(dataset, batch_size=2), model, flow_steps=1, num_samples=2
    )
    original = sampling.decode_composition_logits
    call_count = 0

    def counted_sample_batch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sampling, "decode_composition_logits", counted_sample_batch)
    result = decode_logit_batches(
        saved_logits, model.max_num_atoms, cpu_workers=1, cpu_task_size=1
    )

    generated = result.samples
    assert result.infeasible_graph_indices == [2, 3]
    assert call_count == 1
    assert len(generated) == 2
    assert all(int(sample.target_index) == 0 for sample in generated)
    output = capsys.readouterr().out
    assert "Skipping formula=O40F4Na12P8Ti8, space_group=43" in output
    assert "(2 trajectories)" in output


def test_decode_logits_file_skips_model_and_writes_samples(tmp_path):
    model = DiscreteFlowModule(
        optimizer_config=OPTIMIZER_CONFIG, decoder=DECODER_CONFIG, **MODEL_CONFIG
    ).eval()
    dataset = [make_condition("Ga4Te4", 194, 118, target_index=0)]
    saved_logits = collect_flow_logits(
        DataLoader(dataset, batch_size=1), model, flow_steps=1, num_samples=1
    )
    logits_path = tmp_path / "sample.logits.pt"
    output_path = tmp_path / "sample.pt"
    torch.save(
        {"time": 1.25, "logit_batches": saved_logits, "args": {"num_evals": 1}},
        logits_path,
    )

    result, cpu_decode_time = decode_logits_file(
        logits_path, output_path, cpu_workers=1
    )

    assert len(result.samples) == 1
    assert cpu_decode_time >= 0
    output = torch.load(output_path, map_location="cpu", weights_only=False)
    assert output["gpu_flow_time"] == 1.25
    assert output["reused_logits_path"] == str(logits_path)
    assert output["args"]["num_samples"] == 1
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

    from models.pl_models.count_conserving import _decoded_data_list
    from models.sampling import SamplingData

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

    samples = _decoded_data_list(
        batch,
        {
            0: [(decoded_zero, decoded_inf, -2.0)],
            1: [(decoded_zero, decoded_inf, -1.0)],
        },
        rank_candidates=True,
    )

    assert len(samples) == 2
    assert [int(sample.target_index) for sample in samples] == [10, 20]
    assert [int(sample.candidate_rank) for sample in samples] == [1, 1]
    assert [float(sample.decoder_log_score) for sample in samples] == [-2.0, -1.0]


@pytest.mark.parametrize(
    "sampling_mode,enforce_composition,num_trajectories",
    [
        ("n-shot", True, 3),
        ("n-shot", False, 3),
        ("top-n", True, 1),
        ("greedy", True, 3),
        ("greedy", False, 3),
    ],
)
def test_shared_sampler_decodes_all_modes_without_mutating_flow_state(
    sampling_mode, enforce_composition, num_trajectories
):
    model = DiscreteFlowModule(
        optimizer_config=OPTIMIZER_CONFIG, decoder=DECODER_CONFIG, **MODEL_CONFIG
    ).eval()
    calls = []
    handle = model.decoder.register_forward_hook(
        lambda module, inputs, output: calls.append(inputs[1].clone())
    )
    records = [make_condition("Ga4Te4", 194, 118, target_index=9)]
    batches = collect_flow_logits(
        DataLoader(records, batch_size=1),
        model,
        flow_steps=3,
        num_samples=3,
        sampling_mode=sampling_mode,
    )
    handle.remove()
    assert len(calls) == 3
    assert all(time.numel() == num_trajectories for time in calls)
    torch.testing.assert_close(torch.stack(calls)[:, 0], torch.arange(3) / 3)
    data, zero_logits, inf_logits = batches[0]
    original_x = data.x.clone()
    original_zero = zero_logits.clone()
    original_inf = inf_logits.clone()
    result = decode_logit_batches(
        batches,
        model.max_num_atoms,
        sampling_mode=sampling_mode,
        num_samples=3,
        enforce_composition=enforce_composition,
        cpu_workers=1,
    )
    assert len(result.samples) == 3
    assert result.infeasible_graph_indices == []
    torch.testing.assert_close(data.x, original_x)
    torch.testing.assert_close(zero_logits, original_zero)
    torch.testing.assert_close(inf_logits, original_inf)
    assert all(int(sample.target_index) == 9 for sample in result.samples)
    if enforce_composition:
        sampled = Batch.from_data_list(result.samples)
        assert torch.equal(
            decode_composition(sampled, 118).round(), sampled.composition
        )
    if sampling_mode == "top-n":
        scores = [float(sample.decoder_log_score) for sample in result.samples]
        assert scores == sorted(scores, reverse=True)
        assert [int(sample.candidate_rank) for sample in result.samples] == [1, 2, 3]
    else:
        assert all("candidate_rank" not in sample for sample in result.samples)


@pytest.mark.parametrize(
    "sampling_mode,failed",
    [("n-shot", [0, 1]), ("top-n", [0]), ("greedy", [0, 1])],
)
def test_all_infeasible_inputs_return_failures_without_raising(sampling_mode, failed):
    model = DiscreteFlowModule(
        optimizer_config=OPTIMIZER_CONFIG, decoder=DECODER_CONFIG, **MODEL_CONFIG
    ).eval()
    batch = Batch.from_data_list([make_condition("Ga", 225, 118, target_index=0)])
    result = sample_batch(
        model,
        batch,
        flow_steps=1,
        num_samples=2,
        sampling_mode=sampling_mode,
        cpu_workers=1,
    )
    assert result.samples == []
    assert result.infeasible_graph_indices == failed


def test_greedy_updates_every_step_with_masked_argmax():
    model = DiscreteFlowModule(
        optimizer_config=OPTIMIZER_CONFIG, decoder=DECODER_CONFIG, **MODEL_CONFIG
    ).eval()
    states = []
    rng_states = []

    class RecordingDecoder(torch.nn.Module):
        def forward(self, data, time):
            states.append(data.clone())
            rng_states.append(torch.random.get_rng_state())
            step = len(states) - 1
            zero_logits = torch.zeros((data.x_0_dof.numel(), 119))
            inf_logits = torch.zeros((*data.x_inf_dof.shape, 9))
            zero_logits[:, 31 if step % 2 == 0 else 52] = 5
            # H is absent from Ga4Te4 and must remain masked despite a high score.
            zero_logits[:, 1] = 100
            inf_logits[:, :, step + 1] = 5
            return zero_logits, inf_logits

    model.decoder = RecordingDecoder()
    batch = Batch.from_data_list([make_condition("Ga4Te4", 194, 118, 0)])
    data, zero_logits, inf_logits = model.sample_logits(batch, 3, greedy=True)

    assert len(states) == 3
    assert torch.all(states[0].x == 0)
    for step, expected_element in [(1, 31), (2, 52)]:
        state = states[step]
        assert torch.all(state.x_0_dof == expected_element)
        expected_inf = torch.zeros_like(state.x_inf_dof)
        expected_inf[:, [30, 51]] = step
        torch.testing.assert_close(state.x_inf_dof, expected_inf)
        torch.testing.assert_close(
            state.x, create_x_matrix(expected_inf, state.x_0_dof, state.zero_dof)
        )
    assert all(torch.equal(rng_states[0], state) for state in rng_states)
    assert torch.equal(rng_states[0], torch.random.get_rng_state())
    decoded = decode_logit_batches(
        [(data, zero_logits, inf_logits)],
        8,
        sampling_mode="greedy",
        enforce_composition=False,
    ).samples[0]
    assert torch.all(decoded.x_0_dof == 31)
    assert torch.all(decoded.x_inf_dof[:, [30, 51]] == 3)
    assert torch.all(decoded.x_inf_dof[:, 0] == 0)
    assert torch.equal(rng_states[0], torch.random.get_rng_state())


@pytest.mark.parametrize("enforce_composition", [False, True])
@pytest.mark.parametrize("flow_steps", [1, 3])
@pytest.mark.parametrize("space_group", [1, 194])
def test_greedy_zero_source_is_repeatable_and_cached_decode_agrees(
    tmp_path, enforce_composition, flow_steps, space_group
):
    model = DiscreteFlowModule(
        optimizer_config=OPTIMIZER_CONFIG, decoder=DECODER_CONFIG, **MODEL_CONFIG
    ).eval()
    batch = Batch.from_data_list([make_condition("Ga4Te4", space_group, 118, 10)])
    results = []
    for seed in (11, 37):
        torch.manual_seed(seed)
        batches = collect_flow_logits(
            [batch], model, flow_steps=flow_steps, num_samples=2, sampling_mode="greedy"
        )
        rng_state = torch.random.get_rng_state()
        decoded = decode_logit_batches(
            batches,
            model.max_num_atoms,
            num_samples=2,
            sampling_mode="greedy",
            enforce_composition=enforce_composition,
            cpu_workers=1,
        )
        assert torch.equal(rng_state, torch.random.get_rng_state())
        assert len(decoded.samples) == 2
        results.extend(decoded.samples)

    for sample in results:
        torch.testing.assert_close(sample.x, results[0].x)
        assert int(sample.target_index) == 10
        assert "candidate_rank" not in sample
    if enforce_composition:
        sampled = Batch.from_data_list(results)
        torch.testing.assert_close(
            decode_composition(sampled, 118).round(), sampled.composition
        )

    logits_path = tmp_path / "greedy.logits.pt"
    output_path = tmp_path / "greedy.pt"
    torch.save(
        {
            "logit_batches": batches,
            "args": {
                "sampling_mode": "greedy",
                "num_samples": 2,
                "enforce_composition": enforce_composition,
            },
        },
        logits_path,
    )
    cached, _ = decode_logits_file(logits_path, output_path, cpu_workers=1)
    assert len(cached.samples) == 2
    for sample in cached.samples:
        torch.testing.assert_close(sample.x, results[0].x)
    settings = torch.load(output_path, weights_only=False)["args"]
    assert settings["sampling_mode"] == "greedy"
    assert settings["num_samples"] == 2


def test_greedy_cli_saves_and_reuses_trajectories(tmp_path, monkeypatch):
    model = DiscreteFlowModule(
        optimizer_config=OPTIMIZER_CONFIG, decoder=DECODER_CONFIG, **MODEL_CONFIG
    ).eval()
    monkeypatch.setattr(sample_wy, "load_model", lambda *a, **kw: (model, None, None))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    parser = sample_wy.build_parser()
    sample_wy.main(
        parser.parse_args(
            [
                "--model_path",
                "unused",
                "--formula",
                "Ga4Te4",
                "--space_group",
                "194",
                "--sampling-mode",
                "greedy",
                "--num-samples",
                "2",
                "--flow_steps",
                "3",
                "--cpu-workers",
                "1",
                "--save_path",
                str(tmp_path / "samples"),
            ]
        )
    )

    def fail_load(*args, **kwargs):
        raise AssertionError("cached greedy sampling must not load the model")

    monkeypatch.setattr(sample_wy, "load_model", fail_load)
    sample_wy.main(
        parser.parse_args(
            [
                "--reuse-logits",
                "--logits_path",
                str(tmp_path / "samples.logits.pt"),
                "--save_path",
                str(tmp_path / "reused"),
                "--cpu-workers",
                "1",
            ]
        )
    )
    fresh = torch.load(tmp_path / "samples.pt", weights_only=False)
    reused = torch.load(tmp_path / "reused.pt", weights_only=False)
    assert fresh["args"]["sampling_mode"] == reused["args"]["sampling_mode"] == "greedy"
    assert fresh["args"]["num_samples"] == reused["args"]["num_samples"] == 2
    assert len(fresh["generated_samples"]) == len(reused["generated_samples"]) == 2
    for original, cached in zip(
        fresh["generated_samples"], reused["generated_samples"]
    ):
        torch.testing.assert_close(original.x, cached.x)
