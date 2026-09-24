import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

import scripts.sample_wy as sample_wy
from models.pl_models.count_conserving import DecodingResult
from models.sampling import collect_flow_logits, make_condition


def test_sample_records_preserve_target_identifiers():
    batch = Batch.from_data_list(
        [
            make_condition("Ga4Te4", 194, 118, target_index=0),
            make_condition("Ni2Cu2", 65, 118, target_index=1),
        ]
    )
    assert batch.target_index.tolist() == [0, 1]
    assert batch.space_group.tolist() == [194, 65]


@pytest.mark.parametrize(
    "sampling_mode, trajectories", [("n-shot", 3), ("top-n", 1), ("greedy", 3)]
)
def test_flow_trajectory_count_is_separate_from_output_count(
    sampling_mode, trajectories
):
    calls = []

    class FakeModel:
        def sample_logits(self, batch, flow_steps, *, greedy):
            calls.append((flow_steps, greedy))
            return batch, None, None

    records = [
        make_condition("Ga4Te4", 194, 118, target_index=10),
        make_condition("Ga4Te4", 225, 118, target_index=10),
    ]
    logits = collect_flow_logits(
        DataLoader(records, batch_size=1),
        FakeModel(),
        flow_steps=50,
        num_samples=3,
        sampling_mode=sampling_mode,
    )
    assert calls == [(50, sampling_mode == "greedy")] * 2
    assert [data.num_graphs for data, _, _ in logits] == [trajectories, trajectories]
    assert logits[0][0].target_index.tolist() == [10] * trajectories
    assert logits[1][0].target_index.tolist() == [10] * trajectories
    assert logits[0][0].sampling_group.tolist() == [0] * trajectories
    assert logits[1][0].sampling_group.tolist() == [1] * trajectories


@pytest.mark.parametrize("requested", [None, 7])
@pytest.mark.parametrize("saved_count_key", ["num_samples", "num_evals"])
def test_reuse_top_n_keeps_or_overrides_count_and_writes_effective_settings(
    tmp_path, monkeypatch, requested, saved_count_key
):
    logits_path = tmp_path / "saved.logits.pt"
    torch.save(
        {
            "logit_batches": [],
            "gpu_flow_time": 1.5,
            "args": {
                "sampling_mode": "top-n",
                saved_count_key: 20,
                "fixed_site_beam_size": 512,
            },
        },
        logits_path,
    )
    calls = []

    def fake_decode(batches, max_variable_count, **options):
        calls.append(options)
        return DecodingResult(
            [Data(target_index=torch.tensor(0)) for _ in range(options["num_samples"])],
            [],
        )

    def fail_load(*args, **kwargs):
        raise AssertionError("logits reuse must not load a model")

    monkeypatch.setattr(sample_wy, "decode_logit_batches", fake_decode)
    monkeypatch.setattr(sample_wy, "load_model", fail_load)
    command = [
        "--save_path",
        str(tmp_path / "result"),
        "--logits_path",
        str(logits_path),
        "--reuse-logits",
    ]
    if requested is not None:
        command.extend(
            ["--num-samples", str(requested), "--fixed-site-beam-size", "1024"]
        )
    args = sample_wy.build_parser().parse_args(command)
    sample_wy.main(args)

    count = 20 if requested is None else requested
    beam = 512 if requested is None else 1024
    assert calls[0]["num_samples"] == count
    assert calls[0]["fixed_site_beam_size"] == beam
    output = torch.load(tmp_path / "result.pt", weights_only=False)
    assert len(output["generated_samples"]) == count
    assert output["args"]["num_samples"] == count
    assert output["args"]["fixed_site_beam_size"] == beam
    assert "num_evals" not in output["args"]
    assert output["gpu_flow_time"] == 1.5
    assert output["reused_logits_path"] == str(logits_path)


@pytest.mark.parametrize("sampling_mode", ["n-shot", "greedy"])
def test_reuse_cannot_change_stored_trajectory_count(tmp_path, sampling_mode):
    path = tmp_path / "logits.pt"
    torch.save(
        {
            "logit_batches": [],
            "args": {"sampling_mode": sampling_mode, "num_samples": 20},
        },
        path,
    )
    with pytest.raises(ValueError, match="new flow trajectories"):
        sample_wy.decode_logits_file(path, tmp_path / "output.pt", num_samples=7)


@pytest.mark.parametrize("saved_mode", ["n-shot", "top-n"])
def test_greedy_cannot_reuse_random_flow_trajectories(tmp_path, saved_mode):
    path = tmp_path / "logits.pt"
    torch.save({"logit_batches": [], "args": {"sampling_mode": saved_mode}}, path)
    with pytest.raises(ValueError, match="changing sampling_mode"):
        sample_wy.decode_logits_file(
            path, tmp_path / "output.pt", sampling_mode="greedy"
        )


def test_existing_cli_aliases_map_to_descriptive_names():
    args = sample_wy.build_parser().parse_args(
        [
            "--save_path",
            "samples",
            "--num_evals",
            "20",
            "--topn-beam-size",
            "256",
            "--no-count_conserving",
        ]
    )
    assert args.num_samples == 20
    assert args.fixed_site_beam_size == 256
    assert args.enforce_composition is False
