import hydra
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir

from models.common.composition import formula_to_counts
from models.common.utils import PROJECT_ROOT
from models.pl_models.flow import DiscreteFlowModule
from models.pl_models.time_embedding import DiffCSPTimeEncoder, FlowTimeEncoder


def make_model(encoding, *overrides):
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "conf"), version_base="1.3"
    ):
        config = compose(
            config_name="default",
            overrides=[
                "model/decoder=crystal_gnn",
                f"model/time={encoding}",
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


def test_diffcsp_matches_known_frequencies_and_distinguishes_endpoints():
    encoder = DiffCSPTimeEncoder(16)
    # Reference channels: highest, middle and lowest frequencies of DiffCSP PE.
    torch.testing.assert_close(
        encoder.frequencies[[0, 16, 32]], torch.tensor([1.0, 0.01, 0.0001])
    )
    captured = []
    hook = encoder.projection.register_forward_pre_hook(
        lambda module, args: captured.append(args[0])
    )
    encoder(torch.tensor([0.0, 0.5, 1.0]))
    hook.remove()
    features = captured[0]
    torch.testing.assert_close(features[0, :33], torch.zeros(33))
    torch.testing.assert_close(features[0, 33:], torch.ones(33))
    torch.testing.assert_close(
        features[:, 0], torch.tensor([0.0, 0.47942554, 0.84147098])
    )
    torch.testing.assert_close(
        features[:, 33], torch.tensor([1.0, 0.87758256, 0.54030231])
    )


def test_ablation_has_identical_trainable_initialization_and_rng_state():
    torch.manual_seed(42)
    baseline = make_model("fourier")
    baseline_rng = torch.get_rng_state()
    torch.manual_seed(42)
    diffcsp = make_model("diffcsp")
    torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
    baseline_params = dict(baseline.named_parameters())
    diffcsp_params = dict(diffcsp.named_parameters())
    assert baseline_params.keys() == diffcsp_params.keys()
    for name in baseline_params:
        torch.testing.assert_close(
            baseline_params[name], diffcsp_params[name], rtol=0, atol=0
        )
    assert isinstance(baseline.decoder.time_encoder, FlowTimeEncoder)
    assert isinstance(diffcsp.decoder.time_encoder, DiffCSPTimeEncoder)


def test_time_config_controls_frequencies_projection_and_phase():
    model = make_model(
        "diffcsp",
        "model.decoder.time.num_frequencies=4",
        "model.decoder.time.max_period=1000.0",
        "model.decoder.time.time_scale=2.0",
    )
    encoder = model.decoder.time_encoder
    reference = DiffCSPTimeEncoder(32, num_frequencies=4, max_period=1000.0)
    reference.load_state_dict(encoder.state_dict())
    torch.testing.assert_close(
        encoder.frequencies, torch.tensor([1.0, 0.1, 0.01, 0.001])
    )
    assert encoder.projection[0].in_features == 8
    t = torch.tensor([0.0, 0.25, 0.5])
    torch.testing.assert_close(encoder(t), reference(2 * t), rtol=0, atol=0)

    fourier = make_model(
        "fourier",
        "model.decoder.time.num_frequencies=4",
        "model.decoder.time.min_frequency=2.0",
        "model.decoder.time.max_frequency=16.0",
        "model.decoder.time.include_raw_time=false",
    ).decoder.time_encoder
    torch.testing.assert_close(
        fourier.frequencies, torch.pi * torch.tensor([2.0, 4.0, 8.0, 16.0])
    )
    assert fourier.projection[0].in_features == 8


def test_diffcsp_backward_and_checkpoint_roundtrip(tmp_path):
    model = make_model(
        "diffcsp",
        "model.decoder.time.num_frequencies=4",
        "model.decoder.time.max_period=1000.0",
        "model.decoder.time.time_scale=2.0",
    )
    batch = model._build_source_from_compositions(
        formula_to_counts("Li4O4", 10)[None], fixed_space_group=2
    )
    loss = model(batch)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in model.decoder.time_encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    checkpoint = tmp_path / "diffcsp_time.ckpt"
    torch.save(
        {
            "pytorch-lightning_version": pl.__version__,
            "hyper_parameters": dict(model.hparams),
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    restored = DiscreteFlowModule.load_from_checkpoint(
        checkpoint, weights_only=False
    ).eval()
    assert isinstance(restored.decoder.time_encoder, DiffCSPTimeEncoder)
    model.eval()
    with torch.no_grad():
        expected = model.decoder(batch, torch.tensor([0.35]))
        actual = restored.decoder(batch, torch.tensor([0.35]))
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
