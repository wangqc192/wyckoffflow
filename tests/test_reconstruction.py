import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.trainer.states import TrainerFn
from torch_geometric.loader import DataLoader

import models.common.reconstruction as reconstruction
from models.common.lookup_tables import chemical_symbols
from models.common.utils import PROJECT_ROOT
from models.common.wyckoff_template import WyckoffTemplate
from models.pl_models.count_conserving import DecodingResult
from models.pl_models.flow import DiscreteFlowModule
from models.pl_models.model_utils import create_wyckoff_graph, get_degrees_of_freedom
from models.run import build_callbacks
from scripts.eval_gwa import evaluate as evaluate_gwa


def compose_config(*overrides):
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "conf"), version_base="1.3"
    ):
        return compose(config_name="default", overrides=list(overrides))


def template_graph(occupancy, label):
    template = WyckoffTemplate.from_crystalflow(occupancy)
    degrees = get_degrees_of_freedom(template.space_group)
    zero = degrees == 0
    x = torch.zeros(len(degrees), 31, dtype=torch.long)
    for orbit, element in zip(template.wyckoff_letters, template.atom_types):
        position = ord(orbit[-1]) - ord("a")
        number = chemical_symbols.index(element)
        if zero[position]:
            x[position, 0] = number
        else:
            x[position, number] += 1
    graph = create_wyckoff_graph(template.space_group, x[zero, 0], x[~zero, 1:])
    graph.composition = torch.zeros(1, 31)
    for element, count in template.formula_counts.items():
        graph.composition[0, chemical_symbols.index(element)] = count
    graph.aflow_label = label
    return graph


def small_model():
    return DiscreteFlowModule(
        optimizer_config={"_target_": "torch.optim.AdamW", "lr": 1e-3},
        num_elements=30,
        max_num_atoms=4,
        flow_source="zeros",
        decoder={
            "_target_": "models.pl_models.gnn.WyckoffGNN",
            "hidden_dim": 8,
            "dof_pos_sg_emb_size": 4,
            "num_gnn_layers": 1,
            "gnn_activation": "SiLU",
            "mlp_hidden_layers": 2,
            "mlp_activation": "SiLU",
            "composition_encoder_dim": 4,
        },
    )


def test_reconstruction_accepts_equivalence_and_keeps_missing_targets(
    monkeypatch, tmp_path
):
    label = "AB_oC4_65_a_c:Cu-Ni"
    dataset = [template_graph("65_Cu1x2a_Ni1x2c", label) for _ in range(2)]
    calls = []
    decoding_inputs = []

    def sample_logits(batch, flow_steps):
        calls.append((batch.num_graphs, flow_steps))
        return batch, None, None

    def decode(data, *args, **kwargs):
        decoding_inputs.append((data, *args[:2]))
        # First sample misses; second matches through an equivalent setting.
        samples = [
            template_graph("65_Cu1x2a_Ni1x2b", label),
            template_graph("65_Ni1x2a_Cu1x2c", label),
        ]
        for sample in samples:
            sample.target_index = torch.tensor(0)
        return DecodingResult(samples, [2, 3])

    def decode_independent(data, zero_logits, inf_logits):
        decoding_inputs.append((data, zero_logits, inf_logits))
        empty = dataset[0].clone()
        empty.x.zero_()
        samples = [
            empty,
            template_graph("65_Cu1x2a_Ni1x4g", label),
            template_graph("65_Ni1x2a_Cu1x2c", label),
            template_graph("65_Cu1x2a_Ni1x2b", label),
        ]
        for index, sample in enumerate(samples):
            sample.target_index = torch.tensor(index // 2)
            sample.composition = dataset[0].composition.clone()
        return DecodingResult(samples, [])

    monkeypatch.setattr(reconstruction, "decode_composition_logits", decode)
    monkeypatch.setattr(reconstruction, "decode_independent_logits", decode_independent)
    model = SimpleNamespace(sample_logits=sample_logits, max_num_atoms=4)
    summary = reconstruction.evaluate_reconstruction(
        model,
        dataset,
        tmp_path,
        num_samples=2,
        flow_steps=3,
        batch_size=4,
        cpu_workers=1,
    )

    assert calls == [(4, 3)]
    assert all(a is b for a, b in zip(*decoding_inputs))
    assert summary["total_materials"] == 2
    assert summary["match_rate"] == 0.5
    assert summary["match_rate_top1"] == 0
    assert summary["materials_with_generated_samples"] == 1
    details = pd.read_csv(tmp_path / "details.csv")
    assert details.generated_count.tolist() == [2, 0]
    assert details.matched.tolist() == [True, False]
    samples = pd.read_csv(tmp_path / "samples.csv")
    assert samples.candidate_rank.tolist() == [1, 2]
    unconstrained = summary["no_composition"]
    assert unconstrained["enforce_composition"] is False
    assert unconstrained["match_rate"] == 0.5
    assert unconstrained["match_rate_top1"] == 0.5
    assert unconstrained["composition_accuracy"] == 0.5
    raw_samples = pd.read_csv(tmp_path / "no_composition/samples.csv")
    assert raw_samples.wyckoff_occupancy.iloc[0] == "65"
    assert raw_samples.composition_correct.tolist() == [False, False, True, True]
    assert raw_samples.target_index.tolist() == [0, 0, 1, 1]
    raw_details = pd.read_csv(tmp_path / "no_composition/details.csv")
    assert raw_details.generated_count.tolist() == [2, 2]
    assert raw_details.matched.tolist() == [False, True]
    _, hits = evaluate_gwa(
        pd.DataFrame({"wyckoff_spglib": [label] * 2}), raw_samples, 2
    )
    assert hits == unconstrained["matched_materials"]


def test_actual_sampling_is_reproducible_and_restores_rng(tmp_path):
    graph = template_graph("1_Li1x1a", "A_aP1_1_a:Li")
    model = small_model().eval()
    logged = {}
    model.log = lambda name, value, **kwargs: logged.update({name: value})
    trainer = SimpleNamespace(
        sanity_checking=False,
        state=SimpleNamespace(fn=TrainerFn.FITTING),
        current_epoch=99,
        global_step=100,
        is_global_zero=True,
        datamodule=SimpleNamespace(val_datasets=[[graph, graph.clone()]]),
        strategy=SimpleNamespace(broadcast=lambda value, src: value),
    )
    callback = reconstruction.ValidationReconstruction(
        tmp_path, num_samples=2, flow_steps=2, batch_size=2, cpu_workers=1, seed=7
    )
    state = torch.get_rng_state().clone()
    callback.on_validation_epoch_end(trainer, model)
    assert torch.equal(state, torch.get_rng_state())
    first = (tmp_path / "epoch_0099" / "samples.csv").read_text()
    raw_first = (tmp_path / "epoch_0099/no_composition/samples.csv").read_text()
    trainer.current_epoch = 199
    callback.on_validation_epoch_end(trainer, model)
    assert (tmp_path / "epoch_0199" / "samples.csv").read_text() == first
    assert (tmp_path / "epoch_0199/no_composition/samples.csv").read_text() == raw_first
    assert torch.equal(state, torch.get_rng_state())
    assert logged["val/gwa_top1"] == logged["val/gwa_top2"] == 1.0
    assert logged["val/composition_accuracy"] == 1.0
    raw = json.loads((tmp_path / "epoch_0199/no_composition/summary.json").read_text())
    assert logged["val/gwa_top1_no_composition"] == raw["match_rate_top1"]
    assert logged["val/gwa_top2_no_composition"] == raw["match_rate"]
    assert (
        logged["val/composition_accuracy_no_composition"] == raw["composition_accuracy"]
    )
    assert raw["generated_samples"] == 4
    assert len(pd.read_csv(tmp_path / "epoch_0099" / "samples.csv")) == 4


@pytest.mark.parametrize(
    "epoch,sanity,fitting", [(98, False, True), (99, True, True), (99, False, False)]
)
def test_reconstruction_runs_only_at_scheduled_training_validation(
    tmp_path, epoch, sanity, fitting
):
    callback = reconstruction.ValidationReconstruction(tmp_path / "reconstruction")
    trainer = SimpleNamespace(
        current_epoch=epoch,
        sanity_checking=sanity,
        state=SimpleNamespace(
            fn=TrainerFn.FITTING if fitting else TrainerFn.VALIDATING
        ),
    )
    callback.on_validation_epoch_end(trainer, None)
    assert not (tmp_path / "reconstruction").exists()


def test_nonzero_rank_logs_broadcast_metrics_without_sampling(tmp_path):
    logged = {}
    trainer = SimpleNamespace(
        current_epoch=99,
        sanity_checking=False,
        state=SimpleNamespace(fn=TrainerFn.FITTING),
        is_global_zero=False,
        strategy=SimpleNamespace(
            broadcast=lambda value, src: {
                "match_rate": 0.8,
                "match_rate_top1": 0.6,
                "composition_accuracy": 1.0,
                "no_composition": {
                    "match_rate": 0.5,
                    "match_rate_top1": 0.3,
                    "composition_accuracy": 0.7,
                },
            }
        ),
    )
    model = SimpleNamespace(
        log=lambda name, value, **kwargs: logged.update({name: value})
    )
    callback = reconstruction.ValidationReconstruction(tmp_path / "reconstruction")
    callback.on_validation_epoch_end(trainer, model)
    assert logged == {
        "val/gwa_top1": 0.6,
        "val/gwa_top20": 0.8,
        "val/composition_accuracy": 1.0,
        "val/gwa_top1_no_composition": 0.3,
        "val/gwa_top20_no_composition": 0.5,
        "val/composition_accuracy_no_composition": 0.7,
    }
    assert not (tmp_path / "reconstruction").exists()


class SmallDataModule(pl.LightningDataModule):
    def setup(self, stage):
        self.train_dataset = [template_graph("1_Li1x1a", "A_aP1_1_a:Li")]
        self.val_datasets = [self.train_dataset]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return DataLoader(self.val_datasets[0], batch_size=1)


def test_training_saves_periodic_metrics_and_best_gwa_and_resumes(
    monkeypatch, tmp_path
):
    def fake_evaluate(model, dataset, directory, **kwargs):
        (directory / "no_composition").mkdir(parents=True, exist_ok=True)
        if model.current_epoch == 5:
            best_gwa = next(
                callback
                for callback in model.trainer.callbacks
                if isinstance(callback, ModelCheckpoint)
                and callback.monitor == "val/gwa_top20"
            )
            assert best_gwa.best_model_score.item() == pytest.approx(0.8)
            best_raw = next(
                callback
                for callback in model.trainer.callbacks
                if isinstance(callback, ModelCheckpoint)
                and callback.monitor == "val/gwa_top20_no_composition"
            )
            assert best_raw.best_model_score.item() == pytest.approx(0.5)
        # A worse later reconstruction must not replace the best checkpoint.
        rate = {1: 0.8, 3: 0.6, 5: 0.9}[model.current_epoch]
        raw_rate = {1: 0.3, 3: 0.5, 5: 0.4}[model.current_epoch]
        return {
            "match_rate": rate,
            "match_rate_top1": rate / 2,
            "composition_accuracy": 1.0,
            "no_composition": {
                "match_rate": raw_rate,
                "match_rate_top1": raw_rate / 2,
                "composition_accuracy": 0.7,
            },
        }

    monkeypatch.setattr(reconstruction, "evaluate_reconstruction", fake_evaluate)
    monkeypatch.setattr(
        "models.run.HydraConfig.get",
        lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path))),
    )
    config = compose_config(
        "train.reconstruction.every_n_epochs=2",
        "train.checkpoint.every_n_epochs=2",
    )
    callbacks = build_callbacks(config)
    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=4,
        callbacks=callbacks,
        logger=CSVLogger(tmp_path, name="logs"),
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=1,
    )
    trainer.fit(small_model(), datamodule=SmallDataModule())

    directory = tmp_path / "checkpoints"
    best = torch.load(directory / "best_gwa_epoch_0001.ckpt", weights_only=False)
    assert best["epoch"] == 1
    best_raw = torch.load(
        directory / "best_gwa_no_composition_epoch_0003.ckpt", weights_only=False
    )
    assert best_raw["epoch"] == 3
    assert (directory / "epoch_0001.ckpt").exists()
    assert (directory / "epoch_0003.ckpt").exists()
    best_loss_paths = list(directory.glob("best_epoch_*.ckpt"))
    assert len(best_loss_paths) == 1
    best_loss = torch.load(best_loss_paths[0], weights_only=False)
    assert best_loss_paths[0].name == f"best_epoch_{best_loss['epoch']:04d}.ckpt"
    assert [p.name for p in directory.glob("best_gwa_epoch_*.ckpt")] == [
        "best_gwa_epoch_0001.ckpt"
    ]
    assert [p.name for p in directory.glob("best_gwa_no_composition_epoch_*.ckpt")] == [
        "best_gwa_no_composition_epoch_0003.ckpt"
    ]
    assert sorted(p.name for p in (tmp_path / "reconstruction").iterdir()) == [
        "epoch_0001",
        "epoch_0003",
    ]
    metrics = pd.read_csv(Path(trainer.logger.log_dir) / "metrics.csv")
    recorded = metrics.dropna(subset=["val/gwa_top20"])
    assert recorded.epoch.tolist() == [1, 3]
    assert np.allclose(recorded["val/gwa_top20"], [0.8, 0.6])
    assert np.allclose(recorded["val/gwa_top20_no_composition"], [0.3, 0.5])

    config.resume_from = str(directory / "last.ckpt")
    resumed_callbacks = build_callbacks(config)
    resumed = pl.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=6,
        callbacks=resumed_callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    resumed.fit(
        small_model(), datamodule=SmallDataModule(), ckpt_path=config.resume_from
    )
    best = torch.load(directory / "best_gwa_epoch_0005.ckpt", weights_only=False)
    assert best["epoch"] == 5
    best_raw = torch.load(
        directory / "best_gwa_no_composition_epoch_0003.ckpt", weights_only=False
    )
    assert best_raw["epoch"] == 3
    assert [p.name for p in directory.glob("best_gwa_epoch_*.ckpt")] == [
        "best_gwa_epoch_0005.ckpt"
    ]
    assert [p.name for p in directory.glob("best_gwa_no_composition_epoch_*.ckpt")] == [
        "best_gwa_no_composition_epoch_0003.ckpt"
    ]
    summary = json.loads(
        (tmp_path / "reconstruction/epoch_0005/summary.json").read_text()
    )
    assert summary["completed_epochs"] == 6
    assert summary["match_rate"] == 0.9
    assert summary["no_composition"]["match_rate"] == 0.4
    raw_summary = json.loads(
        (tmp_path / "reconstruction/epoch_0005/no_composition/summary.json").read_text()
    )
    assert raw_summary == summary["no_composition"]


def test_space_group_training_does_not_install_reconstruction(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "models.run.HydraConfig.get",
        lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path))),
    )
    callbacks = build_callbacks(compose_config("experiment=space_group"))
    assert len(callbacks) == 2
    assert all(isinstance(callback, ModelCheckpoint) for callback in callbacks)
