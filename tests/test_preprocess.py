import importlib
import math

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from pymatgen.core import Lattice, Structure
from torch_geometric.loader import DataLoader

from models.common.utils import PROJECT_ROOT
from models.pl_data.dataset import CrystalDataset
from models.pl_data.preprocess import (
    _composition_from_wyckoff_matrix,
    preprocess,
    preprocess_dataframe,
)


@pytest.fixture
def cif_frame():
    # CIFs contain P1 coordinates, as in MatterGen; spglib must rediscover SG 225.
    conventional = Structure.from_spacegroup(
        225, Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    primitive = conventional.get_primitive_structure()
    return pd.DataFrame(
        {
            "material_id": ["salt-conventional", "salt-primitive"],
            "cif": [structure.to(fmt="cif") for structure in (conventional, primitive)],
            "space_group": ["Fm-3m", "Fm-3m"],
            "energy_above_hull": [0.0, 0.0],
        }
    )


def test_cif_templates_preserve_equivalents_and_conventional_composition(cif_frame):
    parsed = preprocess_dataframe(cif_frame)
    assert parsed.space_group.tolist() == ["225", "225"]
    assert parsed.aflow_label.nunique() == 1
    assert parsed.e_form_per_atom.isna().all()
    assert parsed.material_id.tolist() == cif_frame.material_id.tolist()

    expected = torch.zeros(101)
    expected[11] = expected[17] = 4
    for row in parsed.itertuples():
        assert len(row.wyckoff_set) > 1
        for matrix in row.wyckoff_element_matrix:
            counts = _composition_from_wyckoff_matrix(
                matrix[None],
                np.array(row.degrees_of_freedom) == 0,
                row.multiplicities,
                100,
            )
            torch.testing.assert_close(counts, expected)


@pytest.mark.parametrize("input_type", ["frame", "records", "record", "series"])
def test_cif_dataset_input_forms_and_batching(cif_frame, input_type):
    data = {
        "frame": cif_frame,
        "records": cif_frame.to_dict("records"),
        "record": cif_frame.iloc[0].to_dict(),
        "series": cif_frame.iloc[0],
    }[input_type]
    dataset = CrystalDataset(data, num_elements=100)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch.num_graphs == len(dataset)
    assert batch.composition.shape == (len(dataset), 101)
    assert batch.composition.sum(dim=1).tolist() == [8] * len(dataset)
    assert batch.material_id == cif_frame.material_id.tolist()[: len(dataset)]
    assert torch.isnan(batch.e_form_per_atom).all()
    assert len(set(dataset.prototype_keys)) == 1


def test_existing_mp20_labels_and_formation_energies_are_preserved():
    raw = pd.read_csv(PROJECT_ROOT / "data/mini/train.csv", nrows=8)
    # Existing labels take priority even if CIF parsing would fail.
    raw["cif"] = "unused"
    parsed = preprocess_dataframe(raw)
    assert parsed.aflow_label.tolist() == raw.wyckoff_spglib.tolist()
    np.testing.assert_allclose(parsed.e_form_per_atom, raw.formation_energy_per_atom)
    assert len(CrystalDataset(parsed)) == len(raw)


def test_missing_energy_does_not_require_a_cif():
    dataset = CrystalDataset({"wyckoff_spglib": "AB_cF8_225_a_b:Cl-Na"})
    assert dataset[0].space_group.item() == 225
    assert math.isnan(dataset[0].e_form_per_atom.item())


def test_parallel_cache_matches_serial_and_reuses_cache(
    cif_frame, tmp_path, monkeypatch
):
    source = tmp_path / "input.csv"
    output = tmp_path / "cache" / "input.pt"
    cif_frame.to_csv(source, index=False)
    expected = preprocess_dataframe(cif_frame)
    actual = preprocess(source, output, num_workers=2, chunk_size=1)
    assert actual.aflow_label.tolist() == expected.aflow_label.tolist()
    assert actual.material_id.tolist() == expected.material_id.tolist()
    for left, right in zip(
        actual.wyckoff_element_matrix, expected.wyckoff_element_matrix
    ):
        np.testing.assert_array_equal(left, right)
    assert not source.with_suffix(".pt").exists()

    module = importlib.import_module("models.pl_data.preprocess")

    def unexpected_rebuild(*args, **kwargs):
        pytest.fail("An unchanged source should reuse its cache")

    monkeypatch.setattr(module, "preprocess_dataframe", unexpected_rebuild)
    reused = preprocess(source, output)
    assert reused.aflow_label.tolist() == actual.aflow_label.tolist()


def test_cache_tracks_source_and_symmetry_settings(cif_frame, tmp_path):
    output = tmp_path / "cache.pt"
    first_source = tmp_path / "first.csv"
    second_source = tmp_path / "second.csv"
    cif_frame.iloc[:1].to_csv(first_source, index=False)
    changed = cif_frame.iloc[:1].copy()
    changed["material_id"] = "different-source"
    changed.to_csv(second_source, index=False)
    preprocess(first_source, output)
    actual = preprocess(second_source, output)
    assert actual.material_id.tolist() == ["different-source"]

    # A small displacement is cubic at 0.1 Å, but not at 0.001 Å.
    displaced = Structure.from_str(cif_frame.cif.iloc[0], fmt="cif")
    displaced.translate_sites([0], [0.002, 0.003, 0.004])
    changed["cif"] = displaced.to(fmt="cif")
    changed.to_csv(second_source, index=False)
    assert preprocess(second_source, output, symprec=0.1).space_group.tolist() == [
        "225"
    ]
    assert preprocess(second_source, output, symprec=0.001).space_group.tolist() != [
        "225"
    ]


def test_alex_config_trains_without_test_csv(cif_frame, tmp_path):
    for split in ("train", "val"):
        cif_frame.to_csv(tmp_path / f"{split}.csv", index=False)
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "conf"), version_base="1.3"
    ):
        config = compose(
            config_name="default",
            overrides=[
                "data=alex_mp_20",
                f"data.root_path={tmp_path}",
                f"data.cache_path={tmp_path}/cache",
                "data.preprocess.num_workers=0",
                "model.decoder.num_gnn_layers=1",
                "model.decoder.hidden_dim=8",
                "model.decoder.dof_pos_sg_emb_size=4",
                "model.decoder.composition_encoder_dim=4",
            ],
        )
    datamodule = hydra.utils.instantiate(config.data.datamodule, _recursive_=False)
    datamodule.prepare_data()
    datamodule.setup("fit")
    assert len(datamodule.train_dataset) == len(cif_frame)
    assert len(datamodule.val_datasets[0]) == len(cif_frame)
    assert datamodule.test_datasets == []
    assert (tmp_path / "cache/train.pt").exists()
    assert (tmp_path / "cache/val.pt").exists()
    model = hydra.utils.instantiate(
        config.model, optimizer_config=config.optim, _recursive_=False
    )
    loss = model(next(iter(datamodule.train_dataloader())))["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_invalid_cif_reports_material_id():
    with pytest.raises(ValueError, match="bad-material"):
        preprocess_dataframe(
            pd.DataFrame([{"material_id": "bad-material", "cif": "bad"}])
        )
