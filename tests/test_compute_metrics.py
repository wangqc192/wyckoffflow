import json
from argparse import Namespace

import pandas as pd
import torch
from pymatgen.core import Lattice, Structure

from scripts.compute_metrics import compute_metrics, main, structure_is_valid
from scripts.compute_metrics_crystalflow import load_crystalflow_records
from scripts.compute_metrics_diffcsppp import load_diffcsppp_records


def write_cif(path, structure):
    structure.to(filename=path)


def test_structure_is_valid_uses_distance_cutoff():
    valid = Structure(Lattice.cubic(4), ["Na", "Cl"], [[0, 0, 0], [0.5] * 3])
    invalid = Structure(Lattice.cubic(4), ["Na", "Cl"], [[0, 0, 0], [0.01] * 3])

    assert structure_is_valid(valid)
    assert not structure_is_valid(invalid)


def test_compute_metrics_uses_best_candidate_per_target(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    target0 = Structure(Lattice.cubic(4), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    target1 = Structure(Lattice.cubic(4), ["Li"], [[0, 0, 0]])
    wrong = Structure(Lattice.cubic(4), ["Na", "Cl"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    write_cif(samples / "0.cif", wrong)
    write_cif(samples / "1.cif", target0)
    (samples / "2.cif").write_text("not a cif", encoding="utf-8")
    manifest = pd.DataFrame(
        {
            "sample_index": [0, 1, 2],
            "target_index": [0, 0, 1],
        }
    )

    metrics, details = compute_metrics(manifest, samples, [target0, target1])

    assert metrics["match_rate"] == 0.5
    assert metrics["rms_dist"] == 0
    assert metrics["num_targets"] == 2
    assert metrics["num_matched"] == 1
    assert details.loc[0, "matched_candidate_index"] == 1
    assert details.loc[1, "num_errors"] == 1


def test_main_reads_notebook_layout_and_writes_outputs(tmp_path):
    root = tmp_path / "crystalflow"
    samples = root / "samples" / "cif"
    samples.mkdir(parents=True)
    target = Structure(Lattice.cubic(4), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    write_cif(samples / "0.cif", target)
    pd.DataFrame({"sample_index": [0], "target_index": [0]}).to_csv(
        root / "manifest.csv", index=False
    )
    gt_file = tmp_path / "targets.csv"
    pd.DataFrame({"cif": [target.to(fmt="cif")]}).to_csv(gt_file, index=False)

    main(
        Namespace(
            root_path=root,
            gt_file=gt_file,
            manifest=None,
            samples_dir=None,
            cif_column=None,
            output=None,
            details=None,
        )
    )

    metrics = json.loads((root / "eval_metrics.json").read_text())
    assert metrics["match_rate"] == 1
    assert metrics["rms_dist"] == 0
    assert (root / "eval_details.csv").is_file()


def test_compute_metrics_uses_input_index_for_multi_material_manifest(tmp_path):
    samples = tmp_path / "cif"
    samples.mkdir()
    target = Structure(Lattice.cubic(4), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    write_cif(samples / "10.cif", target)
    manifest = pd.DataFrame(
        {
            "input_index": [10],
            "material_index": [0],
            "candidate_index": [3],
        }
    )

    metrics, details = compute_metrics(manifest, samples, [target])

    assert metrics["match_rate"] == 1
    assert details.loc[0, "matched_input_index"] == 10
    assert details.loc[0, "matched_candidate_index"] == 3


def test_compute_metrics_counts_targets_without_candidates(tmp_path):
    samples = tmp_path / "cif"
    samples.mkdir()
    target = Structure(
        Lattice.cubic(4),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    write_cif(samples / "0.cif", target)
    write_cif(samples / "1.cif", target)
    manifest = pd.DataFrame(
        {
            "input_index": [0, 1],
            "target_index": [0, 2],
            "candidate_index": [0, 0],
        }
    )

    metrics, details = compute_metrics(manifest, samples, [target, target, target])

    assert metrics["num_targets"] == 3
    assert metrics["num_matched"] == 2
    assert metrics["match_rate"] == 2 / 3
    assert details.loc[1, "num_generated"] == 0
    assert not details.loc[1, "matched"]


def test_compute_metrics_passes_matcher_tolerances_to_workers(tmp_path):
    samples = tmp_path / "cif"
    samples.mkdir()
    target = Structure(
        Lattice.cubic(4),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    write_cif(samples / "0.cif", target)
    manifest = pd.DataFrame(
        {
            "input_index": [0],
            "target_index": [0],
            "candidate_index": [0],
        }
    )

    for workers in (1, 2):
        metrics, _ = compute_metrics(
            manifest,
            samples,
            [target],
            workers=workers,
            show_progress=False,
            ltol=0.11,
            stol=0.22,
            angle_tol=3.3,
        )

        assert metrics["matcher"] == {
            "ltol": 0.11,
            "stol": 0.22,
            "angle_tol": 3.3,
            "primitive_cell": True,
            "scale": True,
        }


def test_compute_metrics_supports_parallel_workers(tmp_path):
    samples = tmp_path / "cif"
    samples.mkdir()
    target = Structure(
        Lattice.cubic(4),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    write_cif(samples / "0.cif", target)
    write_cif(samples / "1.cif", target)
    manifest = pd.DataFrame(
        {
            "input_index": [0, 1],
            "target_index": [0, 1],
            "candidate_index": [0, 0],
        }
    )

    metrics, details = compute_metrics(manifest, samples, [target, target], workers=2)

    assert metrics["workers"] == 2
    assert metrics["match_rate"] == 1
    assert details["matched"].all()


def test_load_diffcsppp_pt_uses_explicit_indices_and_atom_counts(tmp_path):
    path = tmp_path / "samples_shard0.pt"
    torch.save(
        {
            "metadata": {},
            "input_indices": torch.tensor([7, 3]),
            "frac_coords": torch.tensor(
                [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]
            ),
            "atom_types": torch.tensor([11, 17, 3]),
            "lengths": torch.tensor([[4.0, 4.0, 4.0], [3.0, 3.0, 3.0]]),
            "angles": torch.tensor([[90.0, 90.0, 90.0], [90.0, 90.0, 90.0]]),
            "num_atoms": torch.tensor([2, 1]),
        },
        path,
    )

    records = load_diffcsppp_records(path, show_progress=False)

    assert sorted(records) == [3, 7]
    assert records[7]["atom_types"].tolist() == [11, 17]
    assert records[3]["atom_types"].tolist() == [3]


def test_load_crystalflow_pt_uses_manifest_order_without_indices(tmp_path):
    path = tmp_path / "samples.pt"
    torch.save(
        {
            "crystal_list": [
                {
                    "frac_coords": [[0.0, 0.0, 0.0]],
                    "atom_types": [11],
                    "lengths": [4.0, 4.0, 4.0],
                    "angles": [90.0, 90.0, 90.0],
                },
                {
                    "frac_coords": [[0.0, 0.0, 0.0]],
                    "atom_types": [17],
                    "lengths": [5.0, 5.0, 5.0],
                    "angles": [90.0, 90.0, 90.0],
                },
            ]
        },
        path,
    )

    records = load_crystalflow_records(
        path, sample_indices=[12, 4], show_progress=False
    )

    assert sorted(records) == [4, 12]
    assert records[12]["atom_types"].tolist() == [11]
    assert records[4]["atom_types"].tolist() == [17]


def test_nextcrystal_manifest_can_be_reconstructed_from_native_outputs(tmp_path):
    from scripts.compute_metrics_nextcrystal import build_manifest_from_outputs

    assignments = tmp_path / "postprocessed_assignments.csv"
    pd.DataFrame(
        {
            "cif_name": [0, 0, 1],
            "Formula pretty": ["NaCl", "NaCl", "Li"],
            "NAtoms": [2, 2, 1],
            "Spacegroup Number": [1, 2, 1],
            "Assignments": [
                '["Na:a, Cl:a", "Unable to find an assignment"]',
                '["Na:a, Cl:b"]',
                '["Li:a"]',
            ],
        }
    ).to_csv(assignments, index=False)
    query_file = tmp_path / "mp_test.json"
    query_file.write_text("[{}, {}, {}]", encoding="utf-8")

    manifest = build_manifest_from_outputs(assignments, query_file)

    assert manifest.to_dict("list") == {
        "query_index": [1, 2, 3],
        "input_number": [1, 1, 2],
        "source_input_id": [0, 0, 1],
        "candidate_rank": [1, 2, 1],
        "formula": ["NaCl", "NaCl", "Li"],
        "num_atoms": [2, 2, 1],
        "spacegroup_number": [1, 2, 1],
    }


def test_nextcrystal_manifest_and_cif_loader_use_shared_csp_metric(tmp_path):
    from scripts.compute_metrics_nextcrystal import (
        build_manifest,
        load_nextcrystal_records,
    )
    from scripts.compute_metrics_nextcrystal import (
        compute_metrics as compute_nextcrystal_metrics,
    )

    samples = tmp_path / "sample_structures"
    samples.mkdir()
    target0 = Structure(
        Lattice.cubic(4),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    target1 = Structure(Lattice.cubic(5), ["Li"], [[0, 0, 0]])
    write_cif(samples / "1.cif", target0)
    write_cif(samples / "2.cif", target1)
    raw_manifest = pd.DataFrame(
        {
            "query_index": [1, 2],
            "source_input_id": [0, 1],
            "candidate_rank": [1, 1],
        }
    )

    manifest = build_manifest(raw_manifest)
    records = load_nextcrystal_records(samples, raw_manifest, show_progress=False)
    assert manifest.to_dict("list") == {
        "target_index": [0, 1],
        "input_index": [1, 2],
        "candidate_index": [1, 1],
    }
    assert sorted(records) == [1, 2]

    metrics, details = compute_nextcrystal_metrics(
        raw_manifest,
        samples,
        [target0, target1],
        show_progress=False,
    )
    assert metrics["match_rate"] == 1
    assert details["matched"].all()
