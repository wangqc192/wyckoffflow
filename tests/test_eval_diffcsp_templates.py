import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

from scripts.eval_diffcsp_templates import (
    MATCHER_KWARGS,
    build_summary,
    evaluate_crystal,
)


def crystal_dict(structure):
    return {
        "lengths": np.asarray(structure.lattice.abc),
        "angles": np.asarray(structure.lattice.angles),
        "frac_coords": structure.frac_coords,
        "atom_types": np.asarray([site.specie.Z for site in structure]),
    }


def test_evaluate_crystal_matches_equivalent_structure():
    target = Structure(
        Lattice.cubic(4),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )

    result = evaluate_crystal(
        crystal_dict(target.copy()),
        target,
        StructureMatcher(**MATCHER_KWARGS),
    )

    assert result[:6] == (True, True, True, True, True, True)
    assert result[6] == 0


def test_build_summary_aggregates_candidate_hits_by_material():
    details = pd.DataFrame(
        {
            "target_index": [0, 0, 1],
            "constructed": [True, True, True],
            "geometry_valid": [True, True, False],
            "structure_valid": [True, True, False],
            "composition_valid": [True, True, True],
            "matcher_matched": [False, True, False],
            "matched": [False, True, False],
            "target_valid": [True, True, False],
        }
    )

    summary = build_summary(details, total_targets=3, top_k=1)

    assert summary["metric"] == "DiffCSP StructureMatcher Top-1"
    assert summary["top_k"] == 1
    assert summary["matched_candidates"] == 1
    assert summary["matched_materials"] == 1
    assert summary["valid_target_materials"] == 1
    assert summary["total_materials"] == 3
    assert summary["match_rate"] == 1 / 3
    assert summary["raw_matcher_match_rate"] == 1 / 3
