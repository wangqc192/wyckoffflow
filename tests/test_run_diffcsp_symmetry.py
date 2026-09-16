import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_diffcsp_symmetry.py"
SPEC = importlib.util.spec_from_file_location("run_diffcsp_symmetry", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_validate_query_preserves_repeated_orbits():
    assert MODULE.validate_query(
        {
            "spacegroup_number": 194,
            "wyckoff_letters": ["4f", "4f"],
            "atom_types": ["Ga", "Te"],
        }
    ) == (194, ["4f", "4f"], ["Ga", "Te"])


def test_validate_query_rejects_mismatched_orbit_and_element_lists():
    with pytest.raises(ValueError, match="equal nonzero length"):
        MODULE.validate_query(
            {
                "spacegroup_number": 194,
                "wyckoff_letters": ["4f"],
                "atom_types": ["Ga", "Te"],
            }
        )
