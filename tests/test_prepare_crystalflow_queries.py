from scripts.prepare_crystalflow_queries import wyckoff_to_query


def test_wyckoff_to_query_accepts_uppercase_wyckoff_letters():
    assert wyckoff_to_query("47_O1x8A_Sr1x1b") == {
        "spacegroup_number": 47,
        "wyckoff_letters": ["8A", "1b"],
        "atom_types": ["O", "Sr"],
    }
