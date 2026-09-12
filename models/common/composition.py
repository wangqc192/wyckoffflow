"""Composition parsing and decoding for formula-conditioned generation.

A composition is represented as a length ``num_elements + 1`` integer count
vector where the index equals the atomic number (index 0 is unused, to align
with the embedding conventions elsewhere in the codebase) and the value is the
number of atoms in the conventional cell. Counts are kept exactly as supplied;
the formula is never reduced to the smallest integer ratio.
"""

import chemparse
import torch

from .lookup_tables import chemical_symbols


def formula_to_counts(formula, num_elements):
    """Parse a complete chemical formula into a count vector.

    ``Ti4O8`` -> a vector with ``counts[22] == 4`` and ``counts[8] == 8``;
    the ``4:8`` counts are not reduced.

    """
    counts = torch.zeros(num_elements + 1)
    for element, amount in chemparse.parse_formula(formula).items():
        counts[chemical_symbols.index(element)] = int(amount)
    return counts
