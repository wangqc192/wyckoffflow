"""Wyckoff graph and symmetry-template conversion helpers."""

from __future__ import annotations

from typing import Any

import torch

from .lookup_tables import chemical_symbols, wyckoff_label_to_index
from .wyckoff_template import WyckoffTemplate


def query_from_gwa_sequence(sequence: str) -> dict[str, Any]:
    """Convert a complete ``G-W-A-W-A-…`` sequence to a symmetry query."""

    return WyckoffTemplate.from_gwa(sequence).to_diffcsppp_query()


def wyckoff_multiplicity(label: str) -> int:
    """Return the multiplicity from a label such as ``4f``."""

    return WyckoffTemplate.wyckoff_multiplicity(label)


def decode_wyckoff_elements(matrix: torch.Tensor) -> tuple[list[str], list[str]]:
    """Decode one model ``x`` matrix into repeated elements and Wyckoff labels.

    A zero-degree-of-freedom site stores its atomic number in column 0; a
    variable site stores occupation counts in the element columns. Repeating
    labels here is intentional: symmetry templates use one orbit token for each
    occupied Wyckoff site, including repeated occupations.
    """

    if matrix.ndim != 2 or matrix.shape[1] != len(chemical_symbols):
        raise ValueError("Wyckoff matrix must have shape (num_sites, num_elements + 1)")
    index_to_label = {index: label for label, index in wyckoff_label_to_index.items()}
    elements: list[str] = []
    labels: list[str] = []
    positions, element_indices = torch.nonzero(matrix, as_tuple=True)
    if element_indices.numel() == 0:
        raise ValueError("No elements found in Wyckoff matrix")

    for position, element_index in zip(positions.tolist(), element_indices.tolist()):
        try:
            label = index_to_label[position + 1]
        except KeyError as error:
            raise ValueError(
                f"unknown Wyckoff position index: {position + 1}"
            ) from error
        if element_index == 0:
            atomic_number = int(matrix[position, 0].item())
            if not 1 <= atomic_number < len(chemical_symbols):
                raise ValueError(
                    f"invalid atomic number in zero-DoF site: {atomic_number}"
                )
            elements.append(chemical_symbols[atomic_number])
            labels.append(label)
            continue
        count = int(round(float(matrix[position, element_index].item())))
        if count <= 0:
            continue
        elements.extend([chemical_symbols[element_index]] * count)
        labels.extend([label] * count)

    if not elements:
        raise ValueError("No occupied Wyckoff sites found")
    return elements, labels
