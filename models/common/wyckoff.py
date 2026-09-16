"""Wyckoff graph and symmetry-template conversion helpers."""

from __future__ import annotations

import re
from typing import Any

import torch

from .lookup_tables import chemical_symbols, wyckoff_label_to_index

_MULTIPLICITY_RE = re.compile(r"^(\d+)([A-Za-z])$")


def query_from_gwa_sequence(sequence: str) -> dict[str, Any]:
    """Convert a complete ``G-W-A-W-A-…`` sequence to a symmetry query."""

    tokens = str(sequence).split("-")
    if len(tokens) < 3 or len(tokens) % 2 == 0:
        raise ValueError(f"invalid G-W-A sequence: {sequence}")
    try:
        space_group = int(tokens[0])
    except ValueError as error:
        raise ValueError(
            f"invalid space group in G-W-A sequence: {sequence}"
        ) from error
    if not 1 <= space_group <= 230:
        raise ValueError(f"space group must be in 1..230: {space_group}")

    wyckoff_letters = tokens[1::2]
    atom_types = tokens[2::2]
    for wyckoff in wyckoff_letters:
        if _MULTIPLICITY_RE.fullmatch(wyckoff) is None:
            raise ValueError(f"invalid multiplicity-qualified Wyckoff label: {wyckoff}")
    if any(not element for element in atom_types):
        raise ValueError(f"empty element in G-W-A sequence: {sequence}")
    return {
        "spacegroup_number": space_group,
        "wyckoff_letters": wyckoff_letters,
        "atom_types": atom_types,
    }


def wyckoff_multiplicity(label: str) -> int:
    """Return the multiplicity from a label such as ``4f``."""

    match = _MULTIPLICITY_RE.fullmatch(str(label))
    if match is None:
        raise ValueError(f"invalid multiplicity-qualified Wyckoff label: {label}")
    return int(match.group(1))


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
