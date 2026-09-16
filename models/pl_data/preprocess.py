"""Preprocessing helpers for Wyckoff/AFLOW crystal data.

The MP20 files store a protostructure in the ``wyckoff_spglib`` column.  The
model consumes a normalized representation instead: one row per material with
the parsed space group, elements, equivalent Wyckoff sets, and an encoded
element matrix.  This module keeps that conversion independent from the
PyTorch Geometric dataset wrapper in :mod:`dataset`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aviary.wren.data as aviary_wren_data
import pandas as pd
import torch

from ..common.lookup_tables import (
    chemical_symbols,
    spg_wyckoff,
    spg_wyckoff_degrees_of_freedom,
    spg_wyckoff_multiplicities,
    wyckoff_label_to_index,
)


def _space_group_table(table: dict, space_group: str) -> Any:
    return table[str(space_group)]


def extract_wyckoff_data_and_properties(data_frame_row: pd.Series) -> pd.Series:
    """Parse one MP20 row into the fields used by the graph representation."""

    label_column = "wyckoff_spglib"
    spg, _, elements, wyckoff_set = aviary_wren_data.parse_protostructure_label(
        data_frame_row[label_column]
    )
    e_form_per_atom = data_frame_row["formation_energy_per_atom"]

    return pd.Series(
        {
            "aflow_label": data_frame_row[label_column],
            "space_group": str(spg),
            "elements": elements,
            "wyckoff_set": wyckoff_set,
            "e_form_per_atom": e_form_per_atom,
        }
    )


def map_element_to_index(element: str | int) -> int:
    """"element symbols to number"""

    return chemical_symbols.index(str(element))


def map_wyckoff_label_to_index(wyckoff_label: str | int) -> int:
    """Wyckoff letter to number."""

    # Change to zero-based indexing
    return wyckoff_label_to_index.get(wyckoff_label) - 1


def matrix_list(rows: int, cols: int, value: int = 0) -> list[list[int]]:
    """Create a fresh rectangular integer matrix."""

    return [[value for _ in range(cols)] for _ in range(rows)]


def map_nested_list(func: Callable[[Any], Any], values: Any) -> Any:
    """Apply ``func`` recursively to arbitrarily nested list/tuple values."""

    if isinstance(values, (list, tuple)):
        return [map_nested_list(func, value) for value in values]
    return func(values)


def _composition_from_wyckoff_matrix(
    wyckoff_element_matrix: Any,
    zero_dof: Any,
    multiplicities: Any,
    num_elements: int,
) -> torch.Tensor:
    """Decode one equivalent Wyckoff set into conventional-cell counts."""

    matrix = torch.as_tensor(wyckoff_element_matrix)
    if matrix.ndim != 3 or matrix.shape[0] == 0:
        raise ValueError("wyckoff_element_matrix must contain at least one set")
    matrix = matrix[0]
    zero_dof = torch.as_tensor(zero_dof, dtype=torch.bool)
    multiplicities = torch.as_tensor(multiplicities, dtype=torch.float)
    if matrix.shape[0] != zero_dof.numel():
        raise ValueError("Wyckoff matrix and degrees-of-freedom lengths differ")
    if matrix.shape[0] != multiplicities.numel():
        raise ValueError("Wyckoff matrix and multiplicities lengths differ")
    if num_elements < 1 or matrix.shape[1] < num_elements + 1:
        raise ValueError("Wyckoff matrix has fewer element channels than num_elements")

    composition = torch.zeros(num_elements + 1, dtype=torch.float)

    zero_values = matrix[zero_dof, 0].long()
    zero_multiplicities = multiplicities[zero_dof]
    valid_zero = zero_values > 0
    if valid_zero.any():
        if (zero_values[valid_zero] > num_elements).any():
            raise ValueError("Wyckoff matrix contains an unsupported element")
        composition.index_add_(
            0, zero_values[valid_zero], zero_multiplicities[valid_zero]
        )

    inf_values = matrix[~zero_dof, 1 : num_elements + 1].float()
    inf_multiplicities = multiplicities[~zero_dof]
    composition[1:] += (inf_values * inf_multiplicities.unsqueeze(1)).sum(dim=0)
    return composition


_PROCESSED_COLUMNS = [
    "aflow_label",
    "space_group",
    "elements",
    "wyckoff_set",
    "e_form_per_atom",
    "wyckoff_element_matrix",
    "degrees_of_freedom",
    "multiplicities",
]


def format_wyckoff_element_matrix(
    extracted_wyckoff_row: pd.Series,
) -> list[list[list[int]]]:
    """Encode all equivalent Wyckoff sets for one material.

    Each set has shape ``(number_of_positions, 119)``.  Column zero stores the
    one-based element index for fixed (zero-DoF) positions; the remaining
    columns store element counts for variable positions.
    """

    elements = extracted_wyckoff_row["elements"]
    wyckoff_sets = extracted_wyckoff_row["wyckoff_set"]
    if wyckoff_sets and isinstance(wyckoff_sets[0], str):
        wyckoff_sets = [wyckoff_sets]
    space_group = str(extracted_wyckoff_row["space_group"])
    spg_positions = _space_group_table(spg_wyckoff, space_group)
    wyckoff_dof = _space_group_table(spg_wyckoff_degrees_of_freedom, space_group)

    element_indices = map_nested_list(map_element_to_index, elements)
    wyckoff_set_indices = map_nested_list(map_wyckoff_label_to_index, wyckoff_sets)
    element_dimension = len(chemical_symbols)
    position_dimension = len(spg_positions)
    wem_list: list[list[list[int]]] = []

    for wyckoff_indices, wyckoff_set in zip(wyckoff_set_indices, wyckoff_sets):
        if len(wyckoff_indices) != len(element_indices):
            raise ValueError(
                f"Wyckoff site count {len(wyckoff_indices)} does not match "
                f"element count {len(element_indices)}"
            )
        wem = matrix_list(position_dimension, element_dimension)
        for wyckoff_site_index, element_index, wyckoff_pos in zip(
            wyckoff_indices, element_indices, wyckoff_set
        ):
            if not 0 <= wyckoff_site_index < position_dimension:
                raise ValueError(f"Wyckoff index {wyckoff_site_index} is out of range")
            dof = wyckoff_dof.get(str(wyckoff_pos))
            if dof is None:
                raise ValueError(
                    f"Unknown Wyckoff position {wyckoff_pos!r} for space group "
                    f"{space_group}"
                )
            if dof == 0:
                wem[wyckoff_site_index][0] = element_index
            else:
                wem[wyckoff_site_index][element_index] += 1
        wem_list.append(wem)
    if not wem_list:
        raise ValueError("No equivalent Wyckoff sets were found")
    return wem_list


def fetch_wyckoff_degrees_of_freedom(space_group: str | int) -> list[int]:
    """Return DoF values ordered with Wyckoff position ``a`` first."""

    values = list(
        _space_group_table(spg_wyckoff_degrees_of_freedom, str(space_group)).values()
    )
    values.reverse()
    return values


def fetch_wyckoff_multiplicities(space_group: str | int) -> list[int]:
    """Return multiplicities ordered with Wyckoff position ``a`` first."""

    values = list(
        _space_group_table(spg_wyckoff_multiplicities, str(space_group)).values()
    )
    values.reverse()
    return values


def preprocess_dataframe(wd_df: pd.DataFrame) -> pd.DataFrame:
    """Convert an in-memory raw MP20 table to model-ready columns."""

    required = {"wyckoff_spglib", "formation_energy_per_atom"}
    missing = required.difference(wd_df.columns)
    if missing:
        raise ValueError(f"Raw MP20 file is missing columns: {sorted(missing)}")
    if wd_df.empty:
        return pd.DataFrame(columns=_PROCESSED_COLUMNS)

    parsed = wd_df.apply(extract_wyckoff_data_and_properties, axis=1)
    parsed["wyckoff_element_matrix"] = parsed.apply(
        lambda row: format_wyckoff_element_matrix(row), axis=1
    )
    parsed["degrees_of_freedom"] = parsed["space_group"].map(
        fetch_wyckoff_degrees_of_freedom
    )
    parsed["multiplicities"] = parsed["space_group"].map(fetch_wyckoff_multiplicities)
    return parsed.reset_index(drop=True)


def processed_path(raw_file_path: str | os.PathLike[str]) -> Path:
    """Return the default on-disk cache path for a raw CSV file."""

    return Path(raw_file_path).with_suffix(".pt")


def save_preprocessed(
    data_frame: pd.DataFrame,
    processed_file_path: str | os.PathLike[str],
) -> Path:
    """Save model-ready rows as a PyTorch cache."""

    output_path = Path(processed_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_frame, output_path)
    return output_path


def load_preprocessed(processed_file_path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load a cache written by :func:`save_preprocessed`."""

    return torch.load(Path(processed_file_path), map_location="cpu", weights_only=False)


def preprocess(
    raw_file_path: str | os.PathLike[str],
    processed_file_path: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Read an MP20 CSV, cache its model-ready rows, and return a DataFrame.

    By default the cache is written next to the CSV with the same stem and a
    ``.pt`` suffix.  A cache newer than the raw CSV is reused; pass ``force``
    to rebuild it.
    """

    path = Path(raw_file_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Only supporting '.csv' raw files.")

    cache_path = (
        processed_path(path)
        if processed_file_path is None
        else Path(processed_file_path)
    )
    if (
        not force
        and cache_path.exists()
        and cache_path.stat().st_mtime >= path.stat().st_mtime
    ):
        return load_preprocessed(cache_path)

    parsed = preprocess_dataframe(pd.read_csv(path))
    save_preprocessed(parsed, cache_path)
    return parsed


__all__ = [
    "_composition_from_wyckoff_matrix",
    "extract_wyckoff_data_and_properties",
    "fetch_wyckoff_degrees_of_freedom",
    "fetch_wyckoff_multiplicities",
    "format_wyckoff_element_matrix",
    "map_element_to_index",
    "map_nested_list",
    "map_wyckoff_label_to_index",
    "matrix_list",
    "load_preprocessed",
    "preprocess",
    "preprocess_dataframe",
    "processed_path",
    "save_preprocessed",
]
