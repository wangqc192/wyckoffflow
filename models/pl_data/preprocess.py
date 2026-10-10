"""Convert MP20 labels, CIFs or ALEX structures to equivalent Wyckoff templates."""

from __future__ import annotations

import ast
import json
import logging
import multiprocessing
import os
import tempfile
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

import aviary.wren.data as aviary_wren_data
from aviary.wren.utils import get_protostructure_label_from_spglib

from ..common.lookup_tables import (
    chemical_symbols,
    spg_wyckoff,
    spg_wyckoff_degrees_of_freedom,
    spg_wyckoff_multiplicities,
    wyckoff_label_to_index,
)

log = logging.getLogger(__name__)


def _space_group_table(table: dict, space_group: str) -> Any:
    return table[str(space_group)]


def extract_wyckoff_data_and_properties(
    data_frame_row: pd.Series,
    *,
    symprec: float = 0.1,
    fallback_symprec: float | None = 1e-5,
) -> pd.Series:
    """Parse an existing label, or derive it from the crystal structure."""

    label = data_frame_row.get("wyckoff_spglib")
    if pd.isna(label):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="dict interface is deprecated.*",
                    category=DeprecationWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Issues encountered while parsing CIF:.*fractional coordinates rounded.*",
                )
                cif = data_frame_row.get("cif")
                if pd.notna(cif):
                    structure = Structure.from_str(cif, fmt="cif")
                else:
                    structure_dict = data_frame_row["structure"]
                    if isinstance(structure_dict, str):
                        structure_dict = ast.literal_eval(structure_dict)
                    structure = Structure.from_dict(structure_dict)
                label = get_protostructure_label_from_spglib(
                    structure,
                    raise_errors=True,
                    init_symprec=symprec,
                    fallback_symprec=fallback_symprec,
                )
        except (ValueError, SyntaxError) as exc:
            identifier = data_frame_row.get(
                "material_id", data_frame_row.get("mat_id", data_frame_row.name)
            )
            raise ValueError(
                f"Could not extract Wyckoff template for {identifier}: {exc}"
            ) from exc
    spg, _, elements, wyckoff_set = aviary_wren_data.parse_protostructure_label(label)
    # Aviary returns equivalent settings from a set; keep cache ordering stable
    # across worker processes without changing the element/site correspondence.
    wyckoff_set = sorted(wyckoff_set)
    e_form_per_atom = data_frame_row.get(
        "formation_energy_per_atom", data_frame_row.get("e_form", float("nan"))
    )

    return pd.Series(
        {
            "aflow_label": label,
            "space_group": str(spg),
            "elements": elements,
            "wyckoff_set": wyckoff_set,
            "e_form_per_atom": e_form_per_atom,
        }
    )


def map_element_to_index(element: str | int) -> int:
    """Map an element symbol to its atomic number."""

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


def preprocess_dataframe(
    wd_df: pd.DataFrame,
    *,
    symprec: float = 0.1,
    fallback_symprec: float | None = 1e-5,
) -> pd.DataFrame:
    """Convert raw labels/CIFs/structure dictionaries; missing energies are NaN.

    The CSV's ``space_group`` field may be a symbol or a placeholder. The
    model's numeric space group always comes from the extracted template.
    """

    if not {"wyckoff_spglib", "cif", "structure"}.intersection(wd_df.columns):
        raise ValueError(
            "Raw crystal data requires 'wyckoff_spglib', 'cif' or 'structure'"
        )
    if wd_df.empty:
        return pd.DataFrame(columns=_PROCESSED_COLUMNS)

    parsed = wd_df.apply(
        extract_wyckoff_data_and_properties,
        axis=1,
        symprec=symprec,
        fallback_symprec=fallback_symprec,
    )
    parsed["wyckoff_element_matrix"] = parsed.apply(
        # Dense Python integer lists dominate memory for the 675k-row dataset.
        lambda row: np.asarray(format_wyckoff_element_matrix(row), dtype=np.int32),
        axis=1,
    )
    parsed["degrees_of_freedom"] = parsed["space_group"].map(
        fetch_wyckoff_degrees_of_freedom
    )
    parsed["multiplicities"] = parsed["space_group"].map(fetch_wyckoff_multiplicities)
    if "material_id" in wd_df:
        parsed["material_id"] = wd_df["material_id"]
    elif "mat_id" in wd_df:
        parsed["material_id"] = wd_df["mat_id"]
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
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, suffix=".pt.tmp", delete=False
    ) as handle:
        torch.save(data_frame, handle)
    os.replace(handle.name, output_path)
    return output_path


def load_preprocessed(processed_file_path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load a cache written by :func:`save_preprocessed`."""

    return torch.load(Path(processed_file_path), map_location="cpu", weights_only=False)


def prepare_preprocessed(
    raw_file_path: str | os.PathLike[str],
    processed_file_path: str | os.PathLike[str] | None = None,
    force: bool = False,
    *,
    num_workers: int = 0,
    chunk_size: int = 1000,
    symprec: float = 0.1,
    fallback_symprec: float | None = 1e-5,
) -> Path:
    """Prepare a cache without loading it, suitable for Lightning prepare_data.

    By default the cache is written next to the CSV with the same stem and a
    ``.pt`` suffix. Raw data is read in chunks and optionally processed in
    parallel. Row order and all equivalent templates are preserved; malformed
    structures raise errors instead of silently dropping training samples.
    """

    path = Path(raw_file_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Only supporting '.csv' raw files.")

    cache_path = (
        processed_path(path)
        if processed_file_path is None
        else Path(processed_file_path)
    )
    stat = path.stat()
    metadata = {
        "version": 1,
        "source": str(path.resolve()),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "symprec": symprec,
        "fallback_symprec": fallback_symprec,
    }
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if (
        not force
        and cache_path.exists()
        and cache_path.stat().st_mtime >= stat.st_mtime
    ):
        if metadata_path.exists():
            if json.loads(metadata_path.read_text()) == metadata:
                return cache_path
        elif (
            cache_path == processed_path(path)
            and symprec == 0.1
            and fallback_symprec == 1e-5
        ):
            # Preserve existing MP20 caches written before metadata was added.
            return cache_path

    process_chunk = partial(
        preprocess_dataframe, symprec=symprec, fallback_symprec=fallback_symprec
    )
    frames = []
    row_count = 0

    def collect(chunks):
        nonlocal row_count
        for frame in chunks:
            frames.append(frame)
            row_count += len(frame)
            log.info("Preprocessed %s: %d rows", path.name, row_count)

    with pd.read_csv(path, chunksize=chunk_size) as reader:
        if num_workers > 1:
            with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
                collect(pool.imap(process_chunk, reader))
        else:
            collect(map(process_chunk, reader))

    parsed = pd.concat(frames, ignore_index=True)
    save_preprocessed(parsed, cache_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return cache_path


def preprocess(
    raw_file_path: str | os.PathLike[str],
    processed_file_path: str | os.PathLike[str] | None = None,
    force: bool = False,
    **options: Any,
) -> pd.DataFrame:
    """Read a label/CIF/structure CSV, preparing or reusing its model-ready cache."""

    return load_preprocessed(
        prepare_preprocessed(raw_file_path, processed_file_path, force, **options)
    )


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
    "prepare_preprocessed",
    "processed_path",
    "save_preprocessed",
]
