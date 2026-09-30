"""Dataset wrapper for the preprocessed Wyckoff crystal records."""

from __future__ import annotations

import os
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from aviary.wren.utils import get_prototype_from_protostructure
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .preprocess import (
    _composition_from_wyckoff_matrix,
    preprocess,
    preprocess_dataframe,
)


def _tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = (
        value.clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    )
    return result.to(dtype=dtype) if dtype is not None else result


class CrystalDataset(Dataset):
    """Expose preprocessed crystal dictionaries as PyG ``Data`` objects.

    ``data`` can be a pandas frame, a list of records, a
    single record, or a path to a CSV/``.pt``/pickle cache.  The constructor
    keeps the original skeleton's intentionally small interface while accepting
    these convenient input forms. ``preprocess_options`` forwards CSV cache,
    worker and symmetry settings to :func:`preprocess`.
    """

    def __init__(
        self,
        data: Any,
        num_elements: int = 118,
        transform: Any = None,
        preprocess_options: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if not isinstance(num_elements, int) or num_elements < 1:
            raise ValueError("num_elements must be a positive integer")
        self.num_elements = num_elements
        self.transform = transform
        self.data = self._load(data, **dict(preprocess_options or {}))
        self._prototype_keys: tuple[str, ...] | None = None

    @staticmethod
    def _load(data: Any, **preprocess_options: Any) -> list[Any]:
        if isinstance(data, (str, os.PathLike)):
            path = Path(data)
            if path.suffix.lower() == ".csv":
                data = preprocess(path, **preprocess_options)
            elif path.suffix.lower() in {".pt", ".pth"}:
                data = torch.load(path, map_location="cpu", weights_only=False)
            elif path.suffix.lower() in {".pkl", ".pickle"}:
                with path.open("rb") as handle:
                    data = pickle.load(handle)
            else:
                raise ValueError(
                    "data path must end in .csv, .pt, .pth, .pkl or .pickle"
                )

        symmetry_options = {
            name: preprocess_options[name]
            for name in ("symprec", "fallback_symprec")
            if name in preprocess_options
        }
        if isinstance(data, pd.Series) or (
            isinstance(data, Mapping)
            and {
                "wyckoff_spglib",
                "cif",
            }.intersection(data)
        ):
            data = pd.DataFrame([data])
        if isinstance(data, pd.DataFrame):
            if {"wyckoff_spglib", "cif"}.intersection(data.columns):
                data = preprocess_dataframe(data, **symmetry_options)
            return data.to_dict("records")
        if isinstance(data, (pd.Series, Mapping, Data)):
            return [data]
        try:
            records = list(data)
        except TypeError as exc:
            raise TypeError("data must be a record or a sequence of records") from exc
        if (
            records
            and isinstance(records[0], Mapping)
            and {"wyckoff_spglib", "cif"}.intersection(records[0])
        ):
            records = preprocess_dataframe(
                pd.DataFrame(records), **symmetry_options
            ).to_dict("records")
        return records

    def __len__(self) -> int:
        return len(self.data)

    @property
    def prototype_keys(self) -> tuple[str, ...]:
        """Return one canonical prototype label for each material record."""

        if self._prototype_keys is None:
            keys = []
            for record in self.data:
                if isinstance(record, pd.Series):
                    record = record.to_dict()
                if isinstance(record, Mapping):
                    label = record.get("aflow_label") or record.get("wyckoff_spglib")
                else:
                    label = getattr(record, "aflow_label", None)
                if label is None:
                    raise ValueError(
                        "prototype sampling requires aflow_label or "
                        "wyckoff_spglib in every record"
                    )
                keys.append(get_prototype_from_protostructure(str(label)))
            self._prototype_keys = tuple(keys)
        return self._prototype_keys

    def __getitem__(self, index: int) -> Data:
        record = self.data[index]
        data = record.clone() if isinstance(record, Data) else self._to_data(record)

        # One material may have several equivalent Wyckoff orderings.  They are
        # interchangeable training views, so choose one without changing the
        # cached record.
        if hasattr(data, "x") and hasattr(data, "num_pos") and data.x.ndim == 2:
            n_pos = int(data.num_pos.reshape(-1)[0])
            n_sets = int(getattr(data, "num_sets", torch.tensor([1])).reshape(-1)[0])
            if n_sets > 1 and data.x.shape[0] == n_sets * n_pos:
                selected = torch.randint(n_sets, (1,)).item()
                data.x = data.x[selected * n_pos : (selected + 1) * n_pos]
            if hasattr(data, "zero_dof"):
                data.x_0_dof = data.x[data.zero_dof, 0]
                data.x_inf_dof = data.x[~data.zero_dof, 1 : self.num_elements + 1]

        if self.transform is not None:
            data = self.transform(data)
        return data

    def _to_data(self, record: Any) -> Data:
        if isinstance(record, pd.Series):
            record = record.to_dict()
        if not isinstance(record, Mapping):
            raise TypeError(f"Unsupported record type: {type(record).__name__}")

        required = {"wyckoff_element_matrix", "degrees_of_freedom", "multiplicities"}
        if required.issubset(record):
            return self._wyckoff_to_data(record)

        # Permit loading a Data object serialized with ``Data.to_dict``.
        if "x" in record or "edge_index" in record:
            values = dict(record)
            if "x" in values:
                values["x"] = _tensor(values["x"])
            if "edge_index" in values:
                values["edge_index"] = _tensor(values["edge_index"], dtype=torch.long)
            return Data.from_dict(values)
        missing = ", ".join(sorted(required.difference(record)))
        raise ValueError(f"record is missing Wyckoff fields: {missing}")

    def wyckoff_data_to_graph(self, record: Mapping[str, Any] | pd.Series) -> Data:
        """Public conversion hook for callers that already have one row."""

        if isinstance(record, pd.Series):
            record = record.to_dict()
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping or pandas Series")
        return self._wyckoff_to_data(record)

    def _wyckoff_to_data(self, record: Mapping[str, Any]) -> Data:
        matrix = _tensor(record["wyckoff_element_matrix"], dtype=torch.long)
        if matrix.ndim != 3 or matrix.shape[0] == 0:
            raise ValueError(
                "wyckoff_element_matrix must have shape (sets, positions, channels)"
            )
        degrees = _tensor(record["degrees_of_freedom"], dtype=torch.long).reshape(-1)
        multiplicities = _tensor(record["multiplicities"], dtype=torch.float).reshape(
            -1
        )
        if (
            matrix.shape[1] != degrees.numel()
            or degrees.numel() != multiplicities.numel()
        ):
            raise ValueError(
                "Wyckoff matrix, degrees_of_freedom and multiplicities must align"
            )

        zero_dof = degrees == 0
        composition = _composition_from_wyckoff_matrix(
            matrix, zero_dof, multiplicities, self.num_elements
        ).unsqueeze(0)
        num_pos = degrees.numel()
        positions = torch.arange(num_pos, dtype=torch.long)
        edge_index = torch.stack(
            [positions.repeat_interleave(num_pos), positions.repeat(num_pos)]
        )
        space_group = int(record["space_group"])
        energy = float(record.get("e_form_per_atom", float("nan")))

        graph = Data(
            x=matrix.flatten(0, 1),
            edge_index=edge_index,
            space_group=torch.tensor(space_group, dtype=torch.long),
            e_form_per_atom=torch.tensor(energy, dtype=torch.float),
            multiplicities=multiplicities,
            degrees_of_freedom=degrees,
            wyckoff_pos_idx=positions,
            num_pos=torch.tensor([num_pos], dtype=torch.long),
            num_nodes=torch.tensor([num_pos], dtype=torch.long),
            num_sets=torch.tensor([matrix.shape[0]], dtype=torch.long),
            zero_dof=zero_dof,
            num_0_dof=zero_dof.sum(),
            num_inf_dof=(~zero_dof).sum(),
            composition=composition,
        )
        # These fields are useful for traceability but are not model inputs.
        for name in ("aflow_label", "elements", "wyckoff_set", "material_id"):
            if name in record and record[name] is not None:
                setattr(graph, name, record[name])
        return graph

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(len={len(self)}, num_elements={self.num_elements})"
        )


__all__ = ["CrystalDataset"]
