"""A single Wyckoff template and its model/input representations.

The project uses the same physical template in three serializations:

* a model ``G-W-A-W-A-...`` sequence;
* CrystalFlow's ``G_ElementNxMultiplicityLetter_...`` string; and
* DiffCSP++'s ``spacegroup_number``/``wyckoff_letters``/``atom_types`` query.

``WyckoffTemplate`` is deliberately a *single-template* object.  It does not
know about data frames, candidate ranks, or target batches.  Code that handles
those concerns can keep using ordinary pandas data frames outside this class.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from .lookup_tables import (
    chemical_symbols,
    spg_wyckoff_multiplicities,
    wyckoff_label_to_index,
)

_GWA_TOKEN_RE = re.compile(r"^(?P<multiplicity>[1-9][0-9]*)(?P<letter>[A-Za-z])$")
_ORBIT_RE = re.compile(
    r"^(?P<element>[A-Z][a-z]?)(?P<count>[1-9][0-9]*)x"
    r"(?P<multiplicity>[1-9][0-9]*)(?P<letter>[A-Za-z])$"
)
_ELEMENT_SET = frozenset(chemical_symbols[1:])


@dataclass(frozen=True, slots=True, eq=False)
class WyckoffTemplate:
    """One space-group Wyckoff template.

    ``wyckoff_letters`` and ``atom_types`` contain one entry per occupied
    orbit.  Repeated entries are intentional: for example,
    ``1_Li2x1a_O1x1a`` is stored as ``1a, 1a, 1a`` with species
    ``Li, Li, O``.  This representation is exactly what DiffCSP++ consumes,
    while :meth:`to_crystalflow` combines repeated ``(element, orbit)`` pairs
    into CrystalFlow occupation counts.
    """

    spacegroup_number: int
    wyckoff_letters: tuple[str, ...]
    atom_types: tuple[str, ...]

    _GWA_KEYS: ClassVar[tuple[str, ...]] = (
        "sequence",
        "structure_sequence",
        "generated_structure_sequence",
        "target_structure_sequence",
    )

    def __post_init__(self) -> None:
        spacegroup_number = int(self.spacegroup_number)
        wyckoff_letters = tuple(str(value) for value in self.wyckoff_letters)
        atom_types = tuple(str(value) for value in self.atom_types)

        if not 1 <= spacegroup_number <= 230:
            raise ValueError(
                "spacegroup_number must be in the range 1..230: "
                f"{spacegroup_number}"
            )
        if not wyckoff_letters:
            raise ValueError("a Wyckoff template must contain at least one orbit")
        if len(wyckoff_letters) != len(atom_types):
            raise ValueError(
                "wyckoff_letters and atom_types must have equal lengths: "
                f"{len(wyckoff_letters)} != {len(atom_types)}"
            )

        multiplicities = spg_wyckoff_multiplicities[str(spacegroup_number)]
        for label in wyckoff_letters:
            match = _GWA_TOKEN_RE.fullmatch(label)
            if match is None:
                raise ValueError(
                    "invalid multiplicity-qualified Wyckoff label: " f"{label}"
                )
            letter = match.group("letter")
            if letter not in multiplicities:
                raise ValueError(
                    f"Wyckoff letter {letter!r} is not available in "
                    f"space group {spacegroup_number}"
                )
            expected = int(multiplicities[letter])
            actual = int(match.group("multiplicity"))
            if actual != expected:
                raise ValueError(
                    f"Wyckoff label {label!r} has multiplicity {actual}, "
                    f"expected {expected} in space group {spacegroup_number}"
                )
        for element in atom_types:
            if element not in _ELEMENT_SET:
                raise ValueError(f"invalid chemical element symbol: {element}")

        object.__setattr__(self, "spacegroup_number", spacegroup_number)
        object.__setattr__(self, "wyckoff_letters", wyckoff_letters)
        object.__setattr__(self, "atom_types", atom_types)

    @property
    def space_group(self) -> int:
        """Compatibility alias for callers using ``space_group`` naming."""

        return self.spacegroup_number

    @property
    def num_orbits(self) -> int:
        return len(self.wyckoff_letters)

    @staticmethod
    def wyckoff_multiplicity(label: str) -> int:
        """Return the integer multiplicity from a label such as ``4f``."""

        match = _GWA_TOKEN_RE.fullmatch(str(label))
        if match is None:
            raise ValueError(f"invalid multiplicity-qualified Wyckoff label: {label}")
        return int(match.group("multiplicity"))

    @staticmethod
    def _normalise_label(label: str) -> str:
        match = _GWA_TOKEN_RE.fullmatch(str(label))
        if match is None:
            raise ValueError(f"invalid multiplicity-qualified Wyckoff label: {label}")
        return f"{int(match.group('multiplicity'))}{match.group('letter')}"

    @classmethod
    def from_gwa(cls, sequence: str) -> "WyckoffTemplate":
        """Build a template from ``G-W-A-W-A-...`` model output."""

        tokens = str(sequence).split("-")
        if len(tokens) < 3 or len(tokens) % 2 == 0:
            raise ValueError(f"invalid G-W-A sequence: {sequence}")
        try:
            spacegroup_number = int(tokens[0])
        except ValueError as error:
            raise ValueError(f"invalid space group in G-W-A sequence: {sequence}") from error
        return cls(
            spacegroup_number,
            tuple(tokens[1::2]),
            tuple(tokens[2::2]),
        )

    @classmethod
    def from_diffcsppp_query(cls, query: Mapping[str, Any]) -> "WyckoffTemplate":
        """Build a template from a DiffCSP++ structured query."""

        required = {"spacegroup_number", "wyckoff_letters", "atom_types"}
        missing = required.difference(query)
        if missing:
            raise ValueError(f"DiffCSP++ query is missing fields: {sorted(missing)}")
        labels = query["wyckoff_letters"]
        atom_types = query["atom_types"]
        if isinstance(labels, (str, bytes)) or isinstance(atom_types, (str, bytes)):
            raise ValueError("query Wyckoff labels and atom types must be sequences")
        return cls(
            int(query["spacegroup_number"]),
            tuple(labels),
            tuple(atom_types),
        )

    @classmethod
    def from_diffcsp_query(cls, query: Mapping[str, Any]) -> "WyckoffTemplate":
        """Alias for :meth:`from_diffcsppp_query`."""

        return cls.from_diffcsppp_query(query)

    @classmethod
    def from_crystalflow(cls, value: str | Mapping[str, Any]) -> "WyckoffTemplate":
        """Build a template from CrystalFlow's ``wyckoff`` representation.

        ``value`` may be the raw ``G_...`` string or a row-like mapping with a
        ``wyckoff``/``wyckoff_occupancy`` field.
        """

        if isinstance(value, Mapping):
            if "wyckoff" in value:
                value = value["wyckoff"]
            elif "wyckoff_occupancy" in value:
                value = value["wyckoff_occupancy"]
            else:
                raise ValueError(
                    "CrystalFlow row must contain 'wyckoff' or "
                    "'wyckoff_occupancy'"
                )

        spacegroup_token, *orbits = str(value).split("_")
        if not orbits:
            raise ValueError(f"invalid CrystalFlow Wyckoff template: {value}")
        try:
            spacegroup_number = int(spacegroup_token)
        except ValueError as error:
            raise ValueError(f"invalid space group in CrystalFlow template: {value}") from error

        labels: list[str] = []
        atom_types: list[str] = []
        for orbit in orbits:
            match = _ORBIT_RE.fullmatch(orbit)
            if match is None:
                raise ValueError(f"invalid CrystalFlow Wyckoff orbit: {orbit}")
            element = match.group("element")
            count = int(match.group("count"))
            label = f"{match.group('multiplicity')}{match.group('letter')}"
            labels.extend([label] * count)
            atom_types.extend([element] * count)
        return cls(spacegroup_number, tuple(labels), tuple(atom_types))

    @classmethod
    def from_occupancy(cls, value: str | Mapping[str, Any]) -> "WyckoffTemplate":
        """Alias for :meth:`from_crystalflow` used by extracted model CSVs."""

        return cls.from_crystalflow(value)

    @classmethod
    def from_model_output(cls, value: Any) -> "WyckoffTemplate":
        """Build a template from a model sequence, query, row, or graph sample.

        The graph form is the PyTorch-Geometric sample emitted by WyckoffFlow:
        it contains ``x`` and ``space_group``.  Keeping this decoder here makes
        the model output conversion independent of the downstream structure
        generator.
        """

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            if "_" in value and "-" not in value:
                return cls.from_crystalflow(value)
            return cls.from_gwa(value)

        if isinstance(value, Mapping):
            for key in cls._GWA_KEYS:
                if key in value and value[key] is not None:
                    return cls.from_gwa(str(value[key]))
            if "query" in value and isinstance(value["query"], Mapping):
                return cls.from_diffcsppp_query(value["query"])
            if {"spacegroup_number", "wyckoff_letters", "atom_types"}.issubset(value):
                return cls.from_diffcsppp_query(value)
            if "wyckoff" in value or "wyckoff_occupancy" in value:
                return cls.from_crystalflow(value)
            if "x" in value:
                return cls._from_graph(value)

        for key in cls._GWA_KEYS:
            sequence = getattr(value, key, None)
            if sequence is not None:
                return cls.from_gwa(str(sequence))
        if hasattr(value, "x"):
            return cls._from_graph(value)

        raise TypeError(
            "unsupported Wyckoff model output; expected a G-W-A string, "
            "CrystalFlow string, DiffCSP++ query, or graph sample"
        )

    @classmethod
    def _from_graph(cls, value: Any) -> "WyckoffTemplate":
        def get_field(name: str) -> Any:
            if isinstance(value, Mapping):
                return value[name]
            return getattr(value, name)

        spacegroup_value = (
            value.get("spacegroup_number")
            if isinstance(value, Mapping) and "spacegroup_number" in value
            else None
        )
        if spacegroup_value is None:
            spacegroup_value = (
                value.get("space_group")
                if isinstance(value, Mapping) and "space_group" in value
                else getattr(value, "spacegroup_number", None)
            )
        if spacegroup_value is None:
            spacegroup_value = getattr(value, "space_group")

        spacegroup_number = _scalar(spacegroup_value)
        x = get_field("x")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.ndim != 2 or not 2 <= x.shape[1] <= len(chemical_symbols):
            raise ValueError(
                "Wyckoff graph x must have shape "
                f"(num_sites, 2..{len(chemical_symbols)}); got {tuple(x.shape)}"
            )
        positions, element_indices = torch.nonzero(x, as_tuple=True)

        index_to_label = {
            index: label for label, index in wyckoff_label_to_index.items()
        }
        labels: list[str] = []
        atom_types: list[str] = []
        for position, element_index in zip(
            positions.tolist(), element_indices.tolist()
        ):
            try:
                letter = index_to_label[position + 1]
            except KeyError as error:
                raise ValueError(
                    f"unknown Wyckoff position index: {position + 1}"
                ) from error
            if element_index == 0:
                atomic_number = int(round(float(x[position, 0].item())))
                if not 1 <= atomic_number < len(chemical_symbols):
                    raise ValueError(
                        "invalid atomic number in zero-DoF Wyckoff site: "
                        f"{atomic_number}"
                    )
                labels.append(
                    cls._graph_label(
                        value,
                        letter,
                        position,
                        spacegroup_number,
                    )
                )
                atom_types.append(chemical_symbols[atomic_number])
                continue

            count = int(round(float(x[position, element_index].item())))
            if count <= 0:
                continue
            labels.extend(
                [
                    cls._graph_label(
                        value,
                        letter,
                        position,
                        spacegroup_number,
                    )
                ]
                * count
            )
            atom_types.extend([chemical_symbols[element_index]] * count)

        if not labels:
            raise ValueError("Wyckoff graph contains no occupied sites")
        return cls(spacegroup_number, tuple(labels), tuple(atom_types))

    @staticmethod
    def _graph_label(
        value: Any,
        letter: str,
        position: int,
        spacegroup_number: int,
    ) -> str:
        multiplicities = None
        if isinstance(value, Mapping):
            multiplicities = value.get("multiplicities")
        else:
            multiplicities = getattr(value, "multiplicities", None)
        if multiplicities is not None:
            multiplicity = _scalar_at(multiplicities, position)
        else:
            multiplicity = None
        if multiplicity is None:
            try:
                table = spg_wyckoff_multiplicities[str(spacegroup_number)]
                multiplicity = table.get(str(letter))
                if multiplicity is None:
                    multiplicity = table[str(letter).lower()]
            except KeyError as error:
                raise ValueError(
                    "cannot infer Wyckoff multiplicity for graph position "
                    f"{letter!r} in space group {spacegroup_number}"
                ) from error
        return f"{multiplicity}{letter}"

    @classmethod
    def from_protostructure_set(cls, value: str) -> tuple["WyckoffTemplate", ...]:
        """Return all equivalent templates encoded by ``wyckoff_spglib``."""

        from aviary.wren.data import parse_protostructure_label

        spacegroup, _, elements, wyckoff_sets = parse_protostructure_label(str(value))
        multiplicities = spg_wyckoff_multiplicities[str(spacegroup)]
        templates = []
        for wyckoff_set in wyckoff_sets:
            labels = []
            for letter in wyckoff_set:
                letter = str(letter)
                multiplicity = multiplicities.get(letter)
                if multiplicity is None:
                    multiplicity = multiplicities[letter.lower()]
                labels.append(f"{multiplicity}{letter}")
            templates.append(cls(int(spacegroup), tuple(labels), tuple(elements)))
        return tuple(templates)

    @classmethod
    def from_protostructure(cls, value: str) -> "WyckoffTemplate":
        """Build the first template from a ``wyckoff_spglib`` label."""

        templates = cls.from_protostructure_set(value)
        if not templates:
            raise ValueError(f"protostructure has no Wyckoff settings: {value}")
        return templates[0]

    @property
    def formula_counts(self) -> dict[str, int]:
        """Return complete conventional-cell element counts."""

        counts: Counter[str] = Counter()
        for label, element in zip(self.wyckoff_letters, self.atom_types):
            counts[element] += self.wyckoff_multiplicity(label)
        return {
            element: counts[element]
            for element in sorted(counts, key=chemical_symbols.index)
        }

    @property
    def formula(self) -> str:
        """Return the complete conventional-cell formula."""

        return "".join(
            element if count == 1 else f"{element}{count}"
            for element, count in self.formula_counts.items()
        )

    @property
    def formula_with_counts(self) -> str:
        """Return the complete formula while retaining explicit ``1`` counts."""

        return "".join(
            f"{element}{count}" for element, count in self.formula_counts.items()
        )

    def to_gwa(self) -> str:
        """Serialize to the model's ``G-W-A-W-A-...`` sequence."""

        tokens = [str(self.spacegroup_number)]
        for label, element in zip(self.wyckoff_letters, self.atom_types):
            tokens.extend((label, element))
        return "-".join(tokens)

    def to_diffcsppp_query(self) -> dict[str, Any]:
        """Serialize to the DiffCSP++ structured query."""

        return {
            "spacegroup_number": self.spacegroup_number,
            "wyckoff_letters": list(self.wyckoff_letters),
            "atom_types": list(self.atom_types),
        }

    def to_diffcsp_query(self) -> dict[str, Any]:
        """Alias for :meth:`to_diffcsppp_query`."""

        return self.to_diffcsppp_query()

    def _occupancy_entries(self, canonical: bool = True) -> tuple[str, ...]:
        counts: Counter[tuple[str, str]] = Counter(
            (element, self._normalise_label(label))
            for label, element in zip(self.wyckoff_letters, self.atom_types)
        )
        entries = tuple(
            f"{element}{count}x{label}"
            for (element, label), count in counts.items()
        )
        if canonical:
            return tuple(sorted(entries))
        return entries

    def occupancy_key(self) -> tuple[str, ...]:
        """Return an order-independent key for template comparison."""

        return (str(self.spacegroup_number), *self._occupancy_entries())

    def to_crystalflow(self) -> str:
        """Serialize to CrystalFlow's ``wyckoff`` string."""

        return "_".join(self.occupancy_key())

    def equivalent_to(self, other: "WyckoffTemplate") -> bool:
        """Return whether two templates differ only in orbit ordering."""

        if not isinstance(other, WyckoffTemplate):
            return False
        return self.occupancy_key() == other.occupancy_key()

    def matches_protostructure(self, value: str) -> bool:
        """Match this template against any equivalent target setting."""

        return any(
            self.equivalent_to(target)
            for target in self.from_protostructure_set(value)
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WyckoffTemplate) and self.equivalent_to(other)

    def __hash__(self) -> int:
        return hash(self.occupancy_key())

    def __str__(self) -> str:
        return self.to_crystalflow()


def _scalar(value: Any) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _scalar_at(value: Any, index: int) -> int | None:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape") and getattr(value, "ndim", 0) > 1:
        value = value.reshape(-1)
    try:
        item = value[index]
    except (IndexError, TypeError):
        return None
    if hasattr(item, "item"):
        item = item.item()
    return int(round(float(item)))


# Keep the misspelled name available for old experiment code that used the
# user's original spelling before the class was given its canonical name.
WyckoffTemplete = WyckoffTemplate
