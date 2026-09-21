"""Shared structure-reconstruction evaluation utilities.

The evaluation scripts intentionally keep the DiffCSP-compatible ``Crystal``
representation local.  This module does not import the DiffCSP package or any
of its scripts; it only uses the representation and validity rules needed by
CSP reconstruction evaluation.
"""

from __future__ import annotations

import itertools
import json
import multiprocessing
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import smact
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Lattice, Structure
from smact.screening import pauling_test
from tqdm.auto import tqdm

MATCHER_KWARGS = {
    "ltol": 0.3,
    "stol": 0.5,
    "angle_tol": 10,
    "primitive_cell": True,
    "scale": True,
}

RECORD_FIELDS = ("frac_coords", "atom_types", "lengths", "angles")
SAMPLE_INDEX_COLUMNS = (
    "input_index",
    "source_index",
    "diffcsp_index",
    "sample_index",
    "candidate_index",
)

CrystalRecord = dict[str, np.ndarray]


def smact_validity(
    comp: Sequence[int],
    count: Sequence[int],
    use_pauling_test: bool = True,
    include_alloys: bool = True,
) -> bool:
    """Return whether a reduced composition passes the SMACT checks."""
    elem_symbols = tuple(Element.from_Z(int(elem)).symbol for elem in comp)
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [element[1] for element in space.items()]
    electronegativities = [element.pauling_eneg for element in smact_elems]
    oxidation_states = [element.oxidation_states for element in smact_elems]

    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys and all(symbol in smact.metals for symbol in elem_symbols):
        return True

    count = tuple(int(value) for value in count)
    threshold = max(count)
    oxidation_state_count = 1
    for states in oxidation_states:
        oxidation_state_count *= len(states)
    if oxidation_state_count > 1e7:
        return False

    for states in itertools.product(*oxidation_states):
        charge_balanced, _ = smact.neutral_ratios(
            states,
            stoichs=[(value,) for value in count],
            threshold=threshold,
        )
        if not charge_balanced:
            continue
        if use_pauling_test:
            try:
                electronegativities_valid = pauling_test(
                    states,
                    electronegativities,
                )
            except TypeError:
                # Missing electronegativity data should not reject a balanced
                # composition, matching the original DiffCSP utility.
                electronegativities_valid = True
        else:
            electronegativities_valid = True
        if electronegativities_valid:
            return True
    return False


def structure_validity(structure: Structure, cutoff: float = 0.5) -> bool:
    """Check minimum inter-atomic distance and a non-degenerate cell volume."""
    distance_matrix = structure.distance_matrix
    padded = distance_matrix + np.diag(
        np.ones(distance_matrix.shape[0]) * (cutoff + 10.0)
    )
    return bool(padded.min() >= cutoff and structure.volume >= 0.1)


class Crystal:
    """Local copy of the DiffCSP crystal representation used for CSP metrics."""

    def __init__(
        self,
        crys_array_dict: Mapping[str, Any],
        compute_valid: bool = True,
        compute_fp: bool = True,
        ignore_smact: bool = False,
    ) -> None:
        self.frac_coords = np.asarray(crys_array_dict["frac_coords"])
        self.atom_types = np.asarray(crys_array_dict["atom_types"])
        self.lengths = np.asarray(crys_array_dict["lengths"])
        self.angles = np.asarray(crys_array_dict["angles"])

        if self.atom_types.ndim > 1:
            self.atom_types = np.argmax(self.atom_types, axis=-1) + 1

        self.dict = {
            "frac_coords": self.frac_coords,
            "atom_types": self.atom_types,
            "lengths": self.lengths,
            "angles": self.angles,
        }
        self.get_structure()
        self.get_composition()

        self.ignore_smact = ignore_smact
        if compute_valid:
            self.get_validity()
        else:
            self.valid = True
            self.comp_valid = True
            self.struct_valid = True

        if compute_fp:
            self.get_fingerprints()
        else:
            self.comp_fp = None
            self.struct_fp = None

    def get_structure(self) -> None:
        self.constructed = False
        if not (
            np.isfinite(self.lengths).all()
            and np.isfinite(self.angles).all()
            and np.isfinite(self.frac_coords).all()
        ):
            self.invalid_reason = "nan_value"
            return
        if self.lengths.size != 3 or np.any(self.lengths <= 0):
            self.invalid_reason = "non_positive_lattice"
            return
        try:
            self.structure = Structure(
                lattice=Lattice.from_parameters(
                    *(self.lengths.tolist() + self.angles.tolist())
                ),
                species=self.atom_types,
                coords=self.frac_coords,
                coords_are_cartesian=False,
            )
        except Exception:
            self.invalid_reason = "construction_raises_exception"
            return
        if self.structure.volume < 0.1:
            self.invalid_reason = "unrealistically_small_lattice"
            return
        self.constructed = True

    def get_composition(self) -> None:
        if self.atom_types.size == 0:
            self.elems = ()
            self.comps = ()
            return
        elem_counter = Counter(self.atom_types.tolist())
        composition = [
            (elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())
        ]
        elems, counts = zip(*composition)
        counts_array = np.asarray(counts, dtype=int)
        counts_array //= np.gcd.reduce(counts_array)
        self.elems = tuple(int(elem) for elem in elems)
        self.comps = tuple(counts_array.tolist())

    def get_validity(self) -> None:
        if not self.elems or len(self.elems) >= 8:
            self.comp_valid = False
        elif self.ignore_smact:
            self.comp_valid = True
        else:
            self.comp_valid = smact_validity(self.elems, self.comps)

        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = bool(self.comp_valid and self.struct_valid)

    def get_fingerprints(self) -> None:
        """Compute the optional DiffCSP fingerprints for API compatibility."""
        from matminer.featurizers.composition.composite import ElementProperty
        from matminer.featurizers.site.fingerprint import CrystalNNFingerprint

        if not self.constructed:
            self.comp_fp = None
            self.struct_fp = None
            return

        composition_fingerprint = ElementProperty.from_preset("magpie")
        crystal_nn_fingerprint = CrystalNNFingerprint.from_preset("ops")
        composition = Composition(Counter(self.atom_types.tolist()))
        self.comp_fp = composition_fingerprint.featurize(composition)
        try:
            site_fingerprints = [
                crystal_nn_fingerprint.featurize(self.structure, index)
                for index in range(len(self.structure))
            ]
        except Exception:
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.asarray(site_fingerprints).mean(axis=0)


def structure_to_record(structure: Structure) -> CrystalRecord:
    """Convert a pymatgen structure into the four arrays consumed by Crystal."""
    return {
        "frac_coords": np.asarray(structure.frac_coords, dtype=float),
        "atom_types": np.asarray([site.specie.Z for site in structure], dtype=int),
        "lengths": np.asarray(structure.lattice.abc, dtype=float),
        "angles": np.asarray(structure.lattice.angles, dtype=float),
    }


def structure_to_crystal(structure: Structure) -> Crystal:
    return Crystal(structure_to_record(structure), compute_fp=False)


def composition_key(
    structure: Structure,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the reduced composition as atomic-number and count tuples."""
    elements = sorted(structure.composition.elements, key=lambda element: element.Z)
    counts = np.asarray(
        [int(round(structure.composition[element])) for element in elements],
        dtype=int,
    )
    counts //= np.gcd.reduce(counts)
    return tuple(element.Z for element in elements), tuple(counts.tolist())


def manifest_columns(manifest: pd.DataFrame) -> tuple[str, str, str]:
    """Return target, sample-record, and candidate-label columns."""
    target_column = next(
        (
            column
            for column in ("target_index", "material_index")
            if column in manifest.columns
        ),
        None,
    )
    if target_column is None:
        raise ValueError("Manifest must contain target_index or material_index")

    sample_column = next(
        (column for column in SAMPLE_INDEX_COLUMNS if column in manifest.columns),
        None,
    )
    if sample_column is None:
        raise ValueError(
            "Manifest must contain one of: " + ", ".join(SAMPLE_INDEX_COLUMNS)
        )

    candidate_column = next(
        (
            column
            for column in ("candidate_index", "sample_index", sample_column)
            if column in manifest.columns
        ),
        sample_column,
    )
    return target_column, sample_column, candidate_column


def resolve_pt_paths(
    sample_path: Path | str | Sequence[Path | str],
) -> list[Path]:
    """Resolve a PT file, a directory, or an explicit sequence of PT files."""
    if isinstance(sample_path, (str, Path)):
        path = Path(sample_path)
        if path.is_file():
            paths = [path]
        elif path.is_dir():
            paths = sorted(path.glob("samples.pt"))
            paths += sorted(path.glob("samples_shard*.pt"))
            if not paths:
                paths = sorted(path.glob("*.pt"))
        else:
            raise FileNotFoundError(path)
    else:
        paths = [Path(path) for path in sample_path]

    if not paths:
        raise FileNotFoundError(f"No PT sample files found under {sample_path}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def load_targets(
    gt_file: Path | str,
    manifest: pd.DataFrame,
    cif_column: str | None = None,
) -> list[Structure]:
    """Load only the target rows referenced by a manifest."""
    target_column, _, _ = manifest_columns(manifest)
    target_indices = manifest[target_column].astype(int)
    if target_indices.empty or target_indices.min() < 0:
        raise ValueError(f"Manifest {target_column} contains no valid target indices")

    targets_frame = pd.read_csv(gt_file, nrows=int(target_indices.max()) + 1)
    if cif_column is None:
        cif_column = next(
            (
                column
                for column in ("cif", "cif.conv")
                if column in targets_frame.columns
            ),
            None,
        )
    if cif_column is None or cif_column not in targets_frame.columns:
        raise ValueError("Ground-truth CSV must contain a CIF column")
    if len(targets_frame) <= int(target_indices.max()):
        raise ValueError("Manifest target index is outside the ground-truth CSV")

    return [
        Structure.from_str(value, fmt="cif") for value in targets_frame[cif_column]
    ]


def _as_record(value: Any) -> CrystalRecord:
    if isinstance(value, Crystal):
        return {
            "frac_coords": np.asarray(value.frac_coords),
            "atom_types": np.asarray(value.atom_types),
            "lengths": np.asarray(value.lengths),
            "angles": np.asarray(value.angles),
        }
    if isinstance(value, Structure):
        return structure_to_record(value)
    missing = [field for field in RECORD_FIELDS if field not in value]
    if missing:
        raise ValueError(f"sample crystal is missing fields: {missing}")
    return {
        "frac_coords": np.asarray(value["frac_coords"]),
        "atom_types": np.asarray(value["atom_types"]),
        "lengths": np.asarray(value["lengths"]),
        "angles": np.asarray(value["angles"]),
    }


def _normalise_prediction_mapping(
    pred_crys: Mapping[int, Any] | Sequence[Any],
) -> dict[int, CrystalRecord]:
    if isinstance(pred_crys, Mapping):
        return {int(index): _as_record(value) for index, value in pred_crys.items()}
    return {index: _as_record(value) for index, value in enumerate(pred_crys)}


def _normalise_target_records(gt_crys: Sequence[Any]) -> list[CrystalRecord]:
    return [_as_record(value) for value in gt_crys]


def _evaluate_target(
    target_index: int,
    target_record: CrystalRecord,
    target_valid: bool,
    target_composition: tuple[tuple[int, ...], tuple[int, ...]],
    candidates: Sequence[tuple[int, int]],
    records: Mapping[int, CrystalRecord],
    matcher: StructureMatcher,
) -> dict[str, object]:
    best_rms: float | None = None
    best_candidate_index: int | None = None
    best_input_index: int | None = None
    valid_generated = 0
    errors = 0

    try:
        target_crystal = Crystal(target_record, compute_fp=False)
        target_valid = bool(target_valid and target_crystal.valid)
    except Exception:
        target_crystal = None
        target_valid = False

    for input_index, candidate_index in candidates:
        candidate_record = records.get(input_index)
        if candidate_record is None:
            errors += 1
            continue
        try:
            candidate_crystal = Crystal(candidate_record, compute_fp=False)
        except Exception:
            errors += 1
            continue

        if not candidate_crystal.valid:
            continue
        valid_generated += 1
        if (
            not target_valid
            or target_crystal is None
            or composition_key(candidate_crystal.structure) != target_composition
        ):
            continue

        try:
            rms = matcher.get_rms_dist(
                candidate_crystal.structure,
                target_crystal.structure,
            )
        except Exception:
            errors += 1
            continue
        if rms is None:
            continue
        rms_value = float(rms[0])
        if best_rms is None or rms_value < best_rms:
            best_rms = rms_value
            best_candidate_index = candidate_index
            best_input_index = input_index

    return {
        "target_index": target_index,
        "matched": best_rms is not None,
        "matched_candidate_index": best_candidate_index,
        "matched_input_index": best_input_index,
        "rms_dist": best_rms,
        "num_generated": len(candidates),
        "num_valid_generated": valid_generated,
        "num_errors": errors,
        "target_valid": target_valid,
    }


_WORKER_RECORDS: dict[int, CrystalRecord] = {}
_WORKER_MATCHER: StructureMatcher | None = None


def _initialize_rec_worker(
    matcher_kwargs: Mapping[str, object] | None = None,
) -> None:
    global _WORKER_MATCHER
    _WORKER_MATCHER = StructureMatcher(**(matcher_kwargs or MATCHER_KWARGS))


def _evaluate_rec_task(
    task: tuple[
        int,
        CrystalRecord,
        bool,
        tuple[tuple[int, ...], tuple[int, ...]],
        list[tuple[int, int]],
    ],
) -> dict[str, object]:
    if _WORKER_MATCHER is None:
        raise RuntimeError("reconstruction worker was not initialized")
    return _evaluate_target(*task, records=_WORKER_RECORDS, matcher=_WORKER_MATCHER)


class RecEval:
    """Evaluate CSP reconstruction candidates against ground-truth crystals.

    ``manifest`` is optional for compatibility with the original DiffCSP++
    one-prediction-per-target API.  When supplied, rows are grouped by target
    and the best matching candidate in each target group is retained.
    """

    def __init__(
        self,
        pred_crys: Mapping[int, Any] | Sequence[Any],
        gt_crys: Sequence[Any],
        manifest: pd.DataFrame | None = None,
        target_composition_valid: Sequence[bool] | None = None,
        workers: int = 1,
        show_progress: bool = True,
        ltol: float = MATCHER_KWARGS["ltol"],
        stol: float = MATCHER_KWARGS["stol"],
        angle_tol: float = MATCHER_KWARGS["angle_tol"],
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.preds = _normalise_prediction_mapping(pred_crys)
        self.gts = _normalise_target_records(gt_crys)
        if not self.gts:
            raise ValueError("at least one ground-truth crystal is required")
        if target_composition_valid is not None and len(target_composition_valid) != len(
            self.gts
        ):
            raise ValueError("target_composition_valid must match the target count")
        if manifest is None:
            if len(self.preds) != len(self.gts):
                raise ValueError(
                    "manifest is required when prediction and target counts differ"
                )
            input_indices = list(self.preds)
            manifest = pd.DataFrame(
                {
                    "target_index": list(range(len(self.gts))),
                    "input_index": input_indices,
                    "candidate_index": input_indices,
                }
            )

        self.manifest = manifest
        self.workers = workers
        self.show_progress = show_progress
        self.matcher_kwargs = {
            **MATCHER_KWARGS,
            "ltol": float(ltol),
            "stol": float(stol),
            "angle_tol": float(angle_tol),
        }
        self.matcher = StructureMatcher(**self.matcher_kwargs)
        self.details: pd.DataFrame | None = None
        self._metrics: dict[str, object] | None = None

        self._target_valid: list[bool] = []
        self._target_compositions: list[
            tuple[tuple[int, ...], tuple[int, ...]]
        ] = []
        for record in self.gts:
            try:
                crystal = Crystal(record, compute_fp=False)
                self._target_valid.append(bool(crystal.valid))
                self._target_compositions.append(composition_key(crystal.structure))
            except Exception:
                self._target_valid.append(False)
                self._target_compositions.append(((), ()))
        if target_composition_valid is not None:
            self._target_valid = [
                valid and bool(composition_valid)
                for valid, composition_valid in zip(
                    self._target_valid, target_composition_valid
                )
            ]

    def _tasks(self) -> list[
        tuple[
            int,
            CrystalRecord,
            bool,
            tuple[tuple[int, ...], tuple[int, ...]],
            list[tuple[int, int]],
        ]
    ]:
        target_column, sample_column, candidate_column = manifest_columns(self.manifest)
        groups: dict[int, list[tuple[int, int]]] = {
            target_index: [] for target_index in range(len(self.gts))
        }
        for row in self.manifest.itertuples(index=False):
            target_index = int(getattr(row, target_column))
            if target_index < 0 or target_index >= len(self.gts):
                raise ValueError(
                    f"Manifest {target_column} contains target index "
                    f"outside the target count: {target_index}"
                )
            groups[target_index].append(
                (
                    int(getattr(row, sample_column)),
                    int(getattr(row, candidate_column)),
                )
            )

        return [
            (
                target_index,
                self.gts[target_index],
                self._target_valid[target_index],
                self._target_compositions[target_index],
                groups[target_index],
            )
            for target_index in range(len(self.gts))
        ]

    def get_match_rate_and_rms(self) -> dict[str, object]:
        if self._metrics is not None:
            return {
                "match_rate": self._metrics["match_rate"],
                "rms_dist": self._metrics["rms_dist"],
            }

        tasks = self._tasks()
        if self.workers == 1:
            rows = [
                _evaluate_target(
                    *task,
                    records=self.preds,
                    matcher=self.matcher,
                )
                for task in tqdm(
                    tasks,
                    total=len(tasks),
                    desc="Evaluating structures",
                    unit="target",
                    disable=not self.show_progress,
                )
            ]
        else:
            context = multiprocessing.get_context("fork")
            # The prediction arrays can be large.  With fork the workers inherit
            # this read-only mapping instead of receiving a pickled copy in
            # every process.
            global _WORKER_RECORDS
            _WORKER_RECORDS = self.preds
            with ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=context,
                initializer=_initialize_rec_worker,
                initargs=(self.matcher_kwargs,),
            ) as executor:
                results = executor.map(_evaluate_rec_task, tasks, chunksize=1)
                rows = list(
                    tqdm(
                        results,
                        total=len(tasks),
                        desc="Evaluating structures",
                        unit="target",
                        disable=not self.show_progress,
                    )
                )

        self.details = pd.DataFrame(rows)
        matched = self.details["matched"].astype(bool)
        matched_rms = self.details.loc[matched, "rms_dist"].to_numpy(dtype=float)
        num_targets = len(self.gts)
        self._metrics = {
            "match_rate": float(matched.mean()),
            "rms_dist": float(matched_rms.mean()) if len(matched_rms) else None,
            "num_targets": num_targets,
            "num_matched": int(matched.sum()),
            "num_generated": int(self.details["num_generated"].sum()),
            "num_valid_generated": int(
                self.details["num_valid_generated"].sum()
            ),
            "workers": self.workers,
            "matcher": self.matcher_kwargs.copy(),
            "validity": "embedded DiffCSP Crystal",
        }
        return {
            "match_rate": self._metrics["match_rate"],
            "rms_dist": self._metrics["rms_dist"],
        }

    def get_metrics(self) -> dict[str, object]:
        self.get_match_rate_and_rms()
        assert self._metrics is not None
        return self._metrics.copy()


def write_metrics(
    output_path: Path | str,
    details_path: Path | str,
    metrics: Mapping[str, object],
    details: pd.DataFrame,
) -> None:
    """Write the standard JSON metrics and per-target details files."""
    output_path = Path(output_path)
    details_path = Path(details_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    details.to_csv(details_path, index=False)
