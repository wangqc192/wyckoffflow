"""Compatibility entry point for DiffCSP++ CSP reconstruction metrics.

The implementation lives in ``compute_metrics_diffcsppp.py``.  CrystalFlow
uses the separate ``compute_metrics_crystalflow.py`` entry point because its PT
payload has a different layout.
"""

from __future__ import annotations

try:
    from .compute_metrics_diffcsppp import (
        compute_metrics,
        load_diffcsppp_records,
        main,
    )
    from .eval_utils import (
        MATCHER_KWARGS,
        Crystal,
        RecEval,
        composition_key,
        manifest_columns,
        structure_to_crystal,
        structure_validity,
    )
except ImportError:  # direct ``python scripts/compute_metrics.py``
    from compute_metrics_diffcsppp import (  # type: ignore[no-redef]
        compute_metrics,
        load_diffcsppp_records,
        main,
    )
    from eval_utils import (  # type: ignore[no-redef]
        MATCHER_KWARGS,
        Crystal,
        RecEval,
        composition_key,
        manifest_columns,
        structure_to_crystal,
        structure_validity,
    )

# Kept for callers of the previous PT-loader name.
load_pt_structures = load_diffcsppp_records
structure_is_valid = structure_validity

__all__ = [
    "MATCHER_KWARGS",
    "Crystal",
    "RecEval",
    "composition_key",
    "compute_metrics",
    "load_diffcsppp_records",
    "load_pt_structures",
    "main",
    "manifest_columns",
    "structure_to_crystal",
    "structure_validity",
    "structure_is_valid",
]


if __name__ == "__main__":
    try:
        from .compute_metrics_diffcsppp import cli
    except ImportError:
        from compute_metrics_diffcsppp import cli

    cli()
