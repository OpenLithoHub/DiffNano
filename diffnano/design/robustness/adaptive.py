"""Adaptive robust optimization -- re-exported from diff_surrogate (canonical).

The canonical implementation lives in ``diff_surrogate.adaptive_robust``.
This module re-exports the public API so that existing DiffNano code
continues to work unchanged.
"""

from diff_surrogate.adaptive_robust import (  # noqa: F401
    AdaptiveRobustOptimizer,
    FabricableSubspaceProjection,
    axial_samples,
    correlated_perturbation,
)

__all__ = [
    "AdaptiveRobustOptimizer",
    "FabricableSubspaceProjection",
    "axial_samples",
    "correlated_perturbation",
]
