"""Process-variation-robust optimization (C5 + C7 mechanisms)."""

from diffnano.design.robustness.core import (
    antithetic_sampler,
    apply_perturbation_to_density,
    linewidth_perturbation,
    relaxed_heaviside_perturbation,
    reparameterize_sample,
    robust_gradient_step,
)

__all__ = [
    "reparameterize_sample",
    "linewidth_perturbation",
    "apply_perturbation_to_density",
    "relaxed_heaviside_perturbation",
    "antithetic_sampler",
    "robust_gradient_step",
    "AdaptiveRobustOptimizer",
    "FabricableSubspaceProjection",
    "axial_samples",
    "correlated_perturbation",
    "MultiAxisPerturbation",
    "sidewall_angle_perturbation",
    "thickness_perturbation",
    "corner_rounding_perturbation",
    "CornerSpec",
    "corner_optimization_step",
    "DEFAULT_CORNERS",
]


def __getattr__(name):
    if name in (
        "AdaptiveRobustOptimizer",
        "FabricableSubspaceProjection",
        "axial_samples",
        "correlated_perturbation",
    ):
        from diffnano.design.robustness import adaptive

        return getattr(adaptive, name)
    if name in (
        "MultiAxisPerturbation",
        "sidewall_angle_perturbation",
        "thickness_perturbation",
        "corner_rounding_perturbation",
    ):
        from diffnano.design.robustness import subspace

        return getattr(subspace, name)
    if name in (
        "CornerSpec",
        "corner_optimization_step",
        "DEFAULT_CORNERS",
    ):
        from diffnano.design.robustness import corner_opt

        return getattr(corner_opt, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
