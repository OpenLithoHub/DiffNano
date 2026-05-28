"""Process-variation-robust optimization (C5 mechanism)."""

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
]
