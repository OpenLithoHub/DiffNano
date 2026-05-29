"""Process-variation-robust differentiable optimization (C5 mechanism).

This module provides:
- C5.1: Reparameterization-trick sampling of process variations
- C5.2: Distance-field perturbation kernel (differentiable level-set shift)
- C5.3: Relaxed-Heaviside boundary perturbation
- C5.4: Variance-reduced robust gradient (antithetic sampling)

The simplified v0.1 scope supports linewidth ±5 nm via differentiable
distance-field shift of the level set, with K=4–8 Monte Carlo samples
per gradient step.

Tier 3 module (release after CN priority confirmation).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = [
    "reparameterize_sample",
    "linewidth_perturbation",
    "robust_gradient_step",
    "antithetic_sampler",
]


# ---------------------------------------------------------------------------
# C5.1: Reparameterization-trick sampling
# ---------------------------------------------------------------------------


def reparameterize_sample(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    n_samples: int = 8,
    base_distribution: str = "normal",
) -> torch.Tensor:
    """Sample perturbation deltas via the reparameterization trick.

    δ = μ + σ · ε,  ε ~ N(0, I)

    The gradient flows through both the FoM and the perturbation
    distribution parameters (μ, σ).

    Parameters
    ----------
    mu : Tensor
        Mean of the perturbation distribution.
    sigma : Tensor
        Standard deviation of the perturbation distribution.
    n_samples : int
        Number of Monte Carlo samples (K).
    base_distribution : str
        Base distribution type ("normal" only for now).

    Returns
    -------
    deltas : Tensor, shape ``(n_samples, *mu.shape)``
        Perturbation samples.
    """
    device = mu.device
    dtype = mu.dtype
    sigma = sigma.to(device=device, dtype=dtype)
    eps = torch.randn(n_samples, *mu.shape, device=device, dtype=dtype)
    return mu + sigma * eps


# ---------------------------------------------------------------------------
# C5.2: Distance-field perturbation kernel
# ---------------------------------------------------------------------------


def linewidth_perturbation(
    sdf: torch.Tensor,
    delta_nm: torch.Tensor,
    pixel_size_nm: float = 5.0,
) -> torch.Tensor:
    """Perturb geometry via differentiable shift of the SDF level set.

    This implements T(θ, δ) as a shift of the zero-crossing of the
    signed distance field by δ_nm nanometers, which is equivalent to
    changing the linewidth by δ_nm.

    Parameters
    ----------
    sdf : Tensor, shape ``(H, W)``
        Signed distance field of the geometry.
    delta_nm : Tensor, scalar or shape ``(H, W)``
        Linewidth perturbation in nanometers.
    pixel_size_nm : float
        Physical size of one pixel.

    Returns
    -------
    perturbed_sdf : Tensor, shape ``(H, W)``
        Perturbed signed distance field.
    """
    delta_pixels = delta_nm / pixel_size_nm
    return sdf - delta_pixels


def apply_perturbation_to_density(
    density: torch.Tensor,
    delta_nm: torch.Tensor,
    beta: float = 10.0,
    pixel_size_nm: float = 5.0,
) -> torch.Tensor:
    """Apply linewidth perturbation directly to a density field.

    For density-parameterized geometries (no explicit SDF), the perturbation
    is applied as a morphological erosion/dilation via the perturbed threshold.

    Parameters
    ----------
    density : Tensor, shape ``(H, W)``
        Original density field.
    delta_nm : Tensor, scalar
        Linewidth perturbation in nanometers (positive = wider features).
    beta : float
        Sigmoid sharpness for re-binarization.
    pixel_size_nm : float
        Pixel size in nm.

    Returns
    -------
    perturbed_density : Tensor, shape ``(H, W)``
    """
    # Approximate SDF from density field (threshold-based)
    # A true SDF requires a contour; here we use a simplified version
    # where we shift the effective threshold
    delta_norm = delta_nm / (pixel_size_nm * 10)  # normalize
    perturbed = torch.sigmoid(beta * (density - 0.5 + delta_norm))
    return perturbed


# ---------------------------------------------------------------------------
# C5.3: Relaxed Heaviside boundary perturbation
# ---------------------------------------------------------------------------


def relaxed_heaviside_perturbation(
    sdf: torch.Tensor,
    delta_nm: torch.Tensor,
    beta: float = 10.0,
    pixel_size_nm: float = 5.0,
) -> torch.Tensor:
    """Perturb binary boundaries via relaxed Heaviside (sigmoid).

    C5.3 mechanism: binary boundaries are smoothed via sigmoid with
    steepness β, so boundary perturbations admit continuous gradients.

    Parameters
    ----------
    sdf : Tensor, shape ``(H, W)``
        Signed distance field.
    delta_nm : Tensor, scalar
        Perturbation amount in nm.
    beta : float
        Sigmoid steepness (higher = sharper boundary).
    pixel_size_nm : float
        Pixel size in nm.

    Returns
    -------
    perturbed_mask : Tensor, shape ``(H, W)``
    """
    shifted_sdf = linewidth_perturbation(sdf, delta_nm, pixel_size_nm)
    return torch.sigmoid(-beta * shifted_sdf)


# ---------------------------------------------------------------------------
# C5.4: Variance-reduced robust gradient (antithetic sampling)
# ---------------------------------------------------------------------------


def antithetic_sampler(
    sigma: float | torch.Tensor,
    shape: tuple[int, ...] | torch.Size,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate antithetic (paired) samples for variance reduction.

    Returns 2K samples where samples[i] and samples[i+K] are antithetic
    pairs (ε and -ε), reducing the variance of the Monte Carlo estimator
    by a factor of ~2 for symmetric distributions.

    Parameters
    ----------
    sigma : float or Tensor
        Standard deviation of the perturbation.
    shape : tuple
        Shape of each sample.
    device : device
    dtype : dtype

    Returns
    -------
    samples : Tensor, shape ``(2K, *shape)``
    """
    K = 4  # base number of pairs
    eps = torch.randn(K, *shape, device=device, dtype=dtype)
    return torch.cat([eps, -eps], dim=0) * sigma


# ---------------------------------------------------------------------------
# Robust optimization loop
# ---------------------------------------------------------------------------


def robust_gradient_step(
    params: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma_nm: float = 5.0,
    n_samples: int = 8,
    perturbation_fn: Callable | None = None,
    antithetic: bool = True,
) -> torch.Tensor:
    """Compute the robust gradient estimate via differentiable Monte Carlo.

    Evaluates 𝔼_{δ ~ p(δ)} [FoM(T(θ, δ))] using the reparameterization
    trick and returns the gradient-ready loss (call .backward() on it).

    Parameters
    ----------
    params : Tensor
        Current design parameters (θ).
    forward_fn : callable
        ``forward_fn(perturbed_params) -> loss`` — computes the figure of
        merit loss from the (possibly perturbed) parameters.
    sigma_nm : float
        Standard deviation of the linewidth perturbation in nm.
    n_samples : int
        Number of Monte Carlo samples (K).  If *antithetic* is True,
        ``n_samples`` should be even and will be split into pairs.
    perturbation_fn : callable, optional
        ``perturbation_fn(params, delta) -> perturbed_params``.  Defaults
        to additive perturbation.
    antithetic : bool
        Use antithetic sampling for variance reduction.

    Returns
    -------
    robust_loss : Tensor, scalar
        Mean loss over perturbation samples.  Call ``.backward()`` to get
        the robust gradient w.r.t. *params*.
    """
    if perturbation_fn is None:
        perturbation_fn = _default_perturbation

    losses = []
    half = n_samples // 2
    saved_eps = []

    for i in range(n_samples):
        # Per-element noise for spatial parameters
        if antithetic and n_samples % 2 == 0:
            if i < half:
                eps = torch.randn_like(params)
                saved_eps.append(eps)
            else:
                eps = -saved_eps[i - half]  # antithetic pair
        else:
            eps = torch.randn_like(params)

        perturbed = perturbation_fn(params, eps * sigma_nm)
        losses.append(forward_fn(perturbed))

    return torch.stack(losses).mean()


def _default_perturbation(params: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Default additive perturbation."""
    return params + delta
