"""Adaptive robust optimization (C7 — inspired by BOSON-1).

Provides:
- Axial sampling: O(2N+1) corner samples instead of exhaustive O(3^N)
- Adaptive refinement: focus samples on worst-case corners
- Curriculum: start axial (cheap), add random samples as optimization converges
- Fabricable subspace projection: discretize continuous density to fabricable geometry

Integrates with ``diff_surrogate.adaptive_corner.AdaptiveMultiCornerEvaluator``
for uncertainty-based corner weighting and adaptive corner skipping.

References
----------
- Ma et al. (2024), BOSON-1: arXiv:2411.08210 (adaptive sampling, fabricable subspace)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from diff_surrogate.adaptive_corner import AdaptiveMultiCornerEvaluator as _DSAdaptiveEvaluator
from diff_surrogate.robust_design import CornerSpec as _DSCornerSpec

__all__ = [
    "AdaptiveRobustOptimizer",
    "FabricableSubspaceProjection",
    "axial_samples",
    "correlated_perturbation",
]


def axial_samples(
    n_dims: int,
    sigma: float | torch.Tensor,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate 2N+1 axial samples for N variation sources.

    Samples: nominal (origin) + 2N axial points at ±σ along each axis.

    Parameters
    ----------
    n_dims : int
        Number of variation dimensions (N).
    sigma : float or Tensor
        Perturbation magnitude per axis.
    device, dtype

    Returns
    -------
    samples : Tensor, shape ``(2N+1, N)``
    """
    nominal = torch.zeros(1, n_dims, device=device, dtype=dtype)
    axial = []
    for i in range(n_dims):
        pos = torch.zeros(1, n_dims, device=device, dtype=dtype)
        neg = torch.zeros(1, n_dims, device=device, dtype=dtype)
        pos[0, i] = sigma
        neg[0, i] = -sigma
        axial.append(pos)
        axial.append(neg)
    return torch.cat([nominal] + axial, dim=0)


def correlated_perturbation(
    params: torch.Tensor,
    cov_cholesky: torch.Tensor,
    n_samples: int = 8,
) -> torch.Tensor:
    """Sample correlated multi-axis perturbations.

    Uses Cholesky decomposition of the joint covariance matrix:
        δ = L · ε,  ε ~ N(0, I)

    Parameters
    ----------
    params : Tensor, shape ``(*param_shape)``
        Design parameters (used for shape/device/dtype).
    cov_cholesky : Tensor, shape ``(N, N)``
        Lower-triangular Cholesky factor of the covariance matrix.
    n_samples : int

    Returns
    -------
    deltas : Tensor, shape ``(n_samples, N)``
        Correlated perturbation samples.
    """
    N = cov_cholesky.shape[0]
    eps = torch.randn(n_samples, N, device=params.device, dtype=params.dtype)
    return eps @ cov_cholesky.T


class FabricableSubspaceProjection:
    """Project continuous density field to nearest fabricable geometry.

    Uses Gumbel-softmax relaxation for differentiable projection to
    discrete height levels, followed by minimum-CD enforcement via
    erosion-dilation.

    Parameters
    ----------
    n_levels : int
        Number of discretized height levels.
    min_cd_pixels : int
        Minimum critical dimension in pixels.
    temperature : float
        Gumbel-softmax temperature (lower = harder projection).
    """

    def __init__(
        self,
        n_levels: int = 4,
        min_cd_pixels: int = 2,
        temperature: float = 1.0,
    ):
        self.n_levels = n_levels
        self.min_cd_pixels = min_cd_pixels
        self.temperature = temperature

    def project(self, density: torch.Tensor) -> torch.Tensor:
        """Project density to fabricable subspace (differentiable).

        Parameters
        ----------
        density : Tensor, shape ``(H, W)``
            Continuous density field ∈ [0, 1].

        Returns
        -------
        projected : Tensor, shape ``(H, W)``
            Projected density (approximately discrete).
        """
        levels = torch.linspace(0, 1, self.n_levels, device=density.device, dtype=density.dtype)

        # Compute distances to each level
        distances = torch.abs(density.unsqueeze(-1) - levels.unsqueeze(0).unsqueeze(0))
        # Gumbel-softmax style soft assignment
        weights = torch.softmax(-distances / max(self.temperature, 0.01), dim=-1)
        projected = (weights * levels).sum(dim=-1)

        # Minimum CD enforcement via morphological opening (erosion + dilation)
        if self.min_cd_pixels > 0:
            projected = _morphological_opening(projected, self.min_cd_pixels)

        return projected

    def projection_loss(self, density: torch.Tensor) -> torch.Tensor:
        """Penalty for distance from fabricable subspace.

        Encourages the density to stay near discrete levels.
        """
        projected = self.project(density)
        return ((density - projected) ** 2).mean()


def _morphological_opening(density: torch.Tensor, radius: int) -> torch.Tensor:
    """Differentiable approximation of morphological opening.

    Erosion followed by dilation using min/max pooling with smoothing.
    """
    if radius <= 0:
        return density

    kernel_size = 2 * radius + 1

    # Pad the input so output size matches input
    # Identity element for erosion (local min) of [0,1] density is 1.0
    padded = torch.nn.functional.pad(
        density.unsqueeze(0).unsqueeze(0),
        [radius] * 4,
        mode="constant",
        value=1.0,
    )

    # Approximate erosion via local min (negative max_pool)
    eroded = -torch.nn.functional.max_pool2d(
        -padded,
        kernel_size,
        stride=1,
        padding=0,
    )

    # Re-pad for dilation
    eroded_padded = torch.nn.functional.pad(
        eroded,
        [radius] * 4,
        mode="constant",
        value=0.0,
    )

    # Approximate dilation via local max
    dilated = torch.nn.functional.max_pool2d(
        eroded_padded,
        kernel_size,
        stride=1,
        padding=0,
    )

    return dilated.squeeze(0).squeeze(0)


class AdaptiveRobustOptimizer:
    """Adaptive robust optimizer using axial sampling with curriculum.

    Combines O(2N+1) axial sampling with adaptive worst-case refinement
    and progressive random sampling for capturing interaction effects.

    Parameters
    ----------
    n_variation_dims : int
        Number of variation sources (N).
    sigma : float
        Perturbation magnitude (standard deviation per axis).
    cov_matrix : Tensor, shape ``(N, N)``, optional
        Covariance matrix for correlated perturbations. Identity if None.
    n_random_budget : int
        Additional random samples to add in curriculum phase.
    refinement_top_k : int
        Number of worst-case samples to refine around.
    device : str or torch.device
    """

    def __init__(
        self,
        n_variation_dims: int = 3,
        sigma: float = 5.0,
        cov_matrix: torch.Tensor | None = None,
        n_random_budget: int = 16,
        refinement_top_k: int = 3,
        device: str | torch.device = "cpu",
        corner_evaluator: Any | None = None,
    ):
        self.n_dims = n_variation_dims
        self.sigma = sigma
        self.n_random_budget = n_random_budget
        self.refinement_top_k = refinement_top_k
        self._device = torch.device(device)

        self._corner_evaluator = corner_evaluator
        if corner_evaluator is not None:
            if not isinstance(corner_evaluator, _DSAdaptiveEvaluator):
                raise TypeError(
                    "corner_evaluator must be an AdaptiveMultiCornerEvaluator "
                    "from diff_surrogate"
                )

        if cov_matrix is not None:
            self.cov_chol = torch.linalg.cholesky(cov_matrix)
        else:
            self.cov_chol = (
                torch.eye(
                    n_variation_dims,
                    device=self._device,
                    dtype=torch.float64,
                )
                * sigma
            )

    def compute_robust_loss(
        self,
        params: torch.Tensor,
        forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        perturbation_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        curriculum_frac: float = 0.0,
    ) -> torch.Tensor:
        """Compute adaptive robust loss estimate.

        Parameters
        ----------
        params : Tensor
            Design parameters θ.
        forward_fn : callable
            ``forward_fn(params, perturbation_delta) -> loss``.

            Note: forward_fn signature differs from robust_gradient_step. Here it takes
            (params, perturbation_delta) rather than (perturbed_params).
        perturbation_fn : callable
            ``perturbation_fn(params, delta) -> perturbed_params``.
        curriculum_frac : float
            Fraction of random samples to add (0 = axial only, 1 = full).

        Returns
        -------
        robust_loss : Tensor, scalar
        """
        # Phase 1: Axial samples
        axial = axial_samples(
            self.n_dims,
            self.sigma,
            device=self._device,
            dtype=params.dtype,
        )
        # Phase 2: Random samples based on curriculum
        n_random = int(self.n_random_budget * curriculum_frac)
        if n_random > 0:
            eps = torch.randn(n_random, self.n_dims, device=self._device, dtype=params.dtype)
            random_samples = eps @ self.cov_chol.T
            all_samples = torch.cat([axial, random_samples], dim=0)
        else:
            all_samples = axial

        # Evaluate loss at all samples
        losses = []
        for i in range(all_samples.shape[0]):
            delta = all_samples[i]
            perturbed = perturbation_fn(params, delta)
            losses.append(forward_fn(perturbed, delta))

        loss_stack = torch.stack(losses)

        # Adaptive weighting: emphasize worst-case samples
        with torch.no_grad():
            sorted_indices = torch.argsort(loss_stack, descending=True)
            top_k = min(self.refinement_top_k, loss_stack.shape[0])

        # Weighted combination: uniform + extra weight on worst cases
        uniform_loss = loss_stack.mean()
        worst_loss = loss_stack[sorted_indices[:top_k]].mean()

        return 0.7 * uniform_loss + 0.3 * worst_loss

    def compute_robust_loss_with_corners(
        self,
        params: torch.Tensor,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        """Compute robust loss using diff-surrogate's AdaptiveMultiCornerEvaluator.

        This is an alternative to :meth:`compute_robust_loss` that delegates
        multi-corner evaluation to the optionally-provided corner evaluator
        (from ``diff_surrogate``).  When no evaluator is configured or when
        ``diff-surrogate`` is not installed, falls back to a simple
        single-corner evaluation.

        Parameters
        ----------
        params:
            Design parameters.
        forward_fn:
            ``forward_fn(design) -> Tensor``.
        loss_fn:
            ``loss_fn(output) -> Tensor`` (scalar).

        Returns
        -------
        loss : Tensor
        info : dict
        """
        if self._corner_evaluator is not None:
            return self._corner_evaluator.evaluate(params, forward_fn, loss_fn)

        output = forward_fn(params)
        return loss_fn(output), {"per_corner_loss": [loss_fn(output).item()],
                                  "weights": [1.0], "uncertainties": [0.0],
                                  "skipped": []}

    def optimize(
        self,
        params: torch.Tensor,
        forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        perturbation_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_steps: int = 200,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """Run adaptive robust optimization.

        Parameters
        ----------
        params : Tensor
            Initial design parameters.
        forward_fn : callable
        perturbation_fn : callable
        n_steps : int
        lr : float
        verbose : bool

        Returns
        -------
        params : Tensor
            Optimized parameters.
        loss_history : list of float
        """
        params = params.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([params], lr=lr)
        loss_history = []

        for step in range(n_steps):
            curriculum_frac = min(1.0, step / n_steps)

            loss = self.compute_robust_loss(
                params,
                forward_fn,
                perturbation_fn,
                curriculum_frac=curriculum_frac,
            )

            opt.zero_grad()
            loss.backward()

            if params.grad is not None and torch.isnan(params.grad).any():
                if verbose:
                    print(f"Step {step}: NaN gradient, stopping.")
                break

            opt.step()
            loss_history.append(loss.item())

            if verbose and step % 20 == 0:
                print(f"Step {step:4d}: loss={loss.item():.6f}")

        return params.detach(), loss_history
