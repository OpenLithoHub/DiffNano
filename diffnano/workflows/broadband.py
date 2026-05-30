"""Broadband multi-wavelength optimization workflow.

Optimizes a nanophotonic device across multiple wavelengths by minimizing
a weighted sum of objectives evaluated at each wavelength via RCWA.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = ["BroadbandOptimizer"]


class BroadbandOptimizer:
    """Multi-wavelength optimizer using weighted RCWA evaluation.

    Parameters
    ----------
    solver
        A Solver implementing the forward method (typically RCWASolver).
    wavelengths_nm : list of float
        Wavelengths to optimize over.
    weights : list of float, optional
        Weight per wavelength (uniform if None).
    grid_shape : tuple[int, int]
        Density field shape.
    n_layers : int
        Number of device layers.
    device : str or torch.device
    """

    def __init__(
        self,
        solver,
        wavelengths_nm: Sequence[float] = (500.0, 532.0, 600.0),
        weights: Sequence[float] | None = None,
        grid_shape: tuple[int, int] = (32, 32),
        n_layers: int = 5,
        eps_low: float = 1.0,
        eps_high: float = 12.0,
        device: str | torch.device = "cpu",
    ):
        self.solver = solver
        self.wavelengths = list(wavelengths_nm)
        self.n_wl = len(self.wavelengths)
        self.weights = (
            torch.tensor(weights, dtype=torch.float64, device=device)
            if weights is not None
            else torch.ones(self.n_wl, dtype=torch.float64, device=device) / self.n_wl
        )
        self.grid_shape = grid_shape
        self.n_layers = n_layers
        self.eps_low = eps_low
        self.eps_high = eps_high
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    def objective(
        self,
        density: torch.Tensor,
        target_order: int = 0,
    ) -> torch.Tensor:
        """Compute weighted multi-wavelength objective.

        Minimizes the negative weighted sum of diffraction efficiencies
        at the target order across all wavelengths.

        Parameters
        ----------
        density : Tensor, shape ``(H, W)``
            Design density field.
        target_order : int
            Diffraction order to maximize.

        Returns
        -------
        loss : Tensor, scalar
        """
        # Convert 2D density to layered geometry
        H, W = density.shape
        layer_thickness = H // self.n_layers
        remainder = H % self.n_layers
        if layer_thickness == 0:
            layer_thickness = 1

        layers = []
        y0 = 0
        for i in range(self.n_layers):
            extra = 1 if i < remainder else 0
            y1 = min(y0 + layer_thickness + extra, H)
            avg_density = density[y0:y1, :].mean(dim=0)
            layers.append(self.eps_low + (self.eps_high - self.eps_low) * avg_density)
            y0 = y1

        geometry = torch.stack(layers)

        result = self.solver.forward(geometry, wavelengths=self.wavelengths)

        if hasattr(self.solver, "diffraction_efficiency"):
            efficiencies = self.solver.diffraction_efficiency(
                geometry,
                wavelengths=self.wavelengths,
                order=target_order,
            )
        else:
            idx = target_order + self.solver.fourier_orders
            idx = max(0, min(idx, result.field.shape[-1] - 1))
            efficiencies = result.field[:, idx]

        # Weighted negative efficiency (minimize = maximize efficiency)
        loss = -(self.weights * efficiencies).sum()

        return loss

    def optimize(
        self,
        n_steps: int = 200,
        lr: float = 0.01,
        target_order: int = 0,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """Run broadband optimization.

        Parameters
        ----------
        n_steps : int
        lr : float
        target_order : int
        verbose : bool

        Returns
        -------
        density : Tensor, shape ``(H, W)``
        loss_history : list of float
        """
        from diffnano.design.projection import (
            beta_continuation_schedule,
            heaviside_projection,
        )

        density = torch.rand(
            *self.grid_shape,
            device=self._device,
            dtype=torch.float64,
        )
        density = density.detach().requires_grad_(True)

        opt = torch.optim.Adam([density], lr=lr)
        loss_history = []

        for step in range(n_steps):
            beta = beta_continuation_schedule(step, n_steps, beta_start=1.0, beta_end=32.0)
            projected = heaviside_projection(density, beta=beta)

            loss = self.objective(projected, target_order=target_order)

            opt.zero_grad()
            loss.backward()

            if density.grad is not None and torch.isnan(density.grad).any():
                if verbose:
                    print(f"Step {step}: NaN gradient, stopping.")
                break

            opt.step()
            with torch.no_grad():
                density.clamp_(0.0, 1.0)

            loss_history.append(loss.item())

            if verbose and step % 20 == 0:
                print(f"Step {step:4d}: loss={loss.item():.6f}")

        return density.detach(), loss_history
