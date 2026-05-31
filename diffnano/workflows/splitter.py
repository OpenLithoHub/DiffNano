"""Beam splitter / power divider inverse design workflow.

Uses RCWA to compute diffraction efficiencies for a periodic grating that
splits an incident beam into target diffraction orders (typically +1 and -1).
The forward model returns the splitting ratio, insertion loss, and per-order
S-parameters (transmission efficiencies).
"""

from __future__ import annotations

import torch

from diffnano.solvers.rcwa import RCWASolver

__all__ = ["SplitterDesigner"]


class SplitterDesigner:
    """Beam splitter inverse design workflow.

    Wraps an RCWASolver to evaluate a beam splitter grating geometry.
    The grating period is set so that the +/- first diffraction orders
    are the primary output channels.

    Parameters
    ----------
    wavelength_nm : float
        Operating wavelength in nm.
    period_nm : float
        Grating period in nm.  Should be ``> wavelength`` to open +/- 1
        diffraction orders but small enough to suppress higher orders.
    n_fourier_orders : int
        Fourier orders retained on each side (total = 2 * n + 1).
    n_grid : int
        Number of real-space grid points per period (controls Toeplitz resolution).
    eps_low : float
        Permittivity of the low-index material (e.g. air).
    eps_high : float
        Permittivity of the high-index material (e.g. silicon).
    thickness_nm : float
        Total grating thickness in nm (distributed evenly across layers).
    n_layers : int
        Number of grating layers for the RCWA model.
    device : str
    """

    def __init__(
        self,
        wavelength_nm: float = 1550.0,
        period_nm: float = 2200.0,
        n_fourier_orders: int = 5,
        n_grid: int = 64,
        eps_low: float = 1.0,
        eps_high: float = 12.0,
        thickness_nm: float = 500.0,
        n_layers: int = 4,
        device: str = "cpu",
    ):
        self.wavelength_nm = wavelength_nm
        self.period_nm = period_nm
        self.n_fourier_orders = n_fourier_orders
        self.n_grid = n_grid
        self.eps_low = eps_low
        self.eps_high = eps_high
        self.thickness_nm = thickness_nm
        self.n_layers = n_layers
        self._device = torch.device(device)

        self.solver = RCWASolver(
            fourier_orders=n_fourier_orders,
            wavelength_nm=wavelength_nm,
            period_nm=(period_nm, period_nm),
            eps_ambient=1.0,
            eps_substrate=1.0,
            device=device,
            solver_backend="eig",
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def density_to_eps_layers(self, density: torch.Tensor) -> torch.Tensor:
        """Convert a 1D density profile to layered permittivity.

        Parameters
        ----------
        density : Tensor, shape ``(n_grid,)``
            Material density in [0, 1] (0 = eps_low, 1 = eps_high).

        Returns
        -------
        eps_layers : Tensor, shape ``(n_layers, n_grid)``
        """
        eps = self.eps_low + (self.eps_high - self.eps_low) * density
        return eps.unsqueeze(0).expand(self.n_layers, -1).clone()

    def transmission_efficiency(
        self,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total transmission efficiency of the beam splitter.

        Runs RCWA and returns the sum of all transmitted diffraction order
        efficiencies (normalized to incident power).

        Parameters
        ----------
        geometry : Tensor
            Either:
            - 1D ``(n_grid,)`` density profile -> converted internally
            - 2D ``(n_layers, n_grid)`` permittivity profiles -> used directly

        Returns
        -------
        efficiency : Tensor, scalar
            Total transmitted power fraction (sum across all orders).
        """
        if geometry.dim() == 1:
            eps_layers = self.density_to_eps_layers(geometry)
        else:
            eps_layers = geometry

        result = self.solver.forward(
            eps_layers,
            wavelengths=[self.wavelength_nm],
            source={
                "theta": 0.0,
                "polarization": "TE",
                "thickness_nm": self.thickness_nm / self.n_layers,
            },
        )
        return result.field.sum()

    def s_parameters(
        self,
        geometry: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute per-order S-parameters (transmission efficiencies).

        Parameters
        ----------
        geometry : Tensor
            1D density profile ``(n_grid,)`` or 2D permittivity ``(n_layers, n_grid)``.

        Returns
        -------
        result : dict
            ``"efficiencies"``: Tensor, shape ``(n_fourier_total,)`` --
            normalized diffraction efficiency per Fourier order.
            ``"splitting_ratio"``: Tensor, scalar --
            min(T_{+1}, T_{-1}) / max(T_{+1}, T_{-1}), ranges [0, 1].
            ``"insertion_loss"``: Tensor, scalar --
            1 - sum of all transmitted efficiencies.
            ``"T_plus1"``: Tensor, scalar -- efficiency in +1 order.
            ``"T_minus1"``: Tensor, scalar -- efficiency in -1 order.
        """
        if geometry.dim() == 1:
            eps_layers = self.density_to_eps_layers(geometry)
        else:
            eps_layers = geometry

        result = self.solver.forward(
            eps_layers,
            wavelengths=[self.wavelength_nm],
            source={
                "theta": 0.0,
                "polarization": "TE",
                "thickness_nm": self.thickness_nm / self.n_layers,
            },
        )

        eff = result.field.squeeze(0)  # (n_fourier_total,)
        n = eff.shape[0]
        center = n // 2

        T_plus1 = eff[center + 1] if center + 1 < n else torch.tensor(0.0, device=self._device)
        T_minus1 = eff[center - 1] if center - 1 >= 0 else torch.tensor(0.0, device=self._device)

        total = eff.sum()
        max_T = torch.max(T_plus1, T_minus1).clamp(min=1e-12)
        splitting_ratio = torch.min(T_plus1, T_minus1) / max_T
        insertion_loss = 1.0 - total

        return {
            "efficiencies": eff,
            "splitting_ratio": splitting_ratio,
            "insertion_loss": insertion_loss,
            "T_plus1": T_plus1,
            "T_minus1": T_minus1,
        }

    def objective(
        self,
        density: torch.Tensor,
        target_orders: tuple[int, ...] = (-1, 1),
    ) -> torch.Tensor:
        """Inverse design objective: maximize target order power, minimize others.

        Parameters
        ----------
        density : Tensor, shape ``(n_grid,)``
            Material density field.
        target_orders : tuple of int
            Diffraction orders to maximize (default: -1 and +1).

        Returns
        -------
        loss : Tensor, scalar
            Negative total efficiency in target orders (minimize to maximize).
        """
        s = self.s_parameters(density)
        eff = s["efficiencies"]
        n = eff.shape[0]
        center = n // 2

        target_eff = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        for order in target_orders:
            idx = center + order
            if 0 <= idx < n:
                target_eff = target_eff + eff[idx]

        return -target_eff

    def optimize(
        self,
        n_steps: int = 100,
        lr: float = 0.02,
        target_orders: tuple[int, ...] = (-1, 1),
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """Run gradient-based optimization of the beam splitter.

        Parameters
        ----------
        n_steps : int
        lr : float
        target_orders : tuple of int
        verbose : bool

        Returns
        -------
        density : Tensor, shape ``(n_grid,)``
        loss_history : list of float
        """
        from diffnano.design.projection import (
            beta_continuation_schedule,
            heaviside_projection,
        )

        density = (
            torch.rand(self.n_grid, device=self._device, dtype=torch.float64)
            .detach()
            .requires_grad_(True)
        )

        opt = torch.optim.Adam([density], lr=lr)
        loss_history = []

        for step in range(n_steps):
            beta = beta_continuation_schedule(step, n_steps, beta_start=1.0, beta_end=32.0)
            projected = heaviside_projection(density, beta=beta)

            loss = self.objective(projected, target_orders=target_orders)

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
