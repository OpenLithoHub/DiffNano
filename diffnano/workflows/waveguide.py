"""Waveguide coupler and mode converter inverse design workflow.

Provides:
- Waveguide mode computation (eigenmode of slab/strip waveguide)
- Mode overlap integral (differentiable figure of merit)
- Bend and mode converter topology optimization
- Supports both FDTD and FDFD backends via the Solver protocol

References
----------
- Christiansen & Sigmund (2021), Inverse design in photonics
"""

from __future__ import annotations

import math

import torch

__all__ = ["WaveguideDesigner"]


def _waveguide_mode_1d(
    eps_r: torch.Tensor,
    wavelength_nm: float,
    dl: float = 20.0,
    n_modes: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute waveguide eigenmodes for a 1D slab waveguide.

    Solves the 1D eigenvalue problem:
        (1/dl^2) d²E/dx² + k₀² eps(x) E = β² E

    where β = k₀ n_eff.

    Parameters
    ----------
    eps_r : Tensor, shape ``(N,)``
        Permittivity profile of the waveguide cross-section.
    wavelength_nm : float
        Wavelength.
    dl : float
        Grid spacing in nm.
    n_modes : int
        Number of modes to return.

    Returns
    -------
    n_eff : Tensor, shape ``(n_modes,)``
        Effective indices.
    modes : Tensor, shape ``(n_modes, N)``
        Mode field profiles.
    """
    N = eps_r.shape[0]
    device = eps_r.device
    k0 = 2 * math.pi / wavelength_nm

    # Build second-derivative operator (finite differences) / dl^2
    inv_dl2 = 1.0 / (dl * dl)
    diag_main = torch.full((N,), -2.0 * inv_dl2, device=device, dtype=torch.float64)
    diag_off = torch.ones(N - 1, device=device, dtype=torch.float64) * inv_dl2
    D2 = torch.diag(diag_main) + torch.diag(diag_off, 1) + torch.diag(diag_off, -1)

    # Eigenvalue problem: (D2 + k0^2 * diag(eps)) E = beta^2 * E
    M = D2 + k0**2 * torch.diag(eps_r)

    eigenvalues, eigenvectors = torch.linalg.eigh(M)

    # beta^2 = eigenvalue, n_eff^2 = beta^2 / k0^2
    beta_sq = eigenvalues
    n_eff_sq = beta_sq / (k0**2)
    n_eff = torch.sqrt(torch.clamp(n_eff_sq, min=0.0))

    # Sort by decreasing n_eff (guided modes first)
    idx = torch.argsort(n_eff, descending=True)

    n_eff_sorted = n_eff[idx[:n_modes]]
    modes_sorted = eigenvectors[:, idx[:n_modes]].T  # (n_modes, N)

    # Normalize modes
    for i in range(min(n_modes, modes_sorted.shape[0])):
        norm = modes_sorted[i].norm()
        if norm > 0:
            modes_sorted[i] = modes_sorted[i] / norm

    return n_eff_sorted, modes_sorted


class WaveguideDesigner:
    """Waveguide inverse design workflow.

    Supports:
    - Waveguide mode computation and overlap calculation
    - Topology optimization of waveguide bends and mode converters
    - Works with any Solver backend (FDFD, FDTD)

    Parameters
    ----------
    wavelength_nm : float
        Operating wavelength.
    grid_shape : tuple[int, int]
        ``(H, W)`` simulation grid.
    dl : float
        Grid spacing in nm.
    n_core : float
        Core refractive index.
    n_clad : float
        Cladding refractive index.
    waveguide_width_nm : float
        Waveguide width.
    device : str or torch.device
    solver : Solver or None
        Optional solver backend. If None, uses built-in mode analysis.
    """

    def __init__(
        self,
        wavelength_nm: float = 1550.0,
        grid_shape: tuple[int, int] = (60, 60),
        dl: float = 20.0,
        n_core: float = 2.5,
        n_clad: float = 1.0,
        waveguide_width_nm: float = 400.0,
        device: str | torch.device = "cpu",
        solver=None,
    ):
        self.wavelength_nm = wavelength_nm
        self.grid_shape = grid_shape
        self.dl = dl
        self.n_core = n_core
        self.n_clad = n_clad
        self.waveguide_width_nm = waveguide_width_nm
        self._device = torch.device(device)
        self.solver = solver

    @property
    def device(self) -> torch.device:
        return self._device

    def waveguide_eps(
        self,
        geometry: torch.Tensor | None = None,
        y_offset: int = 0,
    ) -> torch.Tensor:
        """Create a waveguide permittivity map.

        Parameters
        ----------
        geometry : Tensor, shape ``(H, W)``, optional
            Design region density (overwrites the waveguide in the design region).
        y_offset : int
            Vertical offset of the waveguide center.

        Returns
        -------
        eps_r : Tensor, shape ``(H, W)``
        """
        H, W = self.grid_shape
        eps_clad = self.n_clad**2
        eps_core = self.n_core**2

        width_px = int(self.waveguide_width_nm / self.dl)
        center_y = H // 2 + y_offset

        eps_r = torch.full((H, W), eps_clad, dtype=torch.float64, device=self._device)

        y_start = max(0, center_y - width_px // 2)
        y_end = min(H, center_y + width_px // 2)
        eps_r[y_start:y_end, :] = eps_core

        if geometry is not None:
            # Use geometry as density for the entire grid
            eps_r = eps_clad + (eps_core - eps_clad) * geometry

        return eps_r

    def fundamental_mode(
        self,
        eps_r: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the fundamental mode of the waveguide.

        Parameters
        ----------
        eps_r : Tensor, shape ``(H, W)``, optional
            Permittivity map. Uses default waveguide if None.

        Returns
        -------
        n_eff : Tensor, scalar
            Effective index of fundamental mode.
        mode_profile : Tensor, shape ``(H, W)``
            2D mode field profile.
        """
        if eps_r is None:
            eps_r = self.waveguide_eps()

        # Average along propagation direction (dim=1, x-axis) to get 1D cross-section
        eps_1d = eps_r.mean(dim=1)  # (W,) — cross-section perpendicular to propagation

        n_effs, modes_1d = _waveguide_mode_1d(
            eps_1d,
            self.wavelength_nm,
            dl=self.dl,
            n_modes=1,
        )

        # Expand to 2D by replicating
        mode_2d = modes_1d[0].unsqueeze(0).expand(self.grid_shape[0], -1)

        return n_effs[0], mode_2d

    def mode_overlap(
        self,
        field: torch.Tensor,
        target_mode: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mode overlap integral (differentiable).

        overlap = |∫ E * E_target* dA|² / (∫ |E|² dA * ∫ |E_target|² dA)

        Parameters
        ----------
        field : Tensor, shape ``(H, W)``
            Simulated field profile.
        target_mode : Tensor, shape ``(H, W)``
            Target mode profile.

        Returns
        -------
        overlap : Tensor, scalar
            Mode overlap ∈ [0, 1].
        """
        field_flat = field.flatten()
        target_flat = target_mode.flatten()

        numerator = torch.abs(field_flat @ target_flat.conj()) ** 2
        denominator = (field_flat.norm() ** 2) * (target_flat.norm() ** 2 + 1e-12)

        return numerator / denominator

    def transmission_loss(
        self,
        field: torch.Tensor,
        input_mode: torch.Tensor,
    ) -> torch.Tensor:
        """Compute transmission loss (negative log overlap).

        Parameters
        ----------
        field : Tensor, shape ``(H, W)``
            Output field from simulation.
        input_mode : Tensor, shape ``(H, W)``
            Input mode (what we want to transmit).

        Returns
        -------
        loss : Tensor, scalar
            -log(overlap), minimized when overlap = 1.
        """
        overlap = self.mode_overlap(field, input_mode)
        return -torch.log(overlap + 1e-8)

    def optimize_bend(
        self,
        bend_radius_nm: float = 2000.0,
        n_steps: int = 200,
        lr: float = 0.01,
        design_region: tuple[int, int, int, int] | None = None,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """Optimize a waveguide bend using mode overlap as figure of merit.

        Topology-optimizes the bend region to maximize transmission
        (mode overlap between input and output).

        Parameters
        ----------
        bend_radius_nm : float
            Bend radius (for geometry setup).
        n_steps : int
            Optimization steps.
        lr : float
            Learning rate.
        design_region : tuple, optional
            ``(y_start, y_end, x_start, x_end)`` of the design region.
        verbose : bool

        Returns
        -------
        density : Tensor, shape ``(H, W)``
            Optimized density in the design region.
        loss_history : list of float
        """
        from diffnano.design.projection import (
            beta_continuation_schedule,
            heaviside_projection,
        )

        H, W = self.grid_shape

        if design_region is None:
            # Default: center region
            margin = H // 4
            design_region = (margin, H - margin, margin, W - margin)

        y0, y1, x0, x1 = design_region
        dh = y1 - y0
        dw = x1 - x0

        density = torch.rand(dh, dw, device=self._device, dtype=torch.float64)
        density = density.detach().requires_grad_(True)

        # Compute input mode (straight waveguide)
        eps_straight = self.waveguide_eps()
        _, input_mode = self.fundamental_mode(eps_straight)

        # Output mode (could be same or different for mode converter)
        output_mode = input_mode

        opt = torch.optim.Adam([density], lr=lr)
        loss_history = []

        for step in range(n_steps):
            beta = beta_continuation_schedule(step, n_steps, beta_start=1.0, beta_end=16.0)
            projected = heaviside_projection(density, beta=beta)

            # Build full geometry with design region
            eps_r = self.waveguide_eps()
            eps_r[y0:y1, x0:x1] = self.n_clad**2 + (self.n_core**2 - self.n_clad**2) * projected

            if self.solver is not None:
                result = self.solver.forward(
                    eps_r,
                    wavelengths=[self.wavelength_nm],
                    source={"type": "gaussian_pulse", "amplitude": 1.0},
                )
                # Handle different field shapes from different solver types
                field = result.field
                if field.dim() == 3:
                    output_field = field[0]  # (H, W)
                elif field.dim() == 2:
                    output_field = field
                else:
                    output_field = field.reshape(H, W)
            else:
                # Analytical estimate: mode overlap between input and
                # the perturbed waveguide mode at the output
                _, output_field = self.fundamental_mode(eps_r)

            loss = self.transmission_loss(output_field, output_mode)

            opt.zero_grad()
            loss.backward()

            if density.grad is not None and torch.isnan(density.grad).any():
                break

            opt.step()
            with torch.no_grad():
                density.clamp_(0.0, 1.0)

            loss_history.append(loss.item())

            if verbose and step % 20 == 0:
                overlap = self.mode_overlap(output_field.detach(), output_mode).item()
                print(f"Step {step:4d}: loss={loss.item():.6f}, overlap={overlap:.4f}")

        # Build full output density
        full_density = torch.zeros(H, W, device=self._device, dtype=torch.float64)
        full_density[y0:y1, x0:x1] = density.detach()

        return full_density, loss_history

    def optimize_mode_converter(
        self,
        n_steps: int = 200,
        lr: float = 0.01,
        design_region: tuple[int, int, int, int] | None = None,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """Optimize a mode converter (fundamental to higher-order mode).

        Parameters
        ----------
        n_steps : int
        lr : float
        design_region : tuple, optional
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

        H, W = self.grid_shape

        if design_region is None:
            margin = H // 4
            design_region = (margin, H - margin, margin, W - margin)

        y0, y1, x0, x1 = design_region
        dh = y1 - y0
        dw = x1 - x0

        density = torch.rand(dh, dw, device=self._device, dtype=torch.float64)
        density = density.detach().requires_grad_(True)

        # Input: fundamental mode
        eps_straight = self.waveguide_eps()
        eps_1d = eps_straight.mean(dim=1)
        n_effs, modes_1d = _waveguide_mode_1d(eps_1d, self.wavelength_nm, dl=self.dl, n_modes=2)

        input_mode = modes_1d[0].unsqueeze(0).expand(H, -1)

        # Target: second-order mode (if it exists)
        if modes_1d.shape[0] > 1:
            target_mode = modes_1d[1].unsqueeze(0).expand(H, -1)
        else:
            target_mode = input_mode

        opt = torch.optim.Adam([density], lr=lr)
        loss_history = []

        for step in range(n_steps):
            beta = beta_continuation_schedule(step, n_steps, beta_start=1.0, beta_end=16.0)
            projected = heaviside_projection(density, beta=beta)

            eps_r = self.waveguide_eps()
            eps_r[y0:y1, x0:x1] = self.n_clad**2 + (self.n_core**2 - self.n_clad**2) * projected

            _, output_field = self.fundamental_mode(eps_r)

            loss = self.transmission_loss(output_field, target_mode)

            opt.zero_grad()
            loss.backward()

            if density.grad is not None and torch.isnan(density.grad).any():
                break

            opt.step()
            with torch.no_grad():
                density.clamp_(0.0, 1.0)

            loss_history.append(loss.item())

            if verbose and step % 20 == 0:
                overlap = self.mode_overlap(output_field.detach(), target_mode).item()
                print(f"Step {step:4d}: loss={loss.item():.6f}, overlap={overlap:.4f}")

        full_density = torch.zeros(H, W, device=self._device, dtype=torch.float64)
        full_density[y0:y1, x0:x1] = density.detach()

        return full_density, loss_history
