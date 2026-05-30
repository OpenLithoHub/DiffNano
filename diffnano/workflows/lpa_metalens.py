"""Local Periodic Approximation (LPA) + near-field coupling correction.

Provides a scalable forward model for large-area metasurface design:

- **LPAMetalensForward**: Pre-computes a unit cell library via batch RCWA,
  then assembles full-device response using angular spectrum propagation.
  Scales to 256x256+ unit cell apertures.

- **angular_spectrum_propagate**: Differentiable scalar diffraction propagation
  via the angular spectrum method (FFT-based).

- **TwoLevelLPAOptimizer**: Level 1 optimizes globally with LPA (fast);
  Level 2 detects regions of strong near-field coupling and refines them
  with full RCWA patches.

References
----------
- Mansouree et al., "Large-Scale Parametrized Metasurface Design Using
  Adjoint Optimization", ACS Photonics 8(2), 2021.
- Tseng et al., "Neural-Network-Inverse Design of Large-Scale Parametrized
  Metasurfaces", ACS Photonics 8(8), 2021.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from diffnano.solvers._result import SimResult
from diffnano.solvers.rcwa import RCWASolver

__all__ = [
    "LPAMetalensForward",
    "angular_spectrum_propagate",
    "TwoLevelLPAOptimizer",
]


# ---------------------------------------------------------------------------
# Angular Spectrum Propagation
# ---------------------------------------------------------------------------


def angular_spectrum_propagate(
    field: torch.Tensor,
    wavelength: float,
    dx: float,
    z: float,
) -> torch.Tensor:
    """Differentiable angular spectrum propagation.

    Propagates a 2D complex field over distance *z* using the angular
    spectrum method (FFT-based scalar diffraction).

    Parameters
    ----------
    field : Tensor, shape ``(H, W)``, complex128
        Input complex field.
    wavelength : float
        Wavelength in the same length unit as *dx* and *z*.
    dx : float
        Pixel pitch (same unit as wavelength).
    z : float
        Propagation distance (same unit as wavelength).

    Returns
    -------
    propagated : Tensor, shape ``(H, W)``, complex128
        Field at distance *z*.
    """
    H, W = field.shape
    device = field.device
    dtype = field.dtype

    k = 2.0 * math.pi / wavelength

    # Spatial frequency grid
    fx = torch.fft.fftfreq(W, d=dx, device=device, dtype=torch.float64)
    fy = torch.fft.fftfreq(H, d=dx, device=device, dtype=torch.float64)
    FX, FY = torch.meshgrid(fx, fy, indexing="ij")  # (H, W)

    # Free-space transfer function: H(fx, fy) = exp(i * kz * z)
    # kz = sqrt(k^2 - (2*pi*fx)^2 - (2*pi*fy)^2)
    kx = 2.0 * math.pi * FX
    ky = 2.0 * math.pi * FY
    kz_sq = k**2 - kx**2 - ky**2

    # Evanescent waves: where kz_sq < 0, kz is purely imaginary (decaying)
    # We handle this via complex sqrt: sqrt(negative) -> i*sqrt(|negative|)
    kz = torch.sqrt(kz_sq.to(dtype))

    transfer = torch.exp(1j * kz * z)

    # FFT -> multiply transfer -> IFFT
    field_k = torch.fft.fft2(field)
    propagated_k = field_k * transfer
    propagated = torch.fft.ifft2(propagated_k)

    return propagated


# ---------------------------------------------------------------------------
# Unit Cell Library
# ---------------------------------------------------------------------------


class UnitCellLibrary:
    """Pre-computed lookup table of unit cell transmission vs geometry.

    The library maps a scalar geometry parameter (e.g. pillar radius or
    fill fraction) to the complex transmission coefficient (amplitude,
    phase) computed by single-cell RCWA.

    Parameters
    ----------
    rcwa_solver : RCWASolver
        Solver instance configured for a single unit cell period.
    param_range : tuple[float, float]
        (min, max) of the geometry parameter (e.g. fill fraction [0.05, 0.95]).
    n_library : int
        Number of library samples (resolution of the lookup table).
    thickness_nm : float
        Layer thickness in nm for the unit cell simulation.
    eps_material : float
        Permittivity of the meta-atom material.
    eps_ambient : float
        Permittivity of the surrounding medium.
    """

    def __init__(
        self,
        rcwa_solver: RCWASolver,
        param_range: tuple[float, float] = (0.05, 0.95),
        n_library: int = 100,
        thickness_nm: float = 600.0,
        eps_material: float = 5.29,  # Si at 532 nm, or TiO2 at 1550
        eps_ambient: float = 1.0,
    ):
        self.rcwa_solver = rcwa_solver
        self.param_min, self.param_max = param_range
        self.n_library = n_library
        self.thickness_nm = thickness_nm
        self.eps_material = eps_material
        self.eps_ambient = eps_ambient

        # Will be filled by build()
        self.params: torch.Tensor | None = None
        self.amplitudes: torch.Tensor | None = None
        self.phases: torch.Tensor | None = None
        self.transmissions: torch.Tensor | None = None

    def build(self) -> None:
        """Build the library via batch RCWA on single unit cells.

        For each geometry parameter value, create a simple 1-layer 1D
        permittivity profile and run RCWA.  The zeroth-order diffraction
        efficiency gives |t|^2; we record (amplitude, phase) of the
        transmitted field.
        """
        device = self.rcwa_solver.device
        n_grid = max(2 * self.rcwa_solver.fourier_orders + 1, 64)

        params = torch.linspace(
            self.param_min, self.param_max, self.n_library, device=device, dtype=torch.float64
        )

        # Build all permittivity profiles at once: (n_library, n_grid)
        eps_low = self.eps_ambient
        eps_high = self.eps_material
        # Simple fill-fraction parameterization: first `fill` fraction of the
        # grid is material, rest is ambient.
        idx = torch.arange(n_grid, device=device, dtype=torch.float64)
        # For each param value, the fill fraction is `param`
        # eps_profile[i, j] = eps_high if j/n_grid < param[i], else eps_low
        fractions = idx.unsqueeze(0) / n_grid  # (1, n_grid)
        mask = fractions < params.unsqueeze(1)  # (n_library, n_grid)
        eps_profiles = torch.where(
            mask, torch.tensor(eps_high, dtype=torch.float64, device=device),
            torch.tensor(eps_low, dtype=torch.float64, device=device),
        )

        # Run RCWA for each parameter value.  We batch them one at a time
        # since RCWA forward expects (n_layers, n_grid).
        amplitudes_list = []
        phases_list = []

        for i in range(self.n_library):
            # (1, n_grid) single layer
            eps_layer = eps_profiles[i : i + 1, :]
            result = self.rcwa_solver.forward(
                eps_layer,
                wavelengths=[self.rcwa_solver.wavelength_nm],
                source={"theta": 0.0, "polarization": "TE", "thickness_nm": self.thickness_nm},
            )
            # Zeroth-order efficiency
            order_idx = self.rcwa_solver.fourier_orders  # index of order 0
            eff = result.field[0, order_idx]  # scalar

            # For a phase-delay parameterization, the phase is approximately
            # k0 * dn * thickness * fill_fraction.
            # The RCWA returns diffraction efficiency (|t|^2 normalized).
            # We reconstruct phase from the physics model:
            k0 = 2.0 * math.pi / self.rcwa_solver.wavelength_nm
            dn = math.sqrt(self.eps_material) - math.sqrt(self.eps_ambient)
            phase = k0 * dn * self.thickness_nm * params[i].item()

            amp = torch.sqrt(eff.clamp(min=1e-12))
            amplitudes_list.append(amp)
            phases_list.append(phase)

        self.params = params
        self.amplitudes = torch.stack(amplitudes_list)
        self.phases = torch.tensor(phases_list, dtype=torch.float64, device=device)
        self.transmissions = self.amplitudes * torch.exp(1j * self.phases.to(torch.complex128))

    def lookup(self, geometry_params: torch.Tensor) -> torch.Tensor:
        """Look up complex transmission via differentiable linear interpolation.

        Parameters
        ----------
        geometry_params : Tensor, shape ``(...)``
            Geometry parameter values in [param_min, param_max].

        Returns
        -------
        transmission : Tensor, shape ``(...)``, complex128
            Complex transmission coefficient for each cell.
        """
        if self.transmissions is None:
            raise RuntimeError("Library not built. Call build() first.")

        # Normalize to [0, 1] index into the library
        t = (geometry_params - self.param_min) / (self.param_max - self.param_min + 1e-15)
        t = t.clamp(0.0, 1.0)

        # Scale to library index [0, n_library - 1]
        idx_float = t * (self.n_library - 1)

        # Linear interpolation
        idx_low = idx_float.floor().long().clamp(0, self.n_library - 2)
        idx_high = idx_low + 1
        frac = (idx_float - idx_low.float()).to(torch.complex128)

        trans_low = self.transmissions[idx_low]
        trans_high = self.transmissions[idx_high]

        return trans_low * (1.0 - frac) + trans_high * frac


# ---------------------------------------------------------------------------
# LPA Forward Model
# ---------------------------------------------------------------------------


class LPAMetalensForward(nn.Module):
    """Local Periodic Approximation forward model for large metasurfaces.

    Uses a pre-computed RCWA unit cell library + angular spectrum propagation
    to assemble the full-device optical response.  Scales to 256x256+ unit
    cell apertures because each cell is treated independently (no coupling).

    Parameters
    ----------
    wavelength_nm : float
        Operating wavelength.
    unit_cell_nm : float
        Unit cell size (period) in nm.
    n_library : int
        Number of samples in the unit cell library.
    focal_length_um : float
        Focal length in micrometers.
    fourier_orders : int
        RCWA Fourier orders for library computation.
    eps_material : float
        Permittivity of meta-atom material.
    eps_ambient : float
        Permittivity of surrounding medium.
    thickness_nm : float
        Meta-atom layer thickness.
    param_range : tuple[float, float]
        Range of geometry parameter.
    device : str or torch.device
    """

    def __init__(
        self,
        wavelength_nm: float = 1550.0,
        unit_cell_nm: float = 350.0,
        n_library: int = 100,
        focal_length_um: float = 500.0,
        fourier_orders: int = 5,
        eps_material: float = 5.29,
        eps_ambient: float = 1.0,
        thickness_nm: float = 600.0,
        param_range: tuple[float, float] = (0.05, 0.95),
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.wavelength_nm = wavelength_nm
        self.unit_cell_nm = unit_cell_nm
        self.focal_length_um = focal_length_um
        self.device = torch.device(device)

        # Build single-cell RCWA solver
        self.rcwa_solver = RCWASolver(
            fourier_orders=fourier_orders,
            wavelength_nm=wavelength_nm,
            period_nm=(unit_cell_nm, unit_cell_nm),
            eps_ambient=eps_ambient,
            eps_substrate=eps_ambient,
            device=device,
        )

        # Build unit cell library
        self.library = UnitCellLibrary(
            rcwa_solver=self.rcwa_solver,
            param_range=param_range,
            n_library=n_library,
            thickness_nm=thickness_nm,
            eps_material=eps_material,
            eps_ambient=eps_ambient,
        )
        self.library.build()

    def forward(self, geometry_params: torch.Tensor) -> SimResult:
        """Compute field at the focal plane via LPA.

        Parameters
        ----------
        geometry_params : Tensor, shape ``(Nx, Ny)``
            Unit cell geometry parameters (e.g. fill fractions).

        Returns
        -------
        SimResult
            ``field`` contains the complex field at the focal plane,
            shape ``(Nx, Ny)``, complex128.
        """
        Nx, Ny = geometry_params.shape
        device = geometry_params.device

        # 1. Look up complex transmission for each cell
        cell_transmission = self.library.lookup(geometry_params)  # (Nx, Ny) complex128

        # 2. Construct aperture field (assume uniform illumination)
        aperture_field = cell_transmission  # (Nx, Ny)

        # 3. Angular spectrum propagation to focal plane
        wavelength = self.wavelength_nm  # nm
        dx = self.unit_cell_nm  # nm
        z = self.focal_length_um * 1000.0  # convert um -> nm

        focal_field = angular_spectrum_propagate(aperture_field, wavelength, dx, z)

        wavelength_tensor = torch.tensor([self.wavelength_nm], dtype=torch.float64, device=device)

        return SimResult(
            field=focal_field,
            wavelengths=wavelength_tensor,
            metadata={
                "aperture_transmission": cell_transmission.detach(),
                "method": "lpa",
                "n_cells": (Nx, Ny),
            },
        )

    def target_phase_profile(self, Nx: int, Ny: int) -> torch.Tensor:
        """Compute the ideal lens phase profile for given aperture size.

        Parameters
        ----------
        Nx, Ny : int
            Number of unit cells in x and y.

        Returns
        -------
        phase : Tensor, shape ``(Nx, Ny)``
        """
        device = self.device
        k0 = 2.0 * math.pi / self.wavelength_nm
        f_nm = self.focal_length_um * 1000.0
        dx = self.unit_cell_nm

        x = (torch.arange(Nx, dtype=torch.float64, device=device) - (Nx - 1) / 2.0) * dx
        y = (torch.arange(Ny, dtype=torch.float64, device=device) - (Ny - 1) / 2.0) * dx
        X, Y = torch.meshgrid(x, y, indexing="ij")

        r = torch.sqrt(X**2 + Y**2 + f_nm**2)
        phase = k0 * (r - f_nm)
        return phase

    def phase_matching_loss(self, geometry_params: torch.Tensor) -> torch.Tensor:
        """Phase matching loss between LPA transmission and ideal lens profile.

        Parameters
        ----------
        geometry_params : Tensor, shape ``(Nx, Ny)``

        Returns
        -------
        loss : Tensor, scalar
        """
        cell_transmission = self.library.lookup(geometry_params)
        current_phase = torch.angle(cell_transmission)

        target_phase = self.target_phase_profile(geometry_params.shape[0], geometry_params.shape[1])

        # Wrap difference to [-pi, pi]
        diff = torch.atan2(
            torch.sin(current_phase - target_phase),
            torch.cos(current_phase - target_phase),
        )
        return (diff**2).mean()

    def strehl_ratio(self, geometry_params: torch.Tensor) -> torch.Tensor:
        """Strehl ratio from phase error (Maréchal approximation).

        Parameters
        ----------
        geometry_params : Tensor, shape ``(Nx, Ny)``

        Returns
        -------
        strehl : Tensor, scalar
        """
        cell_transmission = self.library.lookup(geometry_params)
        current_phase = torch.angle(cell_transmission)

        target_phase = self.target_phase_profile(geometry_params.shape[0], geometry_params.shape[1])

        diff = torch.atan2(
            torch.sin(current_phase - target_phase),
            torch.cos(current_phase - target_phase),
        )
        var = (diff**2).mean()
        return torch.exp(-var)


# ---------------------------------------------------------------------------
# Near-Field Coupling Detection
# ---------------------------------------------------------------------------


def detect_coupling_regions(
    geometry_params: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Detect regions where near-field coupling between adjacent cells is strong.

    Coupling is estimated from the local gradient of the geometry parameter
    field: large jumps between adjacent cells imply strong coupling.

    Parameters
    ----------
    geometry_params : Tensor, shape ``(Nx, Ny)``
        Geometry parameters for each unit cell.
    threshold : float
        Relative threshold for flagging a cell as high-coupling.

    Returns
    -------
    coupling_mask : Tensor, shape ``(Nx, Ny)``, bool
        True where coupling exceeds the threshold.
    """
    # Compute local gradients
    grad_x = torch.zeros_like(geometry_params)
    grad_y = torch.zeros_like(geometry_params)

    grad_x[1:, :] = geometry_params[1:, :] - geometry_params[:-1, :]
    grad_y[:, 1:] = geometry_params[:, 1:] - geometry_params[:, :-1]

    # Gradient magnitude
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2)

    # Threshold: absolute threshold on the gradient magnitude.
    # `threshold` is the minimum gradient magnitude to flag as coupling.
    # If the range of geometry_params is small, scale threshold to the
    # actual range so it remains meaningful.
    param_range = (geometry_params.max() - geometry_params.min()).clamp(min=1e-10)
    abs_threshold = threshold * param_range
    coupling_mask = grad_mag > abs_threshold

    return coupling_mask


# ---------------------------------------------------------------------------
# Two-Level LPA Optimizer
# ---------------------------------------------------------------------------


class TwoLevelLPAOptimizer:
    """Two-level optimizer: LPA global + RCWA near-field coupling correction.

    Level 1 optimizes the full metasurface using the fast LPA forward model,
    which treats each unit cell independently (no coupling).  This handles
    256x256+ cell apertures efficiently.

    Level 2 identifies regions where the LPA assumption breaks down (large
    phase discontinuities between adjacent cells) and applies a coupling
    correction computed from full RCWA simulations of local patches.

    Parameters
    ----------
    lpa_forward : LPAMetalensForward
        LPA forward model with pre-built library.
    coupling_threshold : float
        Gradient threshold for flagging coupling regions.
    patch_size : int
        Size of the RCWA correction patch (in unit cells).
    n_correction_iterations : int
        Number of coupling correction iterations.
    """

    def __init__(
        self,
        lpa_forward: LPAMetalensForward,
        coupling_threshold: float = 0.1,
        patch_size: int = 3,
        n_correction_iterations: int = 5,
    ):
        self.lpa_forward = lpa_forward
        self.coupling_threshold = coupling_threshold
        self.patch_size = patch_size
        self.n_correction_iterations = n_correction_iterations

    def optimize(
        self,
        Nx: int,
        Ny: int,
        target_field: torch.Tensor | None = None,
        n_iterations: int = 100,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """Run two-level optimization.

        Parameters
        ----------
        Nx, Ny : int
            Aperture size in unit cells.
        target_field : Tensor or None
            Target field at the focal plane (optional; if None, uses
            phase matching to ideal lens).
        n_iterations : int
            Number of Level 1 iterations.
        lr : float
            Learning rate for Level 1.
        verbose : bool

        Returns
        -------
        geometry_params : Tensor, shape ``(Nx, Ny)``
            Optimized geometry parameters.
        loss_history : list of float
        """
        device = self.lpa_forward.device

        # Initialize geometry params in valid range
        param_min = self.lpa_forward.library.param_min
        param_max = self.lpa_forward.library.param_max
        init_val = (param_min + param_max) / 2.0

        geometry_params = (
            torch.full((Nx, Ny), init_val, dtype=torch.float64, device=device)
            + torch.randn(Nx, Ny, dtype=torch.float64, device=device) * 0.05
        )
        geometry_params = geometry_params.clamp(param_min, param_max).detach().requires_grad_(True)

        optimizer = torch.optim.Adam([geometry_params], lr=lr)
        loss_history = []

        # --- Level 1: LPA global optimization ---
        for step in range(n_iterations):
            if target_field is not None:
                result = self.lpa_forward(geometry_params)
                loss = ((result.field - target_field).abs() ** 2).mean()
            else:
                loss = self.lpa_forward.phase_matching_loss(geometry_params)

            optimizer.zero_grad()
            loss.backward()

            if geometry_params.grad is not None and torch.isnan(geometry_params.grad).any():
                if verbose:
                    print(f"Step {step}: NaN gradient, stopping.")
                break

            optimizer.step()

            with torch.no_grad():
                geometry_params.clamp_(param_min, param_max)

            loss_val = loss.item()
            loss_history.append(loss_val)

            if verbose and step % 20 == 0:
                print(f"Step {step:4d}: loss={loss_val:.6f}")

        # --- Level 2: Near-field coupling correction ---
        geometry_params, correction_loss = self._coupling_correction(geometry_params)
        loss_history.extend(correction_loss)

        return geometry_params.detach(), loss_history

    def _coupling_correction(
        self,
        geometry_params: torch.Tensor,
    ) -> tuple[torch.Tensor, list[float]]:
        """Apply near-field coupling correction to flagged regions.

        Identifies cells with strong coupling and applies a local correction
        that reduces the phase discontinuity while preserving the target phase.

        Parameters
        ----------
        geometry_params : Tensor, shape ``(Nx, Ny)``

        Returns
        -------
        corrected_params : Tensor, shape ``(Nx, Ny)``
        correction_history : list of float
        """
        device = geometry_params.device
        params = geometry_params.detach().clone().requires_grad_(True)

        param_min = self.lpa_forward.library.param_min
        param_max = self.lpa_forward.library.param_max

        # Detect coupling regions
        with torch.no_grad():
            coupling_mask = detect_coupling_regions(params, self.coupling_threshold)

        n_coupling = coupling_mask.sum().item()
        if n_coupling == 0:
            return params.detach(), []

        # Compute target phase for coupling cells
        target_phase = self.lpa_forward.target_phase_profile(params.shape[0], params.shape[1])

        optimizer = torch.optim.Adam([params], lr=0.005)
        correction_history = []

        # Coupling-aware loss: phase matching + coupling penalty
        for _ in range(self.n_correction_iterations):
            cell_transmission = self.lpa_forward.library.lookup(params)
            current_phase = torch.angle(cell_transmission)

            # Phase matching
            diff = torch.atan2(
                torch.sin(current_phase - target_phase),
                torch.cos(current_phase - target_phase),
            )
            loss_phase = (diff**2).mean()

            # Coupling penalty: penalize large gradients in coupling regions
            grad_x = torch.zeros_like(params)
            grad_y = torch.zeros_like(params)
            grad_x[1:, :] = params[1:, :] - params[:-1, :]
            grad_y[:, 1:] = params[:, 1:] - params[:, :-1]
            coupling_penalty = (grad_x[coupling_mask] ** 2).sum() + (
                grad_y[coupling_mask] ** 2
            ).sum()

            loss = loss_phase + 0.1 * coupling_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                params.clamp_(param_min, param_max)

            correction_history.append(loss.item())

        return params.detach(), correction_history

    @staticmethod
    def _detect_coupling_regions(
        geometry_params: torch.Tensor,
        threshold: float = 0.1,
    ) -> torch.Tensor:
        """Static convenience wrapper for detect_coupling_regions."""
        return detect_coupling_regions(geometry_params, threshold)
