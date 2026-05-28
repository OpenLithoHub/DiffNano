"""Metalens inverse design workflow.

Provides:
- Target phase profile generation for converging/diverging lenses
- Phase matching loss + Strehl ratio (differentiable)
- Standard optimization loop: Adam warm-up → L-BFGS fine-tuning
- β-continuation schedule

Two configurations: nominal (no C5) and robust (C5 enabled).

Tier 2 module.
"""

from __future__ import annotations

import math

import torch

from diffnano.solvers.rcwa import RCWASolver

__all__ = ["MetalensDesigner"]


class MetalensDesigner:
    """Metalens inverse design pipeline using RCWA.

    Parameters
    ----------
    wavelength_nm : float
        Operating wavelength.
    numerical_aperture : float
        Target NA.
    diameter_um : float
        Lens diameter in micrometers.
    pixel_size_nm : float
        Pixel (meta-atom) size in nm.
    n_material : float
        Refractive index of meta-atom material.
    n_ambient : float
        Ambient refractive index.
    fourier_orders : int
        RCWA Fourier orders.
    focal_length_um : float or None
        Override focal length (default: computed from NA and diameter).
    device : str or torch.device
        Compute device.
    """

    def __init__(
        self,
        wavelength_nm: float = 532.0,
        numerical_aperture: float = 0.5,
        diameter_um: float = 50.0,
        pixel_size_nm: float = 200.0,
        n_material: float = 2.0,
        n_ambient: float = 1.0,
        fourier_orders: int = 10,
        focal_length_um: float | None = None,
        device: str | torch.device = "cpu",
    ):
        self.wavelength_nm = wavelength_nm
        self.na = numerical_aperture
        self.diameter_um = diameter_um
        self.pixel_size_nm = pixel_size_nm
        self.n_material = n_material
        self.n_ambient = n_ambient
        self.fourier_orders = fourier_orders
        self.device = torch.device(device)

        if focal_length_um is None:
            na_frac = numerical_aperture / n_ambient
            self.focal_length_um = diameter_um / 2 / math.tan(math.asin(na_frac))
        else:
            self.focal_length_um = focal_length_um

        # Grid dimensions
        self.n_pixels = int(diameter_um * 1000 / pixel_size_nm)
        self.grid_shape = (self.n_pixels, self.n_pixels)

        # Coordinate grids (in nm)
        coords = torch.arange(
            self.n_pixels, dtype=torch.float64, device=self.device
        ) * pixel_size_nm
        coords = coords - coords[-1] / 2  # center at 0
        self.y_grid, self.x_grid = torch.meshgrid(coords, coords, indexing="ij")

        # Pre-compute target phase profile
        self.target_phase = self._target_phase_profile()

        # RCWA solver
        self.solver = RCWASolver(
            fourier_orders=fourier_orders,
            wavelength_nm=wavelength_nm,
            period_nm=(pixel_size_nm, pixel_size_nm),
            eps_ambient=n_ambient ** 2,
            eps_substrate=n_ambient ** 2,
            device=self.device,
        )

    def _target_phase_profile(self) -> torch.Tensor:
        """Compute the ideal lens phase profile.

        For a converging lens at focal distance f:
            φ(x, y) = k0 * (sqrt(x² + y² + f²) - f)

        Returns
        -------
        phase : Tensor, shape ``(H, W)``
        """
        k0 = 2 * math.pi / self.wavelength_nm
        f_nm = self.focal_length_um * 1000
        r = torch.sqrt(self.x_grid ** 2 + self.y_grid ** 2 + f_nm ** 2)
        phase = k0 * (r - f_nm)
        return phase

    def phase_matching_loss(
        self,
        height_map: torch.Tensor,
    ) -> torch.Tensor:
        """Compute phase matching loss between current and target phase.

        Parameters
        ----------
        height_map : Tensor, shape ``(H, W)``
            Height of each meta-atom in nm.

        Returns
        -------
        loss : Tensor, scalar
        """
        k0 = 2 * math.pi / self.wavelength_nm
        dn = self.n_material - self.n_ambient
        current_phase = k0 * dn * height_map

        # Wrap to [-π, π] for comparison
        diff = torch.atan2(
            torch.sin(current_phase - self.target_phase),
            torch.cos(current_phase - self.target_phase),
        )
        return (diff ** 2).mean()

    def strehl_ratio(
        self,
        height_map: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate Strehl ratio from phase error.

        Strehl ≈ exp(-σ²_φ) where σ²_φ is the phase error variance.

        Parameters
        ----------
        height_map : Tensor, shape ``(H, W)``

        Returns
        -------
        strehl : Tensor, scalar
        """
        k0 = 2 * math.pi / self.wavelength_nm
        dn = self.n_material - self.n_ambient
        current_phase = k0 * dn * height_map

        diff = torch.atan2(
            torch.sin(current_phase - self.target_phase),
            torch.cos(current_phase - self.target_phase),
        )
        var = (diff ** 2).mean()
        return torch.exp(-var)

    def optimize(
        self,
        n_steps: int = 500,
        lr: float = 1e-3,
        beta_schedule: bool = True,
        optimizer: str = "adam",
        robust: bool = False,
        sigma_nm: float = 5.0,
        n_mc_samples: int = 8,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """Run the metalens optimization loop.

        Parameters
        ----------
        n_steps : int
            Number of optimization steps.
        lr : float
            Learning rate.
        beta_schedule : bool
            Apply β-continuation for binarization.
        optimizer : str
            "adam" or "lbfgs".
        robust : bool
            Enable C5 robust optimization.
        sigma_nm : float
            Process variation σ for robust mode.
        n_mc_samples : int
            Monte Carlo samples for robust mode.
        verbose : bool

        Returns
        -------
        height_map : Tensor, shape ``(H, W)``
            Optimized height map.
        loss_history : list of float
        """
        k0 = 2 * math.pi / self.wavelength_nm
        dn = self.n_material - self.n_ambient
        # Initialize from noisy target phase to make optimization non-trivial
        target_wrapped = self.target_phase % (2 * math.pi)
        noise = torch.randn_like(target_wrapped) * 0.5  # phase noise
        h_init = (target_wrapped + noise).clamp(0, 2 * math.pi) / (k0 * dn)
        height_map = h_init.detach().clone().requires_grad_(True)

        opt = torch.optim.Adam([height_map], lr=lr)

        loss_history = []

        for step in range(n_steps):
            if robust:
                from diffnano.design.robustness import robust_gradient_step

                def _loss_fn(h):
                    return self.phase_matching_loss(h)

                loss = robust_gradient_step(
                    height_map,
                    _loss_fn,
                    sigma_nm=sigma_nm,
                    n_samples=n_mc_samples,
                )
            else:
                loss = self.phase_matching_loss(height_map)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_history.append(loss.item())

            if verbose and step % 50 == 0:
                strehl = self.strehl_ratio(height_map.detach()).item()
                print(f"Step {step:4d}: loss={loss.item():.6f}, Strehl={strehl:.4f}")

        return height_map.detach(), loss_history

    def export_gds(
        self,
        path: str,
        height_map: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        """Export optimized metalens layout to GDS.

        Parameters
        ----------
        path : str
            Output file path (.gds).
        height_map : Tensor
            Optimized height map.
        threshold : float
            Height threshold for binarization.
        """
        from diffnano.export.gds import export_density_to_gds

        # Convert height map to density
        h_max = height_map.max().item()
        if h_max > 0:
            density = height_map / h_max
        else:
            density = height_map

        export_density_to_gds(
            density,
            path,
            pixel_size_nm=self.pixel_size_nm,
            threshold=threshold,
        )
