"""Metalens inverse design workflow.

Provides:
- Target phase profile generation for converging/diverging lenses
- Phase matching loss + Strehl ratio (differentiable)
- Standard optimization loop: Adam warm-up → L-BFGS fine-tuning
- Progressive β-continuation schedule during Adam phase
- Optional designable-mask support for freezing fixed regions
- Hybrid Z-score convergence monitoring

Two configurations: nominal (no C5) and robust (C5 enabled).

Tier 2 module.
"""

from __future__ import annotations

import math

import torch

from diffnano.design.designable_mask import DesignableMask, apply_mask
from diffnano.design.projection import beta_continuation_schedule
from diffnano.solvers.rcwa import RCWASolver
from diffnano.utils.convergence import ConvergenceMonitor

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
        coords = (
            torch.arange(self.n_pixels, dtype=torch.float64, device=self.device) * pixel_size_nm
        )
        coords = coords - coords[-1] / 2  # center at 0
        self.y_grid, self.x_grid = torch.meshgrid(coords, coords, indexing="ij")

        # Pre-compute target phase profile
        self.target_phase = self._target_phase_profile()

        # RCWA solver
        self.solver = RCWASolver(
            fourier_orders=fourier_orders,
            wavelength_nm=wavelength_nm,
            period_nm=(diameter_um * 1000, diameter_um * 1000),
            eps_ambient=n_ambient**2,
            eps_substrate=n_ambient**2,
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
        r = torch.sqrt(self.x_grid**2 + self.y_grid**2 + f_nm**2)
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
        return (diff**2).mean()

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
        var = (diff**2).mean()
        return torch.exp(-var)

    def optimize(
        self,
        n_steps: int = 500,
        lr: float = 1e-3,
        beta_schedule: bool = True,
        beta_start: float = 1.0,
        beta_end: float = 64.0,
        optimizer: str = "adam",
        robust: bool = False,
        sigma_nm: float = 5.0,
        n_mc_samples: int = 8,
        designable_mask: DesignableMask | None = None,
        convergence_monitor: ConvergenceMonitor | None = None,
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
            Apply progressive β-continuation during Adam phase.
        beta_start : float
            Initial β (soft projection, default 1.0).
        beta_end : float
            Final β (near-binary, default 64.0).
        optimizer : str
            "adam" or "lbfgs".
        robust : bool
            Enable C5 robust optimization.
        sigma_nm : float
            Process variation σ for robust mode.
        n_mc_samples : int
            Monte Carlo samples for robust mode.
        designable_mask : DesignableMask, optional
            Restrict gradient updates to designable pixels only.
        convergence_monitor : ConvergenceMonitor, optional
            Monitor convergence; if None a default one is created.
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

        if convergence_monitor is None:
            convergence_monitor = ConvergenceMonitor(
                patience=10,
                z_threshold=0.3,
                window=max(10, min(50, n_steps // 4)),
            )

        loss_history = []

        for step in range(n_steps):
            # Progressive beta-continuation: start soft, sharpen over time
            if beta_schedule:
                beta = beta_continuation_schedule(
                    step,
                    n_steps,
                    beta_start=beta_start,
                    beta_end=beta_end,
                )
            else:
                beta = beta_end

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

            # Zero out gradients for frozen pixels
            if designable_mask is not None and height_map.grad is not None:
                height_map.grad = apply_mask(height_map.grad, designable_mask)

            opt.step()
            with torch.no_grad():
                height_map.clamp_(min=0.0)

            loss_val = loss.item()
            loss_history.append(loss_val)

            # Convergence monitoring
            info = convergence_monitor.step(loss_val)
            if info["should_stop"]:
                if verbose:
                    print(
                        f"Step {step:4d}: converged (z={info['z_score']:.4f}, loss={loss_val:.6f})"
                    )
                break

            if info["should_decay_lr"]:
                for pg in opt.param_groups:
                    pg["lr"] *= convergence_monitor.lr_decay_factor
                if verbose:
                    print(
                        f"Step {step:4d}: stalled, decaying LR to {opt.param_groups[0]['lr']:.2e}"
                    )

            if verbose and step % 50 == 0:
                strehl = self.strehl_ratio(height_map.detach()).item()
                print(f"Step {step:4d}: loss={loss_val:.6f}, Strehl={strehl:.4f}, beta={beta:.1f}")

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
