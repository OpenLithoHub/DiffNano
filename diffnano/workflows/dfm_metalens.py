"""DFM-aware metalens workflow (C4 embodiment).

Single density parameterization driving both:
(a) Hopkins forward lithography model (aerial image → printed mask)
(b) diffnano RCWA forward solve (optical response)

Both pipelines call constraints_shared primitives; gradients from both
flow back to the same parameter tensor via a unified autograd graph.

Tier 3 module (release after CN priority confirmation).
"""

from __future__ import annotations

import math

import torch

from diffnano.design.constraints_shared import combined_fabrication_penalty
from diffnano.design.parameterization import DensityField
from diffnano.design.projection import beta_continuation_schedule
from diffnano.solvers.litho import HopkinsLithoModel
from diffnano.solvers.rcwa import RCWASolver

__all__ = ["DFMMetalensDesigner"]


class DFMMetalensDesigner:
    """DFM-native metalens inverse design (C4 embodiment).

    Uses a density field parameterization θ ∈ [0,1]^{H×W} that is shared
    between the lithography forward model and the EM forward model.

    Parameters
    ----------
    wavelength_nm : float
        Device operating wavelength (e.g. 940 nm).
    numerical_aperture : float
        Target NA.
    diameter_um : float
        Lens diameter.
    pixel_size_nm : float
        Pixel size (must match lithography resolution).
    n_material : float
        Meta-atom refractive index.
    n_ambient : float
        Ambient index.
    fourier_orders : int
        RCWA Fourier orders.
    litho_wavelength_nm : float
        Lithography exposure wavelength.
    litho_na : float
        Lithography projection NA.
    device : str or torch.device
    """

    def __init__(
        self,
        wavelength_nm: float = 940.0,
        numerical_aperture: float = 0.5,
        diameter_um: float = 200.0,
        pixel_size_nm: float = 5.0,
        n_material: float = 2.0,
        n_ambient: float = 1.0,
        fourier_orders: int = 10,
        litho_wavelength_nm: float = 193.0,
        litho_na: float = 1.35,
        device: str | torch.device = "cpu",
    ):
        self.wavelength_nm = wavelength_nm
        self.na = numerical_aperture
        self.diameter_um = diameter_um
        self.pixel_size_nm = pixel_size_nm
        self.n_material = n_material
        self.n_ambient = n_ambient
        self.max_height_nm = 2 * wavelength_nm
        self.device = torch.device(device)

        self.n_pixels = int(diameter_um * 1000 / pixel_size_nm)
        self.grid_shape = (self.n_pixels, self.n_pixels)

        # Density field parameterization (shared θ)
        self.density_param = DensityField(
            grid_shape=self.grid_shape,
            eps_low=n_ambient**2,
            eps_high=n_material**2,
            beta=1.0,
        )

        # RCWA solver (EM forward model)
        self.solver = RCWASolver(
            fourier_orders=fourier_orders,
            wavelength_nm=wavelength_nm,
            period_nm=(pixel_size_nm, pixel_size_nm),
            eps_ambient=n_ambient**2,
            eps_substrate=n_ambient**2,
            device=self.device,
        )

        # Forward lithography model
        self.litho_model = HopkinsLithoModel(
            wavelength_nm=litho_wavelength_nm,
            na=litho_na,
            sigma_source=0.8,
            n_kernels=4,
            pixel_size_nm=pixel_size_nm,
            resist_threshold=0.5,
            resist_beta=20.0,
            device=self.device,
        )

        # Target phase profile
        coords = (
            torch.arange(self.n_pixels, dtype=torch.float64, device=self.device) * pixel_size_nm
        )
        coords = coords - coords[-1] / 2
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing="ij")
        k0 = 2 * math.pi / wavelength_nm
        na_frac = numerical_aperture / n_ambient
        f_nm = diameter_um * 1000 / 2 / math.tan(math.asin(na_frac))
        self.target_phase = k0 * (torch.sqrt(x_grid**2 + y_grid**2 + f_nm**2) - f_nm)

    def _optical_loss(
        self,
        printed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Optical FoM on the *printed* mask (key to C4)."""
        k0 = 2 * math.pi / self.wavelength_nm
        height = printed_mask * self.max_height_nm
        current_phase = k0 * (self.n_material - self.n_ambient) * height
        target_wrapped = self.target_phase % (2 * math.pi)
        current_wrapped = current_phase % (2 * math.pi)
        diff = torch.atan2(
            torch.sin(current_wrapped - target_wrapped),
            torch.cos(current_wrapped - target_wrapped),
        )
        return (diff**2).mean()

    def _litho_loss(self, mask: torch.Tensor) -> torch.Tensor:
        """Lithography EPE via Hopkins forward model."""
        result = self.litho_model.forward(mask)
        return result["epe"]

    def total_loss(
        self,
        density: torch.Tensor,
        lambda_optical: float = 1.0,
        lambda_litho: float = 0.1,
        lambda_fab: float = 0.01,
        beta: float = 10.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the unified autograd graph loss (C4 mechanism).

        Parameters
        ----------
        density : Tensor, shape ``(H, W)``
            Shared design parameter θ ∈ (0, 1).
        """
        # Shared parameterization → mask
        mask = self.density_param(density, beta=beta)

        # Lithography forward: M(θ) → aerial image → printed P(θ)
        litho_result = self.litho_model.forward(mask)
        printed = litho_result["printed_contour"]
        loss_litho = litho_result["epe"]

        # Optical forward on the *printed* contour
        loss_optical = self._optical_loss(printed)

        # Fabrication constraints
        loss_fab = combined_fabrication_penalty(mask)

        total = lambda_optical * loss_optical + lambda_litho * loss_litho + lambda_fab * loss_fab

        breakdown = {
            "total": total,
            "optical": loss_optical.detach(),
            "litho": loss_litho.detach(),
            "fab": loss_fab.detach(),
        }
        return total, breakdown

    def optimize(
        self,
        n_steps: int = 500,
        lr: float = 1e-2,
        lambda_optical: float = 1.0,
        lambda_litho: float = 0.1,
        lambda_fab: float = 0.01,
        robust: bool = False,
        sigma_nm: float = 5.0,
        n_mc_samples: int = 8,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float], list[dict]]:
        """Run the DFM-metalens optimization loop."""
        density = torch.rand(*self.grid_shape, device=self.device, dtype=torch.float64)
        density = density.detach().requires_grad_(True)

        opt = torch.optim.Adam([density], lr=lr)
        loss_history = []
        breakdown_history = []

        for step in range(n_steps):
            beta = beta_continuation_schedule(step, n_steps, beta_start=1.0, beta_end=16.0)

            if robust:
                from diffnano.design.robustness import robust_gradient_step

                # Safe: _loss_fn is called synchronously within robust_gradient_step,
                # so beta is the current step's value when the closure executes.
                def _loss_fn(d):
                    t, _ = self.total_loss(d, lambda_optical, lambda_litho, lambda_fab, beta)
                    return t

                loss = robust_gradient_step(
                    density,
                    _loss_fn,
                    sigma_nm=sigma_nm,
                    n_samples=n_mc_samples,
                )
                _, breakdown = self.total_loss(
                    density.detach(),
                    lambda_optical,
                    lambda_litho,
                    lambda_fab,
                    beta,
                )
            else:
                loss, breakdown = self.total_loss(
                    density,
                    lambda_optical,
                    lambda_litho,
                    lambda_fab,
                    beta,
                )

            if torch.isnan(loss):
                if verbose:
                    print(f"Step {step}: NaN loss, stopping.")
                break

            opt.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            if density.grad is not None:
                if torch.isnan(density.grad).any():
                    if verbose:
                        print(f"Step {step}: NaN gradient, stopping.")
                    break
                torch.nn.utils.clip_grad_norm_([density], max_norm=1.0)

            opt.step()

            # Clamp density to valid range
            with torch.no_grad():
                density.clamp_(0.0, 1.0)

            loss_history.append(loss.item())
            breakdown_history.append(
                {k: v.item() if hasattr(v, "item") else v for k, v in breakdown.items()}
            )

            if verbose and step % 50 == 0:
                print(
                    f"Step {step:4d}: total={loss.item():.6f} "
                    f"opt={breakdown['optical']:.4f} "
                    f"litho={breakdown['litho']:.4f} "
                    f"fab={breakdown['fab']:.4f}"
                )

        return density.detach(), loss_history, breakdown_history

    def decoupled_baseline(
        self,
        n_steps: int = 500,
        lr: float = 1e-2,
        lambda_optical: float = 1.0,
        lambda_fab: float = 0.01,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """Decoupled baseline: optical-only optimization, then litho check."""
        density = torch.rand(*self.grid_shape, device=self.device, dtype=torch.float64)
        density = density.detach().requires_grad_(True)

        opt = torch.optim.Adam([density], lr=lr)
        loss_history = []

        for step in range(n_steps):
            beta = beta_continuation_schedule(step, n_steps, beta_start=1.0, beta_end=16.0)

            mask = self.density_param(density, beta=beta)
            loss = self._optical_loss(mask) + lambda_fab * combined_fabrication_penalty(mask)

            if torch.isnan(loss):
                break

            opt.zero_grad()
            loss.backward()

            if density.grad is not None:
                if torch.isnan(density.grad).any():
                    break
                torch.nn.utils.clip_grad_norm_([density], max_norm=1.0)

            opt.step()
            with torch.no_grad():
                density.clamp_(0.0, 1.0)

            loss_history.append(loss.item())

            if verbose and step % 50 == 0:
                print(f"[Decoupled] Step {step:4d}: loss={loss.item():.6f}")

        return density.detach(), loss_history
