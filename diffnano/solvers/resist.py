"""Differentiable resist model (C6 — inspired by TorchResist).

Analytical resist model with interpretable parameters: acid diffusion,
post-exposure bake (PEB) diffusion, and development contrast.

Clean-room reimplementation from the paper (arXiv:2502.06838), not
from TorchResist source.

References
----------
- Geng et al. (2025), TorchResist: arXiv:2502.06838, SPIE 2025
"""

from __future__ import annotations

import torch

from diffnano.solvers._result import SimResult

__all__ = ["DifferentiableResistModel"]


class DifferentiableResistModel:
    """Differentiable analytical resist model.

    Models the photoresist processing chain:
        exposure → acid generation → PEB diffusion → development

    All parameters are differentiable and can be calibrated to a
    target process node.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        ``(H, W)`` grid dimensions.
    dl : float
        Grid spacing in nm.
    acid_diffusion_length_nm : float
        Acid diffusion length during PEB (nm).
    development_contrast : float
        Resist contrast (higher = sharper development).
    threshold_dose : float
        Normalized threshold dose for clearing.
    peb_diffusion_nm : float
        Post-exposure bake diffusion length (nm).
    device : str or torch.device
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (64, 64),
        dl: float = 5.0,
        acid_diffusion_length_nm: float = 20.0,
        development_contrast: float = 10.0,
        threshold_dose: float = 0.5,
        peb_diffusion_nm: float = 10.0,
        device: str | torch.device = "cpu",
    ):
        self.grid_shape = grid_shape
        self.dl = dl
        self._device = torch.device(device)

        # Differentiable parameters
        self.acid_diffusion = torch.tensor(
            acid_diffusion_length_nm,
            dtype=torch.float64,
            device=self._device,
        )
        self.contrast = torch.tensor(
            development_contrast,
            dtype=torch.float64,
            device=self._device,
        )
        self.threshold = torch.tensor(
            threshold_dose,
            dtype=torch.float64,
            device=self._device,
        )
        self.peb_diffusion = torch.tensor(
            peb_diffusion_nm,
            dtype=torch.float64,
            device=self._device,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def _gaussian_blur(
        self,
        field: torch.Tensor,
        sigma_nm: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Gaussian blur with differentiable sigma.

        Uses separable 1D Gaussian convolution.
        """
        sigma_px = sigma_nm / self.dl
        # Use fixed kernel size to maintain gradient flow through sigma
        k_size = max(7, int(6 * max(sigma_px.item(), 2.0)) + 1)
        if k_size % 2 == 0:
            k_size += 1

        # Clamp kernel size to input dimensions
        H, W = field.shape
        k_size = min(k_size, min(H, W))
        if k_size % 2 == 0:
            k_size -= 1
        if k_size < 3:
            return field

        k_half = k_size // 2

        x = torch.arange(-k_half, k_half + 1, device=field.device, dtype=field.dtype)
        kernel_1d = torch.exp(-(x**2) / (2 * sigma_px**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Separable 2D convolution
        padded = torch.nn.functional.pad(
            field.unsqueeze(0).unsqueeze(0),
            [k_half] * 4,
            mode="reflect",
        )
        h_kernel = kernel_1d.reshape(1, 1, 1, -1)
        h_blurred = torch.nn.functional.conv2d(padded, h_kernel, padding=0)
        v_kernel = kernel_1d.reshape(1, 1, -1, 1)
        result = torch.nn.functional.conv2d(h_blurred, v_kernel, padding=0)

        return result.squeeze(0).squeeze(0)

    def forward(
        self,
        exposure_dose: torch.Tensor,
        wavelengths=None,
        *,
        source=None,
    ) -> SimResult:
        """Simulate the resist processing chain.

        Parameters
        ----------
        exposure_dose : Tensor, shape ``(H, W)``
            Aerial image / exposure dose distribution.
        wavelengths : ignored
        source : ignored

        Returns
        -------
        SimResult
            ``field`` contains the developed resist profile (1 = resist remains,
            0 = cleared).
        """
        dose = exposure_dose.to(self._device).to(torch.float64)

        # Step 1: Acid generation (linear response for simplicity)
        acid_concentration = dose

        # Step 2: Acid diffusion during PEB
        acid_diffused = self._gaussian_blur(acid_concentration, self.acid_diffusion)

        # Step 3: PEB diffusion of reaction products
        peb_result = self._gaussian_blur(acid_diffused, self.peb_diffusion)

        # Step 4: Development (sigmoid with controllable contrast)
        resist_thickness = torch.sigmoid(self.contrast * (self.threshold - peb_result))

        return SimResult(
            field=resist_thickness.unsqueeze(0),
            wavelengths=torch.tensor([0.0], device=self._device),
            metadata={
                "model": "resist",
                "acid_diffusion_nm": self.acid_diffusion.item(),
                "contrast": self.contrast.item(),
                "threshold": self.threshold.item(),
                "peb_diffusion_nm": self.peb_diffusion.item(),
            },
        )

    def parameters(self) -> list[torch.Tensor]:
        """Return differentiable parameters for calibration."""
        return [self.acid_diffusion, self.contrast, self.threshold, self.peb_diffusion]

    def calibrate(
        self,
        target_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        n_steps: int = 100,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> list[float]:
        """Calibrate resist parameters to match target (dose, resist_profile) pairs.

        Parameters
        ----------
        target_pairs : list of (dose, target_resist) tuples
        n_steps : int
        lr : float
        verbose : bool

        Returns
        -------
        loss_history : list of float
        """
        # Ensure parameters require grad
        for p in self.parameters():
            p.requires_grad_(True)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_history = []

        for step in range(n_steps):
            total_loss = 0.0
            for dose, target in target_pairs:
                pred = self.forward(dose).field.squeeze(0)
                target_dev = target.to(self._device).to(torch.float64)
                loss = ((pred - target_dev) ** 2).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

                # Constrain parameters
                with torch.no_grad():
                    self.acid_diffusion.clamp_(min=1.0)
                    self.contrast.clamp_(min=1.0)
                    self.peb_diffusion.clamp_(min=0.0)
                    self.threshold.clamp_(0.0, 1.0)

                total_loss += loss.item()

            avg_loss = total_loss / len(target_pairs)
            loss_history.append(avg_loss)

            if verbose and step % 20 == 0:
                print(f"Step {step}: loss={avg_loss:.6f}")

        return loss_history
