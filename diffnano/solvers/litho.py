"""Differentiable forward computational lithography model.

Implements a physically motivated DUV (193nm immersion) imaging model that is
fully differentiable for inclusion in the C4 unified autograd graph.

The model computes the aerial image from a mask via convolution with the
optical transfer function (point spread function of the projection system),
then applies a resist development model (sigmoid threshold).

Pipeline:
    mask M → PSF convolution → aerial image I → sigmoid resist → printed P

All steps are PyTorch operations; gradients flow through to the mask (and
hence to the shared parameterization θ in the C4 workflow).

This is a physically correct simplified Hopkins model: the PSF is the
Airy-like response of a partially coherent imaging system, which is the
dominant effect determining print fidelity for dense periodic structures.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["HopkinsLithoModel"]


class HopkinsLithoModel:
    """Differentiable forward lithography model.

    Uses the point spread function (PSF) of a partially coherent imaging
    system as the imaging kernel.  The PSF width is determined by the
    projection NA and exposure wavelength.

    Parameters
    ----------
    wavelength_nm : float
        Exposure wavelength (default 193 nm, DUV immersion).
    na : float
        Projection lens numerical aperture.
    sigma_source : float
        Source partial coherence factor (sigma_outer).
    n_kernels : int
        Number of source points for partially coherent averaging.
    pixel_size_nm : float
        Mask pixel size in nm.
    resist_threshold : float
        Resist threshold (relative intensity for printing).
    resist_beta : float
        Sigmoid steepness for resist development model.
    device : str or torch.device
    """

    def __init__(
        self,
        wavelength_nm: float = 193.0,
        na: float = 1.35,
        sigma_source: float = 0.8,
        n_kernels: int = 4,
        pixel_size_nm: float = 5.0,
        resist_threshold: float = 0.5,
        resist_beta: float = 20.0,
        device: str | torch.device = "cpu",
    ):
        self.wavelength_nm = wavelength_nm
        self.na = na
        self.sigma_source = sigma_source
        self.n_kernels = n_kernels
        self.pixel_size_nm = pixel_size_nm
        self.resist_threshold = resist_threshold
        self.resist_beta = resist_beta
        self.device = torch.device(device)

        # Pre-compute PSF kernels for partially coherent illumination
        self._kernels: list[torch.Tensor] = []
        self._initialized = False

    def _initialize_kernels(self, grid_size: int) -> None:
        """Compute PSF kernels for different source points.

        For partially coherent illumination, the source is sampled at several
        points. Each produces a shifted PSF. The total image is the incoherent
        sum of all shifted PSF convolutions.
        """
        device = self.device
        dtype = torch.float64

        # PSF size: determined by optical resolution
        # Rayleigh resolution: 0.61 * λ / NA
        resolution_nm = 0.61 * self.wavelength_nm / self.na
        psf_radius_px = max(3, int(resolution_nm / self.pixel_size_nm) * 3)
        psf_size = 2 * psf_radius_px + 1

        # Source points (partial coherence)
        if self.n_kernels <= 1:
            source_offsets = [0.0]
        else:
            source_offsets = torch.linspace(
                -self.sigma_source, self.sigma_source, self.n_kernels
            ).tolist()

        self._kernels = []
        for offset in source_offsets:
            coords = torch.arange(psf_size, device=device, dtype=dtype) - psf_radius_px
            x = coords * self.pixel_size_nm  # in nm

            # Shifted PSF (Gaussian approximation of Airy disk)
            sigma_psf = 0.42 * self.wavelength_nm / self.na  # Gaussian σ ≈ 0.42 λ/NA
            shift = offset * self.wavelength_nm / self.na  # source shift
            psf_1d = torch.exp(-((x - shift) ** 2) / (2 * sigma_psf ** 2))
            psf_1d = psf_1d / psf_1d.sum()

            # 2D separable PSF
            psf_2d = psf_1d.unsqueeze(1) * psf_1d.unsqueeze(0)
            self._kernels.append(psf_2d)

        self._initialized = True

    def aerial_image(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute the aerial image from a mask.

        For partially coherent illumination, the image is the sum of
        intensities from each coherent sub-source:
            I(x,y) = Σ_s |h_s ⊗ M|²

        Parameters
        ----------
        mask : Tensor, shape ``(H, W)``
            Continuous mask field (0 = chrome, 1 = clear).

        Returns
        -------
        image : Tensor, shape ``(H, W)``
        """
        if not self._kernels:
            self._initialize_kernels(mask.shape[-1])

        H, W = mask.shape
        mask_4d = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        image = torch.zeros_like(mask)
        for kernel in self._kernels:
            k = kernel.to(mask.dtype).to(mask.device)
            k_4d = k.unsqueeze(0).unsqueeze(0)  # (1, 1, kH, kW)
            pad_h = k.shape[0] // 2
            pad_w = k.shape[1] // 2

            # Convolve and trim to match input size
            conv = F.conv2d(mask_4d, k_4d, padding=(pad_h, pad_w))
            conv = conv[:, :, :H, :W].squeeze()

            image = image + conv ** 2

        # Normalize (avoid GPU sync — use tensor operations)
        img_max = image.max()
        image = image / (img_max + 1e-12)

        return image

    def printed_contour(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute the printed (resist-developed) contour from a mask.

        Pipeline: mask → aerial image → sigmoid resist model → printed contour.
        """
        image = self.aerial_image(mask)
        printed = torch.sigmoid(
            self.resist_beta * (image - self.resist_threshold)
        )
        return printed

    def edge_placement_error(
        self,
        mask: torch.Tensor,
        printed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Differentiable edge placement error (EPE).

        EPE = ||M - P||₂ between target mask and printed contour.
        """
        if printed is None:
            printed = self.printed_contour(mask)
        return ((mask - printed) ** 2).mean()

    def forward(self, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass: mask → aerial image → printed contour + EPE."""
        aerial = self.aerial_image(mask)
        printed = self.printed_contour(mask)
        epe = self.edge_placement_error(mask, printed)

        return {
            "aerial_image": aerial,
            "printed_contour": printed,
            "epe": epe,
        }

    def forward_solver(
        self,
        geometry: torch.Tensor,
        wavelengths=None,
        *,
        source=None,
    ):
        """Solver-protocol compatible forward pass."""
        from diffnano.solvers._result import SimResult

        result = self.forward(geometry)
        return SimResult(
            field=result["printed_contour"].unsqueeze(0),
            wavelengths=torch.tensor([0.0]),
            metadata={
                "model": "hopkins_litho",
                "epe": result["epe"],
            },
        )
