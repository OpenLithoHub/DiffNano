"""Waveguide coupler and mode converter inverse design workflow.

Stub for v0.2 — defines the interface.
"""

from __future__ import annotations

import torch

__all__ = ["WaveguideDesigner"]


class WaveguideDesigner:
    """Waveguide inverse design workflow (stub).

    Parameters
    ----------
    wavelength_nm : float
    device : str
    """

    def __init__(
        self,
        wavelength_nm: float = 1550.0,
        device: str = "cpu",
    ):
        self.wavelength_nm = wavelength_nm
        self.device = device

    def mode_overlap(
        self,
        field: torch.Tensor,
        target_mode: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mode overlap integral (differentiable).

        Parameters
        ----------
        field : Tensor
            Simulated field profile.
        target_mode : Tensor
            Target mode profile.

        Returns
        -------
        overlap : Tensor, scalar
            Mode overlap ∈ [0, 1].
        """
        field_flat = field.flatten()
        target_flat = target_mode.flatten()
        overlap = torch.abs(field_flat @ target_flat.conj()) / (
            field_flat.norm() * target_flat.norm() + 1e-12
        )
        return overlap ** 2
