"""Beam splitter / power divider inverse design workflow.

Stub for v0.2 — defines the interface and a simple 1×2 splitter example.
"""

from __future__ import annotations

import torch

__all__ = ["SplitterDesigner"]


class SplitterDesigner:
    """Beam splitter inverse design workflow (stub).

    Parameters
    ----------
    wavelength_nm : float
        Operating wavelength.
    device : str
    """

    def __init__(
        self,
        wavelength_nm: float = 1550.0,
        device: str = "cpu",
    ):
        self.wavelength_nm = wavelength_nm
        self.device = device

    def transmission_efficiency(
        self,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Compute splitting efficiency (stub).

        Parameters
        ----------
        geometry : Tensor
            Device geometry.

        Returns
        -------
        efficiency : Tensor, scalar
        """
        # Placeholder: return mean of geometry as proxy
        return geometry.mean()
