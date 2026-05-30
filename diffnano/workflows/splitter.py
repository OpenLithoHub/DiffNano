"""Beam splitter / power divider inverse design workflow.

PLACEHOLDER / UNIMPLEMENTED — This module is a non-functional stub.
``SplitterDesigner.transmission_efficiency`` returns ``geometry.mean()``
as a dummy proxy, not a real EM simulation. Do not use for evaluation.

Stub for v0.2 — defines the interface and a simple 1×2 splitter example.
"""

from __future__ import annotations

import torch

__all__ = ["SplitterDesigner"]


class SplitterDesigner:
    """Beam splitter inverse design workflow (stub).

    PLACEHOLDER / UNIMPLEMENTED — This class does not perform real EM
    simulation. ``transmission_efficiency`` returns ``geometry.mean()``.

    Parameters
    ----------
    wavelength_nm : float
        Operating wavelength.
    device : str
    """

    # NOTE: Placeholder implementation — not functional, do not use for evaluation

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

        PLACEHOLDER / UNIMPLEMENTED — Returns ``geometry.mean()``, not a
        real EM simulation result. Do not use for evaluation.

        Parameters
        ----------
        geometry : Tensor
            Device geometry.

        Returns
        -------
        efficiency : Tensor, scalar
        """
        # NOTE: Placeholder implementation — not functional, do not use for evaluation
        # Placeholder: return mean of geometry as proxy
        return geometry.mean()
