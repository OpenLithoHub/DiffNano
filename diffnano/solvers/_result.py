"""Simulation result container (avoids circular import)."""

from __future__ import annotations

import torch


class SimResult:
    """Container for simulation outputs.

    Attributes
    ----------
    field : torch.Tensor
        Electromagnetic field data. Shape depends on solver and geometry.
    wavelengths : torch.Tensor
        Wavelengths evaluated, shape ``(W,)``.
    metadata : dict
        Solver-specific metadata (e.g., S-matrix, Fourier orders).
    """

    __slots__ = ("field", "wavelengths", "metadata")

    def __init__(
        self,
        field: torch.Tensor,
        wavelengths: torch.Tensor,
        metadata: dict | None = None,
    ):
        self.field = field
        self.wavelengths = wavelengths
        self.metadata = metadata or {}

    @property
    def device(self) -> torch.device:
        return self.field.device
