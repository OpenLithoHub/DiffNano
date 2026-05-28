"""Backend-agnostic solver interface and solver implementations.

The `Solver` Protocol defines a uniform interface so RCWA, FDTD, FDFD,
and future backends (e.g., R-DIT) are interchangeable from the workflow layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch

from diffnano.solvers._result import SimResult

__all__ = [
    "Solver", "SimResult", "RCWASolver", "FDFDSolver2D",
    "FDTDSolver2D", "FDTDSolver3D", "NeuralSurrogate",
    "HopkinsLithoModel", "LearnedFabModel", "DifferentiableResistModel",
]


@runtime_checkable
class Solver(Protocol):
    """Backend-agnostic forward-solver interface.

    Any solver (RCWA, FDTD, FDFD, R-DIT) implementing this protocol can
    be dropped into DiffNano workflows without modification.
    """

    def forward(
        self,
        geometry: torch.Tensor,
        wavelengths: Sequence[float] | torch.Tensor,
        *,
        source: dict | None = None,
    ) -> SimResult:
        """Run a forward simulation.

        Parameters
        ----------
        geometry : torch.Tensor
            Device geometry (e.g., permittivity map or density field).
        wavelengths : sequence of float or Tensor
            Wavelengths to evaluate, in nanometers.
        source : dict, optional
            Source configuration (solver-specific).

        Returns
        -------
        SimResult
        """
        ...

    @property
    def device(self) -> torch.device:
        """Device on which computations are performed."""
        ...


# Deferred import to avoid circular dependency
def __getattr__(name):
    if name == "RCWASolver":
        from diffnano.solvers.rcwa import RCWASolver
        return RCWASolver
    if name == "FDFDSolver2D":
        from diffnano.solvers.fdfd2d import FDFDSolver2D
        return FDFDSolver2D
    if name == "FDTDSolver2D":
        from diffnano.solvers.fdtd2d import FDTDSolver2D
        return FDTDSolver2D
    if name == "FDTDSolver3D":
        from diffnano.solvers.fdtd3d import FDTDSolver3D
        return FDTDSolver3D
    if name == "NeuralSurrogate":
        from diffnano.solvers.surrogate import NeuralSurrogate
        return NeuralSurrogate
    if name == "HopkinsLithoModel":
        from diffnano.solvers.litho import HopkinsLithoModel
        return HopkinsLithoModel
    if name == "LearnedFabModel":
        from diffnano.solvers.fab_model import LearnedFabModel
        return LearnedFabModel
    if name == "DifferentiableResistModel":
        from diffnano.solvers.resist import DifferentiableResistModel
        return DifferentiableResistModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
