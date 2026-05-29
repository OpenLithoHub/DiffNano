"""Multi-corner deterministic process-window optimization.

Port of OpenLithoHub's process-window corner-sweep pattern (``process_window.py``)
into the DiffNano robustness framework.  Instead of Monte Carlo sampling over a
continuous perturbation distribution, this module evaluates a **small, fixed set
of deterministic perturbation corners** (e.g. +/- linewidth bias, dose offsets)
and aggregates the per-corner losses into a single weighted objective.

This is the nanophotonic / nanopatterned analogue of the lithographic four-corner
dose/focus sweep: fast, deterministic, and fully differentiable.

Usage
-----
>>> from diffnano.design.robustness.corner_opt import (
...     CornerSpec, corner_optimization_step,
... )
>>> corners = [
...     CornerSpec("nominal",  delta_nm= 0.0, weight=2.0),
...     CornerSpec("wide",     delta_nm= 5.0, weight=1.0),
...     CornerSpec("narrow",   delta_nm=-5.0, weight=1.0),
... ]
>>> loss = corner_optimization_step(
...     params=density_tensor,
...     forward_fn=fom_loss,
...     corners=corners,
...     optimizer=optimizer,
... )
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from diffnano.design.robustness.core import linewidth_perturbation

__all__ = [
    "CornerSpec",
    "corner_optimization_step",
]


# ---------------------------------------------------------------------------
# Corner specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CornerSpec:
    """One deterministic perturbation corner in the optimisation sweep.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"wide"``, ``"narrow"``).
    delta_nm : float
        Linewidth perturbation in nanometers applied at this corner.
        Positive = wider features, negative = narrower.
    weight : float
        Loss weight for this corner.  The total objective is the
        weighted sum ``nominal_weight * L_nominal + sum(w_i * L_i)``.
    """

    name: str
    delta_nm: float
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Default corner set
# ---------------------------------------------------------------------------

DEFAULT_CORNERS: tuple[CornerSpec, ...] = (
    CornerSpec("wide", delta_nm=5.0, weight=1.0),
    CornerSpec("narrow", delta_nm=-5.0, weight=1.0),
)


# ---------------------------------------------------------------------------
# Corner optimisation step
# ---------------------------------------------------------------------------


def corner_optimization_step(
    params: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    corners: Sequence[CornerSpec] = DEFAULT_CORNERS,
    nominal_weight: float = 1.0,
    optimizer: torch.optim.Optimizer | None = None,
    pixel_size_nm: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One optimisation step that jointly minimises nominal + corner losses.

    Evaluates ``forward_fn`` at the nominal (unperturbed) design and at every
    corner defined by *corners*.  The total loss is the weighted sum::

        total = nominal_weight * L_nominal
              + sum(corner.weight * L_corner for corner in corners)

    If *optimizer* is provided the gradients are back-propagated and the
    optimizer is stepped in-place.  If *optimizer* is ``None`` the caller
    is responsible for calling ``total_loss.backward()`` and stepping their
    own optimizer (useful when this function is composed into a larger
    training loop that manages the gradient lifecycle).

    Parameters
    ----------
    params : Tensor
        Current design parameters (geometry / density tensor).  Must carry
        ``requires_grad=True`` if an optimizer is supplied.
    forward_fn : callable
        ``forward_fn(geometry) -> scalar loss``.  The geometry tensor passed
        in is the (possibly perturbed) SDF or density field.
    corners : sequence of CornerSpec
        Deterministic perturbation corners to evaluate.
    nominal_weight : float
        Weight applied to the nominal (unperturbed) loss term.
    optimizer : torch.optim.Optimizer, optional
        Optimizer whose ``zero_grad`` / ``step`` will be called.
    pixel_size_nm : float
        Physical size of one pixel in nanometers, forwarded to
        :func:`~diffnano.design.robustness.core.linewidth_perturbation`.

    Returns
    -------
    total_loss : Tensor, scalar
        Weighted loss across the nominal point and all corners.
    params : Tensor
        The (updated) parameter tensor (same object that was passed in).

    Raises
    ------
    ValueError
        If *corners* is empty and *nominal_weight* is zero (no objective to
        minimise).
    """
    if not corners and nominal_weight == 0.0:
        raise ValueError(
            "corner_optimization_step: corners is empty and nominal_weight is "
            "zero — nothing to optimise."
        )

    if optimizer is not None:
        optimizer.zero_grad()

    # --- Nominal loss -------------------------------------------------------
    loss_nominal = forward_fn(params)

    total_loss = nominal_weight * loss_nominal

    # --- Corner losses ------------------------------------------------------
    for corner in corners:
        delta = params.new_tensor(corner.delta_nm, dtype=params.dtype)

        # Use the existing linewidth_perturbation from core.py (SDF shift).
        # For density-parameterized designs, callers can wrap forward_fn to
        # apply apply_perturbation_to_density instead.
        perturbed = linewidth_perturbation(params, delta, pixel_size_nm=pixel_size_nm)

        loss_corner = forward_fn(perturbed)
        total_loss = total_loss + corner.weight * loss_corner

    # --- Back-propagate & step ----------------------------------------------
    total_loss.backward()
    if optimizer is not None:
        optimizer.step()

    return total_loss, params
