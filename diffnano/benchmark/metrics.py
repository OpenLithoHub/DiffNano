"""Benchmark metrics for evaluating nanophotonic designs.

Provides standardized figures of merit:
- Transmission efficiency
- Strehl ratio
- Bandwidth
- Bandgap/midgap ratio
"""

from __future__ import annotations

import torch

__all__ = [
    "transmission_efficiency",
    "strehl_ratio_from_phase",
    "strehl_ratio_from_field",
    "bandgap_ratio",
]


def transmission_efficiency(
    transmitted: torch.Tensor,
    incident: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute transmission efficiency.

    Parameters
    ----------
    transmitted : Tensor
        Transmitted power per diffraction order, shape ``(W, N_orders)``
        or scalar.
    incident : Tensor, optional
        Incident power.  Defaults to 1.0.

    Returns
    -------
    efficiency : Tensor, scalar
    """
    total = transmitted.sum(dim=-1)
    if incident is None:
        return total
    return total / (incident + 1e-12)


def strehl_ratio_from_phase(
    phase_error: torch.Tensor,
) -> torch.Tensor:
    """Strehl ratio from phase error variance.

    Strehl ≈ exp(-σ²_φ)  (Marechal approximation).

    Parameters
    ----------
    phase_error : Tensor, shape ``(...)``
        Phase error in radians.

    Returns
    -------
    strehl : Tensor, scalar
    """
    var = (phase_error**2).mean()
    return torch.exp(-var)


def strehl_ratio_from_field(
    field: torch.Tensor,
    target_field: torch.Tensor,
) -> torch.Tensor:
    """Strehl ratio from field intensity at focal point.

    Strehl = I_peak / I_peak_ideal

    Parameters
    ----------
    field : Tensor, shape ``(H, W)`` or ``(N, H, W)``
        Simulated intensity.
    target_field : Tensor
        Ideal (aberration-free) intensity.

    Returns
    -------
    strehl : Tensor, scalar
    """
    I_peak = field.max()
    I_ideal = target_field.max()
    return I_peak / (I_ideal + 1e-12)


def bandgap_ratio(
    frequencies: torch.Tensor,
    transmission: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Compute bandgap/midgap ratio from transmission spectrum.

    Parameters
    ----------
    frequencies : Tensor, shape ``(N,)``
        Frequency points.
    transmission : Tensor, shape ``(N,)``
        Transmission at each frequency.
    threshold : float
        Transmission threshold below which a bandgap exists.

    Returns
    -------
    ratio : Tensor, scalar
        gap/midgap ratio.  Returns 0 if no gap detected.
    """
    in_gap = transmission < threshold
    if not in_gap.any():
        return torch.tensor(0.0, device=frequencies.device)

    indices = torch.where(in_gap)[0]
    f_low = frequencies[indices[0]]
    f_high = frequencies[indices[-1]]
    gap = f_high - f_low
    mid = (f_high + f_low) / 2
    return gap / (mid + 1e-12)
