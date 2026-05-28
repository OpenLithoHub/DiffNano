"""Reference designs and datasets for benchmarking.

Provides canonical test cases from the nanophotonics literature.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "silicon_grating_1d",
    "metalens_devlin2016",
]


def silicon_grating_1d(
    period_nm: float = 400.0,
    fill_factor: float = 0.5,
    n_grid: int = 100,
    eps_si: float = 12.0,
    eps_air: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a 1D silicon grating benchmark geometry.

    Parameters
    ----------
    period_nm : float
        Grating period in nm.
    fill_factor : float
        Duty cycle (fraction of Si).
    n_grid : int
        Number of grid points per period.
    eps_si : float
        Silicon permittivity.
    eps_air : float
        Air permittivity.
    device : str

    Returns
    -------
    eps_profile : Tensor, shape ``(n_grid,)``
        Permittivity sampled over one period.
    """
    x = torch.linspace(0, 1, n_grid, device=device, dtype=torch.float64)
    eps = torch.where(x < fill_factor, eps_si, eps_air)
    return eps


def metalens_devlin2016(
    n_pixels: int = 250,
    wavelength_nm: float = 532.0,
    na: float = 0.8,
    diameter_um: float = 50.0,
    n_material: float = 2.0,
    n_ambient: float = 1.0,
    device: str = "cpu",
) -> dict:
    """Reference metalens design from Devlin et al. 2016 (Science).

    Returns the target phase profile and an initial height map.

    Parameters
    ----------
    n_pixels : int
        Number of meta-atoms across the aperture.
    wavelength_nm : float
    na : float
    diameter_um : float
    n_material : float
    n_ambient : float
    device : str

    Returns
    -------
    dict with keys:
        "target_phase": Tensor (H, W)
        "pixel_size_nm": float
        "focal_length_um": float
    """
    pixel_size_nm = diameter_um * 1000 / n_pixels
    f = diameter_um / 2 / math.tan(math.asin(na / n_ambient))

    coords = torch.arange(n_pixels, dtype=torch.float64, device=device) * pixel_size_nm
    coords = coords - coords[-1] / 2
    y, x = torch.meshgrid(coords, coords, indexing="ij")

    k0 = 2 * math.pi / wavelength_nm
    f_nm = f * 1000
    target_phase = k0 * (torch.sqrt(x ** 2 + y ** 2 + f_nm ** 2) - f_nm)

    return {
        "target_phase": target_phase,
        "pixel_size_nm": pixel_size_nm,
        "focal_length_um": f,
    }
