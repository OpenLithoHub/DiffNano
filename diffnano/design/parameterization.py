"""Geometry parameterizations with differentiable rasterization.

Supports three parameterization modes:
- HeightMap: height → phase profile (thin-element approximation)
- DensityField: density → permittivity (Heaviside projection)
- BSplineCurve: B-spline control points → binary mask (distance-field rasterization)

The B-spline + differentiable-distance-field machinery is referenced by
C4.2 and C5.2 dependent claims.  Tier 3 module.
"""

from __future__ import annotations

import math

import torch

__all__ = ["HeightMap", "DensityField", "BSplineCurve", "signed_distance_field"]


# ---------------------------------------------------------------------------
# Signed distance field (core primitive)
# ---------------------------------------------------------------------------


def signed_distance_field(
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    contours: torch.Tensor,
) -> torch.Tensor:
    """Compute differentiable signed distance field from closed contour(s).

    Uses soft-min for differentiable distance aggregation and a differentiable
    winding number for inside/outside classification.

    Parameters
    ----------
    grid_x, grid_y : Tensor, shape ``(H, W)``
        Coordinate grids.
    contours : Tensor, shape ``(N_contours, N_pts, 2)``
        Closed polygon contours defining the structure boundary.

    Returns
    -------
    sdf : Tensor, shape ``(H, W)``
        Negative inside the contour, positive outside.
    """
    n_contours = contours.shape[0]
    all_dists = []

    for ci in range(n_contours):
        pts = contours[ci]  # (N_pts, 2)
        n_pts = pts.shape[0]
        closed = torch.cat([pts, pts[:1]], dim=0)

        for pi in range(n_pts):
            a = closed[pi]
            b = closed[pi + 1]

            ab = b - a
            ab_len_sq = (ab**2).sum() + 1e-12

            ap_x = grid_x - a[0]
            ap_y = grid_y - a[1]

            t = (ap_x * ab[0] + ap_y * ab[1]) / ab_len_sq
            t = torch.clamp(t, 0.0, 1.0)

            closest_x = a[0] + t * ab[0]
            closest_y = a[1] + t * ab[1]

            dist = torch.sqrt((grid_x - closest_x) ** 2 + (grid_y - closest_y) ** 2 + 1e-12)
            all_dists.append(dist)

    # Soft-min for differentiable aggregation (avoids zero-gradient dead zones)
    dists = torch.stack(all_dists, dim=-1)  # (H, W, n_segments)
    softmin_temp = 10.0
    weights = torch.softmax(-softmin_temp * dists, dim=-1)
    sdf = (weights * dists).sum(dim=-1)

    # Differentiable winding number for inside/outside
    sign = _winding_number(grid_x, grid_y, contours)
    sdf = sdf * (1 - 2 * sign)

    return sdf


def _winding_number(
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    contours: torch.Tensor,
) -> torch.Tensor:
    """Compute differentiable winding number via atan2.

    Returns a value near 1 inside, 0 outside the contour.
    """
    inside = torch.zeros_like(grid_x)

    for ci in range(contours.shape[0]):
        pts = contours[ci]
        n = pts.shape[0]
        closed = torch.cat([pts, pts[:1]], dim=0)

        angles_sum = torch.zeros_like(grid_x)
        for i in range(n):
            dx1 = grid_x - closed[i, 0]
            dy1 = grid_y - closed[i, 1]
            dx2 = grid_x - closed[i + 1, 0]
            dy2 = grid_y - closed[i + 1, 1]

            cross = dx1 * dy2 - dy1 * dx2
            dot = dx1 * dx2 + dy1 * dy2
            angle = torch.atan2(cross, dot)
            angles_sum = angles_sum + angle

        winding = angles_sum / (2 * math.pi)
        inside = inside + torch.sigmoid(20.0 * (winding.abs() - 0.5))

    return inside.clamp(0.0, 1.0)


def _sdf_smooth(
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    control_points: torch.Tensor,
    n_eval: int = 100,
) -> torch.Tensor:
    """Compute smooth SDF from B-spline control points via differentiable rasterization.

    Uses soft-min for distance and winding number for sign.
    """
    t_vals = torch.linspace(0, 1, n_eval, device=control_points.device, dtype=control_points.dtype)
    curve = _eval_bspline_closed(control_points, t_vals)

    gx = grid_x.unsqueeze(-1)
    gy = grid_y.unsqueeze(-1)
    cx = curve[:, 0].reshape(1, 1, -1)
    cy = curve[:, 1].reshape(1, 1, -1)

    dist_sq = (gx - cx) ** 2 + (gy - cy) ** 2
    dists = torch.sqrt(dist_sq + 1e-12)

    # Soft-min for differentiable distance aggregation
    softmin_temp = 10.0
    weights = torch.softmax(-softmin_temp * dists, dim=-1)
    min_dist = (weights * dists).sum(dim=-1)

    # Differentiable winding number
    contours = curve.unsqueeze(0)
    inside = _winding_number(grid_x, grid_y, contours)

    return min_dist * (1 - 2 * inside)


def _eval_bspline_closed(
    control_points: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a closed uniform cubic B-spline (vectorized, differentiable)."""
    N = control_points.shape[0]

    t_scaled = t * N
    segments = t_scaled.floor().long() % N
    fracs = t_scaled - t_scaled.floor()

    # Standard uniform cubic B-spline: segment i blends ctrl pts i, i+1, i+2, i+3
    idx0 = segments % N
    idx1 = (segments + 1) % N
    idx2 = (segments + 2) % N
    idx3 = (segments + 3) % N

    p0 = control_points[idx0]
    p1 = control_points[idx1]
    p2 = control_points[idx2]
    p3 = control_points[idx3]

    f = fracs.unsqueeze(-1)
    curve = (
        (1 - f) ** 3 / 6 * p0
        + (3 * f**3 - 6 * f**2 + 4) / 6 * p1
        + (-3 * f**3 + 3 * f**2 + 3 * f + 1) / 6 * p2
        + f**3 / 6 * p3
    )
    return curve


# ---------------------------------------------------------------------------
# HeightMap parameterization
# ---------------------------------------------------------------------------


class HeightMap:
    """Height map → phase profile (thin-element approximation).

    Converts a height field to a phase map via:
        phase(x, y) = k0 * (n_material - n_ambient) * h(x, y)

    Parameters
    ----------
    grid_shape : tuple[int, int]
        Spatial grid dimensions ``(H, W)``.
    wavelength_nm : float
        Operating wavelength.
    n_material : float
        Refractive index of the material.
    n_ambient : float
        Refractive index of the ambient medium.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        wavelength_nm: float = 532.0,
        n_material: float = 2.0,
        n_ambient: float = 1.0,
    ):
        self.grid_shape = grid_shape
        self.wavelength_nm = wavelength_nm
        self.n_material = n_material
        self.n_ambient = n_ambient

    def forward(self, height_map: torch.Tensor) -> torch.Tensor:
        """Convert height map to phase profile.

        Parameters
        ----------
        height_map : Tensor, shape ``(H, W)``
            Physical height in nanometers.

        Returns
        -------
        phase : Tensor, shape ``(H, W)``
            Phase in radians.
        """
        k0 = 2 * math.pi / self.wavelength_nm
        dn = self.n_material - self.n_ambient
        return k0 * dn * height_map

    def __call__(self, height_map: torch.Tensor) -> torch.Tensor:
        return self.forward(height_map)


# ---------------------------------------------------------------------------
# DensityField parameterization
# ---------------------------------------------------------------------------


class DensityField:
    """Density → permittivity via Heaviside projection with β-continuation.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        ``(H, W)`` spatial grid.
    eps_low : float
        Permittivity of void.
    eps_high : float
        Permittivity of material.
    beta : float
        Projection sharpness (higher = more binary).
    eta : float
        Projection threshold (default 0.5).
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        eps_low: float = 1.0,
        eps_high: float = 12.0,
        beta: float = 1.0,
        eta: float = 0.5,
    ):
        self.grid_shape = grid_shape
        self.eps_low = eps_low
        self.eps_high = eps_high
        self.beta = beta
        self.eta = eta

    def forward(
        self,
        density: torch.Tensor,
        beta: float | None = None,
    ) -> torch.Tensor:
        """Map density field to permittivity.

        Parameters
        ----------
        density : Tensor, shape ``(H, W)``
            Continuous density ∈ [0, 1].
        beta : float, optional
            Override projection sharpness for this call.

        Returns
        -------
        permittivity : Tensor, shape ``(H, W)``
        """
        b = beta if beta is not None else self.beta
        # Smoothed Heaviside projection
        projected = torch.sigmoid(b * (density - self.eta))
        return self.eps_low + (self.eps_high - self.eps_low) * projected

    def set_beta(self, beta: float) -> None:
        self.beta = beta

    def __call__(self, density: torch.Tensor, beta: float | None = None) -> torch.Tensor:
        return self.forward(density, beta)


# ---------------------------------------------------------------------------
# BSplineCurve parameterization
# ---------------------------------------------------------------------------


class BSplineCurve:
    """B-spline control points → binary mask via differentiable distance-field rasterization.

    The distance-field machinery exposes differentiable shifts of the level set,
    which is reused by the C5 perturbation kernel.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        ``(H, W)`` rasterization grid.
    pixel_size_nm : float
        Physical size of one pixel in nanometers.
    n_eval : int
        Number of points to evaluate on the spline curve.
    threshold : float
        SDF threshold for binarization (default 0 → exact boundary).
    beta : float
        Sigmoid sharpness for soft binarization.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        pixel_size_nm: float = 5.0,
        n_eval: int = 100,
        threshold: float = 0.0,
        beta: float = 10.0,
    ):
        self.grid_shape = grid_shape
        self.pixel_size_nm = pixel_size_nm
        self.n_eval = n_eval
        self.threshold = threshold
        self.beta = beta

        # Pre-compute coordinate grids
        H, W = grid_shape
        y_coords = torch.arange(H, dtype=torch.float64) * pixel_size_nm
        x_coords = torch.arange(W, dtype=torch.float64) * pixel_size_nm
        self.grid_y, self.grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    def forward(
        self,
        control_points: torch.Tensor,
        shift: torch.Tensor | None = None,
        beta: float | None = None,
    ) -> torch.Tensor:
        """Rasterize B-spline control points to a density mask.

        Parameters
        ----------
        control_points : Tensor, shape ``(N_ctrl, 2)``
            B-spline control points in nanometers.
        shift : Tensor, optional
            Scalar or per-point shift of the zero level set (for C5 perturbation).
        beta : float, optional
            Override sigmoid sharpness.

        Returns
        -------
        mask : Tensor, shape ``(H, W)``
            Continuous mask ∈ (0, 1).
        """
        b = beta if beta is not None else self.beta
        device = control_points.device

        grid_x = self.grid_x.to(device)
        grid_y = self.grid_y.to(device)

        sdf = _sdf_smooth(grid_x, grid_y, control_points, self.n_eval)

        # Apply level-set shift (C5 perturbation)
        if shift is not None:
            sdf = sdf - shift

        # Soft binarization
        threshold = self.threshold
        mask = torch.sigmoid(-b * (sdf - threshold))
        return mask

    def sdf(
        self,
        control_points: torch.Tensor,
    ) -> torch.Tensor:
        """Return the raw signed distance field (for external use)."""
        device = control_points.device
        return _sdf_smooth(
            self.grid_x.to(device),
            self.grid_y.to(device),
            control_points,
            self.n_eval,
        )

    def __call__(
        self,
        control_points: torch.Tensor,
        shift: torch.Tensor | None = None,
        beta: float | None = None,
    ) -> torch.Tensor:
        return self.forward(control_points, shift, beta)
