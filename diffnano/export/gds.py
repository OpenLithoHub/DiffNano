"""GDS-II layout export via gdstk.

Exports density fields and polygon contours to GDS format for
fabrication (e-beam, DUV, EUV).
"""

from __future__ import annotations

import torch

__all__ = ["export_density_to_gds", "export_polygons_to_gds"]


def export_density_to_gds(
    density: torch.Tensor,
    path: str,
    *,
    pixel_size_nm: float = 5.0,
    threshold: float = 0.5,
    layer: int = 0,
    datatype: int = 0,
) -> None:
    """Export a 2D density field to GDS-II.

    Converts the density to a binary polygon by thresholding and
    tracing the contour.

    Parameters
    ----------
    density : Tensor, shape ``(H, W)``
        Continuous density field ∈ [0, 1].
    path : str
        Output GDS file path.
    pixel_size_nm : float
        Physical size per pixel in nanometers.
    threshold : float
        Binarization threshold.
    layer : int
        GDS layer number.
    datatype : int
        GDS datatype.
    """
    try:
        import gdstk
    except ImportError:
        raise ImportError("gdstk is required for GDS export. Install with: pip install gdstk")

    import numpy as np

    arr = density.detach().cpu().numpy()
    binary = (arr > threshold).astype(np.uint8)

    H, W = binary.shape

    # Use gdstk's bitmap_to_polygon for efficient contour tracing
    polygons = gdstk.bitmap_to_polygon(
        binary,
        dx=pixel_size_nm,
        dy=pixel_size_nm,
        offset=(0, 0),
        scale=1.0,
        layer=layer,
        datatype=datatype,
    )

    lib = gdstk.Library("DiffNano")
    cell = lib.new_cell("DESIGN")
    if polygons:
        cell.add(*polygons)
    lib.write_gds(path)


def export_polygons_to_gds(
    polygons: list[torch.Tensor],
    path: str,
    *,
    layer: int = 0,
    datatype: int = 0,
) -> None:
    """Export a list of polygon contours to GDS-II.

    Parameters
    ----------
    polygons : list of Tensor, each shape ``(N_pts, 2)``
        Polygon vertices in nanometers.
    path : str
        Output GDS file path.
    layer : int
        GDS layer number.
    datatype : int
        GDS datatype.
    """
    try:
        import gdstk
    except ImportError:
        raise ImportError("gdstk is required for GDS export. Install with: pip install gdstk")


    lib = gdstk.Library("DiffNano")
    cell = lib.new_cell("DESIGN")

    for poly in polygons:
        pts = poly.detach().cpu().numpy()
        polygon = gdstk.Polygon(pts, layer=layer, datatype=datatype)
        cell.add(polygon)

    lib.write_gds(path)
