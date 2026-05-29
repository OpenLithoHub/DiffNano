"""Bitmap-based designable-region mask for topology optimization.

Borrowed from mini-vec-engine's ``Bitmap<W>`` API (``iter_set_bits``,
``popcount``, ``and``/``or``).  Instead of a Rust u64 word-array, we use a
PyTorch boolean tensor as the bitmap.  This lets us track "designable" vs
"frozen" pixels and zero out gradients / clamp density for frozen regions
without wasting compute on the entire grid.

Key operations:
- ``from_bounds`` / ``from_circle``: factory constructors
- ``freeze_region``: shrink the designable region
- ``apply_mask``: zero out gradients for frozen pixels
- ``apply_mask_to_density``: clamp frozen pixels to 0 or 1
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "DesignableMask",
    "apply_mask",
    "apply_mask_to_density",
]


class DesignableMask:
    """Boolean bitmap tracking designable (True) vs frozen (False) pixels.

    Parameters
    ----------
    mask : Tensor, shape ``(H, W)``, dtype ``torch.bool``
        ``True`` marks a designable pixel, ``False`` marks a frozen pixel.
    """

    def __init__(self, mask: torch.Tensor):
        if mask.dtype != torch.bool:
            raise ValueError(f"mask must be bool, got {mask.dtype}")
        self._mask = mask.clone()

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_bounds(
        cls,
        shape: tuple[int, int],
        bounds: tuple[
            tuple[int, int] | None,
            tuple[int, int] | None,
        ],
        device: str | torch.device = "cpu",
    ) -> DesignableMask:
        """Create a mask where only a rectangular sub-region is designable.

        Parameters
        ----------
        shape : (H, W)
            Full grid dimensions.
        bounds : ((row_start, row_end), (col_start, col_end))
            Inclusive-exclusive row and column bounds.  ``None`` means
            "full extent" on that axis.
        device : str or torch.device

        Returns
        -------
        DesignableMask
        """
        H, W = shape
        row_bounds = bounds[0] if bounds[0] is not None else (0, H)
        col_bounds = bounds[1] if bounds[1] is not None else (0, W)

        mask = torch.zeros(H, W, dtype=torch.bool, device=device)
        mask[row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1]] = True
        return cls(mask)

    @classmethod
    def from_circle(
        cls,
        shape: tuple[int, int],
        center: tuple[float, float],
        radius: float,
        device: str | torch.device = "cpu",
    ) -> DesignableMask:
        """Create a mask where only a circular region is designable.

        Parameters
        ----------
        shape : (H, W)
            Full grid dimensions.
        center : (cy, cx)
            Center pixel coordinates.
        radius : float
            Radius in pixels.
        device : str or torch.device

        Returns
        -------
        DesignableMask
        """
        H, W = shape
        y = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1)
        x = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0)
        dist_sq = (y - center[0]) ** 2 + (x - center[1]) ** 2
        mask = dist_sq <= radius ** 2
        return cls(mask)

    @classmethod
    def all_designable(
        cls,
        shape: tuple[int, int],
        device: str | torch.device = "cpu",
    ) -> DesignableMask:
        """Create a mask where every pixel is designable."""
        return cls(torch.ones(shape, dtype=torch.bool, device=device))

    # ------------------------------------------------------------------
    # Bitmap-style queries (mirrors Bitmap<W> API)
    # ------------------------------------------------------------------

    @property
    def tensor(self) -> torch.Tensor:
        """Underlying boolean tensor."""
        return self._mask

    @property
    def shape(self) -> tuple[int, ...]:
        return self._mask.shape

    @property
    def device(self) -> torch.device:
        return self._mask.device

    def designable_count(self) -> int:
        """Number of designable (True) pixels — equivalent to ``popcount``."""
        return int(self._mask.sum().item())

    def freeze_count(self) -> int:
        """Number of frozen (False) pixels."""
        return int((~self._mask).sum().item())

    def total_pixels(self) -> int:
        return self._mask.numel()

    # ------------------------------------------------------------------
    # Set operations (mirrors Bitmap<W>.and / .or / .not)
    # ------------------------------------------------------------------

    def and_(self, other: DesignableMask) -> DesignableMask:
        """Intersection — only pixels designable in *both* masks survive."""
        return DesignableMask(self._mask & other._mask)

    def or_(self, other: DesignableMask) -> DesignableMask:
        """Union — pixels designable in *either* mask survive."""
        return DesignableMask(self._mask | other._mask)

    def not_(self) -> DesignableMask:
        """Complement — flip designable / frozen."""
        return DesignableMask(~self._mask)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def freeze_region(self, region: torch.Tensor) -> None:
        """Freeze (remove from designable set) the pixels where *region* is True.

        Parameters
        ----------
        region : Tensor, shape ``(H, W)``, dtype ``bool``
            Pixels to freeze.
        """
        if region.dtype != torch.bool:
            region = region.bool()
        self._mask[region] = False

    def designable_region(self, region: torch.Tensor) -> None:
        """Mark additional pixels as designable.

        Parameters
        ----------
        region : Tensor, shape ``(H, W)``, dtype ``bool``
            Pixels to mark as designable.
        """
        if region.dtype != torch.bool:
            region = region.bool()
        self._mask[region] = True

    # ------------------------------------------------------------------
    # Iteration (mirrors iter_set_bits)
    # ------------------------------------------------------------------

    def iter_set_bits(self) -> list[tuple[int, int]]:
        """Return ``(row, col)`` of every designable pixel.

        Unlike the Rust bitmap which returns a lazy iterator, this returns
        a list because PyTorch tensors don't support Rust-style bit iteration
        natively.  For large masks, prefer vectorized operations
        (``apply_mask``) instead.
        """
        ys, xs = torch.where(self._mask)
        return list(zip(ys.tolist(), xs.tolist()))

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = self.designable_count()
        t = self.total_pixels()
        return (
            f"DesignableMask(shape={self.shape}, "
            f"designable={n}/{t}, frozen={t - n})"
        )


# ----------------------------------------------------------------------
# Functional API
# ----------------------------------------------------------------------


def apply_mask(
    gradient: torch.Tensor,
    mask: DesignableMask,
) -> torch.Tensor:
    """Zero out gradients for frozen pixels.

    Only keeps gradients where the designable mask is True.

    Parameters
    ----------
    gradient : Tensor, shape ``(H, W)``
        Gradient tensor.
    mask : DesignableMask
        Designable region.

    Returns
    -------
    masked_gradient : Tensor
        Same shape as *gradient*, with frozen pixels zeroed.
    """
    return gradient * mask.tensor.to(dtype=gradient.dtype, device=gradient.device)


def apply_mask_to_density(
    density: torch.Tensor,
    mask: DesignableMask,
    frozen_value: float = 0.0,
) -> torch.Tensor:
    """Clamp frozen pixels to a fixed value.

    Parameters
    ----------
    density : Tensor, shape ``(H, W)``
        Current density field.
    mask : DesignableMask
        Designable region.
    frozen_value : float
        Value to assign to frozen pixels (default 0.0).

    Returns
    -------
    clamped : Tensor
        Density with frozen regions set to *frozen_value*.
    """
    m = mask.tensor.to(dtype=density.dtype, device=density.device)
    return density * m + frozen_value * (1.0 - m)
