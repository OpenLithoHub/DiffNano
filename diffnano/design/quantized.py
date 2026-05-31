"""Quantization-aware inverse design with straight-through estimators.

References:
    - Quantized Inverse Design for Photonic Integrated Circuits, ACS Omega, arXiv:2407.10273
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


__all__ = [
    "StraightThroughQuantize",
    "BinarySTE",
    "QuantizationNoiseGuardrail",
    "QuantizedOptimizer",
]


class _STEQuantizeFn(torch.autograd.Function):
    """Autograd function implementing straight-through estimator."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        diffs = (x.unsqueeze(-1) - levels).abs()
        indices = diffs.argmin(dim=-1)
        ctx.save_for_backward(x, levels, indices)
        return levels[indices]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class StraightThroughQuantize(nn.Module):
    """k-level quantization with straight-through estimator."""

    def __init__(self, n_levels: int = 2):
        super().__init__()
        if n_levels < 2:
            raise ValueError("n_levels must be >= 2")
        self.n_levels = n_levels
        levels = torch.linspace(0.0, 1.0, n_levels, dtype=torch.float64)
        self.register_buffer("levels", levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.levels.dtype)
        levels = self.levels.to(x.device)
        return _STEQuantizeFn.apply(x, levels)


class _BinarySTEPFn(torch.autograd.Function):
    """Binary STE: forward hard threshold, backward sigmoid gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x > 0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        sigmoid_grad = torch.sigmoid(x) * (1.0 - torch.sigmoid(x))
        return grad_output * sigmoid_grad


class BinarySTE(nn.Module):
    """Binary STE with sigmoid backward pass."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _BinarySTEPFn.apply(x)


class QuantizationNoiseGuardrail:
    """Checks STE gradient consistency with finite-difference approximation."""

    def __init__(self, eps: float = 1e-4, boundary_tolerance: float = 0.3):
        self.eps = eps
        self.boundary_tolerance = boundary_tolerance

    def check(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        loss_fn,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (direction_cosine, at_boundary) diagnostic tuple."""
        fd = torch.zeros_like(x)
        x_flat = x.detach().flatten()
        for i in range(x_flat.numel()):
            x_plus = x.detach().clone()
            x_minus = x.detach().clone()
            idx = tuple(torch.unravel_index(torch.tensor(i), x.shape))
            x_plus[idx] += self.eps
            x_minus[idx] -= self.eps
            fd_val = (loss_fn(x_plus) - loss_fn(x_minus)) / (2.0 * self.eps)
            fd.view(-1)[i] = fd_val

        grad_flat = grad.detach().flatten()
        fd_flat = fd.flatten()

        norm_g = grad_flat.norm()
        norm_f = fd_flat.norm()
        if norm_g < 1e-12 or norm_f < 1e-12:
            return torch.tensor(0.0), torch.tensor(False)

        cosine = torch.dot(grad_flat, fd_flat) / (norm_g * norm_f)

        dists = (x - 0.5).abs().flatten()
        level_gap = 1.0
        boundary_threshold = 0.05 * level_gap
        at_boundary = (dists < boundary_threshold).any()

        return cosine, at_boundary


class QuantizedOptimizer:
    """Wraps design optimization with STE quantization and compares approaches."""

    def __init__(
        self,
        grid_shape: Tuple[int, int] = (16, 16),
        quantizer: Optional[nn.Module] = None,
        lr: float = 0.05,
        n_steps: int = 50,
        device: str = "cpu",
    ):
        self.grid_shape = grid_shape
        self.quantizer = quantizer or StraightThroughQuantize(n_levels=2)
        self.lr = lr
        self.n_steps = n_steps
        self.device = torch.device(device)

    def _run_single(
        self,
        fom_fn,
        seed: int,
        quantize_aware: bool,
    ) -> Tuple[torch.Tensor, float]:
        torch.manual_seed(seed)
        x = torch.rand(
            *self.grid_shape, dtype=torch.float64, device=self.device, requires_grad=True
        )
        opt = torch.optim.Adam([x], lr=self.lr)

        for step in range(self.n_steps):
            opt.zero_grad()
            if quantize_aware:
                xq = self.quantizer(x)
            else:
                xq = x
            fom = fom_fn(xq)
            loss = -fom
            loss.backward()
            if x.grad is not None and torch.isnan(x.grad).any():
                break
            opt.step()
            with torch.no_grad():
                x.clamp_(0.0, 1.0)

        with torch.no_grad():
            if quantize_aware:
                final = self.quantizer(x).detach()
            else:
                final = self.quantizer(x.detach()).detach()
            final_fom = fom_fn(final).item()
        return final, final_fom

    def compare_approaches(self, fom_fn, n_seeds: int = 8) -> dict:
        """Run both quantization-aware and post-hoc quantization, return comparison."""
        from scipy.stats import wilcoxon

        qa_foms = []
        ph_foms = []

        for seed in range(n_seeds):
            _, qa_fom = self._run_single(fom_fn, seed, quantize_aware=True)
            _, ph_fom = self._run_single(fom_fn, seed, quantize_aware=False)
            qa_foms.append(qa_fom)
            ph_foms.append(ph_fom)

        try:
            _, p_value = wilcoxon(qa_foms, ph_foms)
        except ValueError:
            p_value = 1.0

        qa_wins = sum(1 for q, p in zip(qa_foms, ph_foms) if q > p) / max(n_seeds, 1)

        return {
            "quantization_aware_foms": qa_foms,
            "post_hoc_foms": ph_foms,
            "p_value": p_value,
            "qa_win_rate": qa_wins,
        }
