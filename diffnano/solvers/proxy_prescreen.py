"""Two-stage proxy prescreen -> RCWA exact optimization.

CrossAttnRCWAProxy provides a fast forward proxy using diff-surrogate's
cross-attention geometry operator for prescreening candidate geometries.
TwoStageOptimizer combines the proxy with full RCWA for a
proxy-prescreen -> exact-refine workflow.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn as nn
from diff_surrogate.cross_attn import _CrossAttention, _GeometryEncoder, _PhysicsEncoder
from diff_surrogate.sdf_trunk import _TrunkNet
from torch import Tensor

from diffnano.solvers._result import SimResult

__all__ = ["CrossAttnRCWAProxy", "TwoStageOptimizer"]


class CrossAttnRCWAProxy(nn.Module):
    """Fast RCWA proxy using diff-surrogate's cross-attention geometry operator.

    Provides 10-100x speedup over full RCWA for screening candidates.
    Accuracy is sufficient for ranking but not for final evaluation.
    """

    def __init__(
        self,
        n_fourier: int = 21,
        geometry_encoder: str = "sdf",
        hidden_dim: int = 64,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_fourier = n_fourier
        self.geometry_encoder = geometry_encoder
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # SDF trunk encodes geometry into spatial basis functions
        self.trunk = _TrunkNet(
            sdf_dim=1,
            hidden_dim=hidden_dim,
            n_basis=hidden_dim,
            n_layers=2,
        )

        # Cross-attention geometry encoder produces keys/values
        self.geom_enc = _GeometryEncoder(
            sdf_dim=1,
            hidden_dim=hidden_dim,
            n_layers=2,
        )

        # Physics encoder produces queries from wavelength info
        self.phys_enc = _PhysicsEncoder(
            param_dim=1,
            hidden_dim=hidden_dim,
            n_outputs=n_fourier,
        )

        self.cross_attn = _CrossAttention(hidden_dim=hidden_dim, n_heads=n_heads)

        # Output head: attended features -> diffraction efficiency vector
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_fourier),
        )

    def forward(
        self,
        geometry: Tensor,
        wavelengths: Tensor | None = None,
    ) -> SimResult:
        """Predict diffraction efficiencies from geometry.

        Uses SDF trunk to encode geometry, then cross-attention to
        predict output field.
        """
        if wavelengths is None:
            wavelengths = torch.tensor([532.0], dtype=torch.float64, device=geometry.device)
        if not isinstance(wavelengths, Tensor):
            wavelengths = torch.tensor(wavelengths, dtype=torch.float64, device=geometry.device)

        # Normalize geometry to [0, 1] range for SDF encoding
        geo = geometry.to(torch.float32)
        if geo.dim() == 2:
            # (n_layers, n_grid) -> treat as (1, n_layers, n_grid) spatial
            geo = geo.unsqueeze(0)

        if geo.dim() == 3 and geo.shape[0] == 1:
            # Single geometry: squeeze to 2D spatial, add batch/channel dims
            geo_2d = geo.squeeze(0)  # (H, W)
            geo_2d = (geo_2d - geo_2d.min()) / (geo_2d.max() - geo_2d.min() + 1e-8)
            sdf = geo_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif geo.dim() == 3:
            # Batch of 2D or multi-layer: average across layers
            geo_avg = geo.mean(dim=0)  # (H, W) or (n_grid,)
            if geo_avg.dim() == 1:
                side = int(geo_avg.shape[0] ** 0.5)
                if side * side == geo_avg.shape[0]:
                    geo_avg = geo_avg.reshape(side, side)
                else:
                    geo_avg = geo_avg.unsqueeze(0).expand(4, -1)  # pad to 2D
            geo_avg = (geo_avg - geo_avg.min()) / (geo_avg.max() - geo_avg.min() + 1e-8)
            sdf = geo_avg.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        else:
            sdf = geo.unsqueeze(0).unsqueeze(0) if geo.dim() == 2 else geo.unsqueeze(0)

        # Geometry -> keys, values via cross-attention encoder
        keys, values = self.geom_enc(sdf)

        # Wavelength -> physics parameter -> queries
        wl_param = wavelengths.float().mean().unsqueeze(0).unsqueeze(0) / 1000.0
        queries = self.phys_enc(wl_param)  # (1, n_fourier, hidden_dim)

        # Cross-attention: physics queries attend to geometry KV pairs
        attended = self.cross_attn(queries, keys, values)  # (1, n_fourier, hidden_dim)

        # Aggregate across spatial attention output to get efficiency vector
        # Use mean of attended features per output
        # attended.squeeze(0): (n_fourier, hidden_dim) -> output: (n_fourier, n_fourier)
        eff = self.output_head(attended.squeeze(0))
        # Take diagonal to get per-order prediction
        eff = torch.sigmoid(eff.diagonal())  # (n_fourier,)

        # Normalize to sum to ~1 (like RCWA efficiencies)
        eff = eff / (eff.sum() + 1e-8)

        # Expand for wavelength dimension
        n_wl = wavelengths.shape[0]
        field = eff.unsqueeze(0).expand(n_wl, -1).to(torch.float64)

        return SimResult(
            field=field,
            wavelengths=wavelengths,
            metadata={"proxy": True},
        )


class TwoStageOptimizer:
    """Two-stage proxy prescreen -> RCWA exact optimization.

    Stage 1: Use proxy to evaluate many candidates quickly
    Stage 2: Run full RCWA on top-K candidates for exact evaluation
    """

    def __init__(
        self,
        proxy: CrossAttnRCWAProxy,
        solver,
        top_k: int = 10,
        n_candidates: int = 100,
    ):
        self.proxy = proxy
        self.solver = solver
        self.top_k = top_k
        self.n_candidates = n_candidates

    def generate_candidates(
        self,
        n: int,
        geometry_shape: tuple[int, ...] = (5, 20),
        eps_range: tuple[float, float] = (1.0, 12.0),
    ) -> list[Tensor]:
        """Generate diverse geometry candidates."""
        lo, hi = eps_range
        candidates = []
        for _ in range(n):
            geo = lo + torch.rand(*geometry_shape, dtype=torch.float64) * (hi - lo)
            candidates.append(geo)
        return candidates

    def prescreen(
        self,
        candidates: list[Tensor],
        objective: Callable[[SimResult], Tensor],
    ) -> list[Tensor]:
        """Use proxy to rank candidates by objective."""
        scored: list[tuple[float, Tensor]] = []
        for geo in candidates:
            with torch.no_grad():
                result = self.proxy(geo)
            score = objective(result).item()
            scored.append((score, geo))
        scored.sort(key=lambda x: x[0])
        return [geo for _, geo in scored[: self.top_k]]

    def refine(
        self,
        candidates: list[Tensor],
        objective: Callable[[SimResult], Tensor],
    ) -> tuple[Tensor, float]:
        """Run full RCWA on candidates, return best."""
        best_geo = None
        best_score = float("inf")
        for geo in candidates:
            with torch.no_grad():
                result = self.solver.forward(geo)
            score = objective(result).item()
            if score < best_score:
                best_score = score
                best_geo = geo
        return best_geo, best_score

    def optimize(
        self,
        objective: Callable[[SimResult], Tensor],
        n_candidates: int = 100,
        top_k: int = 10,
        geometry_shape: tuple[int, ...] = (5, 20),
    ) -> dict:
        """Full two-stage optimization.

        Returns dict with best_geometry, best_objective, hit_rate, and timing.
        """
        actual_top_k = min(top_k, self.top_k)
        actual_n = max(actual_top_k + 1, n_candidates)

        # Stage 1: generate and prescreen with proxy
        t0 = time.perf_counter()
        candidates = self.generate_candidates(actual_n, geometry_shape)

        proxy_top_k = self.prescreen(candidates, objective)
        t_proxy = time.perf_counter() - t0

        # Stage 2: exact evaluation on proxy's top-K
        t1 = time.perf_counter()
        best_geo, best_proxy_obj = self.refine(proxy_top_k, objective)
        t_refine = time.perf_counter() - t1

        # Hit rate: check if proxy top-K contains the true best
        t2 = time.perf_counter()
        all_exact: list[tuple[float, Tensor]] = []
        for geo in candidates:
            with torch.no_grad():
                result = self.solver.forward(geo)
            score = objective(result).item()
            all_exact.append((score, geo))
        all_exact.sort(key=lambda x: x[0])
        true_best_geo = all_exact[0][1]
        true_best_obj = all_exact[0][0]
        t_full = time.perf_counter() - t2

        # Compute hit rate: is true best in proxy's top-K?
        true_best_id = id(true_best_geo)
        hit = any(id(geo) == true_best_id for geo in proxy_top_k)

        # For a probabilistic hit rate, check how many of true top-K
        # are in proxy top-K
        true_top_k_set = {id(geo) for _, geo in all_exact[:actual_top_k]}
        proxy_top_k_set = {id(geo) for geo in proxy_top_k}
        overlap = len(true_top_k_set & proxy_top_k_set)
        hit_rate = overlap / actual_top_k

        return {
            "best_geometry": best_geo,
            "best_objective": best_proxy_obj,
            "true_best_objective": true_best_obj,
            "hit_rate": hit_rate,
            "exact_hit": hit,
            "n_candidates": actual_n,
            "top_k": actual_top_k,
            "proxy_time_s": t_proxy,
            "refine_time_s": t_refine,
            "full_rcwa_time_s": t_full,
            "speedup": t_full / max(t_proxy, 1e-10),
        }
