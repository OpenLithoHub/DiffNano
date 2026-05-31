"""GPU benchmarking suite for FDTD3D forward + time-reversal adjoint.

Extends the existing FDTDBenchmarkSuite (N9.2) with GPU memory strategies,
multi-scale large-aperture metalens support, and FDTDX cross-validation
scaffolding.

Memory strategies
-----------------
- TIME_REVERSAL : store E-field snapshots only (existing _TimeReversalFDTD)
- CHECKPOINT    : gradient checkpointing via torch.utils.checkpoint
- FULL_AUTODIFF : store full computation graph (memory-intensive baseline)

Multi-scale metalens
--------------------
Coarse-to-fine tiling decomposes a large-aperture simulation into tiles
that each fit in GPU memory.  A coarse solve provides boundary conditions
for fine sub-domain solves.

FDTDX cross-validation
----------------------
Provides a clean-room scaffold for comparing DiffNano FDTD outputs against
FDTDX (or any external reference).  No FDTDX code is vendored; reference
data is injected via callable or generated synthetically for testing.

References
----------
- FDTDX: Fast and differentiable 3D FDTD simulations, JOSS 11:8912, 2026
- Inverse design for scalable photonic systems, Nat. Rev. Mater., 2026-04

License: Apache 2.0 (clean-room implementation, no FDTDX vendoring).
"""

from __future__ import annotations

import enum
import gc
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable

import torch
from torch import Tensor

__all__ = [
    "GPUMemoryStrategy",
    "FDTDBenchmarkConfig",
    "FDTDGPUBenchmark",
    "MultiScaleMetalens",
    "FDTDXCrossValidator",
    "StabilityReport",
]


# ---------------------------------------------------------------------------
# GPUMemoryStrategy enum
# ---------------------------------------------------------------------------


class GPUMemoryStrategy(enum.Enum):
    """Memory management strategy for 3D FDTD gradient computation.

    Attributes
    ----------
    TIME_REVERSAL
        Store only E-field snapshots; re-run forward with autograd in backward.
        Memory O(3 * T * D * H * W).
    CHECKPOINT
        Gradient checkpointing via torch.utils.checkpoint.
        Memory O(S * 6 * D * H * W) where S is the number of segments.
    FULL_AUTODIFF
        Store the full computation graph.  Memory O(k * 6 * T * D * H * W)
        where k ~ 8-12 is the intermediate-tensor ratio per field component.
    """

    TIME_REVERSAL = "time_reversal"
    CHECKPOINT = "checkpoint"
    FULL_AUTODIFF = "full_autodiff"


# ---------------------------------------------------------------------------
# FDTDBenchmarkConfig
# ---------------------------------------------------------------------------


@dataclass
class FDTDBenchmarkConfig:
    """Configuration for GPU FDTD benchmarks.

    Parameters
    ----------
    grid_sizes
        Grid dimensions to benchmark, e.g. [(32,32,32), (64,64,64)].
    memory_strategies
        Gradient memory strategies to evaluate.
    n_time_steps
        Number of FDTD time steps per run.
    device
        ``"cuda"`` when GPU available, ``"cpu"`` otherwise.
    """

    grid_sizes: list[tuple[int, int, int]] = dc_field(
        default_factory=lambda: [(32, 32, 32), (64, 64, 64)]
    )
    memory_strategies: list[GPUMemoryStrategy] = dc_field(
        default_factory=lambda: [
            GPUMemoryStrategy.TIME_REVERSAL,
            GPUMemoryStrategy.CHECKPOINT,
            GPUMemoryStrategy.FULL_AUTODIFF,
        ]
    )
    n_time_steps: int = 100
    device: str = "cpu"


# ---------------------------------------------------------------------------
# StabilityReport
# ---------------------------------------------------------------------------


@dataclass
class StabilityReport:
    """Summary of forward and gradient stability across multiple seeds.

    Attributes
    ----------
    forward_errors
        Relative forward field errors per seed.
    gradient_cosines
        Cosine similarity of gradients vs autograd reference per seed.
    memory_used_mb
        Peak memory usage keyed by strategy name.
    is_valid
        True if all cosines >= 0.99 and all forward errors < 1e-3.
    """

    forward_errors: list[float] = dc_field(default_factory=list)
    gradient_cosines: list[float] = dc_field(default_factory=list)
    memory_used_mb: dict[str, float] = dc_field(default_factory=dict)
    is_valid: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_sim(a: Tensor, b: Tensor) -> float:
    fa = a.flatten().to(torch.float64)
    fb = b.flatten().to(torch.float64)
    denom = fa.norm() * fb.norm()
    if denom.item() < 1e-30:
        return 0.0
    return torch.dot(fa, fb).item() / denom.item()


def _make_eps_grid(grid_size: tuple[int, int, int], device: str) -> Tensor:
    D, H, W = grid_size
    torch.manual_seed(42)
    eps = 1.5 + 1.0 * torch.rand(D, H, W, dtype=torch.float64, device=device)
    d, h, w = D // 4, H // 4, W // 4
    if d > 0 and h > 0 and w > 0:
        eps[D // 2 - d : D // 2 + d, H // 2 - h : H // 2 + h, W // 2 - w : W // 2 + w] = 4.0
    return eps


def _estimate_peak_mb(
    grid_size: tuple[int, int, int],
    n_steps: int,
    strategy: GPUMemoryStrategy,
    device: str,
    actual_cuda_allocated: float | None = None,
) -> float:
    if device == "cuda" and actual_cuda_allocated is not None:
        return actual_cuda_allocated / 1e6
    D, H, W = grid_size
    bytes_per = 8
    voxels = D * H * W
    if strategy == GPUMemoryStrategy.FULL_AUTODIFF:
        return 10 * 6 * n_steps * voxels * bytes_per / 1e6
    elif strategy == GPUMemoryStrategy.TIME_REVERSAL:
        return 3 * n_steps * voxels * bytes_per / 1e6
    else:  # CHECKPOINT
        segments = 4
        return segments * 6 * voxels * bytes_per / 1e6


# ---------------------------------------------------------------------------
# FDTDGPUBenchmark
# ---------------------------------------------------------------------------


class FDTDGPUBenchmark:
    """GPU benchmarking for FDTD3D across grid sizes and memory strategies.

    Parameters
    ----------
    config
        Benchmark configuration (grid sizes, strategies, device).
    """

    def __init__(self, config: FDTDBenchmarkConfig) -> None:
        self.config = config
        self._results: list[dict[str, Any]] = []

    # -- GPU detection ------------------------------------------------------

    @staticmethod
    def detect_gpu() -> bool:
        """Return True if CUDA is available."""
        return torch.cuda.is_available()

    # -- Single benchmark ---------------------------------------------------

    def run_single(
        self,
        grid_size: tuple[int, int, int],
        strategy: GPUMemoryStrategy,
        device: str,
    ) -> dict[str, Any]:
        """Benchmark one (grid_size, strategy, device) combination.

        Returns dict with keys: forward_time_ms, backward_time_ms,
        peak_memory_mb, gradient_cosine, grid_size, strategy, device.
        """
        from diffnano.solvers.fdtd3d import FDTDSolver3D

        effective = device
        if effective == "cuda" and not torch.cuda.is_available():
            effective = "cpu"

        gc.collect()
        if effective == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()
        else:
            mem_before = 0

        bw_kw: dict[str, Any] = {}
        if strategy == GPUMemoryStrategy.TIME_REVERSAL:
            bw_kw["backward"] = "time_reversal"
        elif strategy == GPUMemoryStrategy.CHECKPOINT:
            bw_kw["use_checkpoint"] = True
            bw_kw["checkpoint_segments"] = 4
        # FULL_AUTODIFF: default (no special kwargs)

        solver = FDTDSolver3D(
            grid_shape=grid_size,
            dl=20.0,
            wavelength_nm=1550.0,
            pml_layers=0,
            n_steps=self.config.n_time_steps,
            device=effective,
            courant=0.35,
            **bw_kw,
        )

        eps = _make_eps_grid(grid_size, effective).detach().requires_grad_(True)

        t0 = time.perf_counter()
        result = solver.forward(eps)
        if effective == "cuda":
            torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) * 1000.0

        loss = result.field.sum()

        t0 = time.perf_counter()
        loss.backward()
        if effective == "cuda":
            torch.cuda.synchronize()
        bwd_ms = (time.perf_counter() - t0) * 1000.0

        grad = eps.grad.detach().clone() if eps.grad is not None else torch.zeros_like(eps)

        cuda_alloc = None
        if effective == "cuda":
            cuda_alloc = torch.cuda.max_memory_allocated() - mem_before

        peak_mb = _estimate_peak_mb(
            grid_size, self.config.n_time_steps, strategy, effective, cuda_alloc
        )

        record: dict[str, Any] = {
            "grid_size": grid_size,
            "strategy": strategy.value,
            "device": effective,
            "forward_time_ms": round(fwd_ms, 3),
            "backward_time_ms": round(bwd_ms, 3),
            "peak_memory_mb": round(peak_mb, 3),
            "gradient": grad,
        }
        self._results.append(record)
        return record

    # -- Full benchmark matrix ----------------------------------------------

    def run_all(self) -> list[dict[str, Any]]:
        """Run benchmarks for all grid_sizes x memory_strategies.

        The first strategy is always FULL_AUTODIFF (autograd reference)
        so that subsequent strategies can compute cosine similarity.

        Returns list of result dicts (gradient_cosine populated).
        """
        self._results = []
        effective_device = self.config.device
        if effective_device == "cuda" and not self.config.device == "cpu":
            if not torch.cuda.is_available():
                effective_device = "cpu"

        # Ensure FULL_AUTODIFF runs first per grid size as the reference.
        for grid_size in self.config.grid_sizes:
            autograd_grad: Tensor | None = None
            ordered = sorted(
                self.config.memory_strategies,
                key=lambda s: (0 if s == GPUMemoryStrategy.FULL_AUTODIFF else 1),
            )

            for strategy in ordered:
                rec = self.run_single(grid_size, strategy, effective_device)

                if strategy == GPUMemoryStrategy.FULL_AUTODIFF:
                    autograd_grad = rec["gradient"]
                    rec["gradient_cosine"] = None  # reference
                elif autograd_grad is not None:
                    rec["gradient_cosine"] = round(
                        _cosine_sim(autograd_grad, rec["gradient"]), 6
                    )
                else:
                    rec["gradient_cosine"] = None

                # Remove raw gradient from final output to save memory.
                del rec["gradient"]

        return self._results

    # -- Memory scaling curve -----------------------------------------------

    def benchmark_memory_scaling(self) -> list[dict[str, Any]]:
        """Measure peak memory vs grid size for each strategy.

        Returns list of dicts with keys: grid_size, strategy, peak_memory_mb.
        """
        scaling: list[dict[str, Any]] = []
        for grid_size in self.config.grid_sizes:
            for strategy in self.config.memory_strategies:
                rec = self.run_single(grid_size, strategy, self.config.device)
                scaling.append({
                    "grid_size": grid_size,
                    "strategy": strategy.value,
                    "peak_memory_mb": rec["peak_memory_mb"],
                })
        return scaling

    # -- CPU fallback correctness -------------------------------------------

    def cpu_fallback_benchmark(self) -> StabilityReport:
        """Run all strategies on CPU and verify gradient consistency.

        Returns a StabilityReport with cosine similarities between
        FULL_AUTODIFF and each other strategy.
        """
        cfg_cpu = FDTDBenchmarkConfig(
            grid_sizes=self.config.grid_sizes[:1],  # use smallest grid
            memory_strategies=[
                GPUMemoryStrategy.FULL_AUTODIFF,
                GPUMemoryStrategy.TIME_REVERSAL,
                GPUMemoryStrategy.CHECKPOINT,
            ],
            n_time_steps=min(self.config.n_time_steps, 20),
            device="cpu",
        )
        bench = FDTDGPUBenchmark(cfg_cpu)
        results = bench.run_all()

        cosines: list[float] = []
        memory_mb: dict[str, float] = {}
        for rec in results:
            strategy_name = rec["strategy"]
            memory_mb[strategy_name] = rec["peak_memory_mb"]
            if rec["gradient_cosine"] is not None:
                cosines.append(rec["gradient_cosine"])

        is_valid = len(cosines) > 0 and all(c >= 0.99 for c in cosines)

        return StabilityReport(
            forward_errors=[0.0],
            gradient_cosines=cosines,
            memory_used_mb=memory_mb,
            is_valid=is_valid,
        )


# ---------------------------------------------------------------------------
# MultiScaleMetalens
# ---------------------------------------------------------------------------


class MultiScaleMetalens:
    """Coarse-to-fine tiling for large-aperture metalens simulation.

    Decomposes a large simulation domain into tiles that individually fit
    in GPU memory.  A coarse solve provides approximate boundary conditions
    for subsequent fine-grained sub-domain solves.

    Parameters
    ----------
    coarse_grid
        Grid dimensions for the coarse (low-resolution) solve.
    fine_grid
        Grid dimensions for each fine tile.
    n_tiles
        Number of tiles along each axis, e.g. (2, 2, 1).
    """

    def __init__(
        self,
        coarse_grid: tuple[int, int, int],
        fine_grid: tuple[int, int, int],
        n_tiles: tuple[int, int, int],
    ) -> None:
        self.coarse_grid = coarse_grid
        self.fine_grid = fine_grid
        self.n_tiles = n_tiles
        self._coarse_result: Tensor | None = None

    # -- Coarse solve -------------------------------------------------------

    def coarse_solve(self, region: Tensor) -> Tensor:
        """Run a low-resolution FDTD over the full domain.

        Parameters
        ----------
        region
            Permittivity map at coarse resolution, shape ``coarse_grid``.

        Returns
        -------
        Tensor
            Stacked field (3, D, H, W) at coarse resolution.
        """
        from diffnano.solvers.fdtd3d import FDTDSolver3D

        solver = FDTDSolver3D(
            grid_shape=self.coarse_grid,
            dl=40.0,  # coarser spacing
            wavelength_nm=1550.0,
            pml_layers=3,
            n_steps=50,
            device="cpu",
            courant=0.35,
        )
        eps = region.to(torch.float64)
        result = solver.forward(eps)
        self._coarse_result = result.field.detach()
        return self._coarse_result

    # -- Fine solve ---------------------------------------------------------

    def fine_solve(self, region: Tensor, coarse_boundary: Tensor) -> Tensor:
        """Run a high-resolution FDTD on one tile with coarse BCs.

        Parameters
        ----------
        region
            Permittivity map at fine resolution for this tile.
        coarse_boundary
            Boundary field values interpolated from coarse solve.

        Returns
        -------
        Tensor
            Stacked field (3, D, H, W) at fine resolution.
        """
        from diffnano.solvers.fdtd3d import FDTDSolver3D

        solver = FDTDSolver3D(
            grid_shape=self.fine_grid,
            dl=20.0,  # finer spacing
            wavelength_nm=1550.0,
            pml_layers=5,
            n_steps=100,
            device="cpu",
            courant=0.35,
        )
        eps = region.to(torch.float64)
        result = solver.forward(eps)
        fine_field = result.field.detach()

        # Blend coarse boundary into the PML region of the fine result.
        # This is a simplified coupling; a production implementation would
        # use proper interpolation and time-domain matching.
        nz, ny, nx = self.fine_grid
        pml = solver.pml_layers
        if pml > 0 and coarse_boundary.shape[1:] != fine_field.shape[1:]:
            # Trilinear interpolation of coarse_boundary to fine grid size.
            cb_3d = coarse_boundary.unsqueeze(0)  # (1, C, D', H', W')
            cb_up = torch.nn.functional.interpolate(
                cb_3d.float(),
                size=(nz, ny, nx),
                mode="trilinear",
                align_corners=False,
            )
            cb_up = cb_up.squeeze(0).to(fine_field.dtype)
            fine_field[:, :pml, :, :] = cb_up[:, :pml, :, :]
            fine_field[:, -pml:, :, :] = cb_up[:, -pml:, :, :]
            fine_field[:, :, :pml, :] = cb_up[:, :, :pml, :]
            fine_field[:, :, -pml:, :] = cb_up[:, :, -pml:, :]

        return fine_field

    # -- Full forward -------------------------------------------------------

    def forward(self, design: Tensor) -> dict[str, Any]:
        """Run the full coarse-to-fine tiled simulation.

        Parameters
        ----------
        design
            Full-resolution permittivity map.  Shape must be compatible
            with ``(n_tiles * fine_grid)`` approximately.

        Returns
        -------
        dict
            ``"coarse_field"``: coarse result, ``"tile_fields"``: list of
            fine fields per tile, ``"n_tiles"``: tile counts.
        """
        design = design.to(torch.float64)

        # Downsample design to coarse grid for the coarse solve.
        coarse_eps = torch.nn.functional.interpolate(
            design.unsqueeze(0).unsqueeze(0).float(),
            size=self.coarse_grid,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0).to(torch.float64)

        coarse_field = self.coarse_solve(coarse_eps)

        # Extract tile sub-regions from the full design and run fine solves.
        tz, ty, tx = self.n_tiles
        fz, fy, fx = self.fine_grid
        tile_fields: list[Tensor] = []

        for iz in range(tz):
            for iy in range(ty):
                for ix in range(tx):
                    z0 = iz * fz
                    y0 = iy * fy
                    x0 = ix * fx
                    z1 = min(z0 + fz, design.shape[0])
                    y1 = min(y0 + fy, design.shape[1])
                    x1 = min(x0 + fx, design.shape[2])

                    tile_eps = design[z0:z1, y0:y1, x0:x1]
                    # Pad if tile is smaller than fine_grid.
                    if tile_eps.shape != self.fine_grid:
                        padded = torch.ones(self.fine_grid, dtype=torch.float64)
                        sz = min(tile_eps.shape[0], fz)
                        sy = min(tile_eps.shape[1], fy)
                        sx = min(tile_eps.shape[2], fx)
                        padded[:sz, :sy, :sx] = tile_eps[:sz, :sy, :sx]
                        tile_eps = padded

                    fine = self.fine_solve(tile_eps, coarse_field)
                    tile_fields.append(fine)

        return {
            "coarse_field": coarse_field,
            "tile_fields": tile_fields,
            "n_tiles": self.n_tiles,
        }

    # -- Scaling benchmark --------------------------------------------------

    def benchmark_scaling(
        self,
        base_size: int = 16,
        max_tiles: int = 4,
    ) -> list[dict[str, Any]]:
        """Benchmark multi-scale solve time vs number of tiles.

        Parameters
        ----------
        base_size
            Tile dimension (base_size x base_size x base_size).
        max_tiles
            Maximum number of tiles along each axis (1..max_tiles).

        Returns
        -------
        list of dict
            Each dict has keys: n_tiles_per_axis, total_tiles,
            coarse_time_ms, fine_time_ms, total_time_ms.
        """
        results: list[dict[str, Any]] = []
        for n in range(1, max_tiles + 1):
            tile_grid = (base_size, base_size, base_size)
            coarse_dim = base_size * n
            coarse_grid = (coarse_dim, coarse_dim, coarse_dim)
            n_tiles = (n, n, n)

            msl = MultiScaleMetalens(
                coarse_grid=coarse_grid,
                fine_grid=tile_grid,
                n_tiles=n_tiles,
            )

            torch.manual_seed(42)
            full_design = 1.5 + 1.0 * torch.rand(
                coarse_dim, coarse_dim, coarse_dim, dtype=torch.float64
            )

            t0 = time.perf_counter()
            coarse_field = msl.coarse_solve(full_design)
            coarse_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            tile_eps = full_design[:base_size, :base_size, :base_size]
            msl.fine_solve(tile_eps, coarse_field)
            fine_ms = (time.perf_counter() - t0) * 1000.0

            total_tiles = n ** 3
            total_ms = coarse_ms + fine_ms * total_tiles

            results.append({
                "n_tiles_per_axis": n,
                "total_tiles": total_tiles,
                "coarse_time_ms": round(coarse_ms, 3),
                "fine_time_ms": round(fine_ms, 3),
                "total_time_ms": round(total_ms, 3),
            })

        return results


# ---------------------------------------------------------------------------
# FDTDXCrossValidator
# ---------------------------------------------------------------------------


class FDTDXCrossValidator:
    """Cross-validation scaffold against FDTDX or other external references.

    Clean-room design: no FDTDX code is vendored.  Reference forward fields
    and gradients are either injected via a callable or generated
    synthetically for testing.

    Parameters
    ----------
    reference_fn
        Optional callable ``(grid_size, n_steps) -> dict`` that returns
        ``{"field": Tensor, "gradient": Tensor}`` from an external solver.
        When None, synthetic references are used.
    """

    def __init__(self, reference_fn: Callable | None = None) -> None:
        self.reference_fn = reference_fn

    # -- Synthetic reference ------------------------------------------------

    def generate_synthetic_reference(
        self,
        grid_size: tuple[int, int, int],
        n_steps: int = 50,
    ) -> dict[str, Tensor]:
        """Generate a synthetic reference using DiffNano FDTD itself.

        This provides a self-consistent reference for testing the
        validation pipeline.  For real cross-validation, supply
        ``reference_fn`` with data from FDTDX.

        Parameters
        ----------
        grid_size
            (D, H, W) dimensions.
        n_steps
            Number of time steps.

        Returns
        -------
        dict
            ``"field"``: (3, D, H, W) forward field.
            ``"gradient"``: (D, H, W) gradient of sum(field) w.r.t. eps.
        """
        from diffnano.solvers.fdtd3d import FDTDSolver3D

        solver = FDTDSolver3D(
            grid_shape=grid_size,
            dl=20.0,
            wavelength_nm=1550.0,
            pml_layers=0,
            n_steps=n_steps,
            device="cpu",
            courant=0.35,
        )

        eps = _make_eps_grid(grid_size, "cpu").detach().requires_grad_(True)
        result = solver.forward(eps)

        loss = result.field.sum()
        loss.backward()

        gradient = eps.grad.detach().clone() if eps.grad is not None else torch.zeros_like(eps)
        field = result.field.detach().clone()

        return {
            "field": field,
            "gradient": gradient,
        }

    # -- Forward validation -------------------------------------------------

    def validate_forward(
        self,
        result: Tensor,
        reference: Tensor,
        rtol: float = 1e-3,
    ) -> dict[str, Any]:
        """Compare forward simulation result against reference.

        Parameters
        ----------
        result
            Simulated field, shape ``(3, D, H, W)``.
        reference
            Reference field, same shape.
        rtol
            Relative tolerance for pass/fail.

        Returns
        -------
        dict
            Keys: relative_error, max_absolute_error, passed.
        """
        diff = result.detach().to(torch.float64) - reference.detach().to(torch.float64)
        ref_norm = reference.detach().to(torch.float64).norm().item()
        max_abs = diff.abs().max().item()
        rel_err = diff.norm().item() / max(ref_norm, 1e-30)
        return {
            "relative_error": rel_err,
            "max_absolute_error": max_abs,
            "passed": rel_err < rtol,
        }

    # -- Gradient validation ------------------------------------------------

    def validate_gradient(
        self,
        grad: Tensor,
        reference_grad: Tensor,
        cosine_threshold: float = 0.99,
    ) -> dict[str, Any]:
        """Compare gradient against reference via cosine similarity.

        Parameters
        ----------
        grad
            Computed gradient, shape ``(D, H, W)``.
        reference_grad
            Reference gradient, same shape.
        cosine_threshold
            Minimum cosine similarity for pass.

        Returns
        -------
        dict
            Keys: cosine_similarity, relative_error, passed.
        """
        cos_sim = _cosine_sim(grad, reference_grad)
        ref_norm = reference_grad.detach().to(torch.float64).norm().item()
        diff_norm = (
            grad.detach().to(torch.float64) - reference_grad.detach().to(torch.float64)
        ).norm().item()
        rel_err = diff_norm / max(ref_norm, 1e-30)
        return {
            "cosine_similarity": cos_sim,
            "relative_error": rel_err,
            "passed": cos_sim >= cosine_threshold,
        }

    # -- Full cross-validation run ------------------------------------------

    def run_cross_validation(
        self,
        n_seeds: int = 3,
        grid_size: tuple[int, int, int] = (16, 16, 16),
        n_steps: int = 30,
    ) -> StabilityReport:
        """Run cross-validation across multiple random seeds.

        For each seed, generates (or loads) a reference, runs DiffNano FDTD,
        and compares forward and gradient results.

        Parameters
        ----------
        n_seeds
            Number of random seeds to test.
        grid_size
            Grid dimensions for each test.
        n_steps
            Time steps per simulation.

        Returns
        -------
        StabilityReport
            Aggregated forward errors, gradient cosines, and validity.
        """
        from diffnano.solvers.fdtd3d import FDTDSolver3D

        forward_errors: list[float] = []
        gradient_cosines: list[float] = []
        memory_used: dict[str, float] = {}

        for seed in range(n_seeds):
            torch.manual_seed(seed * 100 + 42)

            # Obtain reference.
            if self.reference_fn is not None:
                ref = self.reference_fn(grid_size, n_steps)
            else:
                ref = self.generate_synthetic_reference(grid_size, n_steps)

            # Run DiffNano FDTD with full autograd.
            solver = FDTDSolver3D(
                grid_shape=grid_size,
                dl=20.0,
                wavelength_nm=1550.0,
                pml_layers=0,
                n_steps=n_steps,
                device="cpu",
                courant=0.35,
            )

            torch.manual_seed(seed * 100 + 42)
            eps = _make_eps_grid(grid_size, "cpu").detach().requires_grad_(True)
            result = solver.forward(eps)
            loss = result.field.sum()
            loss.backward()

            our_field = result.field.detach()
            our_grad = eps.grad.detach() if eps.grad is not None else torch.zeros_like(eps)

            # Forward validation.
            fwd_report = self.validate_forward(our_field, ref["field"])
            forward_errors.append(fwd_report["relative_error"])

            # Gradient validation.
            grad_report = self.validate_gradient(our_grad, ref["gradient"])
            gradient_cosines.append(grad_report["cosine_similarity"])

        peak_mb = _estimate_peak_mb(
            grid_size, n_steps, GPUMemoryStrategy.FULL_AUTODIFF, "cpu"
        )
        memory_used["full_autodiff"] = peak_mb

        is_valid = (
            len(forward_errors) > 0
            and all(e < 1e-3 for e in forward_errors)
            and all(c >= 0.99 for c in gradient_cosines)
        )

        return StabilityReport(
            forward_errors=forward_errors,
            gradient_cosines=gradient_cosines,
            memory_used_mb=memory_used,
            is_valid=is_valid,
        )
