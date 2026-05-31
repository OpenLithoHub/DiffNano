"""Robust posterior warm start: angle/process-corner aware candidate selection.

Links latent_warm_start with robustness evaluation and quantized geometry
for worst-case quantile scoring of warm-start candidates.

References:
    - Inverse Design of Nanophotonic Color Router Robust to Oblique Incidence,
      Adv. Opt. Mater. 14(4), 2026
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Dict, Tuple, Callable

from diffnano.design.latent_warm_start import ConditionalLatentSampler, StrehlScorer
from diffnano.design.robustness.subspace import MultiAxisPerturbation

__all__ = [
    "AngleSweepScorer",
    "RobustPosteriorWarmStart",
    "ProcessCornerWarmStart",
]


class AngleSweepScorer:
    """Score designs by worst-case FoM across an incident-angle sweep.

    Wraps a base scorer (e.g. StrehlScorer) and evaluates it at multiple
    source angles, returning the minimum (worst-case) FoM.
    """

    def __init__(
        self,
        base_scorer: Callable[[Tensor, Tensor], Tensor],
        angle_range: Tuple[float, float] = (-30.0, 30.0),
        n_angles: int = 7,
    ):
        self.base_scorer = base_scorer
        self.angle_range = angle_range
        self.n_angles = n_angles

    def _angle_conditions(
        self,
        condition: Tensor,
    ) -> List[Tensor]:
        """Generate perturbed condition tensors for each angle in the sweep."""
        lo, hi = self.angle_range
        angles = torch.linspace(lo, hi, self.n_angles, dtype=condition.dtype, device=condition.device)
        conditions = []
        for angle in angles:
            cond = condition.clone()
            cond[-1] = angle / self.angle_range[1]
            conditions.append(cond)
        return conditions

    def score(self, design: Tensor, condition: Tensor) -> Tensor:
        """Score a design by its worst-case FoM across angles.

        Parameters
        ----------
        design : Tensor, shape ``(H, W)``
        condition : Tensor
            Conditioning tensor whose last element encodes the source angle.

        Returns
        -------
        worst_fom : Tensor, scalar
            Minimum FoM across the angle sweep.
        """
        angle_conditions = self._angle_conditions(condition)
        foms = torch.stack([self.base_scorer(design, c) for c in angle_conditions])
        return foms.min()

    def score_with_bands(
        self,
        design: Tensor,
        condition: Tensor,
        n_mc: int = 16,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Score with uncertainty bands via MC sampling over angle perturbations.

        Returns
        -------
        worst_fom : Tensor, scalar
            Mean worst-case FoM across MC replicates.
        lower : Tensor, scalar
            5th percentile of worst-case FoM distribution.
        upper : Tensor, scalar
            95th percentile of worst-case FoM distribution.
        """
        lo, hi = self.angle_range
        worst_foms = []
        for _ in range(n_mc):
            angles = lo + (hi - lo) * torch.rand(
                self.n_angles, dtype=condition.dtype, device=condition.device,
            )
            foms = []
            for angle in angles:
                cond = condition.clone()
                cond[-1] = angle / self.angle_range[1]
                foms.append(self.base_scorer(design, cond))
            worst_foms.append(torch.stack(foms).min())

        worst_foms_t = torch.stack(worst_foms)
        lower = torch.quantile(worst_foms_t, 0.05)
        upper = torch.quantile(worst_foms_t, 0.95)
        return worst_foms_t.mean(), lower, upper


class ProcessCornerWarmStart:
    """Score designs by worst-case FoM across process corner variations.

    Uses MultiAxisPerturbation from robustness/subspace.py to generate
    deterministic process corners and evaluates the worst-case performance.
    """

    def __init__(
        self,
        fom_fn: Callable[[Tensor], Tensor],
        perturbation: Optional[MultiAxisPerturbation] = None,
        n_corners: int = 8,
        corner_sigma_scale: float = 2.0,
    ):
        self.fom_fn = fom_fn
        self.perturbation = perturbation or MultiAxisPerturbation()
        self.n_corners = n_corners
        self.corner_sigma_scale = corner_sigma_scale

    def _generate_corners(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Generate deterministic corner perturbation vectors.

        Uses structured sampling at +/- sigma_scale multiples along each
        perturbation axis, yielding up to n_corners points.
        """
        base = self.perturbation.sigmas.to(device=device, dtype=dtype)
        corners = [
            base * self.corner_sigma_scale,
            -base * self.corner_sigma_scale,
        ]
        signs = torch.tensor(
            [[1, 1, -1, -1], [-1, -1, 1, 1], [1, -1, 1, -1], [-1, 1, -1, 1],
             [1, -1, -1, 1], [-1, 1, 1, -1]],
            dtype=dtype,
            device=device,
        )
        for s in signs:
            corners.append(base * self.corner_sigma_scale * s)
        corners_t = torch.stack(corners)[: self.n_corners]
        return corners_t

    def score(
        self,
        design: Tensor,
        condition: Optional[Tensor] = None,
        n_corners: Optional[int] = None,
    ) -> Tensor:
        """Score a design by worst-case FoM across process corners.

        Parameters
        ----------
        design : Tensor, shape ``(H, W)``
        condition : Tensor, optional
            Unused, kept for API compatibility.
        n_corners : int, optional
            Override the number of corners.

        Returns
        -------
        worst_fom : Tensor, scalar
        """
        n = n_corners or self.n_corners
        corners = self._generate_corners(design.device, design.dtype)[:n]
        foms = []
        for i in range(corners.shape[0]):
            perturbed = self.perturbation.apply(design, corners[i].to(torch.float64))
            foms.append(self.fom_fn(perturbed))
        return torch.stack(foms).min()


class RobustPosteriorWarmStart(nn.Module):
    """Robustness-aware warm start: score candidates by worst-case quantile.

    Extends ConditionalLatentSampler by scoring generated candidates through
    angle or process-corner sweeps before selection, optionally applying
    quantization (STE) before scoring.
    """

    def __init__(
        self,
        latent_sampler: ConditionalLatentSampler,
        robust_scorer: Callable[[Tensor, Tensor], Tensor],
        n_candidates: int = 16,
        quantize_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        super().__init__()
        self.latent_sampler = latent_sampler
        self.robust_scorer = robust_scorer
        self.n_candidates = n_candidates
        self.quantize_fn = quantize_fn

    def _apply_quantize(self, design: Tensor) -> Tensor:
        if self.quantize_fn is not None:
            return self.quantize_fn(design)
        return design

    def sample_robust(
        self,
        condition: Tensor,
        n_candidates: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Sample candidates and score by worst-case robust FoM.

        Returns
        -------
        dict with:
        - ``candidates``: Tensor ``(n_candidates, H, W)``
        - ``robust_scores``: Tensor ``(n_candidates,)``
        - ``best``: Tensor ``(H, W)``
        - ``best_score``: Tensor scalar
        """
        n = n_candidates or self.n_candidates
        candidates = self.latent_sampler.sample_candidates(condition, n)

        scores = []
        for i in range(n):
            design = self._apply_quantize(candidates[i])
            s = self.robust_scorer(design, condition)
            scores.append(s)

        scores_t = torch.stack(scores)
        best_idx = scores_t.argmax()
        return {
            "candidates": candidates,
            "robust_scores": scores_t,
            "best": candidates[best_idx],
            "best_score": scores_t[best_idx],
        }

    def sample_with_decision_gate(
        self,
        condition: Tensor,
        n_candidates: Optional[int] = None,
        decision_gate: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
    ) -> Dict[str, Tensor]:
        """Sample candidates, score robustly, then apply a decision gate.

        The decision_gate callable receives (scores, candidates) and returns
        (accepted_mask, accepted_candidates).  This is designed to accept a
        diff-surrogate DecisionGate or equivalent.

        Returns
        -------
        dict with:
        - ``candidates``: Tensor ``(n_candidates, H, W)``
        - ``robust_scores``: Tensor ``(n_candidates,)``
        - ``accepted_mask``: Tensor ``(n_candidates,)`` bool
        - ``accepted``: Tensor ``(n_accepted, H, W)``
        """
        n = n_candidates or self.n_candidates
        result = self.sample_robust(condition, n)
        scores = result["robust_scores"]
        candidates = result["candidates"]

        if decision_gate is not None:
            mask, accepted = decision_gate(scores, candidates)
        else:
            mask = torch.ones(n, dtype=torch.bool, device=scores.device)
            accepted = candidates

        return {
            "candidates": candidates,
            "robust_scores": scores,
            "accepted_mask": mask,
            "accepted": accepted,
        }

    def compare_robust_vs_nominal(
        self,
        condition: Tensor,
        fom_fn: Callable[[Tensor], Tensor],
        n_seeds: int = 5,
        n_candidates: int = 10,
    ) -> Dict[str, list]:
        """Statistical comparison: robust-scored vs nominal-scored selection.

        For each seed, generates candidates, picks best by robust score and
        by nominal score, then evaluates both at the nominal condition.

        Returns
        -------
        dict with:
        - ``robust_foms``: list of FoMs from robust-selected designs
        - ``nominal_foms``: list of FoMs from nominally-selected designs
        - ``robust_wins``: fraction of seeds where robust selection wins
        """
        robust_foms: list[float] = []
        nominal_foms: list[float] = []

        for seed in range(n_seeds):
            torch.manual_seed(seed)

            candidates = self.latent_sampler.sample_candidates(condition, n_candidates)

            nominal_scores = torch.stack([fom_fn(c) for c in candidates])
            nominal_best = candidates[nominal_scores.argmax()]
            nominal_foms.append(fom_fn(nominal_best).item())

            robust_scores = []
            for c in candidates:
                design = self._apply_quantize(c)
                robust_scores.append(self.robust_scorer(design, condition))
            robust_scores_t = torch.stack(robust_scores)
            robust_best = candidates[robust_scores_t.argmax()]
            robust_foms.append(fom_fn(robust_best).item())

        wins = sum(
            1 for r, n in zip(robust_foms, nominal_foms) if r > n
        ) / max(n_seeds, 1)

        return {
            "robust_foms": robust_foms,
            "nominal_foms": nominal_foms,
            "robust_wins": wins,
        }
