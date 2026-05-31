"""Physics-aware latent/diffusion prior for inverse design warm-start (N8.3).

Multi-candidate generation via latent space sampling around a conditioning
design, batch differentiable refinement through RCWA, and statistical
comparison against random initialization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diffnano.design.representation_learning import LearnedRepresentation

__all__ = [
    "ConditionalLatentSampler",
    "StrehlScorer",
    "WilcoxonComparison",
]


class ConditionalLatentSampler(nn.Module):
    """Multi-candidate warm-start via latent space sampling + EM refinement.

    Consumes diff-surrogate's CandidateSampler interface and combines with
    DiffNano's LearnedRepresentation (VAE) and RCWA for refinement.
    """

    def __init__(
        self,
        vae: LearnedRepresentation,
        rcwa_solver=None,
        latent_dim: int = 8,
        device: str = "cpu",
    ):
        super().__init__()
        self.vae = vae
        self.rcwa_solver = rcwa_solver
        self.latent_dim = latent_dim
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    # -- CandidateSampler protocol compliance --
    def sample(self, condition: torch.Tensor, n_candidates: int) -> torch.Tensor:
        """CandidateSampler protocol: generate *n_candidates* from *condition*."""
        return self.sample_candidates(condition, n_candidates)

    def sample_candidates(
        self,
        condition: torch.Tensor,
        n_candidates: int = 10,
        perturbation_scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample *n_candidates* from latent space around *condition*.

        1. Encode the condition into latent space.
        2. Sample multiple points via Gaussian perturbation.
        3. Decode each to get candidate geometries.

        Parameters
        ----------
        condition : Tensor, shape ``(H, W)``
            Reference design to condition sampling around.
        n_candidates : int
            Number of candidates.
        perturbation_scale : float
            Standard deviation of Gaussian perturbation in latent space.

        Returns
        -------
        candidates : Tensor, shape ``(n_candidates, H, W)``
        """
        z_center = self.vae.encode(condition)  # (latent_dim,)
        z_center = z_center.to(self._device)

        noise = torch.randn(
            n_candidates,
            self.latent_dim,
            dtype=torch.float64,
            device=self._device,
        ) * perturbation_scale
        z_samples = z_center.unsqueeze(0) + noise  # (n_candidates, latent_dim)

        candidates = []
        for i in range(n_candidates):
            geom = self.vae.decode(z_samples[i])  # (H, W)
            candidates.append(geom)
        return torch.stack(candidates)

    def batch_refine(
        self,
        candidates: torch.Tensor,
        fom_fn,
        n_steps: int = 20,
        lr: float = 0.01,
    ) -> tuple[torch.Tensor, list[list[float]]]:
        """Batch refine candidates using differentiable FOM + RCWA.

        Each candidate gets a few gradient steps to improve FOM.

        Parameters
        ----------
        candidates : Tensor, shape ``(n_candidates, H, W)``
            Candidate geometries.
        fom_fn : callable
            ``fom_fn(geometry) -> scalar`` where higher is better (we minimize
            ``-fom``).
        n_steps : int
        lr : float

        Returns
        -------
        refined : Tensor, shape ``(n_candidates, H, W)``
        histories : list of list of float
            Per-candidate loss trajectories.
        """
        n = candidates.shape[0]
        refined_list = []
        histories = []

        for i in range(n):
            geom = candidates[i].clone().to(self._device).to(torch.float64)
            geom.requires_grad_(True)
            opt = torch.optim.Adam([geom], lr=lr)
            hist = []

            for _ in range(n_steps):
                loss = -fom_fn(geom)
                opt.zero_grad()
                loss.backward()

                if geom.grad is not None and torch.isnan(geom.grad).any():
                    break

                opt.step()
                hist.append(loss.item())

            with torch.no_grad():
                geom = geom.clamp(0.0, 1.0)
            refined_list.append(geom.detach())
            histories.append(hist)

        return torch.stack(refined_list), histories

    def score_and_select(
        self,
        candidates: torch.Tensor,
        fom_fn,
        top_k: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score candidates by FOM and return top_k.

        Parameters
        ----------
        candidates : Tensor, shape ``(n_candidates, H, W)``
        fom_fn : callable
        top_k : int

        Returns
        -------
        best : Tensor, shape ``(top_k, H, W)``
        scores : Tensor, shape ``(n_candidates,)``
        indices : Tensor, shape ``(top_k,)``
        """
        n = candidates.shape[0]
        scores = torch.zeros(n, dtype=torch.float64, device=self._device)
        for i in range(n):
            scores[i] = fom_fn(candidates[i]).detach()

        _, indices = torch.topk(scores, min(top_k, n))
        best = candidates[indices]
        return best, scores, indices

    def warm_start_optimize(
        self,
        condition: torch.Tensor,
        fom_fn,
        n_candidates: int = 10,
        top_k: int = 3,
        refine_steps: int = 20,
        lr: float = 0.01,
        perturbation_scale: float = 1.0,
    ) -> dict:
        """Full pipeline: sample -> refine -> select.

        Returns
        -------
        dict with:
        - ``best_geometry``: Tensor ``(H, W)``
        - ``all_candidates``: Tensor ``(n_candidates, H, W)``
        - ``all_scores``: Tensor ``(n_candidates,)``
        - ``refined``: Tensor ``(top_k, H, W)``
        - ``refined_scores``: Tensor ``(top_k,)``
        """
        candidates = self.sample_candidates(
            condition, n_candidates, perturbation_scale
        )
        refined, _ = self.batch_refine(candidates, fom_fn, refine_steps, lr)
        best, scores, indices = self.score_and_select(refined, fom_fn, top_k)

        return {
            "best_geometry": best[0],
            "all_candidates": candidates,
            "all_scores": scores,
            "refined": best,
            "refined_scores": scores[indices],
        }


class StrehlScorer:
    """Score designs by Strehl ratio using RCWA forward simulation.

    The Strehl ratio is the on-axis diffraction efficiency normalized
    to its theoretical maximum.  Higher is better.
    """

    def __init__(self, rcwa_solver, target_focus=None, wavelength: float = 1550.0):
        self.rcwa_solver = rcwa_solver
        self.target_focus = target_focus
        self.wavelength = wavelength

    def score(self, geometry: torch.Tensor) -> torch.Tensor:
        """Compute Strehl ratio for a given geometry.

        Parameters
        ----------
        geometry : Tensor
            Layer geometry compatible with the RCWA solver.

        Returns
        -------
        strehl : Tensor, scalar
        """
        result = self.rcwa_solver.forward(
            geometry,
            wavelengths=[self.wavelength],
        )
        # 0-th order efficiency as Strehl proxy
        center = result.field.shape[-1] // 2
        strehl = result.field[0, center]
        return strehl


class WilcoxonComparison:
    """Statistical comparison of warm-start vs random initialization."""

    @staticmethod
    def compare(
        sampler: ConditionalLatentSampler,
        fom_fn,
        n_seeds: int = 10,
        n_candidates: int = 10,
        grid_size: int = 32,
    ) -> dict:
        """Compare latent warm-start vs random initialization.

        Parameters
        ----------
        sampler : ConditionalLatentSampler
        fom_fn : callable
        n_seeds : int
            Number of random seeds.
        n_candidates : int
        grid_size : int

        Returns
        -------
        dict with:
        - ``warm_start_foms``: list of best FOMs per seed
        - ``random_foms``: list of best FOMs per seed
        - ``p_value``: Wilcoxon signed-rank test p-value
        - ``warm_start_wins``: fraction of seeds where warm-start wins
        """
        from scipy.stats import wilcoxon

        warm_start_foms = []
        random_foms = []

        for seed in range(n_seeds):
            torch.manual_seed(seed)

            condition = torch.rand(
                grid_size, grid_size, dtype=torch.float64, device=sampler.device
            )

            # Warm-start pipeline
            ws_result = sampler.warm_start_optimize(
                condition, fom_fn, n_candidates=n_candidates, top_k=1, refine_steps=10
            )
            warm_start_foms.append(ws_result["refined_scores"][0].item())

            # Random baseline: same number of random geometries, pick best
            random_candidates = torch.rand(
                n_candidates,
                grid_size,
                grid_size,
                dtype=torch.float64,
                device=sampler.device,
            )
            best_random_fom = -float("inf")
            for j in range(n_candidates):
                fom_val = fom_fn(random_candidates[j]).item()
                if fom_val > best_random_fom:
                    best_random_fom = fom_val
            random_foms.append(best_random_fom)

        warm_wins = sum(
            1 for w, r in zip(warm_start_foms, random_foms) if w > r
        ) / max(n_seeds, 1)

        try:
            _, p_value = wilcoxon(warm_start_foms, random_foms)
        except ValueError:
            p_value = 1.0

        return {
            "warm_start_foms": warm_start_foms,
            "random_foms": random_foms,
            "p_value": p_value,
            "warm_start_wins": warm_wins,
        }
