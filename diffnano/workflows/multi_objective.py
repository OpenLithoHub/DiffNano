"""Multi-objective design space exploration with Pareto front (C8).

Provides adaptive weight sampling to discover the Pareto front across
multiple objectives (optical performance, litho EPE, robustness,
fabricability, device footprint).
"""

from __future__ import annotations

import torch

__all__ = ["MultiObjectiveExplorer"]


class MultiObjectiveExplorer:
    """Multi-objective optimizer with adaptive weight sampling.

    Discovers the Pareto front by running optimizations with different
    weight combinations across multiple objectives.

    Parameters
    ----------
    objectives : dict[str, callable]
        ``{"name": fn(density) -> scalar_loss}`` — each objective maps
        a density field to a scalar loss.
    grid_shape : tuple[int, int]
        ``(H, W)`` density field shape.
    n_pareto_points : int
        Number of Pareto points to discover.
    device : str or torch.device
    """

    def __init__(
        self,
        objectives: dict[str, callable],
        grid_shape: tuple[int, int] = (32, 32),
        n_pareto_points: int = 10,
        device: str | torch.device = "cpu",
    ):
        self.objectives = objectives
        self.obj_names = list(objectives.keys())
        self.n_objectives = len(objectives)
        self.grid_shape = grid_shape
        self.n_pareto_points = n_pareto_points
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    def _generate_weights(self, n_points: int) -> list[dict[str, float]]:
        """Generate weight combinations for scalarized optimization.

        Uses Dirichlet-distributed weights to uniformly cover the
        simplex of objective combinations.
        """
        n_obj = self.n_objectives
        alpha = torch.ones(n_obj, dtype=torch.float64)
        weights_list = []

        for _ in range(n_points):
            w = torch._sample_dirichlet(alpha).tolist()
            weights = {name: w[i] for i, name in enumerate(self.obj_names)}
            weights_list.append(weights)

        return weights_list

    def _scalarized_loss(
        self,
        density: torch.Tensor,
        weights: dict[str, float],
    ) -> torch.Tensor:
        """Compute weighted scalarized loss."""
        total = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        for name, fn in self.objectives.items():
            w = weights.get(name, 0.0)
            if w > 0:
                total = total + w * fn(density)
        return total

    def _optimize_single(
        self,
        weights: dict[str, float],
        n_steps: int = 100,
        lr: float = 0.01,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run single-objective optimization with given weights."""
        from diffnano.design.projection import (
            beta_continuation_schedule,
            heaviside_projection,
        )

        density = torch.rand(
            *self.grid_shape, device=self._device, dtype=torch.float64,
        )
        density = density.detach().requires_grad_(True)

        opt = torch.optim.Adam([density], lr=lr)

        for step in range(n_steps):
            beta = beta_continuation_schedule(step, n_steps, beta_start=1.0, beta_end=16.0)
            projected = heaviside_projection(density, beta=beta)

            loss = self._scalarized_loss(projected, weights)

            opt.zero_grad()
            loss.backward()

            if density.grad is not None and torch.isnan(density.grad).any():
                break

            opt.step()
            with torch.no_grad():
                density.clamp_(0.0, 1.0)

        # Evaluate all objectives at the final design
        final_density = density.detach()
        with torch.no_grad():
            from diffnano.design.projection import heaviside_projection
            projected = heaviside_projection(final_density, beta=16.0)
            obj_values = {}
            for name, fn in self.objectives.items():
                obj_values[name] = fn(projected).item()

        return final_density, obj_values

    def explore(
        self,
        n_steps: int = 50,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> list[tuple[torch.Tensor, dict[str, float]]]:
        """Run multi-objective exploration to discover Pareto front.

        Parameters
        ----------
        n_steps : int
            Optimization steps per weight combination.
        lr : float
        verbose : bool

        Returns
        -------
        pareto_points : list of (density, objective_values) tuples
            Pareto-optimal designs and their objective values.
        """
        weights_list = self._generate_weights(self.n_pareto_points)
        all_results = []

        for i, weights in enumerate(weights_list):
            if verbose:
                w_str = ", ".join(f"{k}={v:.2f}" for k, v in weights.items())
                print(f"Point {i+1}/{self.n_pareto_points}: weights=[{w_str}]")

            density, obj_values = self._optimize_single(weights, n_steps, lr)
            all_results.append((density, obj_values))

            if verbose:
                v_str = ", ".join(f"{k}={v:.4f}" for k, v in obj_values.items())
                print(f"  objectives: [{v_str}]")

        # Filter to Pareto-optimal points
        pareto = self._filter_pareto(all_results)

        if verbose:
            print(f"\nPareto front: {len(pareto)} / {len(all_results)} points")

        return pareto

    def _filter_pareto(
        self,
        results: list[tuple[torch.Tensor, dict[str, float]]],
    ) -> list[tuple[torch.Tensor, dict[str, float]]]:
        """Filter results to Pareto-optimal points.

        A point is Pareto-optimal if no other point dominates it
        (i.e., is better or equal in all objectives and strictly
        better in at least one).
        """
        pareto = []
        for i, (dens_i, obj_i) in enumerate(results):
            dominated = False
            for j, (dens_j, obj_j) in enumerate(results):
                if i == j:
                    continue
                # Check if j dominates i
                all_leq = all(
                    obj_j[k] <= obj_i[k] + 1e-8
                    for k in self.obj_names
                )
                any_lt = any(
                    obj_j[k] < obj_i[k] - 1e-8
                    for k in self.obj_names
                )
                if all_leq and any_lt:
                    dominated = True
                    break
            if not dominated:
                pareto.append((dens_i, obj_i))
        return pareto
