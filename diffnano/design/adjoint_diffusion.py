"""Adjoint-guided latent diffusion for nanophotonic inverse design (N11.1).

Extends LatentDiffusionDesigner with TRUE adjoint-gradient guidance via the
guidance_fn interface.  Where PhysicsGuidance uses a soft classifier-style
gradient (decode z -> design -> forward model -> MSE loss -> autograd),
AdjointGuidance wraps an RCWA solver with exact backpropagation through the
full physics, producing mathematically rigorous adjoint gradients.

References:
    Seo et al., "Physics-Guided and Fabrication-Aware Inverse Design of
    Photonic Devices Using Diffusion Models (AdjointDiffusion)",
    ACS Photonics 2026, 13(2):363-372; arXiv:2504.17077

    Clean-room implementation -- mechanism only, no weights from published code.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffnano.design.latent_diffusion import (
    LatentDecoder,
    PhysicsGuidance,
)
from diffnano.solvers.rcwa import RCWASolver

__all__ = [
    "AdjointGuidance",
    "AdjointDiffusionDesigner",
    "AdjointDiffusionBenchmark",
]


class AdjointGuidance(nn.Module):
    """True adjoint-gradient guidance for diffusion sampling.

    Unlike PhysicsGuidance (which decodes z -> design, runs a forward model,
    and backprops through everything), AdjointGuidance uses the RCWA solver's
    native differentiable path combined with ``torch.autograd.grad`` to obtain
    exact adjoint gradients of a physics figure-of-merit w.r.t. the latent z.

    The key difference from soft classifier guidance:
    - PhysicsGuidance: gradient of MSE(predicted_response, target) w.r.t. z
    - AdjointGuidance: gradient of a physics FOM (e.g. diffraction efficiency)
      through the full RCWA solver chain, giving true adjoint information.

    Parameters
    ----------
    solver : RCWASolver
        Differentiable RCWA solver instance.
    decoder : LatentDecoder
        Latent-to-design decoder.
    target_response : Tensor, optional
        Target optical response vector for loss computation.
    loss_fn : callable, optional
        Physics loss: ``loss_fn(predicted_response, target) -> scalar``.
        Defaults to MSE.
    forward_budget : int
        Maximum number of forward solver calls allowed.
    fom_fn : callable, optional
        Figure-of-merit function: ``fom_fn(response) -> scalar``.
        If provided, the gradient is taken w.r.t. ``-fom_fn(response)``
        (maximise FOM).  Otherwise uses ``loss_fn``.
    """

    def __init__(
        self,
        solver: RCWASolver,
        decoder: LatentDecoder,
        target_response: Tensor | None = None,
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        forward_budget: int = 1000,
        fom_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.solver = solver
        self.decoder = decoder
        self.target_response = target_response
        self.loss_fn = loss_fn or F.mse_loss
        self.forward_budget = forward_budget
        self.fom_fn = fom_fn

        self._forward_calls: int = 0
        self._budget_exhausted: bool = False

    @property
    def forward_calls(self) -> int:
        """Number of forward solver calls consumed so far."""
        return self._forward_calls

    @property
    def budget_remaining(self) -> int:
        """Remaining forward solver budget."""
        return max(0, self.forward_budget - self._forward_calls)

    def reset_budget(self) -> None:
        """Reset the forward-call counter."""
        self._forward_calls = 0
        self._budget_exhausted = False

    def _run_solver(self, designs: Tensor) -> Tensor:
        """Run the RCWA solver on decoded designs and track budget.

        Parameters
        ----------
        designs : Tensor, shape ``(batch, H, W)``
            Design density fields in [0, 1].

        Returns
        -------
        response : Tensor, shape ``(batch, n_fourier)``
            Diffraction efficiencies.
        """
        if self._budget_exhausted:
            return torch.zeros(designs.shape[0], self.solver.n_fourier, device=designs.device)

        self._forward_calls += designs.shape[0]
        if self._forward_calls >= self.forward_budget:
            self._budget_exhausted = True

        # Convert design densities to permittivity layers.
        eps_low = self.solver.eps_ambient
        eps_high = self.solver.eps_substrate if self.solver.eps_substrate > 1.0 else 12.0
        eps_layers = eps_low + (eps_high - eps_low) * designs  # (batch, n_grid)
        result = self.solver.forward(eps_layers)
        return result.field  # (batch, n_fourier)

    def _compute_physics_loss(self, response: Tensor, condition: Tensor) -> Tensor:
        """Compute physics loss from solver response.

        Uses ``target_response`` (in solver-output space) when available.
        Falls back to a default FOM: maximise the first-order diffraction
        efficiency (peak of the response spectrum), which is always valid
        regardless of conditioning dimension.
        """
        if self.fom_fn is not None:
            fom = self.fom_fn(response)
            return -fom  # negative: we maximise the FOM

        target = self.target_response
        if target is not None and target.shape[-1] == response.shape[-1]:
            if target.dim() == 1:
                target = target.unsqueeze(0).expand_as(response)
            return self.loss_fn(response, target)

        # Default FOM: maximise peak diffraction efficiency.
        return -response.max(dim=-1).values.mean()

    def guide_score(
        self,
        z: Tensor,
        condition: Tensor,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        """Compute true adjoint gradient of physics loss w.r.t. latent z.

        Parameters
        ----------
        z : Tensor, shape ``(batch, latent_dim)``
            Current latent sample.
        condition : Tensor
            Conditioning vector (target optical response).
        guidance_scale : float
            Scale factor for the gradient.

        Returns
        -------
        Tensor, shape ``(batch, latent_dim)``
            Adjoint gradient w.r.t. z.
        """
        if self._budget_exhausted:
            return torch.zeros_like(z)

        z_param = z.detach().requires_grad_(True)
        designs = self.decoder(z_param)  # (batch, H, W)

        response = self._run_solver(designs)  # (batch, n_fourier)
        loss = self._compute_physics_loss(response, condition)

        grad = torch.autograd.grad(loss, z_param)[0]
        return guidance_scale * grad


class AdjointDiffusionDesigner(nn.Module):
    """Latent diffusion designer with adjoint-gradient guidance option.

    Extends the standard LatentDiffusionDesigner to support both soft
    classifier guidance (PhysicsGuidance) and true adjoint guidance
    (AdjointGuidance) through the ``use_adjoint`` flag.

    Parameters
    ----------
    latent_encoder : LatentEncoder
    latent_decoder : LatentDecoder
    diffusion : ConditionedDiffusion
    quantizer : nn.Module, optional
    soft_guidance : PhysicsGuidance, optional
        Soft classifier guidance (default path).
    adjoint_guidance : AdjointGuidance, optional
        True adjoint guidance.  If None but ``use_adjoint=True`` is
        requested, will raise ValueError.
    """

    def __init__(
        self,
        latent_encoder: nn.Module,
        latent_decoder: LatentDecoder,
        diffusion: nn.Module,
        quantizer: nn.Module | None = None,
        soft_guidance: PhysicsGuidance | None = None,
        adjoint_guidance: AdjointGuidance | None = None,
    ) -> None:
        super().__init__()
        self.encoder = latent_encoder
        self.decoder = latent_decoder
        self.diffusion = diffusion
        self.quantizer = quantizer
        self.soft_guidance = soft_guidance
        self.adjoint_guidance = adjoint_guidance

    def _get_guidance(self, use_adjoint: bool) -> nn.Module | None:
        """Select the appropriate guidance module."""
        if use_adjoint:
            if self.adjoint_guidance is None:
                raise ValueError(
                    "use_adjoint=True but no AdjointGuidance was provided. "
                    "Pass adjoint_guidance to the constructor."
                )
            return self.adjoint_guidance
        return self.soft_guidance

    @torch.no_grad()
    def design(
        self,
        target_response: Tensor,
        n_candidates: int = 16,
        n_diffusion_steps: int = 50,
        guidance_scale: float = 1.0,
        use_adjoint: bool = False,
    ) -> dict[str, Tensor]:
        """Generate design candidates via latent diffusion sampling.

        Parameters
        ----------
        target_response : Tensor, shape ``(cond_dim,)``
        n_candidates : int
        n_diffusion_steps : int
        guidance_scale : float
        use_adjoint : bool
            If True, use AdjointGuidance for true adjoint gradients.
            If False, use PhysicsGuidance (soft classifier guidance).

        Returns
        -------
        dict with ``candidates``, ``latent_samples``, ``guidance_mode``
        """
        guidance = self._get_guidance(use_adjoint)

        # Set target on the guidance module
        if guidance is not None and guidance_scale > 0:
            if hasattr(guidance, "target_response"):
                guidance.target_response = target_response

        z_samples = self._guided_sample(
            target_response,
            n_diffusion_steps,
            guidance_scale,
            n_candidates,
            guidance,
        )

        designs = self.decoder(z_samples)

        return {
            "candidates": designs,
            "latent_samples": z_samples,
            "guidance_mode": "adjoint" if use_adjoint else "soft",
        }

    def _guided_sample(
        self,
        condition: Tensor,
        n_steps: int,
        guidance_scale: float,
        n_samples: int,
        guidance: nn.Module | None,
    ) -> Tensor:
        """Sample with guidance, re-enabling autograd for the guidance call.

        Replicates the core DDPM denoising loop from
        ``ConditionedDiffusion.sample`` but calls the guidance inside
        ``torch.enable_grad()`` so that autograd can compute gradients
        through the decoder and solver.
        """

        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        batch = condition.shape[0]

        cond_expanded = condition.repeat_interleave(n_samples, dim=0)
        latent_dim = self.diffusion.latent_dim
        z = torch.randn(
            batch * n_samples, latent_dim, device=condition.device, dtype=condition.dtype
        )

        step_indices = torch.linspace(
            self.diffusion.n_steps - 1, 0, n_steps, device=condition.device, dtype=torch.long
        )

        for idx in step_indices:
            t = idx.expand(batch * n_samples).to(condition.device)
            noise_pred = self.diffusion(z, t, cond_expanded)

            if guidance is not None and guidance_scale > 0:
                with torch.enable_grad():
                    grad = guidance.guide_score(z, cond_expanded, guidance_scale)
                noise_pred = noise_pred - guidance_scale * grad

            alpha_bar = self.diffusion._cosine_schedule(t.float()).unsqueeze(-1)
            t_prev = (t - 1).clamp(min=0).float()
            alpha_bar_prev = self.diffusion._cosine_schedule(t_prev).unsqueeze(-1)

            alpha = alpha_bar / alpha_bar_prev
            sigma = (
                torch.sqrt(1.0 - alpha_bar_prev)
                / torch.sqrt(1.0 - alpha_bar).clamp(min=1e-8)
                * torch.sqrt(1.0 - alpha)
            )
            sigma = sigma.clamp(min=1e-8)

            z = (
                z - (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar).clamp(min=1e-8) * noise_pred
            ) / torch.sqrt(alpha).clamp(min=1e-8)

            if idx > 0:
                noise = torch.randn_like(z)
                z = z + sigma * noise

        return z

    def compare_adjoint_vs_soft(
        self,
        target_response: Tensor,
        n_candidates: int = 8,
        n_diffusion_steps: int = 20,
        guidance_scale: float = 1.0,
        forward_budget: int = 200,
        scorer: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> dict[str, object]:
        """Compare adjoint guidance vs soft guidance with fixed forward budget.

        Parameters
        ----------
        target_response : Tensor, shape ``(cond_dim,)``
        n_candidates : int
        n_diffusion_steps : int
        guidance_scale : float
        forward_budget : int
            Max forward solver calls for the adjoint path.
        scorer : callable, optional
            ``(design, target) -> scalar`` for FoM evaluation.

        Returns
        -------
        dict with ``adjoint``, ``soft``, each containing ``best_fom``,
        ``forward_calls``, ``wall_time``.
        """
        results: dict[str, dict] = {}

        for mode, use_adjoint in [("soft", False), ("adjoint", True)]:
            if use_adjoint and self.adjoint_guidance is not None:
                self.adjoint_guidance.forward_budget = forward_budget
                self.adjoint_guidance.reset_budget()

            t0 = time.perf_counter()
            design_result = self.design(
                target_response,
                n_candidates=n_candidates,
                n_diffusion_steps=n_diffusion_steps,
                guidance_scale=guidance_scale,
                use_adjoint=use_adjoint,
            )
            wall_time = time.perf_counter() - t0

            candidates = design_result["candidates"]

            if scorer is not None:
                scores = torch.stack(
                    [scorer(candidates[i], target_response) for i in range(candidates.shape[0])]
                )
                best_fom = scores.max().item()
            else:
                best_fom = 0.0

            forward_calls = 0
            if use_adjoint and self.adjoint_guidance is not None:
                forward_calls = self.adjoint_guidance.forward_calls
            elif not use_adjoint and self.soft_guidance is not None:
                forward_calls = n_diffusion_steps * n_candidates

            results[mode] = {
                "best_fom": best_fom,
                "forward_calls": forward_calls,
                "wall_time": wall_time,
            }

        return results


class AdjointDiffusionBenchmark:
    """Benchmark: adjoint guidance vs soft guidance vs classical optimisers.

    Compares the three approaches on a fixed RCWA forward-model budget and
    reports FoM (figure of merit), wall-clock time, and forward-call count.

    Parameters
    ----------
    adjoint_designer : AdjointDiffusionDesigner
        Designer with both soft and adjoint guidance configured.
    classical_fn : callable, optional
        Classical optimiser function:
        ``classical_fn(target, budget) -> dict`` with ``best``, ``fom``, keys.
    scorer : callable
        ``(design, target) -> scalar`` for FoM evaluation.
    grid_size : int
        Design grid dimension.
    """

    def __init__(
        self,
        adjoint_designer: AdjointDiffusionDesigner,
        classical_fn: Callable[[Tensor, int], dict[str, Tensor]] | None = None,
        scorer: Callable[[Tensor, Tensor], Tensor] | None = None,
        grid_size: int = 32,
    ) -> None:
        self.designer = adjoint_designer
        self.classical_fn = classical_fn
        self.scorer = scorer
        self.grid_size = grid_size

    def run(
        self,
        target_response: Tensor,
        forward_budget: int = 200,
        n_candidates: int = 8,
        n_diffusion_steps: int = 20,
        n_seeds: int = 3,
    ) -> dict[str, dict]:
        """Run the three-way benchmark.

        Parameters
        ----------
        target_response : Tensor, shape ``(cond_dim,)``
        forward_budget : int
        n_candidates : int
        n_diffusion_steps : int
        n_seeds : int

        Returns
        -------
        dict with ``adjoint``, ``soft``, and optionally ``classical`` sub-dicts,
        each containing ``foms``, ``best_fom``, ``mean_fom``, ``forward_calls``,
        ``wall_time``.
        """
        all_results: dict[str, dict] = {}

        for mode, use_adjoint in [("soft", False), ("adjoint", True)]:
            foms: list[float] = []
            times: list[float] = []
            fwd_calls: list[int] = []

            for seed in range(n_seeds):
                torch.manual_seed(seed)

                if use_adjoint and self.designer.adjoint_guidance is not None:
                    self.designer.adjoint_guidance.forward_budget = forward_budget
                    self.designer.adjoint_guidance.reset_budget()

                t0 = time.perf_counter()
                result = self.designer.design(
                    target_response,
                    n_candidates=n_candidates,
                    n_diffusion_steps=n_diffusion_steps,
                    use_adjoint=use_adjoint,
                )
                elapsed = time.perf_counter() - t0

                candidates = result["candidates"]

                if self.scorer is not None:
                    scores = torch.stack(
                        [
                            self.scorer(candidates[i], target_response)
                            for i in range(candidates.shape[0])
                        ]
                    )
                    foms.append(scores.max().item())
                else:
                    foms.append(0.0)

                times.append(elapsed)
                if use_adjoint and self.designer.adjoint_guidance is not None:
                    fwd_calls.append(self.designer.adjoint_guidance.forward_calls)
                else:
                    fwd_calls.append(n_diffusion_steps * n_candidates)

            import numpy as np

            foms_arr = np.array(foms)
            all_results[mode] = {
                "foms": foms,
                "best_fom": float(foms_arr.max()),
                "mean_fom": float(foms_arr.mean()),
                "forward_calls": fwd_calls,
                "wall_time": times,
            }

        # Classical optimiser (optional)
        if self.classical_fn is not None:
            classical_foms: list[float] = []
            classical_times: list[float] = []
            classical_fwd: list[int] = []

            for seed in range(n_seeds):
                torch.manual_seed(seed)
                t0 = time.perf_counter()
                cls_result = self.classical_fn(target_response, forward_budget)
                elapsed = time.perf_counter() - t0

                fom = cls_result.get("fom", torch.tensor(0.0))
                if isinstance(fom, Tensor):
                    fom = fom.item()
                classical_foms.append(fom)
                classical_times.append(elapsed)
                classical_fwd.append(forward_budget)

            import numpy as np

            cls_arr = np.array(classical_foms)
            all_results["classical"] = {
                "foms": classical_foms,
                "best_fom": float(cls_arr.max()),
                "mean_fom": float(cls_arr.mean()),
                "forward_calls": classical_fwd,
                "wall_time": classical_times,
            }

        return all_results
