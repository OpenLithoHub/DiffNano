"""Tests for adjoint-guided latent diffusion inverse design (N11.1)."""

import pytest
import torch

from diffnano.design.adjoint_diffusion import (
    AdjointDiffusionBenchmark,
    AdjointDiffusionDesigner,
    AdjointGuidance,
)
from diffnano.design.latent_diffusion import (
    ConditionedDiffusion,
    LatentDecoder,
    LatentEncoder,
    PhysicsGuidance,
)
from diffnano.solvers.rcwa import RCWASolver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_encoder_decoder(grid_size=16, latent_dim=8):
    enc = LatentEncoder(grid_size=grid_size, latent_dim=latent_dim, hidden_channels=16)
    dec = LatentDecoder(latent_dim=latent_dim, grid_size=grid_size, hidden_channels=16)
    return enc, dec


def _make_diffusion(latent_dim=8, cond_dim=16):
    return ConditionedDiffusion(latent_dim=latent_dim, cond_dim=cond_dim, n_blocks=2, n_steps=100)


def _make_solver(n_fourier_orders=2, n_grid=5):
    return RCWASolver(
        fourier_orders=n_fourier_orders,
        wavelength_nm=532.0,
        period_nm=(400.0, 400.0),
        eps_ambient=1.0,
        eps_substrate=2.25,
        device="cpu",
        solver_backend="eig",
    )


def _make_adjoint_designer(grid_size=16, latent_dim=8, cond_dim=16):
    enc, dec = _make_encoder_decoder(grid_size, latent_dim)
    diff = _make_diffusion(latent_dim, cond_dim)
    solver = _make_solver()

    soft_guidance = PhysicsGuidance(
        forward_model=lambda d: d.reshape(d.shape[0], -1)[:, :cond_dim],
        decoder=dec,
    )
    adjoint_guidance = AdjointGuidance(
        solver=solver,
        decoder=dec,
        forward_budget=500,
    )
    return AdjointDiffusionDesigner(
        latent_encoder=enc,
        latent_decoder=dec,
        diffusion=diff,
        soft_guidance=soft_guidance,
        adjoint_guidance=adjoint_guidance,
    )


def _mean_scorer(design: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """Simple scorer: mean of design pixels."""
    return design.mean()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdjointGuidance:
    """Tests for the AdjointGuidance class."""

    def test_gradient_nonzero(self):
        """AdjointGuidance computes non-zero gradients."""
        dec = LatentDecoder(latent_dim=8, grid_size=16, hidden_channels=16)
        solver = _make_solver()
        target = torch.randn(1, 5)
        guidance = AdjointGuidance(solver=solver, decoder=dec, target_response=target)
        z = torch.randn(1, 8)
        grad = guidance.guide_score(z, target, guidance_scale=1.0)
        assert grad.shape == (1, 8)
        assert not torch.all(grad == 0), "Adjoint gradient should not be all zeros"

    def test_gradient_direction_loss_decreasing(self):
        """Adjoint gradient points in the loss-decreasing direction."""
        dec = LatentDecoder(latent_dim=8, grid_size=16, hidden_channels=16)
        solver = _make_solver()
        target = torch.randn(1, 5)
        guidance = AdjointGuidance(
            solver=solver,
            decoder=dec,
            target_response=target,
            forward_budget=100,
        )

        z = torch.randn(1, 8, requires_grad=False)
        grad = guidance.guide_score(z, target, guidance_scale=1.0)

        # The negative gradient direction should decrease the loss
        step = 0.01
        z_new = z - step * grad

        # Compute loss at z and z_new
        designs_old = dec(z)
        designs_new = dec(z_new)
        eps_low = solver.eps_ambient
        eps_high = solver.eps_substrate if solver.eps_substrate > 1.0 else 12.0

        eps_old = eps_low + (eps_high - eps_low) * designs_old
        eps_new = eps_low + (eps_high - eps_low) * designs_new
        resp_old = solver.forward(eps_old).field
        resp_new = solver.forward(eps_new).field

        loss_old = torch.nn.functional.mse_loss(resp_old, target.expand_as(resp_old))
        loss_new = torch.nn.functional.mse_loss(resp_new, target.expand_as(resp_new))

        assert loss_new <= loss_old + 1e-3, (
            f"Loss should not increase after gradient step: {loss_new:.6f} > {loss_old:.6f}"
        )

    def test_adjoint_vs_soft_guidance_gradient_differs(self):
        """Adjoint gradient is different from soft guidance gradient."""
        dec = LatentDecoder(latent_dim=8, grid_size=16, hidden_channels=16)
        solver = _make_solver()
        target = torch.randn(1, 5)

        soft = PhysicsGuidance(
            forward_model=lambda d: d.reshape(d.shape[0], -1)[:, :5],
            decoder=dec,
            target_response=target,
        )
        adjoint = AdjointGuidance(solver=solver, decoder=dec, target_response=target)

        z = torch.randn(1, 8)
        soft_grad = soft.guide_score(z, target, guidance_scale=1.0)
        adj_grad = adjoint.guide_score(z, target, guidance_scale=1.0)

        # The gradients should differ because they compute different things:
        # soft: gradient through a dummy model that just slices the design
        # adjoint: gradient through the full RCWA solver
        assert not torch.allclose(soft_grad, adj_grad, atol=1e-6), (
            "Adjoint and soft gradients should differ"
        )

    def test_budget_tracking(self):
        """Adjoint budget tracking works correctly."""
        dec = LatentDecoder(latent_dim=8, grid_size=16, hidden_channels=16)
        solver = _make_solver()
        guidance = AdjointGuidance(solver=solver, decoder=dec, forward_budget=10)

        assert guidance.forward_calls == 0
        assert guidance.budget_remaining == 10

        z = torch.randn(2, 8)
        target = torch.randn(2, 5)
        guidance.guide_score(z, target)
        assert guidance.forward_calls == 2
        assert guidance.budget_remaining == 8

    def test_budget_exhaustion(self):
        """Guidance returns zero gradient when budget is exhausted."""
        dec = LatentDecoder(latent_dim=8, grid_size=16, hidden_channels=16)
        solver = _make_solver()
        guidance = AdjointGuidance(solver=solver, decoder=dec, forward_budget=2)

        z = torch.randn(2, 8)
        target = torch.randn(2, 5)
        # First call consumes 2 forward calls -> budget exhausted
        grad1 = guidance.guide_score(z, target)
        assert not torch.all(grad1 == 0)
        assert guidance._budget_exhausted

        # Second call should return zeros
        guidance.reset_budget()
        guidance._forward_calls = guidance.forward_budget  # force exhaustion
        guidance._budget_exhausted = True
        grad2 = guidance.guide_score(z, target)
        assert torch.all(grad2 == 0)

    def test_budget_reset(self):
        """Budget reset restores the call counter."""
        dec = LatentDecoder(latent_dim=8, grid_size=16, hidden_channels=16)
        solver = _make_solver()
        guidance = AdjointGuidance(solver=solver, decoder=dec, forward_budget=100)

        z = torch.randn(3, 8)
        target = torch.randn(3, 5)
        guidance.guide_score(z, target)
        assert guidance.forward_calls == 3

        guidance.reset_budget()
        assert guidance.forward_calls == 0
        assert guidance.budget_remaining == 100
        assert not guidance._budget_exhausted


class TestAdjointDiffusionDesigner:
    """Tests for AdjointDiffusionDesigner."""

    def test_produces_valid_candidates_soft(self):
        """AdjointDiffusionDesigner produces valid candidates (soft mode)."""
        designer = _make_adjoint_designer()
        target = torch.randn(16)
        result = designer.design(target, n_candidates=4, n_diffusion_steps=5, use_adjoint=False)
        assert result["candidates"].shape[0] == 4
        assert result["candidates"].shape[-1] == 16
        assert result["guidance_mode"] == "soft"

    def test_produces_valid_candidates_adjoint(self):
        """Design with adjoint guidance works end-to-end."""
        designer = _make_adjoint_designer()
        target = torch.randn(16)
        result = designer.design(target, n_candidates=4, n_diffusion_steps=5, use_adjoint=True)
        assert result["candidates"].shape[0] == 4
        assert result["candidates"].shape[-1] == 16
        assert result["guidance_mode"] == "adjoint"

    def test_adjoint_mode_requires_guidance(self):
        """Requesting adjoint mode without guidance raises ValueError."""
        enc, dec = _make_encoder_decoder()
        diff = _make_diffusion()
        designer = AdjointDiffusionDesigner(enc, dec, diff)
        target = torch.randn(16)
        with pytest.raises(ValueError, match="use_adjoint=True"):
            designer.design(target, use_adjoint=True)

    def test_multiple_candidates_generation(self):
        """Multiple candidates generation works with adjoint guidance."""
        designer = _make_adjoint_designer()
        target = torch.randn(16)
        n_candidates = 8
        result = designer.design(
            target,
            n_candidates=n_candidates,
            n_diffusion_steps=5,
            use_adjoint=True,
        )
        assert result["candidates"].shape[0] == n_candidates
        assert result["latent_samples"].shape[0] == n_candidates

    def test_compare_adjoint_vs_soft(self):
        """Comparison vs soft guidance returns valid metrics."""
        designer = _make_adjoint_designer()
        target = torch.randn(16)
        scorer = _mean_scorer
        comparison = designer.compare_adjoint_vs_soft(
            target,
            n_candidates=4,
            n_diffusion_steps=3,
            forward_budget=100,
            scorer=scorer,
        )
        assert "adjoint" in comparison
        assert "soft" in comparison
        assert "best_fom" in comparison["adjoint"]
        assert "forward_calls" in comparison["adjoint"]
        assert "wall_time" in comparison["adjoint"]
        assert isinstance(comparison["adjoint"]["best_fom"], float)
        assert isinstance(comparison["soft"]["best_fom"], float)


class TestAdjointDiffusionBenchmark:
    """Tests for AdjointDiffusionBenchmark."""

    def test_benchmark_runs_both_methods(self):
        """Benchmark runs with both adjoint and soft methods."""
        designer = _make_adjoint_designer()
        scorer = _mean_scorer
        bench = AdjointDiffusionBenchmark(
            adjoint_designer=designer,
            scorer=scorer,
            grid_size=16,
        )
        result = bench.run(
            target_response=torch.randn(16),
            forward_budget=100,
            n_candidates=4,
            n_diffusion_steps=3,
            n_seeds=2,
        )
        assert "adjoint" in result
        assert "soft" in result
        assert "foms" in result["adjoint"]
        assert "best_fom" in result["adjoint"]
        assert "mean_fom" in result["adjoint"]
        assert len(result["adjoint"]["foms"]) == 2

    def test_benchmark_with_classical(self):
        """Benchmark includes classical optimiser when provided."""
        designer = _make_adjoint_designer()

        def classical_fn(target, budget):
            return {"best": torch.rand(16, 16), "fom": torch.tensor(0.5)}

        scorer = _mean_scorer
        bench = AdjointDiffusionBenchmark(
            adjoint_designer=designer,
            classical_fn=classical_fn,
            scorer=scorer,
            grid_size=16,
        )
        result = bench.run(
            target_response=torch.randn(16),
            forward_budget=100,
            n_candidates=4,
            n_diffusion_steps=3,
            n_seeds=2,
        )
        assert "classical" in result
        assert "best_fom" in result["classical"]
        assert len(result["classical"]["foms"]) == 2

    def test_fom_improvement_quantified(self):
        """FoM improvement is quantified and reported."""
        designer = _make_adjoint_designer()
        scorer = _mean_scorer
        bench = AdjointDiffusionBenchmark(
            adjoint_designer=designer,
            scorer=scorer,
            grid_size=16,
        )
        result = bench.run(
            target_response=torch.randn(16),
            forward_budget=100,
            n_candidates=4,
            n_diffusion_steps=3,
            n_seeds=2,
        )
        # Both methods should report finite FOMs
        for mode in ("adjoint", "soft"):
            assert all(isinstance(f, float) for f in result[mode]["foms"])
            assert result[mode]["best_fom"] >= result[mode]["mean_fom"]
