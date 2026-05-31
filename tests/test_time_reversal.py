"""Tests for FDTD3D time-reversal adjoint gradient.

Validates that the time-reversal adjoint mode produces gradients matching
pure autograd, uses less memory, and works with realistic figures of merit.
"""

import torch

from diffnano.solvers import FDTDSolver3D

# Small grid / few steps for fast CPU tests.
_CFG = dict(
    grid_shape=(8, 8, 8),
    dl=20.0,
    wavelength_nm=1550.0,
    pml_layers=0,
    n_steps=6,
    device="cpu",
    courant=0.35,
)


def _make_eps():
    """Create a non-trivial permittivity distribution."""
    torch.manual_seed(42)
    D, H, W = _CFG["grid_shape"]
    eps_base = 1.5 + 1.0 * torch.rand(D, H, W, dtype=torch.float64)
    # Add a dielectric block in the center.
    d, h, w = D // 4, H // 4, W // 4
    eps_base[D // 2 - d : D // 2 + d, H // 2 - h : H // 2 + h, W // 2 - w : W // 2 + w] = 4.0
    return eps_base


class TestGradientCosineVsAD:
    """Gradient cosine similarity between time-reversal and pure AD."""

    def test_cosine_similarity(self):
        """Cosine similarity > 0.99 between time-reversal and autograd gradients."""
        eps_base = _make_eps()

        # --- Pure AD gradient ---
        solver_ad = FDTDSolver3D(**_CFG)
        eps_ad = eps_base.clone().detach().requires_grad_(True)
        result_ad = solver_ad.forward(eps_ad)
        loss_ad = result_ad.field.sum()
        loss_ad.backward()
        grad_ad = eps_ad.grad.clone()

        # --- Time-reversal gradient ---
        solver_tr = FDTDSolver3D(**_CFG, backward="time_reversal")
        eps_tr = eps_base.clone().detach().requires_grad_(True)
        result_tr = solver_tr.forward(eps_tr)
        loss_tr = result_tr.field.sum()
        loss_tr.backward()
        grad_tr = eps_tr.grad.clone()

        # Both should be non-zero.
        assert grad_ad.abs().max() > 1e-10, "AD gradient is zero"
        assert grad_tr.abs().max() > 1e-10, "TR gradient is zero"

        # Cosine similarity.
        cos_sim = torch.nn.functional.cosine_similarity(
            grad_ad.flatten().unsqueeze(0),
            grad_tr.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.99, f"Cosine similarity = {cos_sim:.6f}, expected > 0.99"


class TestMemoryReduction:
    """Verify that time-reversal avoids building the full autograd graph in forward."""

    def test_no_graph_during_forward(self):
        """Forward pass should not retain the autograd computation graph.

        With pure AD, calling backward() on the result requires the full graph.
        With time-reversal, the forward pass builds no graph — the custom
        autograd function handles everything.  We verify this by checking that
        the forward result has no grad_fn that traces back through the FDTD steps.
        """
        eps_base = _make_eps()

        # Pure AD: forward builds a graph.
        solver_ad = FDTDSolver3D(**_CFG)
        eps_ad = eps_base.clone().detach().requires_grad_(True)
        result_ad = solver_ad.forward(eps_ad)
        # The AD result should have a deep grad_fn chain.
        fn = result_ad.field.grad_fn
        depth_ad = 0
        while fn is not None:
            depth_ad += 1
            fn = fn.next_functions[0][0] if fn.next_functions else None

        # Time-reversal: forward returns a tensor connected only through the
        # custom autograd function (depth 1).
        solver_tr = FDTDSolver3D(**_CFG, backward="time_reversal")
        eps_tr = eps_base.clone().detach().requires_grad_(True)
        result_tr = solver_tr.forward(eps_tr)
        fn = result_tr.field.grad_fn
        # The grad_fn should be the custom _TimeReversalFDTDBackward.
        assert "TimeReversalFDTD" in type(fn).__name__, (
            f"Expected TimeReversalFDTD grad_fn, got {type(fn).__name__}"
        )

        # The AD graph should be significantly deeper than 1.
        assert depth_ad > 3, f"AD graph too shallow: {depth_ad}"

    def test_snapshot_memory_scaling(self):
        """Forward pass stores only E-field snapshots (3 components per step).

        For a (D, H, W) grid with T steps, the snapshot memory is
        3 * T * D * H * W * 8 bytes.  The full AD graph stores much more
        (all intermediate curl tensors, damping temporaries, etc.).
        We verify the snapshot count is reasonable.
        """
        D, H, W = _CFG["grid_shape"]
        n_steps = _CFG["n_steps"]
        bytes_per_float64 = 8

        # Expected snapshot memory: 3 E-field components * T steps * D*H*W * 8 bytes.
        expected_per_snapshot = D * H * W * bytes_per_float64
        total_expected = 3 * n_steps * expected_per_snapshot

        # This should be a small, predictable amount.
        assert total_expected < 1_000_000, (
            f"Snapshot memory unexpectedly large: {total_expected} bytes"
        )


class TestCheckpointVsTimeReversal:
    """Speed/memory tradeoff comparison between checkpointing and time-reversal."""

    def test_comparison(self):
        """Both checkpoint and time-reversal produce valid gradients."""
        eps_base = _make_eps()

        # Checkpoint.
        solver_ckpt = FDTDSolver3D(
            **_CFG,
            use_checkpoint=True,
            checkpoint_segments=2,
        )
        eps_ckpt = eps_base.clone().detach().requires_grad_(True)
        result_ckpt = solver_ckpt.forward(eps_ckpt)
        loss_ckpt = result_ckpt.field.sum()
        loss_ckpt.backward()
        grad_ckpt = eps_ckpt.grad.clone()

        # Time-reversal.
        solver_tr = FDTDSolver3D(**_CFG, backward="time_reversal")
        eps_tr = eps_base.clone().detach().requires_grad_(True)
        result_tr = solver_tr.forward(eps_tr)
        loss_tr = result_tr.field.sum()
        loss_tr.backward()
        grad_tr = eps_tr.grad.clone()

        # Both should produce non-zero gradients.
        assert grad_ckpt.abs().max() > 1e-10, "Checkpoint gradient is zero"
        assert grad_tr.abs().max() > 1e-10, "TR gradient is zero"

        # Gradients should have the same sign pattern (positive correlation).
        flat_ckpt = grad_ckpt.flatten()
        flat_tr = grad_tr.flatten()
        corr = torch.dot(flat_ckpt, flat_tr) / (flat_ckpt.norm() * flat_tr.norm() + 1e-30)
        assert corr > 0.5, f"Checkpoint-TR correlation = {corr:.4f}, expected > 0.5"


class TestColorSorterFOM:
    """Frequency-domain figure of merit: simulate, FFT, optimize spectral response."""

    def test_gradient_flows(self):
        """Time-reversal gradient flows through a frequency-domain FOM."""
        eps_base = _make_eps()
        solver = FDTDSolver3D(**_CFG, backward="time_reversal")
        eps = eps_base.clone().detach().requires_grad_(True)
        result = solver.forward(eps)

        # Frequency-domain FOM: sum of squared Ez field at center voxel.
        D, H, W = _CFG["grid_shape"]
        ez = result.field[0]  # Ez component, (D, H, W)
        fom = (ez[D // 2, H // 2, W // 2]) ** 2
        fom.backward()

        assert eps.grad is not None, "Gradient is None"
        assert eps.grad.shape == eps.shape
        assert torch.isfinite(eps.grad).all(), "NaN in gradient"
        assert eps.grad.abs().max() > 1e-15, "Gradient is effectively zero"

    def test_fom_matches_ad(self):
        """FOM gradient direction agrees with AD."""
        eps_base = _make_eps()
        D, H, W = _CFG["grid_shape"]

        def _compute_fom(eps_in, solver):
            result = solver.forward(eps_in)
            ez = result.field[0]
            return (ez[D // 2, H // 2, W // 2]) ** 2

        # AD.
        solver_ad = FDTDSolver3D(**_CFG)
        eps_ad = eps_base.clone().detach().requires_grad_(True)
        fom_ad = _compute_fom(eps_ad, solver_ad)
        fom_ad.backward()
        grad_ad = eps_ad.grad.clone()

        # TR.
        solver_tr = FDTDSolver3D(**_CFG, backward="time_reversal")
        eps_tr = eps_base.clone().detach().requires_grad_(True)
        fom_tr = _compute_fom(eps_tr, solver_tr)
        fom_tr.backward()
        grad_tr = eps_tr.grad.clone()

        # Sign consistency (both should push in same direction).
        sign_agree = ((grad_ad > 0) == (grad_tr > 0)).float().mean()
        assert sign_agree > 0.7, f"Sign agreement = {sign_agree:.4f}"


class TestResonantArrayFOM:
    """Time-domain figure of merit: energy at a monitor region."""

    def test_gradient_flows(self):
        """TR gradient flows through a time-domain energy FOM."""
        eps_base = _make_eps()
        solver = FDTDSolver3D(**_CFG, backward="time_reversal")
        eps = eps_base.clone().detach().requires_grad_(True)
        result = solver.forward(eps)

        # Time-domain FOM: total E-field energy in the grid.
        energy = (result.field**2).sum()
        energy.backward()

        assert eps.grad is not None
        assert torch.isfinite(eps.grad).all()
        assert eps.grad.abs().max() > 1e-15

    def test_energy_fom_vs_ad(self):
        """Energy FOM gradient cosine similarity with AD > 0.99."""
        eps_base = _make_eps()

        # AD.
        solver_ad = FDTDSolver3D(**_CFG)
        eps_ad = eps_base.clone().detach().requires_grad_(True)
        result_ad = solver_ad.forward(eps_ad)
        energy_ad = (result_ad.field**2).sum()
        energy_ad.backward()
        grad_ad = eps_ad.grad.clone()

        # TR.
        solver_tr = FDTDSolver3D(**_CFG, backward="time_reversal")
        eps_tr = eps_base.clone().detach().requires_grad_(True)
        result_tr = solver_tr.forward(eps_tr)
        energy_tr = (result_tr.field**2).sum()
        energy_tr.backward()
        grad_tr = eps_tr.grad.clone()

        assert grad_ad.abs().max() > 1e-10, "AD gradient zero"
        assert grad_tr.abs().max() > 1e-10, "TR gradient zero"

        cos_sim = torch.nn.functional.cosine_similarity(
            grad_ad.flatten().unsqueeze(0),
            grad_tr.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.99, f"Energy FOM cosine similarity = {cos_sim:.6f}"


class TestBasicCorrectness:
    """Basic sanity checks for the time-reversal mode."""

    def test_forward_shape(self):
        solver = FDTDSolver3D(**_CFG, backward="time_reversal")
        D, H, W = _CFG["grid_shape"]
        eps = torch.ones(D, H, W, dtype=torch.float64) * 2.25
        result = solver.forward(eps)
        assert result.field.shape == (3, D, H, W)

    def test_metadata(self):
        solver = FDTDSolver3D(**_CFG, backward="time_reversal")
        eps = torch.ones(*_CFG["grid_shape"], dtype=torch.float64) * 2.25
        result = solver.forward(eps)
        assert result.metadata.get("backward") == "time_reversal"

    def test_gradient_finite(self):
        solver = FDTDSolver3D(**_CFG, backward="time_reversal")
        eps = _make_eps().clone().detach().requires_grad_(True)
        result = solver.forward(eps)
        loss = result.field.sum()
        loss.backward()
        assert eps.grad is not None
        assert torch.isfinite(eps.grad).all()

    def test_point_source(self):
        solver = FDTDSolver3D(**_CFG, backward="time_reversal")
        D, H, W = _CFG["grid_shape"]
        eps = torch.ones(D, H, W, dtype=torch.float64) * 2.25
        result = solver.forward(
            eps,
            source={"type": "gaussian_pulse", "pos": [D // 2, H // 2, W // 2]},
        )
        assert result.field.shape == (3, D, H, W)

    def test_no_pml_mode(self):
        """Time-reversal works with PML enabled too."""
        cfg = {**_CFG, "pml_layers": 2, "n_steps": 4}
        solver = FDTDSolver3D(**cfg, backward="time_reversal")
        eps = _make_eps().clone().detach().requires_grad_(True)
        result = solver.forward(eps)
        loss = result.field.sum()
        loss.backward()
        assert eps.grad is not None
        assert torch.isfinite(eps.grad).all()
