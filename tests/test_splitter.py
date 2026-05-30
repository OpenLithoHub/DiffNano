"""Regression tests for the beam splitter workflow."""

import pytest
import torch

from diffnano.workflows.splitter import SplitterDesigner


@pytest.fixture
def designer():
    return SplitterDesigner(
        wavelength_nm=1550.0,
        period_nm=2200.0,
        n_fourier_orders=3,
        n_grid=32,
        eps_low=1.0,
        eps_high=12.0,
        thickness_nm=500.0,
        n_layers=3,
        device="cpu",
    )


class TestSplitterTransmissionShape:
    def test_efficiency_scalar(self, designer):
        density = torch.rand(designer.n_grid, dtype=torch.float64)
        eff = designer.transmission_efficiency(density)
        assert eff.numel() == 1

    def test_efficiency_positive(self, designer):
        density = torch.rand(designer.n_grid, dtype=torch.float64)
        eff = designer.transmission_efficiency(density)
        assert eff.item() >= 0.0

    def test_s_params_efficiencies_shape(self, designer):
        density = torch.rand(designer.n_grid, dtype=torch.float64)
        s = designer.s_parameters(density)
        n_fourier_total = 2 * designer.n_fourier_orders + 1
        assert s["efficiencies"].shape == (n_fourier_total,)


class TestSplitterSParameters:
    def test_efficiencies_positive(self, designer):
        density = torch.rand(designer.n_grid, dtype=torch.float64)
        s = designer.s_parameters(density)
        assert (s["efficiencies"] >= 0.0).all()

    def test_efficiencies_sum_le_one(self, designer):
        density = torch.rand(designer.n_grid, dtype=torch.float64)
        s = designer.s_parameters(density)
        assert s["efficiencies"].sum().item() <= 1.0 + 1e-6

    def test_insertion_loss_non_negative(self, designer):
        density = torch.rand(designer.n_grid, dtype=torch.float64)
        s = designer.s_parameters(density)
        assert s["insertion_loss"].item() >= -1e-6

    def test_splitting_ratio_bounded(self, designer):
        density = torch.rand(designer.n_grid, dtype=torch.float64)
        s = designer.s_parameters(density)
        assert 0.0 <= s["splitting_ratio"].item() <= 1.0 + 1e-6


class TestSplitterDifferentiable:
    def test_gradient_through_efficiency(self, designer):
        density = torch.rand(designer.n_grid, dtype=torch.float64, requires_grad=True)
        eff = designer.transmission_efficiency(density)
        eff.backward()
        assert density.grad is not None
        assert torch.isfinite(density.grad).all()

    def test_gradient_through_s_params(self, designer):
        density = torch.rand(designer.n_grid, dtype=torch.float64, requires_grad=True)
        s = designer.s_parameters(density)
        loss = s["T_plus1"] + s["T_minus1"]
        loss.backward()
        assert density.grad is not None
        assert torch.isfinite(density.grad).all()

    def test_gradient_through_objective(self, designer):
        density = torch.rand(designer.n_grid, dtype=torch.float64, requires_grad=True)
        loss = designer.objective(density)
        loss.backward()
        assert density.grad is not None


class TestSplitter5050Design:
    def test_symmetric_grating_balanced(self, designer):
        """A symmetric grating (centred fill) should produce roughly equal +/-1 power."""
        n = designer.n_grid
        density = torch.zeros(n, dtype=torch.float64)
        center = n // 2
        half_fill = n // 4
        density[center - half_fill : center + half_fill] = 1.0

        s = designer.s_parameters(density)

        T_p = s["T_plus1"].item()
        T_m = s["T_minus1"].item()
        total_target = T_p + T_m

        # Both orders should receive non-trivial power
        assert total_target > 0.01, (
            f"Symmetric grating should couple some power to +/- 1 orders, got {total_target:.4f}"
        )

        # Ratio should be close to 1.0 for symmetric structure
        ratio = s["splitting_ratio"].item()
        assert ratio > 0.5, (
            f"Symmetric grating splitting ratio should be > 0.5, got {ratio:.4f}"
        )
