"""Validation tests for the metalens + litho DFM co-optimization flagship case.

Runs a short version (10 steps) of both coupled and decoupled optimization,
verifying finite losses, gradient flow, and that the coupled approach includes
the lithography model in its loss.
"""

import torch
import pytest

from diffnano.workflows.dfm_metalens import DFMMetalensDesigner
from diffnano.design.constraints_shared import combined_fabrication_penalty


@pytest.fixture
def designer():
    return DFMMetalensDesigner(
        wavelength_nm=940.0,
        numerical_aperture=0.3,
        diameter_um=2.0,
        pixel_size_nm=100.0,
        n_material=2.0,
        n_ambient=1.0,
        fourier_orders=3,
        litho_wavelength_nm=193.0,
        litho_na=1.35,
        device="cpu",
    )


class TestFlagshipMetalensCoupled:
    """Coupled (co-design) optimization: optical + litho + fab."""

    def test_coupled_produces_finite_losses(self, designer):
        density, history, breakdown = designer.optimize(
            n_steps=10, lr=1e-2, lambda_optical=1.0, lambda_litho=0.1,
            lambda_fab=0.01, verbose=False,
        )
        assert len(history) == 10
        for val in history:
            assert not torch.isnan(torch.tensor(val))
            assert torch.isfinite(torch.tensor(val))

    def test_coupled_has_litho_component(self, designer):
        _, _, breakdown = designer.optimize(
            n_steps=5, lr=1e-2, verbose=False,
        )
        litho_vals = [b["litho"] for b in breakdown]
        assert all(v > 0 for v in litho_vals), "Litho loss should be positive (model is active)"

    def test_coupled_gradient_flows(self, designer):
        density = torch.rand(*designer.grid_shape, dtype=torch.float64, requires_grad=True)
        total, _ = designer.total_loss(density, lambda_optical=1.0, lambda_litho=0.1,
                                        lambda_fab=0.01, beta=10.0)
        total.backward()
        assert density.grad is not None
        assert not torch.isnan(density.grad).any()
        assert (density.grad.abs() > 0).any(), "Gradients should be nonzero"


class TestFlagshipMetalensDecoupled:
    """Decoupled baseline: optical-only optimization with post-hoc litho eval."""

    def test_decoupled_produces_finite_losses(self, designer):
        density, history = designer.decoupled_baseline(
            n_steps=10, lr=1e-2, verbose=False,
        )
        assert len(history) == 10
        for val in history:
            assert not torch.isnan(torch.tensor(val))
            assert torch.isfinite(torch.tensor(val))

    def test_decoupled_result_has_finite_litho_posthoc(self, designer):
        density, _ = designer.decoupled_baseline(
            n_steps=10, lr=1e-2, verbose=False,
        )
        mask = designer.density_param(density, beta=10.0)
        litho = designer.litho_model.forward(mask)
        epe = litho["epe"]
        assert epe.numel() == 1
        assert not torch.isnan(epe)
        assert torch.isfinite(epe)
        assert epe.item() >= 0


class TestFlagshipMetalensComparison:
    """Head-to-head comparisons ensuring both methods produce valid outputs."""

    def test_both_methods_produce_valid_optical_loss(self, designer):
        d_c, _, _ = designer.optimize(n_steps=10, verbose=False)
        d_d, _ = designer.decoupled_baseline(n_steps=10, verbose=False)

        mask_c = designer.density_param(d_c, beta=10.0)
        mask_d = designer.density_param(d_d, beta=10.0)

        litho_c = designer.litho_model.forward(mask_c)
        litho_d = designer.litho_model.forward(mask_d)

        opt_c = designer._optical_loss(litho_c["printed_contour"])
        opt_d = designer._optical_loss(litho_d["printed_contour"])

        for label, v in [("coupled", opt_c), ("decoupled", opt_d)]:
            assert torch.isfinite(v), f"{label} optical loss is not finite: {v}"
            assert v.item() >= 0, f"{label} optical loss is negative: {v}"

    def test_coupled_litho_epe_evaluated(self, designer):
        d_c, _, _ = designer.optimize(n_steps=10, verbose=False)
        mask_c = designer.density_param(d_c, beta=10.0)
        litho_c = designer.litho_model.forward(mask_c)
        epe_c = litho_c["epe"]
        assert torch.isfinite(epe_c)
        assert epe_c.item() >= 0
