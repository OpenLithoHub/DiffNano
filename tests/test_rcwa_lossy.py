"""Tests for RCWA solver with lossy (complex-permittivity) materials.

Verifies that the solver preserves the imaginary part of permittivity
(absorption) and produces physically correct, differentiable results.
"""

import pytest
import torch

from diffnano.solvers import RCWASolver


def _make_grating(n_layers, n_grid, eps_mean, eps_imag=0.0, seed=42):
    torch.manual_seed(seed)
    x = torch.linspace(0, 4 * 3.14159, n_grid, dtype=torch.float64)
    modulation = 0.5 * torch.sin(x)
    eps_real = (eps_mean + modulation).unsqueeze(0).expand(n_layers, -1).clone()
    eps = torch.complex(eps_real, torch.full_like(eps_real, eps_imag))
    return eps.detach().requires_grad_(True)


class TestLossyMaterials:
    @pytest.fixture
    def solver(self):
        return RCWASolver(
            fourier_orders=3,
            wavelength_nm=532.0,
            period_nm=(400.0, 400.0),
            device="cpu",
        )

    def test_complex_permittivity_runs(self, solver):
        eps = torch.full((3, 50), -10.0 + 1j, dtype=torch.complex128)
        result = solver.forward(eps, wavelengths=[532.0])
        assert result.field.shape[0] == 1
        assert result.field.shape[1] == solver.n_fourier
        assert torch.isfinite(result.field).all()

    def test_absorption_changes_result(self, solver):
        eps_lossless = torch.full((3, 80), 2.25, dtype=torch.float64)
        result_lossless = solver.forward(eps_lossless, wavelengths=[532.0])

        eps_lossy = torch.full((3, 80), 2.25 + 1j, dtype=torch.complex128)
        result_lossy = solver.forward(eps_lossy, wavelengths=[532.0])

        assert not torch.allclose(result_lossless.field, result_lossy.field, atol=1e-8)

    def test_imaginary_part_not_discarded(self, solver):
        eps_real = torch.full((3, 80), 2.25, dtype=torch.complex128)
        eps_complex = torch.full((3, 80), 2.25 + 0.5j, dtype=torch.complex128)

        result_real = solver.forward(eps_real, wavelengths=[532.0])
        result_complex = solver.forward(eps_complex, wavelengths=[532.0])

        diff = (result_real.field - result_complex.field).abs().max()
        assert diff > 1e-10, f"Imaginary permittivity had no effect: diff={diff}"

    def test_metal_grating(self, solver):
        eps_metal = torch.full((5, 100), -10.0 + 1.0j, dtype=torch.complex128)
        result = solver.forward(eps_metal, wavelengths=[532.0])
        assert result.field.shape[0] == 1
        assert result.field.shape[1] == solver.n_fourier
        assert torch.isfinite(result.field).all()
        assert (result.field >= 0).all()

    def test_gradient_complex_no_nan(self, solver):
        eps = _make_grating(3, 50, 2.25, eps_imag=0.5)
        result = solver.forward(eps, wavelengths=[532.0])
        loss = result.field[:, solver.fourier_orders].sum()
        loss.backward()
        assert eps.grad is not None
        assert eps.grad.shape == eps.shape
        assert torch.isfinite(eps.grad).all(), "NaN in gradient for complex permittivity"

    def test_gradient_real_no_nan(self, solver):
        eps = _make_grating(3, 50, 2.25, eps_imag=0.0)
        result = solver.forward(eps, wavelengths=[532.0])
        loss = result.field[:, solver.fourier_orders].sum()
        loss.backward()
        assert eps.grad is not None
        assert eps.grad.shape == eps.shape
        assert torch.isfinite(eps.grad).all(), "NaN in gradient for real permittivity"

    def test_thin_film_transmission(self):
        solver = RCWASolver(
            fourier_orders=3,
            wavelength_nm=532.0,
            period_nm=(400.0, 400.0),
            device="cpu",
        )
        eps_lossless = torch.ones(2, 80, dtype=torch.float64) * 2.25
        t_lossless = solver.transmission(eps_lossless, wavelengths=[532.0])
        assert t_lossless.numel() == 1
        assert t_lossless.item() >= 0

        eps_lossy = torch.ones(2, 80, dtype=torch.complex128) * (2.25 + 0.5j)
        t_lossy = solver.transmission(eps_lossy, wavelengths=[532.0])
        assert t_lossy.numel() == 1
        assert t_lossy.item() >= 0

    def test_multi_wavelength_complex(self, solver):
        eps = torch.full((4, 60), 2.25 + 0.3j, dtype=torch.complex128)
        result = solver.forward(eps, wavelengths=[500.0, 532.0, 600.0])
        assert result.field.shape[0] == 3
        assert torch.isfinite(result.field).all()

    def test_source_config_complex(self, solver):
        eps = torch.full((4, 60), 2.0 + 0.1j, dtype=torch.complex128)
        result = solver.forward(
            eps,
            wavelengths=[532.0],
            source={"theta": 0.1, "polarization": "TE"},
        )
        assert result.field.shape[0] == 1
        assert torch.isfinite(result.field).all()
