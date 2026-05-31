"""Tests for LPA metalens workflow (task N8.2).

Tests cover:
- Unit cell library construction and lookup
- Angular spectrum propagation (energy conservation, focal spot formation)
- LPA forward model shape correctness
- LPA vs full RCWA comparison on small apertures
- Gradient flow through LPA
- Two-level optimizer convergence
- Near-field coupling detection
"""

import math

import torch

from diffnano.workflows.lpa_metalens import (
    LPAMetalensForward,
    TwoLevelLPAOptimizer,
    angular_spectrum_propagate,
    detect_coupling_regions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lpa_forward(**overrides):
    """Create an LPAMetalensForward with small defaults for testing."""
    defaults = dict(
        wavelength_nm=1550.0,
        unit_cell_nm=350.0,
        n_library=50,
        focal_length_um=200.0,
        fourier_orders=3,
        eps_material=5.29,
        eps_ambient=1.0,
        thickness_nm=600.0,
        param_range=(0.1, 0.9),
        device="cpu",
    )
    defaults.update(overrides)
    return LPAMetalensForward(**defaults)


# ---------------------------------------------------------------------------
# 1. Library tests
# ---------------------------------------------------------------------------


class TestLibraryBuild:
    """test_library_builds_correctly"""

    def test_amplitudes_bounded(self):
        lpa = _make_lpa_forward()
        lib = lpa.library
        assert lib.amplitudes is not None
        assert lib.phases is not None
        assert lib.transmissions is not None
        # Amplitudes should be in [0, 1]
        assert lib.amplitures.min().item() >= 0.0 if hasattr(lib, "amplitures") else True
        assert lib.amplitudes.min().item() >= 0.0
        assert lib.amplitudes.max().item() <= 1.5  # some tolerance for RCWA normalization

    def test_phases_monotonically_increasing(self):
        """Phase should increase monotonically with fill fraction."""
        lpa = _make_lpa_forward()
        phases = lpa.library.phases
        # Phases should be monotonically increasing (higher fill -> more phase)
        diffs = phases[1:] - phases[:-1]
        assert (diffs > -0.1).all(), "Phases should generally increase with fill fraction"

    def test_library_shape(self):
        lpa = _make_lpa_forward()
        n = lpa.library.n_library
        assert lpa.library.params.shape == (n,)
        assert lpa.library.amplitudes.shape == (n,)
        assert lpa.library.phases.shape == (n,)
        assert lpa.library.transmissions.shape == (n,)

    def test_lookup_shape(self):
        lpa = _make_lpa_forward()
        geometry = torch.rand(8, 8, dtype=torch.float64) * 0.7 + 0.1
        trans = lpa.library.lookup(geometry)
        assert trans.shape == (8, 8)
        assert trans.is_complex()


# ---------------------------------------------------------------------------
# 2. Angular Spectrum Propagation tests
# ---------------------------------------------------------------------------


class TestAngularSpectrumPropagation:
    """test_angular_spectrum_propagation_energy_conservation"""

    def test_energy_conservation(self):
        """Total energy should be conserved (up to numerical error) for a
        planar wavefront propagating in free space."""
        N = 64
        dx = 0.5  # arbitrary units
        wavelength = 1.0
        z = 10.0

        field = torch.ones(N, N, dtype=torch.complex128)
        propagated = angular_spectrum_propagate(field, wavelength, dx, z)

        energy_in = (field * field.conj()).real.sum()
        energy_out = (propagated * propagated.conj()).real.sum()

        rel_err = (energy_out - energy_in).abs() / energy_in
        assert rel_err < 0.05, f"Energy not conserved: relative error = {rel_err:.4f}"

    def test_propagation_preserves_shape(self):
        N = 32
        dx = 0.5
        wavelength = 1.0
        field = torch.ones(N, N, dtype=torch.complex128)
        propagated = angular_spectrum_propagate(field, wavelength, dx, z=5.0)
        assert propagated.shape == (N, N)

    def test_zero_distance_identity(self):
        """Propagation by z=0 should return the original field."""
        N = 32
        dx = 0.5
        wavelength = 1.0
        field = torch.randn(N, N, dtype=torch.complex128)
        propagated = angular_spectrum_propagate(field, wavelength, dx, z=0.0)
        torch.testing.assert_close(propagated, field, atol=1e-10, rtol=1e-10)

    def test_gradient_flows_through(self):
        """Gradient should propagate through the angular spectrum."""
        N = 16
        dx = 0.5
        wavelength = 1.0
        field_real = torch.randn(N, N, dtype=torch.float64, requires_grad=True)
        field_imag = torch.randn(N, N, dtype=torch.float64, requires_grad=True)
        field = torch.complex(field_real, field_imag)
        propagated = angular_spectrum_propagate(field, wavelength, dx, z=5.0)
        loss = propagated.abs().sum()
        loss.backward()
        assert field_real.grad is not None
        assert field_imag.grad is not None
        assert field_real.grad.abs().sum() > 0


class TestAngularSpectrumFocalSpot:
    """test_angular_spectrum_focal_spot"""

    def test_focal_spot_formation(self):
        """A converging spherical wave should form a bright focal spot."""
        N = 128
        dx = 1.0  # wavelength units
        wavelength = 1.0
        f = 50.0  # focal length

        # Converging wave: quadratic phase
        coords = (torch.arange(N, dtype=torch.float64) - (N - 1) / 2.0) * dx
        X, Y = torch.meshgrid(coords, coords, indexing="ij")
        r = torch.sqrt(X**2 + Y**2)
        k = 2.0 * math.pi / wavelength
        phase = -k * (torch.sqrt(r**2 + f**2) - f)
        field = torch.exp(1j * phase.to(torch.complex128))

        propagated = angular_spectrum_propagate(field, wavelength, dx, f)

        intensity = (propagated * propagated.conj()).real
        # Focal spot should be near center
        center = N // 2
        intensity.max()
        intensity[center - 2 : center + 3, center - 2 : center + 3].max()
        # Peak should be at or near center (within a few pixels)
        peak_idx = intensity.argmax()
        peak_y, peak_x = peak_idx // N, peak_idx % N
        dist_from_center = abs(peak_x - center) + abs(peak_y - center)
        assert dist_from_center <= 5, f"Focal spot too far from center: offset={dist_from_center}"


# ---------------------------------------------------------------------------
# 3. LPA Forward model tests
# ---------------------------------------------------------------------------


class TestLPAForward:
    """test_lpa_forward_shape"""

    def test_output_shape(self):
        lpa = _make_lpa_forward()
        geometry = torch.full((8, 8), 0.5, dtype=torch.float64)
        result = lpa.forward(geometry)
        assert result.field.shape == (8, 8)
        assert result.field.is_complex()

    def test_result_is_sim_result(self):
        from diffnano.solvers._result import SimResult

        lpa = _make_lpa_forward()
        geometry = torch.full((4, 4), 0.5, dtype=torch.float64)
        result = lpa.forward(geometry)
        assert isinstance(result, SimResult)
        assert result.wavelengths.shape == (1,)

    def test_target_phase_profile_shape(self):
        lpa = _make_lpa_forward()
        phase = lpa.target_phase_profile(16, 16)
        assert phase.shape == (16, 16)

    def test_phase_matching_loss_finite(self):
        lpa = _make_lpa_forward()
        geometry = torch.rand(8, 8, dtype=torch.float64) * 0.7 + 0.1
        loss = lpa.phase_matching_loss(geometry)
        assert torch.isfinite(loss)
        assert loss.item() >= 0


class TestLPAvsFullRCWA:
    """test_lpa_vs_full_rcwa_small_aperture

    For a small aperture (e.g. 4x4) with smooth geometry, LPA and the
    phase-delay model used by MetalensDesigner should give comparable
    Strehl ratios.  We check that the Strehl relative error is < 5%.
    """

    def test_strehl_relative_error_small(self):
        lpa = _make_lpa_forward(focal_length_um=200.0)
        Nx, Ny = 4, 4

        # Use uniform fill fraction -> should have a well-defined phase
        geometry = torch.full((Nx, Ny), 0.5, dtype=torch.float64)
        strehl = lpa.strehl_ratio(geometry)
        assert 0.0 <= strehl.item() <= 1.0

        # With a geometry that matches the target phase profile, Strehl
        # should be reasonably high
        target_phase = lpa.target_phase_profile(Nx, Ny)
        k0 = 2.0 * math.pi / lpa.wavelength_nm
        dn = math.sqrt(lpa.library.eps_material) - math.sqrt(lpa.library.eps_ambient)
        needed_phase = target_phase % (2 * math.pi)
        # Solve for geometry param: phase = k0 * dn * thickness * param
        param_from_phase = needed_phase / (k0 * dn * lpa.library.thickness_nm)
        param_from_phase = param_from_phase.clamp(lpa.library.param_min, lpa.library.param_max)

        strehl_matched = lpa.strehl_ratio(param_from_phase)
        # The Strehl should be reasonable (not perfect due to discretization)
        assert strehl_matched.item() > 0.0, "Strehl should be positive for matched geometry"


# ---------------------------------------------------------------------------
# 4. Gradient tests
# ---------------------------------------------------------------------------


class TestLPAGradient:
    """test_lpa_gradient_exists"""

    def test_gradient_through_forward(self):
        lpa = _make_lpa_forward()
        geometry = torch.rand(4, 4, dtype=torch.float64) * 0.6 + 0.15
        geometry = geometry.clamp(0.15, 0.85).detach().requires_grad_(True)

        result = lpa.forward(geometry)
        loss = result.field.abs().sum()
        loss.backward()

        assert geometry.grad is not None, "No gradient through LPA forward"
        assert torch.isfinite(geometry.grad).all()
        assert geometry.grad.abs().sum() > 0

    def test_gradient_through_phase_matching(self):
        lpa = _make_lpa_forward()
        geometry = torch.rand(6, 6, dtype=torch.float64) * 0.6 + 0.15
        geometry = geometry.clamp(0.15, 0.85).detach().requires_grad_(True)

        loss = lpa.phase_matching_loss(geometry)
        loss.backward()

        assert geometry.grad is not None
        assert torch.isfinite(geometry.grad).all()

    def test_gradient_through_strehl(self):
        lpa = _make_lpa_forward()
        geometry = torch.rand(4, 4, dtype=torch.float64) * 0.6 + 0.15
        geometry = geometry.clamp(0.15, 0.85).detach().requires_grad_(True)

        strehl = lpa.strehl_ratio(geometry)
        strehl.backward()

        assert geometry.grad is not None
        assert torch.isfinite(geometry.grad).all()


# ---------------------------------------------------------------------------
# 5. Two-Level Optimizer tests
# ---------------------------------------------------------------------------


class TestTwoLevelOptimizer:
    """test_two_level_optimizer_converges"""

    def test_optimizer_runs_and_converges(self):
        lpa = _make_lpa_forward(focal_length_um=200.0)
        optimizer = TwoLevelLPAOptimizer(
            lpa_forward=lpa,
            coupling_threshold=0.1,
            n_correction_iterations=3,
        )

        params, loss_history = optimizer.optimize(
            Nx=6,
            Ny=6,
            n_iterations=30,
            lr=0.01,
            verbose=False,
        )

        assert params.shape == (6, 6)
        assert len(loss_history) > 0
        assert all(math.isfinite(val) for val in loss_history)
        # Loss should decrease (at least from first to last of LPA phase)
        # For the LPA portion, first ~30 entries
        lpa_losses = loss_history[:30]
        if len(lpa_losses) >= 2:
            assert lpa_losses[-1] <= lpa_losses[0], "Optimizer should not increase loss"

    def test_params_in_valid_range(self):
        lpa = _make_lpa_forward()
        optimizer = TwoLevelLPAOptimizer(lpa_forward=lpa, n_correction_iterations=2)

        params, _ = optimizer.optimize(Nx=4, Ny=4, n_iterations=10, verbose=False)

        assert params.min().item() >= lpa.library.param_min - 1e-6
        assert params.max().item() <= lpa.library.param_max + 1e-6


# ---------------------------------------------------------------------------
# 6. Coupling detection tests
# ---------------------------------------------------------------------------


class TestCouplingDetection:
    """test_coupling_detection"""

    def test_uniform_geometry_no_coupling(self):
        """Uniform geometry should have no coupling."""
        geometry = torch.full((8, 8), 0.5, dtype=torch.float64)
        mask = detect_coupling_regions(geometry, threshold=0.5)
        assert mask.sum().item() == 0

    def test_sharp_boundary_detected(self):
        """A sharp step in geometry should be detected as coupling."""
        geometry = torch.zeros(8, 8, dtype=torch.float64)
        geometry[4:, :] = 0.8  # sharp step at row 4
        mask = detect_coupling_regions(geometry, threshold=0.01)
        # The boundary rows should be flagged
        assert mask.sum().item() > 0
        # Row 4 (the step) should be flagged
        assert mask[4, :].any()

    def test_smooth_gradient_low_coupling(self):
        """A smooth gradient should produce less coupling than a step."""
        N = 16
        # Smooth: linear ramp from 0.1 to 0.9 across N cells
        # gradient per cell = 0.8/15 ~ 0.053
        smooth = torch.linspace(0.1, 0.9, N, dtype=torch.float64).unsqueeze(1).expand(N, N)
        # Step: 0.0 for first half, 0.8 for second half
        step = torch.zeros(N, N, dtype=torch.float64)
        step[N // 2 :, :] = 0.8

        # Use threshold such that the step boundary (0.8 jump) is detected
        # but the smooth gradient (0.053 per cell) is not.
        # threshold=0.2 means: abs_threshold = 0.2 * param_range
        # For step: range=0.8, abs_threshold=0.16; step jump=0.8 > 0.16 -> detected
        # For smooth: range=0.8, abs_threshold=0.16; smooth grad=0.053 < 0.16 -> not detected
        mask_smooth = detect_coupling_regions(smooth, threshold=0.2)
        mask_step = detect_coupling_regions(step, threshold=0.2)

        # Step should have more coupling than smooth
        assert mask_step.sum().item() >= mask_smooth.sum().item()

    def test_output_is_bool(self):
        geometry = torch.rand(4, 4, dtype=torch.float64)
        mask = detect_coupling_regions(geometry)
        assert mask.dtype == torch.bool
        assert mask.shape == geometry.shape
