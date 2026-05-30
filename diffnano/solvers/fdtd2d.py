"""Differentiable 2D FDTD (Finite-Difference Time-Domain) solver.

Yee-grid explicit time-stepping with full PyTorch autograd through every
time step.  Supports:

- TM mode (Ez, Hx, Hy) and TE mode (Hz, Ex, Ey)
- CPML (Convolutional PML) absorbing boundaries
- Differentiable Gaussian pulse source
- Gradient checkpointing for memory efficiency on long runs

References
----------
- Taflove & Hagness (2005), Computational Electrodynamics: The FDTD Method
- Mahlau et al. (2024), FDTDX: arXiv:2412.12360 (time-reversibility reference)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.utils.checkpoint as cp

from diffnano.solvers._result import SimResult

__all__ = ["FDTDSolver2D"]


class _CPMLRegion:
    """Convolutional PML boundary region.

    Implements the CPML formulation from Roden & Gedney (2000) with
    recursive-convolution updating of auxiliary fields.

    Parameters
    ----------
    size : int
        Grid dimension.
    pml_layers : int
        Number of PML layers per side.
    dl : float
        Grid spacing.
    dt : float
        Time step.
    sigma_max : float
        Maximum conductivity.
    order : int
        Grading order (polynomial).
    """

    def __init__(
        self,
        size: int,
        pml_layers: int,
        dl: float,
        dt: float,
        sigma_max: float = 0.8,
        order: int = 3,
    ):
        self.size = size
        self.pml_layers = pml_layers
        self.dl = dl
        self.dt = dt
        self.order = order

        # Build conductivity profile (quadratic grading)
        device = torch.device("cpu")
        sigma = torch.zeros(size, device=device, dtype=torch.float64)
        if pml_layers > 0:
            for i in range(pml_layers):
                d = (pml_layers - i) / pml_layers
                val = sigma_max * d**order
                sigma[i] = val
                sigma[size - 1 - i] = val

        # CPML coefficients
        sigma_scaled = sigma * dl / dt
        self.kappa = 1.0 + (sigma_max - 0.8) * sigma / (sigma.max() + 1e-30)
        self.kappa = self.kappa.clamp(min=1.0)

        alpha = 0.02 * sigma_scaled
        self.b = torch.exp(-(sigma_scaled + alpha) * dt)
        self.c = torch.where(
            sigma_scaled + alpha > 1e-30,
            sigma_scaled * (self.b - 1.0) / (sigma_scaled + alpha + 1e-30),
            torch.zeros_like(sigma_scaled),
        )


class FDTDSolver2D:
    """Differentiable 2D FDTD solver.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        ``(H, W)`` spatial grid.
    dl : float
        Grid spacing in nanometers.
    wavelength_nm : float
        Center wavelength for source.
    polarization : str
        "TM" (Ez, Hx, Hy) or "TE" (Hz, Ex, Ey).
    pml_layers : int
        Number of CPML layers on each boundary.
    n_steps : int
        Default number of time steps.
    courant : float
        Courant number (must be < 1/sqrt(2) for 2D stability).
    device : str or torch.device
    use_checkpoint : bool
        Use gradient checkpointing for memory efficiency.
    checkpoint_segments : int
        Number of checkpoint segments (more = less memory, more compute).
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (100, 100),
        dl: float = 20.0,
        wavelength_nm: float = 1550.0,
        polarization: str = "TM",
        pml_layers: int = 10,
        n_steps: int = 1000,
        courant: float = 0.5,
        device: str | torch.device = "cpu",
        use_checkpoint: bool = False,
        checkpoint_segments: int = 4,
    ):
        self.grid_shape = grid_shape
        self.dl = dl
        self.wavelength_nm = wavelength_nm
        self.polarization = polarization.upper()
        self.pml_layers = pml_layers
        self.n_steps = n_steps
        self.courant = courant
        self._device = torch.device(device)
        self.use_checkpoint = use_checkpoint
        self.checkpoint_segments = checkpoint_segments

        # Speed of light normalization: c = 1
        self.dt = courant * dl / math.sqrt(2.0)
        self.omega = 2 * math.pi / wavelength_nm

        # CPML regions (coefficients built on CPU, cached to device lazily)
        H, W = grid_shape
        self._cpml_x = _CPMLRegion(W, pml_layers, dl, self.dt)
        self._cpml_y = _CPMLRegion(H, pml_layers, dl, self.dt)
        self._cpml_device_cached = False

    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------
    # Lazy CPML device caching
    # ------------------------------------------------------------------

    def _ensure_cpml_on_device(self) -> None:
        """Move CPML damping coefficients to the target device once."""
        if self._cpml_device_cached:
            return
        self._cached_bx = self._cpml_x.b.to(self._device).to(torch.float64)
        self._cached_cx = self._cpml_x.c.to(self._device).to(torch.float64)
        self._cached_by = self._cpml_y.b.to(self._device).to(torch.float64)
        self._cached_cy = self._cpml_y.c.to(self._device).to(torch.float64)
        self._cpml_device_cached = True

    def _cpml_damping(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cached CPML damping coefficients for x and y boundaries."""
        self._ensure_cpml_on_device()
        return self._cached_bx, self._cached_cx, self._cached_by, self._cached_cy

    # ------------------------------------------------------------------
    # Source helpers
    # ------------------------------------------------------------------

    def _source_waveform(
        self,
        step: int,
        source: dict,
    ) -> torch.Tensor:
        """Compute source amplitude at a given time step."""
        src_type = source.get("type", "gaussian_pulse")
        t = step * self.dt
        omega = self.omega
        device = self._device

        if src_type == "gaussian_pulse":
            t0 = source.get("t0", 3.0 / omega)
            spread = source.get("spread", 1.0 / omega)
            amp = source.get("amplitude", 1.0)
            t_t = torch.tensor(t, dtype=torch.float64, device=device)
            t0_t = torch.tensor(t0, dtype=torch.float64, device=device)
            spread_t = torch.tensor(spread, dtype=torch.float64, device=device)
            sin_val = torch.sin(torch.tensor(omega * t, device=device))
            envelope = torch.exp(-((t_t - t0_t) ** 2) / (2 * spread_t**2))
            return amp * sin_val * envelope
        elif src_type == "continuous":
            amp = source.get("amplitude", 1.0)
            ramp = min(1.0, t * omega / (4 * math.pi))
            val = amp * ramp * math.sin(omega * t)
            return torch.tensor(val, dtype=torch.float64, device=device)
        else:
            return torch.tensor(0.0, dtype=torch.float64, device=device)

    def _build_source_mask(self, source: dict) -> torch.Tensor:
        """Pre-build source injection mask for the current source config.

        Returns a (H, W) tensor that is 1 at source locations and 0 elsewhere.
        Used to inject sources via ``field + mask * waveform`` instead of
        ``clone() + scatter``, avoiding per-step memory allocation.
        """
        H, W = self.grid_shape
        mask = torch.zeros(H, W, dtype=torch.float64, device=self._device)

        pos = source.get("pos", None)
        if pos is not None:
            y, x = pos
            if 0 <= y < H and 0 <= x < W:
                mask[y, x] = 1.0
        else:
            row = source.get("row", H // 2)
            if 0 <= row < H:
                mask[row, :] = 1.0

        return mask

    # ------------------------------------------------------------------
    # Time-stepping kernels (no source injection — handled in run loop)
    # ------------------------------------------------------------------

    def _time_step_tm(
        self,
        Ez: torch.Tensor,
        Hx: torch.Tensor,
        Hy: torch.Tensor,
        eps_r: torch.Tensor,
        mu_r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One FDTD time step for TM polarization (Ez, Hx, Hy) with CPML."""
        dt = self.dt
        dx = self.dl
        dy = self.dl

        b_x, c_x, b_y, c_y = self._cpml_damping()

        # Update H fields with CPML damping
        dEz_dy = torch.zeros_like(Ez)
        dEz_dy[:, :-1] = (Ez[:, 1:] - Ez[:, :-1]) / dy
        Hx = b_y.unsqueeze(0) * Hx - (dt / mu_r) * (c_y.unsqueeze(0) * dEz_dy + dEz_dy)

        dEz_dx = torch.zeros_like(Ez)
        dEz_dx[:-1, :] = (Ez[1:, :] - Ez[:-1, :]) / dx
        Hy = b_x.unsqueeze(0) * Hy + (dt / mu_r) * (c_x.unsqueeze(0) * dEz_dx + dEz_dx)

        # Update E field with CPML damping
        dHy_dx = torch.zeros_like(Hy)
        dHy_dx[1:, :] = (Hy[1:, :] - Hy[:-1, :]) / dx
        dHx_dy = torch.zeros_like(Hx)
        dHx_dy[:, 1:] = (Hx[:, 1:] - Hx[:, :-1]) / dy

        Ez = b_x.unsqueeze(0) * b_y.unsqueeze(1) * Ez + (dt / eps_r) * (
            c_x.unsqueeze(0) * dHy_dx + dHy_dx - c_y.unsqueeze(1) * dHx_dy - dHx_dy
        )

        return Ez, Hx, Hy

    def _time_step_te(
        self,
        Hz: torch.Tensor,
        Ex: torch.Tensor,
        Ey: torch.Tensor,
        eps_r: torch.Tensor,
        mu_r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One FDTD time step for TE polarization (Hz, Ex, Ey) with CPML."""
        dt = self.dt
        dx = self.dl
        dy = self.dl

        b_x, c_x, b_y, c_y = self._cpml_damping()

        # Update E fields with CPML
        dHz_dx = torch.zeros_like(Hz)
        dHz_dx[:, 1:] = (Hz[:, 1:] - Hz[:, :-1]) / dx
        Ex = b_x.unsqueeze(0) * Ex + (dt / eps_r) * (c_x.unsqueeze(0) * dHz_dx + dHz_dx)

        dHz_dy = torch.zeros_like(Hz)
        dHz_dy[1:, :] = (Hz[1:, :] - Hz[:-1, :]) / dy
        Ey = b_y.unsqueeze(0) * Ey - (dt / eps_r) * (c_y.unsqueeze(0) * dHz_dy + dHz_dy)

        # Update H field with CPML
        dEy_dx = torch.zeros_like(Ey)
        dEy_dx[:-1, :] = (Ey[1:, :] - Ey[:-1, :]) / dx
        dEx_dy = torch.zeros_like(Ex)
        dEx_dy[:, :-1] = (Ex[:, 1:] - Ex[:, :-1]) / dy

        Hz = b_x.unsqueeze(0) * b_y.unsqueeze(1) * Hz + (dt / mu_r) * (
            c_x.unsqueeze(0) * dEy_dx + dEy_dx - c_y.unsqueeze(1) * dEx_dy - dEx_dy
        )

        return Hz, Ex, Ey

    # ------------------------------------------------------------------
    # Run loops (source injection inlined with pre-built mask)
    # ------------------------------------------------------------------

    def _run_steps(
        self,
        eps_r: torch.Tensor,
        mu_r: torch.Tensor,
        n_steps: int,
        source: dict,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run FDTD for n_steps, returning final E-field and time-series snapshots."""
        H, W = self.grid_shape
        device = self._device
        dtype = torch.float64

        if self.polarization == "TM":
            Ez = torch.zeros(H, W, dtype=dtype, device=device)
            Hx = torch.zeros(H, W, dtype=dtype, device=device)
            Hy = torch.zeros(H, W, dtype=dtype, device=device)

            snapshots = []
            probe_pos = source.get("probe", None)

            for step in range(n_steps):
                Ez, Hx, Hy = self._time_step_tm(Ez, Hx, Hy, eps_r, mu_r)
                waveform = self._source_waveform(step, source)
                Ez = Ez + source_mask * waveform

                if probe_pos is not None:
                    py, px = probe_pos
                    snapshots.append(Ez[py, px].detach().clone())

            return Ez, snapshots
        else:
            Hz = torch.zeros(H, W, dtype=dtype, device=device)
            Ex = torch.zeros(H, W, dtype=dtype, device=device)
            Ey = torch.zeros(H, W, dtype=dtype, device=device)

            snapshots = []
            probe_pos = source.get("probe", None)

            for step in range(n_steps):
                Hz, Ex, Ey = self._time_step_te(Hz, Ex, Ey, eps_r, mu_r)
                waveform = self._source_waveform(step, source)
                Hz = Hz + source_mask * waveform

                if probe_pos is not None:
                    py, px = probe_pos
                    snapshots.append(Hz[py, px].detach().clone())

            return Hz, snapshots

    def _run_steps_checkpointed(
        self,
        eps_r: torch.Tensor,
        mu_r: torch.Tensor,
        n_steps: int,
        source: dict,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run FDTD with gradient checkpointing."""
        seg_len = max(1, n_steps // self.checkpoint_segments)

        def _segment_forward(e_field, h_field_1, h_field_2, eps, mu, start_step, steps):
            if self.polarization == "TM":
                Ez, Hx, Hy = e_field, h_field_1, h_field_2
                for step in range(start_step, start_step + steps):
                    Ez, Hx, Hy = self._time_step_tm(Ez, Hx, Hy, eps, mu)
                    waveform = self._source_waveform(step, source)
                    Ez = Ez + source_mask * waveform
                return Ez, Hx, Hy
            else:
                Hz, Ex, Ey = e_field, h_field_1, h_field_2
                for step in range(start_step, start_step + steps):
                    Hz, Ex, Ey = self._time_step_te(Hz, Ex, Ey, eps, mu)
                    waveform = self._source_waveform(step, source)
                    Hz = Hz + source_mask * waveform
                return Hz, Ex, Ey

        H, W = self.grid_shape
        device = self._device
        dtype = torch.float64

        e_field = torch.zeros(H, W, dtype=dtype, device=device)
        h1 = torch.zeros(H, W, dtype=dtype, device=device)
        h2 = torch.zeros(H, W, dtype=dtype, device=device)

        step_idx = 0
        while step_idx < n_steps:
            steps_this = min(seg_len, n_steps - step_idx)
            e_field, h1, h2 = cp.checkpoint(
                _segment_forward,
                e_field,
                h1,
                h2,
                eps_r,
                mu_r,
                step_idx,
                steps_this,
                use_reentrant=False,
            )
            step_idx += steps_this

        return e_field, []

    def forward(
        self,
        geometry: torch.Tensor,
        wavelengths: Sequence[float] | torch.Tensor | None = None,
        *,
        source: dict | None = None,
    ) -> SimResult:
        """Run 2D FDTD forward simulation.

        Parameters
        ----------
        geometry : Tensor, shape ``(H, W)``
            Relative permittivity map (eps_r).
        wavelengths : sequence or Tensor, optional
            Center wavelength (used for metadata only).
        source : dict, optional
            Source configuration:
            - ``type``: "gaussian_pulse" or "continuous"
            - ``pos``: [y, x] for point source
            - ``row``: row index for line source
            - ``amplitude``: source strength
            - ``probe``: [y, x] for time-series probe

        Returns
        -------
        SimResult
            ``field`` contains the final E-field (TM) or H-field (TE),
            shape ``(1, H, W)``.
        """
        if wavelengths is None:
            wavelengths = [self.wavelength_nm]
        if not isinstance(wavelengths, torch.Tensor):
            wavelengths = torch.tensor(wavelengths, dtype=torch.float64, device=self._device)

        src = source or {"type": "gaussian_pulse"}
        H, W = self.grid_shape

        eps_r = geometry.to(self._device).to(torch.float64)
        if eps_r.dim() == 3:
            eps_r = eps_r.squeeze(0)

        mu_r = torch.ones_like(eps_r)

        source_mask = self._build_source_mask(src)

        if self.use_checkpoint:
            final_field, snapshots = self._run_steps_checkpointed(
                eps_r,
                mu_r,
                self.n_steps,
                src,
                source_mask,
            )
        else:
            final_field, snapshots = self._run_steps(
                eps_r,
                mu_r,
                self.n_steps,
                src,
                source_mask,
            )

        return SimResult(
            field=final_field.unsqueeze(0),
            wavelengths=wavelengths,
            metadata={
                "polarization": self.polarization,
                "n_steps": self.n_steps,
                "dt": self.dt,
                "courant": self.courant,
                "snapshots": snapshots,
            },
        )

    def time_series(
        self,
        eps_r: torch.Tensor,
        probe: tuple[int, int],
        source: dict | None = None,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """Run FDTD and return field time-series at a probe point.

        Parameters
        ----------
        eps_r : Tensor, shape ``(H, W)``
        probe : tuple[int, int]
            ``(y, x)`` probe position.
        source : dict, optional
        n_steps : int, optional
            Override number of time steps.

        Returns
        -------
        ts : Tensor, shape ``(n_steps,)``
            Field amplitude at the probe over time.
        """
        src = source or {"type": "gaussian_pulse"}
        src["probe"] = probe

        H, W = self.grid_shape
        eps_r = eps_r.to(self._device).to(torch.float64)
        if eps_r.dim() == 3:
            eps_r = eps_r.squeeze(0)
        mu_r = torch.ones_like(eps_r)

        source_mask = self._build_source_mask(src)
        steps = n_steps or self.n_steps
        _, snapshots = self._run_steps(eps_r, mu_r, steps, src, source_mask)

        if snapshots:
            return torch.stack(snapshots)
        return torch.zeros(steps, dtype=torch.float64, device=self._device)
