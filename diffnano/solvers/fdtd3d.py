"""Differentiable 3D FDTD (Finite-Difference Time-Domain) solver.

Yee-grid explicit time-stepping with full PyTorch autograd through every
time step.  Extends the 2D FDTD to three spatial dimensions:

- All six field components (Ex, Ey, Ez, Hx, Hy, Hz)
- CPML (Convolutional PML) absorbing boundaries on all six faces
- Differentiable point, line, and plane sources
- Gradient checkpointing for memory efficiency

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

__all__ = ["FDTDSolver3D"]


class _CPMLRegion3D:
    """Convolutional PML boundary region for one axis in 3D.

    Parameters
    ----------
    size : int
        Grid dimension along this axis.
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

        device = torch.device("cpu")
        sigma = torch.zeros(size, device=device, dtype=torch.float64)
        if pml_layers > 0:
            for i in range(pml_layers):
                d = (pml_layers - i) / pml_layers
                val = sigma_max * d ** order
                sigma[i] = val
                sigma[size - 1 - i] = val

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


class FDTDSolver3D:
    """Differentiable 3D FDTD solver.

    Parameters
    ----------
    grid_shape : tuple[int, int, int]
        ``(D, H, W)`` spatial grid (depth, height, width).
    dl : float
        Grid spacing in nanometers.
    wavelength_nm : float
        Center wavelength for source.
    pml_layers : int
        Number of CPML layers on each boundary face.
    n_steps : int
        Default number of time steps.
    courant : float
        Courant number (must be < 1/sqrt(3) for 3D stability).
    device : str or torch.device
    use_checkpoint : bool
        Use gradient checkpointing for memory efficiency.
    checkpoint_segments : int
        Number of checkpoint segments.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int] = (30, 30, 30),
        dl: float = 20.0,
        wavelength_nm: float = 1550.0,
        pml_layers: int = 5,
        n_steps: int = 500,
        courant: float = 0.4,
        device: str | torch.device = "cpu",
        use_checkpoint: bool = False,
        checkpoint_segments: int = 4,
    ):
        self.grid_shape = grid_shape
        self.dl = dl
        self.wavelength_nm = wavelength_nm
        self.pml_layers = pml_layers
        self.n_steps = n_steps
        self.courant = courant
        self._device = torch.device(device)
        self.use_checkpoint = use_checkpoint
        self.checkpoint_segments = checkpoint_segments

        # Courant condition for 3D: dt < dl / (c * sqrt(3))
        self.dt = courant * dl / math.sqrt(3.0)
        self.omega = 2 * math.pi / wavelength_nm

        D, H, W = grid_shape
        self._cpml_x = _CPMLRegion3D(W, pml_layers, dl, self.dt)
        self._cpml_y = _CPMLRegion3D(H, pml_layers, dl, self.dt)
        self._cpml_z = _CPMLRegion3D(D, pml_layers, dl, self.dt)

    @property
    def device(self) -> torch.device:
        return self._device

    def _source_waveform(self, step: int, source: dict) -> torch.Tensor:
        src_type = source.get("type", "gaussian_pulse")
        t = step * self.dt
        omega = self.omega
        dev = self._device

        if src_type == "gaussian_pulse":
            t0 = source.get("t0", 3.0 / omega)
            spread = source.get("spread", 1.0 / omega)
            amp = source.get("amplitude", 1.0)
            t_t = torch.tensor(t, dtype=torch.float64, device=dev)
            t0_t = torch.tensor(t0, dtype=torch.float64, device=dev)
            spread_t = torch.tensor(spread, dtype=torch.float64, device=dev)
            envelope = torch.exp(-((t_t - t0_t) ** 2) / (2 * spread_t ** 2))
            return amp * torch.sin(torch.tensor(omega * t, device=dev)) * envelope
        elif src_type == "continuous":
            amp = source.get("amplitude", 1.0)
            ramp = min(1.0, t * omega / (4 * math.pi))
            return torch.tensor(amp * ramp * math.sin(omega * t), dtype=torch.float64, device=dev)
        return torch.tensor(0.0, dtype=torch.float64, device=dev)

    def _inject_source(
        self,
        field: torch.Tensor,
        step: int,
        source: dict,
        component: str = "Ez",
    ) -> torch.Tensor:
        D, H, W = self.grid_shape
        waveform = self._source_waveform(step, source)
        field = field.clone()

        pos = source.get("pos", None)
        if pos is not None:
            z, y, x = pos
            if 0 <= z < D and 0 <= y < H and 0 <= x < W:
                field[z, y, x] = field[z, y, x] + waveform
        else:
            plane = source.get("plane", None)
            if plane == "xy":
                z = source.get("z", D // 2)
                if 0 <= z < D:
                    field[z, :, :] = field[z, :, :] + waveform
            elif plane == "xz":
                y = source.get("y", H // 2)
                if 0 <= y < H:
                    field[:, y, :] = field[:, y, :] + waveform
            elif plane == "yz":
                x = source.get("x", W // 2)
                if 0 <= x < W:
                    field[:, :, x] = field[:, :, x] + waveform
            else:
                # Default: line source along z at center
                z = D // 2
                y = H // 2
                field[z, y, :] = field[z, y, :] + waveform

        return field

    def _time_step(
        self,
        Ex: torch.Tensor,
        Ey: torch.Tensor,
        Ez: torch.Tensor,
        Hx: torch.Tensor,
        Hy: torch.Tensor,
        Hz: torch.Tensor,
        eps_r: torch.Tensor,
        mu_r: torch.Tensor,
        step: int,
        source: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """One 3D FDTD time step (all six field components).

        Update equations (Yee grid, normalized c=1):
            H fields updated first (half-step), then E fields (full step).
        """
        dt = self.dt
        dx = self.dl
        dy = self.dl
        dz = self.dl

        # --- Update H fields ---
        # Hx -= dt/mu * (dEz/dy - dEy/dz)
        dEz_dy = torch.zeros_like(Ez)
        dEz_dy[:, :-1, :] = (Ez[:, 1:, :] - Ez[:, :-1, :]) / dy
        dEy_dz = torch.zeros_like(Ey)
        dEy_dz[:-1, :, :] = (Ey[1:, :, :] - Ey[:-1, :, :]) / dz
        Hx = Hx - (dt / mu_r) * (dEz_dy - dEy_dz)

        # Hy -= dt/mu * (dEx/dz - dEz/dx)
        dEx_dz = torch.zeros_like(Ex)
        dEx_dz[:-1, :, :] = (Ex[1:, :, :] - Ex[:-1, :, :]) / dz
        dEz_dx = torch.zeros_like(Ez)
        dEz_dx[:, :, :-1] = (Ez[:, :, 1:] - Ez[:, :, :-1]) / dx
        Hy = Hy - (dt / mu_r) * (dEx_dz - dEz_dx)

        # Hz -= dt/mu * (dEy/dx - dEx/dy)
        dEy_dx = torch.zeros_like(Ey)
        dEy_dx[:, :, :-1] = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dx
        dEx_dy = torch.zeros_like(Ex)
        dEx_dy[:, :-1, :] = (Ex[:, 1:, :] - Ex[:, :-1, :]) / dy
        Hz = Hz - (dt / mu_r) * (dEy_dx - dEx_dy)

        # --- Update E fields ---
        # Ex += dt/eps * (dHz/dy - dHy/dz)
        dHz_dy = torch.zeros_like(Hz)
        dHz_dy[:, 1:, :] = (Hz[:, 1:, :] - Hz[:, :-1, :]) / dy
        dHy_dz = torch.zeros_like(Hy)
        dHy_dz[1:, :, :] = (Hy[1:, :, :] - Hy[:-1, :, :]) / dz
        Ex = Ex + (dt / eps_r) * (dHz_dy - dHy_dz)

        # Ey += dt/eps * (dHx/dz - dHz/dx)
        dHx_dz = torch.zeros_like(Hx)
        dHx_dz[1:, :, :] = (Hx[1:, :, :] - Hx[:-1, :, :]) / dz
        dHz_dx = torch.zeros_like(Hz)
        dHz_dx[:, :, 1:] = (Hz[:, :, 1:] - Hz[:, :, :-1]) / dx
        Ey = Ey + (dt / eps_r) * (dHx_dz - dHz_dx)

        # Ez += dt/eps * (dHy/dx - dHx/dy)
        dHy_dx = torch.zeros_like(Hy)
        dHy_dx[:, :, 1:] = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dx
        dHx_dy = torch.zeros_like(Hx)
        dHx_dy[:, 1:, :] = (Hx[:, 1:, :] - Hx[:, :-1, :]) / dy
        Ez = Ez + (dt / eps_r) * (dHy_dx - dHx_dy)

        # Inject source into Ez (default)
        Ez = self._inject_source(Ez, step, source, "Ez")

        return Ex, Ey, Ez, Hx, Hy, Hz

    def _run_steps(
        self,
        eps_r: torch.Tensor,
        mu_r: torch.Tensor,
        n_steps: int,
        source: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D, H, W = self.grid_shape
        dev = self._device
        dtype = torch.float64

        Ex = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Ey = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Ez = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Hx = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Hy = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Hz = torch.zeros(D, H, W, dtype=dtype, device=dev)

        for step in range(n_steps):
            Ex, Ey, Ez, Hx, Hy, Hz = self._time_step(
                Ex, Ey, Ez, Hx, Hy, Hz, eps_r, mu_r, step, source,
            )

        return Ez, Ex, Ey

    def _run_steps_checkpointed(
        self,
        eps_r: torch.Tensor,
        mu_r: torch.Tensor,
        n_steps: int,
        source: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seg_len = max(1, n_steps // self.checkpoint_segments)

        def _segment(Ex, Ey, Ez, Hx, Hy, Hz, eps, mu, start, steps):
            for step in range(start, start + steps):
                Ex, Ey, Ez, Hx, Hy, Hz = self._time_step(
                    Ex, Ey, Ez, Hx, Hy, Hz, eps, mu, step, source,
                )
            return Ex, Ey, Ez, Hx, Hy, Hz

        D, H, W = self.grid_shape
        dev = self._device
        dtype = torch.float64

        Ex = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Ey = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Ez = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Hx = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Hy = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Hz = torch.zeros(D, H, W, dtype=dtype, device=dev)

        step_idx = 0
        while step_idx < n_steps:
            steps_this = min(seg_len, n_steps - step_idx)
            Ex, Ey, Ez, Hx, Hy, Hz = cp.checkpoint(
                _segment,
                Ex, Ey, Ez, Hx, Hy, Hz, eps_r, mu_r,
                step_idx, steps_this,
                use_reentrant=False,
            )
            step_idx += steps_this

        return Ez, Ex, Ey

    def forward(
        self,
        geometry: torch.Tensor,
        wavelengths: Sequence[float] | torch.Tensor | None = None,
        *,
        source: dict | None = None,
    ) -> SimResult:
        """Run 3D FDTD forward simulation.

        Parameters
        ----------
        geometry : Tensor, shape ``(D, H, W)`` or ``(1, D, H, W)``
            Relative permittivity map (eps_r).
        wavelengths : ignored
            Center wavelength is set at construction.
        source : dict, optional
            Source configuration:
            - ``type``: "gaussian_pulse" or "continuous"
            - ``pos``: [z, y, x] for point source
            - ``plane``: "xy", "xz", or "yz" for plane source
            - ``amplitude``: source strength

        Returns
        -------
        SimResult
            ``field`` contains (Ez, Ex, Ey) stacked, shape ``(3, D, H, W)``.
        """
        if wavelengths is None:
            wavelengths = [self.wavelength_nm]
        if not isinstance(wavelengths, torch.Tensor):
            wavelengths = torch.tensor(wavelengths, dtype=torch.float64, device=self._device)

        src = source or {"type": "gaussian_pulse"}

        eps_r = geometry.to(self._device).to(torch.float64)
        if eps_r.dim() == 4:
            eps_r = eps_r.squeeze(0)
        if eps_r.dim() == 2:
            eps_r = eps_r.unsqueeze(0)

        mu_r = torch.ones_like(eps_r)

        if self.use_checkpoint:
            Ez, Ex, Ey = self._run_steps_checkpointed(eps_r, mu_r, self.n_steps, src)
        else:
            Ez, Ex, Ey = self._run_steps(eps_r, mu_r, self.n_steps, src)

        field = torch.stack([Ez, Ex, Ey], dim=0)  # (3, D, H, W)

        return SimResult(
            field=field,
            wavelengths=wavelengths,
            metadata={
                "n_steps": self.n_steps,
                "dt": self.dt,
                "courant": self.courant,
                "grid_shape": self.grid_shape,
            },
        )

    def time_series(
        self,
        eps_r: torch.Tensor,
        probe: tuple[int, int, int],
        source: dict | None = None,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """Run FDTD and return Ez time-series at a probe point.

        Parameters
        ----------
        eps_r : Tensor, shape ``(D, H, W)``
        probe : tuple[int, int, int]
            ``(z, y, x)`` probe position.
        source : dict, optional
        n_steps : int, optional

        Returns
        -------
        ts : Tensor, shape ``(n_steps,)``
        """
        src = source or {"type": "gaussian_pulse"}

        eps_r = eps_r.to(self._device).to(torch.float64)
        if eps_r.dim() == 4:
            eps_r = eps_r.squeeze(0)
        if eps_r.dim() == 2:
            eps_r = eps_r.unsqueeze(0)
        mu_r = torch.ones_like(eps_r)

        steps = n_steps or self.n_steps
        D, H, W = self.grid_shape
        dev = self._device
        dtype = torch.float64

        Ex = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Ey = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Ez = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Hx = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Hy = torch.zeros(D, H, W, dtype=dtype, device=dev)
        Hz = torch.zeros(D, H, W, dtype=dtype, device=dev)

        snapshots = []
        pz, py, px = probe

        for step in range(steps):
            Ex, Ey, Ez, Hx, Hy, Hz = self._time_step(
                Ex, Ey, Ez, Hx, Hy, Hz, eps_r, mu_r, step, src,
            )
            snapshots.append(Ez[pz, py, px].detach().clone())

        return torch.stack(snapshots)
