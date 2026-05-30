"""Photonic crystal inverse design workflow.

Provides:
- Band structure computation via plane-wave expansion (differentiable)
- Topology optimization to maximize bandgap / midgap ratio
- Validation target: reproduce Jensen & Sigmund (2004) bandgap maximization

References
----------
- Joannopoulos et al. (2008), Photonic Crystals: Molding the Flow of Light
- Jensen & Sigmund (2004), Topology optimization of photonic crystal structures
"""

from __future__ import annotations

import math

import torch

__all__ = ["PhCDesigner"]


def _reciprocal_lattice_vectors(
    lattice: str,
    a: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute reciprocal lattice vectors for 2D lattice.

    Parameters
    ----------
    lattice : str
        "square" or "hexagonal" (triangular).
    a : float
        Lattice constant.

    Returns
    -------
    b1, b2 : Tensor, shape ``(2,)``
        Reciprocal lattice vectors.
    """
    if lattice == "square":
        b1 = torch.tensor([2 * math.pi / a, 0.0], dtype=torch.float64)
        b2 = torch.tensor([0.0, 2 * math.pi / a], dtype=torch.float64)
    elif lattice in ("hexagonal", "triangular"):
        b1 = torch.tensor([2 * math.pi / a, -2 * math.pi / (a * math.sqrt(3))], dtype=torch.float64)
        b2 = torch.tensor([0.0, 4 * math.pi / (a * math.sqrt(3))], dtype=torch.float64)
    else:
        raise ValueError(f"Unknown lattice type: {lattice!r}")

    return b1, b2


def _generate_g_points(
    n_max: int,
    b1: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Generate reciprocal lattice points G = n1*b1 + n2*b2.

    Parameters
    ----------
    n_max : int
        Max index |n_i| <= n_max.
    b1, b2 : Tensor, shape ``(2,)``
        Reciprocal lattice vectors.

    Returns
    -------
    G : Tensor, shape ``(N_G, 2)``
    """
    points = []
    for n1 in range(-n_max, n_max + 1):
        for n2 in range(-n_max, n_max + 1):
            G = n1 * b1 + n2 * b2
            points.append(G)
    return torch.stack(points)


def _band_structure_pwe(
    density: torch.Tensor,
    lattice: str,
    a: float,
    n_g: int,
    n_air: float,
    n_material: float,
    k_points: torch.Tensor,
    polarization: str,
) -> torch.Tensor:
    """Compute photonic crystal band structure via plane-wave expansion.

    For TM polarization (Ez):
        Sum_G' [ |k+G|^2 * eta(G-G') ] E_G' = (omega/c)^2 E_G

    where eta is the inverse Fourier transform of 1/eps(r).

    Parameters
    ----------
    density : Tensor, shape ``(H, W)``
        Density field (0 = air, 1 = material).
    lattice : str
        "square" or "hexagonal".
    a : float
        Lattice constant in nm.
    n_g : int
        Plane-wave cutoff (|n_i| <= n_g).
    n_air : float
        Refractive index of air.
    n_material : float
        Refractive index of material.
    k_points : Tensor, shape ``(N_k, 2)``
        Wavevectors in the Brillouin zone.
    polarization : str
        "TM" (Ez) or "TE" (Hz).

    Returns
    -------
    bands : Tensor, shape ``(N_k, n_bands)``
        Eigenfrequencies (omega * a / (2*pi*c)), sorted ascending.
    """
    device = density.device
    H, W = density.shape

    # Permittivity from density
    eps_r = n_air**2 + (n_material**2 - n_air**2) * density

    # Reciprocal lattice vectors
    b1, b2 = _reciprocal_lattice_vectors(lattice, a)

    # Generate G points
    G_points = _generate_g_points(n_g, b1, b2)
    N_G = G_points.shape[0]

    # Compute Fourier coefficients of 1/eps(r) for TM or eps(r) for TE
    # eta(q) = (1/A) integral exp(-i q.r) * f(r) dr
    # where f = 1/eps for TM, f = eps for TE
    A = H * W  # unit cell area in pixel units

    if polarization == "TM":
        field = 1.0 / eps_r.clamp(min=0.1)
    else:
        field = eps_r

    # FFT to get Fourier coefficients
    field_fft = torch.fft.fft2(field) / A

    # Build eta matrix: eta(G, G') = eta(G - G')

    # Map G_diff to FFT indices
    # G = n1*b1 + n2*b2, so G_diff = (n1_i-n1_j)*b1 + (n2_i-n2_j)*b2
    # We need to find the corresponding FFT coefficient index
    n_indices = []
    for n1 in range(-n_g, n_g + 1):
        for n2 in range(-n_g, n_g + 1):
            n_indices.append((n1, n2))

    # Build eta matrix using FFT coefficients
    eta = torch.zeros(N_G, N_G, dtype=torch.complex128, device=device)
    for i in range(N_G):
        for j in range(N_G):
            dn1 = n_indices[i][0] - n_indices[j][0]
            dn2 = n_indices[i][1] - n_indices[j][1]
            # Map to FFT index using fftshift convention
            # FFT output has positive frequencies first, then negative
            fft_idx_h = (dn1 + H // 2) % H
            fft_idx_w = (dn2 + W // 2) % W
            eta[i, j] = field_fft[fft_idx_h, fft_idx_w]

    # Compute bands at each k-point
    all_bands = []
    n_bands = min(6, N_G)

    for ki in range(k_points.shape[0]):
        k = k_points[ki]

        kG = k.unsqueeze(0) + G_points  # (N_G, 2)
        kG_sq = (kG**2).sum(dim=-1)  # (N_G,)

        if polarization == "TM":
            # TM: M_{GG'} = |k+G|^2 * eta(G-G')
            kG_sq_diag = torch.diag(kG_sq)
            M = kG_sq_diag.to(torch.complex128) @ eta
        else:
            # TE: M_{GG'} = (k+G)_x * eta(G-G') * (k+G')_x
            #              + (k+G)_y * eta(G-G') * (k+G')_y
            # Full vector coupling for TE polarization
            kGx = kG[:, 0]  # (N_G,)
            kGy = kG[:, 1]
            kGx_c = kGx.to(torch.complex128)
            kGy_c = kGy.to(torch.complex128)
            M_x = kGx_c.unsqueeze(1) * eta * kGx_c.unsqueeze(0)
            M_y = kGy_c.unsqueeze(1) * eta * kGy_c.unsqueeze(0)
            M = M_x + M_y

        # Make M Hermitian for stable real eigenvalues
        M_herm = (M + M.conj().mT) / 2.0

        eigenvalues = torch.linalg.eigvalsh(M_herm)
        bands_k = eigenvalues[:n_bands]
        freq = torch.sqrt(torch.clamp(bands_k, min=0.0))
        all_bands.append(freq)

    return torch.stack(all_bands)


class PhCDesigner:
    """Photonic crystal topology optimization workflow.

    Optimizes the density field of a unit cell to maximize the
    bandgap / midgap ratio for a target band gap.

    Parameters
    ----------
    lattice : str
        "square" or "hexagonal".
    lattice_constant_nm : float
        Lattice constant in nm.
    n_air : float
        Refractive index of air (or background).
    n_material : float
        Refractive index of dielectric material.
    grid_resolution : int
        Number of pixels per lattice constant.
    n_g : int
        Plane-wave expansion cutoff.
    n_bands : int
        Number of bands to compute.
    polarization : str
        "TM" or "TE".
    target_band_gap : tuple[int, int]
        ``(lower, upper)`` band indices to maximize gap between.
    device : str or torch.device
    """

    def __init__(
        self,
        lattice: str = "square",
        lattice_constant_nm: float = 400.0,
        n_air: float = 1.0,
        n_material: float = 3.5,
        grid_resolution: int = 32,
        n_g: int = 3,
        n_bands: int = 6,
        polarization: str = "TM",
        target_band_gap: tuple[int, int] = (1, 2),
        device: str | torch.device = "cpu",
    ):
        self.lattice = lattice
        self.a = lattice_constant_nm
        self.n_air = n_air
        self.n_material = n_material
        self.grid_resolution = grid_resolution
        self.n_g = n_g
        self.n_bands = n_bands
        self.polarization = polarization
        self.target_band_gap = target_band_gap
        self._device = torch.device(device)

        self.grid_shape = (grid_resolution, grid_resolution)

        # Default k-path for Brillouin zone edge
        self.k_points = self._default_k_path()

    def _default_k_path(self) -> torch.Tensor:
        """Generate default k-point path along Brillouin zone edge.

        For square lattice: Gamma -> X -> M -> Gamma
        """
        a = self.a
        n_per_segment = 5

        if self.lattice == "square":
            pi_a = math.pi / a
            gamma = torch.tensor([0.0, 0.0], dtype=torch.float64)
            X = torch.tensor([pi_a, 0.0], dtype=torch.float64)
            M = torch.tensor([pi_a, pi_a], dtype=torch.float64)

            k_gamma_x = torch.stack(
                [gamma + t * (X - gamma) for t in torch.linspace(0, 1, n_per_segment)]
            )
            k_x_m = torch.stack([X + t * (M - X) for t in torch.linspace(0, 1, n_per_segment)])
            k_m_gamma = torch.stack(
                [M + t * (gamma - M) for t in torch.linspace(0, 1, n_per_segment)]
            )
            return torch.cat([k_gamma_x, k_x_m[1:], k_m_gamma[1:]])
        else:
            # Hexagonal: Gamma -> K -> M -> Gamma
            pi_a = math.pi / a
            gamma = torch.tensor([0.0, 0.0], dtype=torch.float64)
            K = torch.tensor([4 * pi_a / 3, 0.0], dtype=torch.float64)
            M = torch.tensor([pi_a, pi_a / math.sqrt(3)], dtype=torch.float64)

            k_gamma_k = torch.stack(
                [gamma + t * (K - gamma) for t in torch.linspace(0, 1, n_per_segment)]
            )
            k_k_m = torch.stack([K + t * (M - K) for t in torch.linspace(0, 1, n_per_segment)])
            k_m_gamma = torch.stack(
                [M + t * (gamma - M) for t in torch.linspace(0, 1, n_per_segment)]
            )
            return torch.cat([k_gamma_k, k_k_m[1:], k_m_gamma[1:]])

    def band_structure(
        self,
        density: torch.Tensor,
        k_points: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute photonic crystal band structure.

        Parameters
        ----------
        density : Tensor, shape ``(H, W)``
            Unit cell density field (0 = air, 1 = material).
        k_points : Tensor, shape ``(N_k, 2)``, optional
            Override k-point path.

        Returns
        -------
        bands : Tensor, shape ``(N_k, n_bands)``
            Band frequencies.
        """
        k = k_points if k_points is not None else self.k_points
        return _band_structure_pwe(
            density.to(self._device),
            self.lattice,
            self.a,
            self.n_g,
            self.n_air,
            self.n_material,
            k.to(self._device),
            self.polarization,
        )

    def bandgap_ratio(
        self,
        density: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the bandgap / midgap ratio for the target band gap.

        The gap is computed over all k-points in the path. The bandgap
        ratio is:

            gap_ratio = (min_upper - max_lower) / ((min_upper + max_lower) / 2)

        Returns 0 if there is no gap (bands overlap).

        Parameters
        ----------
        density : Tensor, shape ``(H, W)``

        Returns
        -------
        gap_ratio : Tensor, scalar
            Bandgap / midgap ratio (0 if no gap).
        """
        bands = self.band_structure(density)
        lower, upper = self.target_band_gap

        max_lower = bands[:, lower - 1].max()
        min_upper = bands[:, upper].min()

        gap = min_upper - max_lower
        midgap = (min_upper + max_lower) / 2

        # Return negative gap ratio (for minimization as loss)
        # If gap < 0 (bands overlap), return 0 gap
        gap_ratio = torch.clamp(gap, min=0.0) / (midgap + 1e-12)
        return gap_ratio

    def bandgap_loss(
        self,
        density: torch.Tensor,
    ) -> torch.Tensor:
        """Loss function for bandgap maximization (negated gap ratio).

        Minimizing this loss maximizes the bandgap.
        """
        return -self.bandgap_ratio(density)

    def maximize_bandgap(
        self,
        n_steps: int = 200,
        lr: float = 0.01,
        beta_schedule: bool = True,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """Run topology optimization to maximize the bandgap.

        Parameters
        ----------
        n_steps : int
            Number of optimization steps.
        lr : float
            Learning rate.
        beta_schedule : bool
            Apply beta-continuation for binarization.
        verbose : bool

        Returns
        -------
        density : Tensor, shape ``(H, W)``
            Optimized density field.
        loss_history : list of float
        """
        from diffnano.design.projection import (
            beta_continuation_schedule,
            heaviside_projection,
        )

        density = torch.rand(*self.grid_shape, device=self._device, dtype=torch.float64)
        density = density.detach().requires_grad_(True)

        opt = torch.optim.Adam([density], lr=lr)
        loss_history = []

        for step in range(n_steps):
            if beta_schedule:
                beta = beta_continuation_schedule(step, n_steps, beta_start=1.0, beta_end=32.0)
                projected = heaviside_projection(density, beta=beta)
            else:
                projected = density

            loss = self.bandgap_loss(projected)

            opt.zero_grad()
            loss.backward()

            if density.grad is not None and torch.isnan(density.grad).any():
                if verbose:
                    print(f"Step {step}: NaN gradient, stopping.")
                break

            opt.step()

            with torch.no_grad():
                density.clamp_(0.0, 1.0)

            loss_history.append(loss.item())

            if verbose and step % 20 == 0:
                gap = self.bandgap_ratio(projected.detach()).item()
                print(f"Step {step:4d}: loss={loss.item():.6f}, gap_ratio={gap:.4f}")

        return density.detach(), loss_history
