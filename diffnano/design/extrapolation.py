"""Extrapolative inverse design with current-diffusion conditioning (N11.2).

Extends the latent diffusion pipeline with frequency-domain dynamics
conditioning to steer generation toward FOM values outside the training
distribution.  The current-diffusion mechanism encodes spectral
characteristics of the optical response into the latent space, enabling
the model to extrapolate beyond seen performance ranges.

References:
    MetaAI, "Physics-aware current-diffusion for metasurface discovery",
    Nature Machine Intelligence, 2026-01
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffnano.design.latent_diffusion import (
    LatentDecoder,
    LatentDiffusionDesigner,
)

__all__ = [
    "CurrentDiffusionConditioner",
    "ExtrapolationDesigner",
    "ExtrapolationBenchmark",
]


class CurrentDiffusionConditioner(nn.Module):
    """Frequency-domain dynamics conditioner for extrapolative latent diffusion.

    Computes FFT-based features from optical responses and uses them to
    modify latent samples toward extrapolation targets outside the training
    distribution.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent vectors.
    n_freq_features : int
        Number of frequency features to extract from FFT spectrum.
    extrapolation_strength : float
        Scaling factor for the extrapolation shift applied to latents.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        n_freq_features: int = 8,
        extrapolation_strength: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_freq_features = n_freq_features
        self.extrapolation_strength = extrapolation_strength

        # Project frequency features to latent-space shifts
        self.freq_proj = nn.Sequential(
            nn.Linear(n_freq_features, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def compute_frequency_features(self, response: Tensor) -> Tensor:
        """Extract FFT-based frequency-domain features from optical response.

        Computes the magnitude spectrum of the real FFT, then selects the
        top-k dominant frequency components by magnitude.

        Parameters
        ----------
        response : Tensor, shape ``(batch, response_len)`` or ``(response_len,)``
            Optical response vector(s).

        Returns
        -------
        Tensor, shape ``(..., n_freq_features)``
            Frequency feature vector(s) — magnitudes of the top-k spectral bins.
        """
        if response.dim() == 1:
            response = response.unsqueeze(0)

        # Real FFT along the response dimension
        spectrum = torch.fft.rfft(response, dim=-1)
        magnitudes = spectrum.abs()  # (batch, n_freq_bins)

        # Pad or truncate to n_freq_features
        n_bins = magnitudes.shape[-1]
        if n_bins >= self.n_freq_features:
            # Select the top-k by magnitude (averaged across batch)
            _, top_indices = magnitudes.mean(dim=0).topk(self.n_freq_features)
            top_indices, _ = top_indices.sort()
            features = magnitudes[:, top_indices]
        else:
            # Pad with zeros if fewer bins than requested features
            features = F.pad(magnitudes, (0, self.n_freq_features - n_bins))

        return features

    def condition_with_dynamics(self, z: Tensor, freq_features: Tensor) -> Tensor:
        """Modify latent samples toward extrapolation targets using frequency features.

        Parameters
        ----------
        z : Tensor, shape ``(batch, latent_dim)``
            Latent samples from the diffusion process.
        freq_features : Tensor, shape ``(batch, n_freq_features)``
            Frequency-domain features of the target response.

        Returns
        -------
        Tensor, shape ``(batch, latent_dim)``
            Conditioned latent samples shifted toward extrapolation targets.
        """
        shift = self.freq_proj(freq_features)
        return z + self.extrapolation_strength * shift


class ExtrapolationDesigner(nn.Module):
    """Extrapolative designer wrapping LatentDiffusionDesigner.

    Generates design candidates targeting figure-of-merit values outside
    the training distribution by combining latent diffusion sampling with
    current-diffusion frequency-domain conditioning.

    Parameters
    ----------
    base_designer : LatentDiffusionDesigner
        Trained latent diffusion designer.
    conditioner : CurrentDiffusionConditioner
        Frequency-domain conditioner for extrapolation.
    decoder : LatentDecoder
        Latent-to-design decoder (typically the base designer's decoder).
    extrapolation_scale : float
        Global scaling for extrapolation shift magnitude.
    """

    def __init__(
        self,
        base_designer: LatentDiffusionDesigner,
        conditioner: CurrentDiffusionConditioner,
        decoder: LatentDecoder,
        extrapolation_scale: float = 1.5,
    ) -> None:
        super().__init__()
        self.base_designer = base_designer
        self.conditioner = conditioner
        self.decoder = decoder
        self.extrapolation_scale = extrapolation_scale

    @torch.no_grad()
    def design_extrapolative(
        self,
        target_response: Tensor,
        held_out_fom_range: tuple[float, float] | None = None,
        n_candidates: int = 16,
        n_diffusion_steps: int = 50,
        guidance_scale: float = 1.0,
    ) -> dict[str, Tensor]:
        """Generate candidates targeting FOM values outside the training distribution.

        Runs standard latent diffusion sampling, then applies current-diffusion
        conditioning to shift latent samples toward extrapolation targets in
        the frequency domain.

        Parameters
        ----------
        target_response : Tensor, shape ``(cond_dim,)``
            Target optical response vector (extrapolative target).
        held_out_fom_range : tuple (low, high), optional
            The held-out FOM range being targeted.  Used for logging only.
        n_candidates : int
            Number of design candidates to generate.
        n_diffusion_steps : int
            Number of diffusion denoising steps.
        guidance_scale : float
            Scale for physics guidance.

        Returns
        -------
        dict with:
            - ``candidates``: Tensor ``(n_candidates, H, W)`` — decoded designs
            - ``latent_samples``: Tensor ``(n_candidates, latent_dim)`` — raw latents
            - ``conditioned_latents``: Tensor ``(n_candidates, latent_dim)``
              latents after conditioning
            - ``freq_features``: Tensor ``(n_candidates, n_freq_features)`` — frequency features
        """
        # Standard diffusion sampling
        base_result = self.base_designer.design(
            target_response,
            n_candidates=n_candidates,
            n_diffusion_steps=n_diffusion_steps,
            guidance_scale=guidance_scale,
        )
        z_samples = base_result["latent_samples"]

        # Compute frequency features of the target response
        freq_features = self.conditioner.compute_frequency_features(
            target_response.unsqueeze(0).expand(z_samples.shape[0], -1),
        )

        # Apply extrapolation conditioning
        conditioned_z = self.conditioner.condition_with_dynamics(z_samples, freq_features)

        # Scale the shift for stronger extrapolation
        z_shifted = z_samples + self.extrapolation_scale * (conditioned_z - z_samples)
        conditioned_z = z_shifted

        # Decode conditioned latents to designs
        candidates = self.decoder(conditioned_z)

        return {
            "candidates": candidates,
            "latent_samples": z_samples,
            "conditioned_latents": conditioned_z,
            "freq_features": freq_features,
        }

    def evaluate_extrapolation(
        self,
        candidates: Tensor,
        high_fidelity_fn: Callable[[Tensor], dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """Verify extrapolated candidates with high-fidelity simulation.

        Parameters
        ----------
        candidates : Tensor, shape ``(n, H, W)``
            Design candidates from extrapolative design.
        high_fidelity_fn : callable
            ``high_fidelity_fn(designs) -> dict`` with at least ``fom`` key
            containing a Tensor of shape ``(n,)``.

        Returns
        -------
        dict with:
            - ``foms``: Tensor ``(n,)`` — per-candidate figure of merit
            - ``best_fom``: scalar Tensor — maximum FOM
            - ``best_idx``: scalar Tensor — index of best candidate
            - ``mean_fom``: scalar Tensor — mean FOM
        """
        result = high_fidelity_fn(candidates)
        foms = result["fom"]

        if foms.dim() == 0:
            foms = foms.unsqueeze(0)

        best_idx = foms.argmax()
        return {
            "foms": foms,
            "best_fom": foms[best_idx],
            "best_idx": best_idx,
            "mean_fom": foms.mean(),
        }


class ExtrapolationBenchmark:
    """Compare extrapolation vs interpolation on held-out performance ranges.

    Runs the ExtrapolationDesigner on targets from outside the training
    distribution and compares against standard (interpolative) sampling,
    both evaluated with a high-fidelity simulation oracle.

    Parameters
    ----------
    extrapolation_designer : ExtrapolationDesigner
        Designer with extrapolation conditioning.
    base_designer : LatentDiffusionDesigner
        Standard latent diffusion designer (interpolation baseline).
    high_fidelity_fn : callable
        ``high_fidelity_fn(designs) -> dict`` with ``fom`` key.
    scorer : callable, optional
        ``scorer(design, target) -> scalar`` for quick FOM proxy.
    grid_size : int
        Design grid dimension.
    """

    def __init__(
        self,
        extrapolation_designer: ExtrapolationDesigner,
        base_designer: LatentDiffusionDesigner,
        high_fidelity_fn: Callable[[Tensor], dict[str, Tensor]],
        scorer: Callable[[Tensor, Tensor], Tensor] | None = None,
        grid_size: int = 32,
    ) -> None:
        self.extrapolation_designer = extrapolation_designer
        self.base_designer = base_designer
        self.high_fidelity_fn = high_fidelity_fn
        self.scorer = scorer
        self.grid_size = grid_size

    def run(
        self,
        extrapolation_target: Tensor,
        interpolation_target: Tensor,
        held_out_fom_range: tuple[float, float] | None = None,
        n_candidates: int = 8,
        n_diffusion_steps: int = 20,
        n_seeds: int = 3,
    ) -> dict[str, dict]:
        """Run the extrapolation vs interpolation benchmark.

        Parameters
        ----------
        extrapolation_target : Tensor, shape ``(cond_dim,)``
            Target response outside the training distribution.
        interpolation_target : Tensor, shape ``(cond_dim,)``
            Target response within the training distribution.
        held_out_fom_range : tuple, optional
            The held-out FOM range for logging.
        n_candidates : int
        n_diffusion_steps : int
        n_seeds : int

        Returns
        -------
        dict with ``extrapolation`` and ``interpolation`` sub-dicts, each
        containing ``foms``, ``best_fom``, ``mean_fom``, ``diversity``.
        """
        import numpy as np

        results: dict[str, dict] = {}

        for label, target, designer in [
            ("extrapolation", extrapolation_target, None),
            ("interpolation", interpolation_target, None),
        ]:
            foms: list[float] = []
            diversities: list[float] = []

            for seed in range(n_seeds):
                torch.manual_seed(seed)

                if label == "extrapolation":
                    result = self.extrapolation_designer.design_extrapolative(
                        target,
                        held_out_fom_range=held_out_fom_range,
                        n_candidates=n_candidates,
                        n_diffusion_steps=n_diffusion_steps,
                    )
                else:
                    base_result = self.base_designer.design(
                        target,
                        n_candidates=n_candidates,
                        n_diffusion_steps=n_diffusion_steps,
                    )
                    result = base_result

                candidates = result["candidates"]

                # Evaluate with high-fidelity oracle
                hf_result = self.high_fidelity_fn(candidates)
                candidate_foms = hf_result["fom"]
                if candidate_foms.dim() == 0:
                    candidate_foms = candidate_foms.unsqueeze(0)
                foms.append(candidate_foms.max().item())

                # Diversity
                n = candidates.shape[0]
                if n >= 2:
                    flat = candidates.reshape(n, -1)
                    dists = torch.cdist(flat.unsqueeze(0), flat.unsqueeze(0)).squeeze(0)
                    mask = ~torch.eye(n, dtype=torch.bool, device=candidates.device)
                    diversities.append(dists[mask].mean().item())
                else:
                    diversities.append(0.0)

            foms_arr = np.array(foms)
            results[label] = {
                "foms": foms,
                "best_fom": float(foms_arr.max()),
                "mean_fom": float(foms_arr.mean()),
                "diversity": float(np.mean(diversities)),
            }

        return results
