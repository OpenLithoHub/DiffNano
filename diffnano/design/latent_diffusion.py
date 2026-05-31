"""Physics-guided latent diffusion for nanophotonic inverse design (N10.1).

Conditional diffusion model operating in a codomain-backbone latent space,
with RCWA-based classifier guidance for physically-informed sampling.

References:
    - MxDiffusion, Nano Lett. 2026
    - AIGP, 2026-05
    - Diffusion-based EM inverse design, arXiv:2511.05357
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "LatentEncoder",
    "LatentDecoder",
    "PhysicsGuidance",
    "ConditionedDiffusion",
    "LatentDiffusionDesigner",
    "LatentDiffusionBenchmark",
]


class LatentEncoder(nn.Module):
    """Encode (H, W) design structures to latent space via 3-layer CNN.

    Returns variational parameters (mean, log_var) for reparameterized
    sampling during training.
    """

    def __init__(
        self,
        grid_size: int = 32,
        latent_dim: int = 16,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        conv_out = hidden_channels
        self.fc_mu = nn.Linear(conv_out, latent_dim)
        self.fc_logvar = nn.Linear(conv_out, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode batch of designs to latent parameters.

        Parameters
        ----------
        x : Tensor, shape ``(batch, H, W)`` or ``(batch, 1, H, W)``

        Returns
        -------
        mu : Tensor, shape ``(batch, latent_dim)``
        log_var : Tensor, shape ``(batch, latent_dim)``
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class LatentDecoder(nn.Module):
    """Decode latent vectors back to (H, W) design space.

    Uses transposed convolutions with sigmoid output clamped to [0, 1].
    """

    def __init__(
        self,
        latent_dim: int = 16,
        grid_size: int = 32,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.hidden = hidden_channels

        self.fc = nn.Linear(latent_dim, hidden_channels * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, 1, 3, stride=2, padding=1, output_padding=1),
        )
        self.upsample = nn.Upsample(size=(grid_size, grid_size), mode="bilinear", align_corners=False)

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent vectors to designs.

        Parameters
        ----------
        z : Tensor, shape ``(batch, latent_dim)``

        Returns
        -------
        Tensor, shape ``(batch, H, W)``
        """
        h = self.fc(z).reshape(-1, self.hidden, 4, 4)
        h = self.deconv(h)
        h = self.upsample(h)
        return torch.sigmoid(h).squeeze(1)


class _SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb)


class _DenoisingBlock(nn.Module):
    """Residual denoising block with time and condition injection."""

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.net(torch.cat([x, cond], dim=-1))
        return self.norm(x + h)


class ConditionedDiffusion(nn.Module):
    """Conditional U-Net-style denoiser operating on latent vectors.

    Conditioned on target optical response vector.  Uses cosine noise
    schedule for stable training across all timesteps.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent vectors.
    cond_dim : int
        Dimension of the conditioning vector (optical response).
    n_blocks : int
        Number of residual denoising blocks.
    n_steps : int
        Number of diffusion timesteps (used for noise schedule).
    """

    def __init__(
        self,
        latent_dim: int = 16,
        cond_dim: int = 32,
        n_blocks: int = 4,
        n_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.n_steps = n_steps

        self.time_embed = _SinusoidalTimeEmbedding(latent_dim)
        self.cond_proj = nn.Linear(cond_dim, latent_dim)
        self.input_proj = nn.Linear(latent_dim, latent_dim)

        self.blocks = nn.ModuleList([
            _DenoisingBlock(latent_dim, latent_dim * 2) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(latent_dim, latent_dim)

        self.register_buffer("_arange", torch.arange(0, n_steps + 1, dtype=torch.float64))

    def _cosine_schedule(self, t: Tensor) -> Tensor:
        """Cosine noise schedule: beta_t and alpha_bar_t.

        Parameters
        ----------
        t : Tensor, shape ``(batch,)``
            Integer timesteps in [0, n_steps).

        Returns
        -------
        alpha_bar : Tensor, shape ``(batch,)``
        """
        s = 0.008
        steps = self.n_steps
        t_float = t.double()
        alpha_bar = torch.cos(((t_float + s) / (steps + s)) * (math.pi / 2)) ** 2
        alpha_bar = alpha_bar / torch.cos(torch.tensor(s / (steps + s)) * (math.pi / 2)) ** 2
        return alpha_bar.clamp(0.0, 1.0).to(t.dtype)

    def forward(self, noisy_z: Tensor, t: Tensor, condition: Tensor) -> Tensor:
        """Predict noise given noisy latent, timestep, and condition.

        Parameters
        ----------
        noisy_z : Tensor, shape ``(batch, latent_dim)``
        t : Tensor, shape ``(batch,)``
            Integer timesteps.
        condition : Tensor, shape ``(batch, cond_dim)``

        Returns
        -------
        Tensor, shape ``(batch, latent_dim)``
            Predicted noise.
        """
        t_emb = self.time_embed(t)
        c_emb = self.cond_proj(condition)
        h = self.input_proj(noisy_z)

        for block in self.blocks:
            cond = torch.cat([t_emb, c_emb], dim=-1)
            h = block(h, cond)

        return self.output_proj(h)

    @torch.no_grad()
    def sample(
        self,
        condition: Tensor,
        n_steps: int = 50,
        guidance: PhysicsGuidance | None = None,
        guidance_scale: float = 1.0,
        n_samples: int = 1,
    ) -> Tensor:
        """Denoise from pure noise to latent samples.

        Parameters
        ----------
        condition : Tensor, shape ``(batch, cond_dim)`` or ``(cond_dim,)``
        n_steps : int
            Number of denoising steps.
        guidance : PhysicsGuidance, optional
            Classifier-guidance module.
        guidance_scale : float
            Scale for classifier guidance gradient.
        n_samples : int
            Number of samples per condition.

        Returns
        -------
        Tensor, shape ``(batch * n_samples, latent_dim)``
        """
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        batch = condition.shape[0]

        cond_expanded = condition.repeat_interleave(n_samples, dim=0)
        z = torch.randn(batch * n_samples, self.latent_dim, device=condition.device, dtype=condition.dtype)

        step_indices = torch.linspace(self.n_steps - 1, 0, n_steps, device=condition.device, dtype=torch.long)

        for idx in step_indices:
            t = idx.expand(batch * n_samples).to(condition.device)
            noise_pred = self(z, t, cond_expanded)

            if guidance is not None and guidance_scale > 0:
                grad = guidance.guide_score(z, cond_expanded, guidance_scale)
                noise_pred = noise_pred - guidance_scale * grad

            alpha_bar = self._cosine_schedule(t.float()).unsqueeze(-1)
            alpha_bar_prev = self._cosine_schedule((t - 1).clamp(min=0).float()).unsqueeze(-1)

            alpha = alpha_bar / alpha_bar_prev
            sigma = torch.sqrt(1.0 - alpha_bar_prev) / torch.sqrt(1.0 - alpha_bar) * torch.sqrt(1.0 - alpha)
            sigma = sigma.clamp(min=1e-8)

            z = (z - (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar).clamp(min=1e-8) * noise_pred) / torch.sqrt(alpha).clamp(min=1e-8)

            if idx > 0:
                noise = torch.randn_like(z)
                z = z + sigma * noise

        return z


class PhysicsGuidance(nn.Module):
    """Maxwell/RCWA classifier guidance for diffusion sampling.

    Wraps a differentiable forward model (e.g. RCWA) and computes
    gradient-based guidance to steer latent samples toward target
    optical responses.

    Parameters
    ----------
    forward_model : callable
        Differentiable model: ``forward_model(design) -> response``.
    decoder : LatentDecoder
        Decoder to map latent vectors back to design space for forward eval.
    target_response : Tensor, optional
        Target optical response to guide toward.
    loss_fn : callable, optional
        ``loss_fn(predicted, target) -> scalar``.  Defaults to MSE.
    """

    def __init__(
        self,
        forward_model: Callable[[Tensor], Tensor],
        decoder: LatentDecoder,
        target_response: Tensor | None = None,
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.forward_model = forward_model
        self.decoder = decoder
        self.target_response = target_response
        self.loss_fn = loss_fn or F.mse_loss

    def guide_score(
        self,
        z: Tensor,
        condition: Tensor,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        """Compute classifier-guidance gradient w.r.t. latent.

        Parameters
        ----------
        z : Tensor, shape ``(batch, latent_dim)``
            Current latent sample.
        condition : Tensor
            Conditioning vector (e.g. target optical response).
        guidance_scale : float

        Returns
        -------
        Tensor, shape ``(batch, latent_dim)``
            Gradient of physics loss w.r.t. z.
        """
        z_param = z.detach().requires_grad_(True)
        designs = self.decoder(z_param)
        predictions = self.forward_model(designs)

        target = self.target_response
        if target is None:
            target = condition
        if target.dim() == 1:
            target = target.unsqueeze(0).expand_as(predictions)

        loss = self.loss_fn(predictions, target)
        grad = torch.autograd.grad(loss, z_param)[0]
        return guidance_scale * grad


class LatentDiffusionDesigner(nn.Module):
    """Main designer: latent diffusion with physics guidance, STE quantization, robust scoring.

    Ties together latent encoder/decoder, conditioned diffusion, physics
    guidance, STE quantization from quantized.py, and robust warm-start
    scoring from robust_warm_start.py.

    Parameters
    ----------
    latent_encoder : LatentEncoder
    latent_decoder : LatentDecoder
    diffusion : ConditionedDiffusion
    quantizer : nn.Module, optional
        STE quantizer (e.g. StraightThroughQuantize).
    guidance : PhysicsGuidance, optional
        RCWA classifier guidance.
    """

    def __init__(
        self,
        latent_encoder: LatentEncoder,
        latent_decoder: LatentDecoder,
        diffusion: ConditionedDiffusion,
        quantizer: nn.Module | None = None,
        guidance: PhysicsGuidance | None = None,
    ) -> None:
        super().__init__()
        self.encoder = latent_encoder
        self.decoder = latent_decoder
        self.diffusion = diffusion
        self.quantizer = quantizer
        self.guidance = guidance

    def train_step(
        self,
        designs: Tensor,
        optical_responses: Tensor,
    ) -> dict[str, Tensor]:
        """One training step: ELBO (reconstruction + KL) + diffusion loss.

        Parameters
        ----------
        designs : Tensor, shape ``(batch, H, W)``
        optical_responses : Tensor, shape ``(batch, cond_dim)``

        Returns
        -------
        dict with ``total_loss``, ``reconstruction_loss``, ``kl_loss``, ``diffusion_loss``
        """
        batch = designs.shape[0]

        mu, log_var = self.encoder(designs)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        recon = self.decoder(z)
        recon_loss = F.mse_loss(recon, designs)

        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        t = torch.randint(0, self.diffusion.n_steps, (batch,), device=designs.device)
        alpha_bar = self.diffusion._cosine_schedule(t.float()).unsqueeze(-1)
        noise = torch.randn_like(mu)
        noisy_z = torch.sqrt(alpha_bar) * mu + torch.sqrt(1.0 - alpha_bar) * noise

        noise_pred = self.diffusion(noisy_z, t, optical_responses)
        diffusion_loss = F.mse_loss(noise_pred, noise)

        total_loss = recon_loss + 0.01 * kl_loss + diffusion_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "diffusion_loss": diffusion_loss,
        }

    @torch.no_grad()
    def design(
        self,
        target_response: Tensor,
        n_candidates: int = 16,
        n_diffusion_steps: int = 50,
        guidance_scale: float = 1.0,
    ) -> dict[str, Tensor]:
        """Generate design candidates via latent diffusion sampling.

        Parameters
        ----------
        target_response : Tensor, shape ``(cond_dim,)``
        n_candidates : int
        n_diffusion_steps : int
        guidance_scale : float

        Returns
        -------
        dict with ``candidates``, ``latent_samples``
        """
        if self.guidance is not None and guidance_scale > 0:
            self.guidance.target_response = target_response

        z_samples = self.diffusion.sample(
            condition=target_response,
            n_steps=n_diffusion_steps,
            guidance=self.guidance,
            guidance_scale=guidance_scale,
            n_samples=n_candidates,
        )

        designs = self.decoder(z_samples)

        return {
            "candidates": designs,
            "latent_samples": z_samples,
        }

    def _quantize_refine(self, candidates: Tensor) -> Tensor:
        """Apply STE quantization to candidate designs.

        Parameters
        ----------
        candidates : Tensor, shape ``(n, H, W)``

        Returns
        -------
        Tensor, shape ``(n, H, W)``
        """
        if self.quantizer is None:
            return candidates
        return self.quantizer(candidates)

    def _score_with_robust(
        self,
        candidates: Tensor,
        condition: Tensor,
        scorer: Callable[[Tensor, Tensor], Tensor],
    ) -> dict[str, Tensor]:
        """Score candidates using a robust scorer (e.g. AngleSweepScorer).

        Parameters
        ----------
        candidates : Tensor, shape ``(n, H, W)``
        condition : Tensor
        scorer : callable

        Returns
        -------
        dict with ``scores``, ``best``, ``best_idx``
        """
        scores = torch.stack([scorer(candidates[i], condition) for i in range(candidates.shape[0])])
        best_idx = scores.argmax()
        return {
            "scores": scores,
            "best": candidates[best_idx],
            "best_idx": best_idx,
        }

    def compare_vs_warm_start(
        self,
        target_response: Tensor,
        warm_start_fn: Callable[[Tensor], dict[str, Tensor]],
        n_seeds: int = 3,
        n_candidates: int = 8,
        scorer: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> dict[str, list]:
        """Benchmark diffusion designer vs warm-start baseline.

        Parameters
        ----------
        target_response : Tensor
        warm_start_fn : callable
            ``warm_start_fn(target) -> dict`` with ``best`` and ``best_score`` keys.
        n_seeds : int
        n_candidates : int
        scorer : callable, optional

        Returns
        -------
        dict with ``diffusion_foms``, ``warm_start_foms``, ``diffusion_wins``
        """
        diffusion_foms: list[float] = []
        warm_start_foms: list[float] = []

        for seed in range(n_seeds):
            torch.manual_seed(seed)

            diff_result = self.design(target_response, n_candidates=n_candidates)
            ws_result = warm_start_fn(target_response)

            ws_best_score = ws_result.get("best_score", torch.tensor(0.0))
            if scorer is not None:
                diff_candidates = diff_result["candidates"]
                diff_scores = torch.stack([
                    scorer(diff_candidates[i], target_response) for i in range(diff_candidates.shape[0])
                ])
                diff_best_fom = diff_scores.max().item()
                ws_best_fom = scorer(ws_result["best"], target_response).item()
            else:
                diff_best_fom = 0.0
                ws_best_fom = ws_best_score.item() if isinstance(ws_best_score, Tensor) else float(ws_best_score)

            diffusion_foms.append(diff_best_fom)
            warm_start_foms.append(ws_best_fom)

        wins = sum(1 for d, w in zip(diffusion_foms, warm_start_foms) if d > w) / max(n_seeds, 1)

        return {
            "diffusion_foms": diffusion_foms,
            "warm_start_foms": warm_start_foms,
            "diffusion_wins": wins,
        }


class LatentDiffusionBenchmark:
    """Compare latent diffusion vs warm-start+gradient baseline.

    Produces a comparison table across FoM, diversity, and
    distribution-out-generalization metrics.
    """

    def __init__(
        self,
        designer: LatentDiffusionDesigner,
        warm_start_fn: Callable[[Tensor], dict[str, Tensor]],
        scorer: Callable[[Tensor, Tensor], Tensor],
        grid_size: int = 32,
    ) -> None:
        self.designer = designer
        self.warm_start_fn = warm_start_fn
        self.scorer = scorer
        self.grid_size = grid_size

    def _compute_diversity(self, candidates: Tensor) -> float:
        """Mean pairwise L2 distance among candidates."""
        n = candidates.shape[0]
        if n < 2:
            return 0.0
        flat = candidates.reshape(n, -1)
        dists = torch.cdist(flat.unsqueeze(0), flat.unsqueeze(0)).squeeze(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=candidates.device)
        return dists[mask].mean().item()

    def run(
        self,
        target_response: Tensor,
        n_seeds: int = 5,
        n_candidates: int = 8,
    ) -> dict[str, dict]:
        """Run the comparison.

        Parameters
        ----------
        target_response : Tensor
        n_seeds : int
        n_candidates : int

        Returns
        -------
        dict with ``diffusion`` and ``warm_start`` sub-dicts, each containing:
            ``best_fom``, ``mean_fom``, ``diversity``, ``foms``
        """
        diff_foms: list[float] = []
        ws_foms: list[float] = []
        diff_diversities: list[float] = []
        ws_diversities: list[float] = []

        for seed in range(n_seeds):
            torch.manual_seed(seed)

            diff_result = self.designer.design(target_response, n_candidates=n_candidates)
            diff_candidates = diff_result["candidates"]

            diff_scores = torch.stack([
                self.scorer(diff_candidates[i], target_response)
                for i in range(diff_candidates.shape[0])
            ])
            diff_foms.append(diff_scores.max().item())
            diff_diversities.append(self._compute_diversity(diff_candidates))

            ws_result = self.warm_start_fn(target_response)
            if "candidates" in ws_result and ws_result["candidates"].dim() >= 3:
                ws_candidates = ws_result["candidates"]
                ws_scores = torch.stack([
                    self.scorer(ws_candidates[i], target_response)
                    for i in range(ws_candidates.shape[0])
                ])
                ws_foms.append(ws_scores.max().item())
                ws_diversities.append(self._compute_diversity(ws_candidates))
            else:
                best = ws_result.get("best", torch.zeros(self.grid_size, self.grid_size))
                ws_fom = self.scorer(best, target_response).item()
                ws_foms.append(ws_fom)
                ws_diversities.append(0.0)

        import numpy as np
        diff_foms_arr = np.array(diff_foms)
        ws_foms_arr = np.array(ws_foms)

        return {
            "diffusion": {
                "best_fom": float(diff_foms_arr.max()),
                "mean_fom": float(diff_foms_arr.mean()),
                "diversity": float(np.mean(diff_diversities)),
                "foms": diff_foms,
            },
            "warm_start": {
                "best_fom": float(ws_foms_arr.max()),
                "mean_fom": float(ws_foms_arr.mean()),
                "diversity": float(np.mean(ws_diversities)),
                "foms": ws_foms,
            },
        }
