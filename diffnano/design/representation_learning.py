"""Learned design representation for faster design space exploration (C8).

Train a VAE or normalizing flow on a library of high-performing designs,
then optimize in the learned latent space for smoother landscapes and
faster convergence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["LearnedRepresentation"]


class _Encoder(nn.Module):
    """Convolutional encoder: density → latent."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 8, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    """Convolutional decoder: latent → density."""

    def __init__(self, latent_dim: int = 8, out_size: int = 32, hidden: int = 16):
        super().__init__()
        self.out_size = out_size
        self.fc = nn.Linear(latent_dim, hidden * 4 * 4)
        self.hidden = hidden
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, 1, 3, stride=2, padding=1, output_padding=1),
        )
        self.final_upsample = nn.Upsample(
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).reshape(-1, self.hidden, 4, 4)
        h = self.deconv(h)
        h = self.final_upsample(h)
        return torch.sigmoid(h)


class LearnedRepresentation:
    """VAE-based learned representation for design space exploration.

    Trains a variational autoencoder on a library of designs, then
    enables optimization in the latent space for faster convergence.

    Parameters
    ----------
    grid_size : int
        Spatial size of density fields (assumes square).
    latent_dim : int
        Dimension of the latent space.
    hidden_channels : int
        Width of hidden conv layers.
    lr : float
        Learning rate.
    device : str or torch.device
    """

    def __init__(
        self,
        grid_size: int = 32,
        latent_dim: int = 8,
        hidden_channels: int = 16,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ):
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self._device = torch.device(device)

        self.encoder = _Encoder(1, latent_dim, hidden_channels).to(self._device).double()
        self.decoder = _Decoder(latent_dim, grid_size, hidden_channels).to(self._device).double()

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _vae_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * (1 + logvar - mu**2 - logvar.exp()).mean()
        return recon_loss + kl_loss

    def train_vae(
        self,
        designs: list[torch.Tensor],
        n_epochs: int = 50,
        batch_size: int = 16,
        verbose: bool = True,
    ) -> list[float]:
        """Train the VAE on a library of designs.

        Parameters
        ----------
        designs : list of Tensor, shape ``(H, W)``
            Library of density fields.
        n_epochs : int
        batch_size : int
        verbose : bool

        Returns
        -------
        loss_history : list of float
        """
        self.encoder.train()
        self.decoder.train()

        data = torch.stack([d.to(self._device).to(torch.float64).unsqueeze(0) for d in designs])
        n = data.shape[0]
        loss_history = []

        for epoch in range(n_epochs):
            perm = torch.randperm(n)
            total_loss = 0.0

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                batch = data[idx]

                mu, logvar = self.encoder(batch)
                z = self._reparameterize(mu, logvar)
                recon = self.decoder(z)

                loss = self._vae_loss(batch, recon, mu, logvar)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg = total_loss / max(1, n // batch_size)
            loss_history.append(avg)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={avg:.2f}")

        self.encoder.eval()
        self.decoder.eval()
        return loss_history

    def encode(self, design: torch.Tensor) -> torch.Tensor:
        """Encode a design to latent space.

        Parameters
        ----------
        design : Tensor, shape ``(H, W)``

        Returns
        -------
        z : Tensor, shape ``(latent_dim,)``
        """
        x = design.to(self._device).to(torch.float64).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.encoder(x)
        return mu.squeeze(0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to density field.

        Parameters
        ----------
        z : Tensor, shape ``(latent_dim,)``

        Returns
        -------
        density : Tensor, shape ``(H, W)``
        """
        z = z.to(self._device).to(torch.float64).unsqueeze(0)
        with torch.no_grad():
            recon = self.decoder(z)
        return recon.squeeze(0).squeeze(0)

    def optimize_in_latent_space(
        self,
        loss_fn: callable,
        n_steps: int = 100,
        lr: float = 0.05,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """Optimize in latent space for faster convergence.

        Parameters
        ----------
        loss_fn : callable
            ``loss_fn(density) -> scalar_loss``.
        n_steps : int
        lr : float
        verbose : bool

        Returns
        -------
        density : Tensor, shape ``(H, W)``
        loss_history : list of float
        """
        # Freeze decoder parameters during latent optimization
        decoder_params_frozen = []
        for p in self.decoder.parameters():
            decoder_params_frozen.append(p.requires_grad)
            p.requires_grad_(False)

        z = torch.zeros(
            self.latent_dim,
            dtype=torch.float64,
            device=self._device,
            requires_grad=True,
        )
        opt = torch.optim.Adam([z], lr=lr)
        loss_history = []

        for step in range(n_steps):
            recon = self.decoder(z.unsqueeze(0))
            density = recon.squeeze(0).squeeze(0)
            loss = loss_fn(density)

            opt.zero_grad()
            loss.backward()

            if z.grad is not None and torch.isnan(z.grad).any():
                if verbose:
                    print(f"Step {step}: NaN gradient, stopping.")
                break

            opt.step()
            loss_history.append(loss.item())

            if verbose and step % 20 == 0:
                print(f"Step {step:4d}: loss={loss.item():.6f}")

        with torch.no_grad():
            final = self.decoder(z.unsqueeze(0)).squeeze(0).squeeze(0)

        # Restore decoder requires_grad
        for p, frozen in zip(self.decoder.parameters(), decoder_params_frozen):
            p.requires_grad_(frozen)

        return final.detach(), loss_history
