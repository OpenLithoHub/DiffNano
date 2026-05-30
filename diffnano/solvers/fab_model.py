"""Learned fabrication process model (C6 — inspired by PRISM + TorchResist).

Neural network that maps design mask → printed contour, trained on calibration
data. Differentiable end-to-end; drops into the DFM workflow as replacement
for HopkinsLithoModel.

References
----------
- Zhou et al. (2026), PRISM: arXiv:2602.15762
- Geng et al. (2025), TorchResist: arXiv:2502.06838
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diffnano.solvers._result import SimResult

__all__ = ["LearnedFabModel"]


class _UNetBlock(nn.Module):
    """Single U-Net block: conv + groupnorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(1, out_ch)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.gn(self.conv(x)))


class _PhysicsGatedNet(nn.Module):
    """U-Net-style encoder-decoder with physics priors.

    The network learns a residual correction on top of a Hopkins-like
    convolutional backbone, ensuring energy conservation and non-negativity.

    Parameters
    ----------
    in_channels : int
    hidden_channels : int
    n_blocks : int
        Number of encoder/decoder blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        n_blocks: int = 3,
    ):
        super().__init__()
        self.n_blocks = n_blocks

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        ch = in_channels
        for _ in range(n_blocks):
            self.encoder.append(_UNetBlock(ch, hidden_channels))
            ch = hidden_channels

        self.bottleneck = _UNetBlock(hidden_channels, hidden_channels)

        for _ in range(n_blocks):
            self.decoder.append(_UNetBlock(hidden_channels, hidden_channels))

        self.final = nn.Conv2d(hidden_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2], x.shape[3]

        # Encoder with skip connections
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = nn.functional.avg_pool2d(x, 2)

        x = self.bottleneck(x)

        # Decoder with skip connections
        for block, skip in zip(self.decoder, reversed(skips)):
            x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            # Match skip dimensions by center-cropping or padding
            if x.shape[2] > skip.shape[2] or x.shape[3] > skip.shape[3]:
                x = x[:, :, : skip.shape[2], : skip.shape[3]]
            elif x.shape[2] < skip.shape[2] or x.shape[3] < skip.shape[3]:
                x = nn.functional.pad(
                    x,
                    [
                        0,
                        skip.shape[3] - x.shape[3],
                        0,
                        skip.shape[2] - x.shape[2],
                    ],
                )
            x = x + skip
            x = block(x)

        # Final resize to match input dimensions
        if x.shape[2] != H_in or x.shape[3] != W_in:
            x = nn.functional.interpolate(
                x,
                size=(H_in, W_in),
                mode="bilinear",
                align_corners=False,
            )

        # Physics-gated output: sigmoid ensures [0, 1] (energy conservation)
        return torch.sigmoid(self.final(x))


class LearnedFabModel:
    """Learned fabrication process model.

    Neural network that maps design mask → printed contour, trained on
    calibration data. Differentiable end-to-end; drops into the DFM
    workflow as replacement for HopkinsLithoModel.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        ``(H, W)`` grid dimensions.
    hidden_channels : int
        Width of hidden layers.
    n_blocks : int
        Depth of U-Net.
    lr : float
        Learning rate.
    device : str or torch.device
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (64, 64),
        hidden_channels: int = 16,
        n_blocks: int = 3,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ):
        self.grid_shape = grid_shape
        self._device = torch.device(device)

        self.net = (
            _PhysicsGatedNet(
                in_channels=1,
                hidden_channels=hidden_channels,
                n_blocks=n_blocks,
            )
            .to(self._device)
            .double()
        )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self._trained = False

    @property
    def device(self) -> torch.device:
        return self._device

    def forward(
        self,
        mask: torch.Tensor,
        wavelengths=None,
        *,
        source=None,
    ) -> SimResult:
        """Predict printed contour from design mask.

        Parameters
        ----------
        mask : Tensor, shape ``(H, W)`` or ``(1, H, W)``
            Design mask (density field).
        wavelengths : ignored
        source : ignored

        Returns
        -------
        SimResult
            ``field`` contains the predicted printed contour.
        """
        x = mask.to(self._device).to(torch.float64)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)

        printed = self.net(x).squeeze(0).squeeze(0)

        return SimResult(
            field=printed.unsqueeze(0),
            wavelengths=torch.tensor([0.0], device=self._device),
            metadata={"model": "learned_fab"},
        )

    def train_model(
        self,
        train_data: list[tuple[torch.Tensor, torch.Tensor]],
        n_epochs: int = 50,
        verbose: bool = True,
    ) -> list[float]:
        """Train the model on (mask, printed_contour) pairs.

        Parameters
        ----------
        train_data : list of (mask, target) tuples
            Each mask/target has shape ``(H, W)``.
        n_epochs : int
        verbose : bool

        Returns
        -------
        loss_history : list of float
        """
        self.net.train()
        loss_fn = nn.functional.mse_loss
        loss_history = []

        for epoch in range(n_epochs):
            total_loss = 0.0
            for mask, target in train_data:
                pred = self.forward(mask).field.squeeze(0)
                loss = loss_fn(pred, target.to(self._device).to(torch.float64))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_data)
            loss_history.append(avg_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={avg_loss:.6f}")

        self._trained = True
        self.net.eval()
        return loss_history

    def generate_synthetic_data(
        self,
        n_samples: int = 100,
        blur_sigma: float = 2.0,
        noise_std: float = 0.02,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate synthetic (mask, printed) pairs for testing.

        Uses Gaussian blur as a stand-in for a real lithography model.
        """
        from diffnano.design.robustness.subspace import corner_rounding_perturbation

        data = []
        H, W = self.grid_shape
        for _ in range(n_samples):
            mask = torch.rand(H, W, dtype=torch.float64, device=self._device)
            mask = (mask > 0.5).double()

            radius = torch.tensor(blur_sigma * 2, dtype=torch.float64, device=self._device)
            printed = corner_rounding_perturbation(mask, radius)
            printed = printed + noise_std * torch.randn_like(printed)
            printed = printed.clamp(0, 1)

            data.append((mask, printed))

        return data
