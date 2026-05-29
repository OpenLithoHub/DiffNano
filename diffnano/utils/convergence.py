"""Hybrid Z-score convergence monitoring for topology optimization.

Combines standard Z-score (mean / std) with robust Z-score (median / MAD)
to detect convergence even when the loss history has outliers or heavy tails:

    Z_hybrid = Z_standard * (1 - w) + Z_robust * w

where Z_robust = (x - median) / (1.4826 * MAD) and w defaults to 0.5.

Also provides ``ConvergenceMonitor`` that wraps an optimization loop and
suggests early stopping or learning rate adjustments based on the hybrid
Z-score of the recent loss window.
"""

from __future__ import annotations

import math

import torch

__all__ = ["HybridZScore", "ConvergenceMonitor"]


class HybridZScore:
    """Compute a hybrid Z-score blending standard and robust statistics.

    Parameters
    ----------
    window : int
        Number of recent loss values to consider.
    robust_weight : float
        Weight ``w`` for the robust component (0 <= w <= 1).
    """

    def __init__(
        self,
        window: int = 50,
        robust_weight: float = 0.5,
    ):
        if window < 2:
            raise ValueError("window must be >= 2")
        if not 0.0 <= robust_weight <= 1.0:
            raise ValueError("robust_weight must be in [0, 1]")
        self.window = window
        self.robust_weight = robust_weight

    def compute(self, history: list[float] | torch.Tensor) -> float:
        """Compute the hybrid Z-score from a loss history.

        Parameters
        ----------
        history : list[float] or 1-D Tensor
            Loss values (most recent last).

        Returns
        -------
        z_hybrid : float
            Absolute hybrid Z-score of the latest value.
            Returns 0.0 if not enough data or zero dispersion.
        """
        if isinstance(history, torch.Tensor):
            history = history.tolist()

        if len(history) < 2:
            return 0.0

        # Use the most recent ``window`` entries
        recent = history[-self.window:]
        if len(recent) < 2:
            return 0.0

        values = torch.tensor(recent, dtype=torch.float64)
        x_latest = values[-1].item()

        # Standard Z-score
        mean = values.mean().item()
        std = values.std().item()
        if std < 1e-15:
            z_standard = 0.0
        else:
            z_standard = abs((x_latest - mean) / std)

        # Robust Z-score
        median = values.median().item()
        mad = (values - median).abs().median().item()
        # 1.4826 scaling factor makes MAD consistent with std for normal dist
        mad_scaled = 1.4826 * mad
        if mad_scaled < 1e-15:
            z_robust = 0.0
        else:
            z_robust = abs((x_latest - median) / mad_scaled)

        w = self.robust_weight
        return z_standard * (1.0 - w) + z_robust * w


class ConvergenceMonitor:
    """Monitor optimization progress and suggest early-stop / LR adjustments.

    Parameters
    ----------
    patience : int
        How many consecutive checks below threshold before suggesting stop.
    z_threshold : float
        Hybrid Z-score threshold — values below this indicate convergence.
    window : int
        Window size passed to ``HybridZScore``.
    robust_weight : float
        Robust weight passed to ``HybridZScore``.
    lr_decay_factor : float
        Factor to multiply LR by when progress stalls (but not yet stopped).
    stall_patience : int
        How many checks before suggesting LR decay.
    """

    def __init__(
        self,
        patience: int = 5,
        z_threshold: float = 0.5,
        window: int = 50,
        robust_weight: float = 0.5,
        lr_decay_factor: float = 0.5,
        stall_patience: int = 3,
    ):
        self.patience = patience
        self.z_threshold = z_threshold
        self.lr_decay_factor = lr_decay_factor
        self.stall_patience = stall_patience

        self._zscores = HybridZScore(window=window, robust_weight=robust_weight)
        self._converged_count = 0
        self._stall_count = 0
        self._loss_history: list[float] = []

    @property
    def loss_history(self) -> list[float]:
        return self._loss_history

    def step(self, loss: float) -> dict:
        """Record a loss value and return convergence diagnostics.

        Parameters
        ----------
        loss : float
            Current loss.

        Returns
        -------
        info : dict
            Keys:
            - ``z_score``: current hybrid Z-score
            - ``should_stop``: True if convergence detected
            - ``should_decay_lr``: True if stalled (suggests LR decay)
            - ``best_loss``: best loss seen so far
            - ``converged_count``: consecutive below-threshold checks
        """
        self._loss_history.append(loss)

        z = self._zscores.compute(self._loss_history)
        best = min(self._loss_history)

        if z < self.z_threshold and len(self._loss_history) > 10:
            self._converged_count += 1
        else:
            self._converged_count = 0

        # Stall detection: Z-score is low but not low enough for convergence
        if (self.z_threshold <= z < self.z_threshold * 3.0
                and len(self._loss_history) > 10):
            self._stall_count += 1
        else:
            self._stall_count = 0

        return {
            "z_score": z,
            "should_stop": self._converged_count >= self.patience,
            "should_decay_lr": self._stall_count >= self.stall_patience,
            "best_loss": best,
            "converged_count": self._converged_count,
        }

    def reset(self) -> None:
        self._converged_count = 0
        self._stall_count = 0
        self._loss_history.clear()
