"""Soft cumulative ordinal head and category head (SafeGrad v2, stage 2).

Follows the SafeAtlas-VL formulation adapted to ladder supervision:

  * ``K - 1`` thresholds parameterised for monotonicity:
    ``beta_1 = alpha_1``, ``beta_k = beta_{k-1} + softplus(alpha_k)``
  * cumulative probabilities ``p^>_k = sigmoid(r - beta_k)`` from a scalar
    risk projection ``r`` of the frozen backbone's last hidden state
  * Gaussian-smoothed cumulative targets instead of hard threshold targets
    (label smoothing across the ordinal scale, width ``gamma``)
  * continuous risk score ``s = 100 * (mu - 1) / (K - 1)`` in ``[0, 100]``,
    where ``mu`` is the expected level index (1-based) under the induced
    categorical distribution
  * a category head over the ladder harm categories plus ``none``
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from safegrad.pipeline.utils import LEVELS_ORDERED

K_LEVELS: int = len(LEVELS_ORDERED)


class OrdinalThresholds(nn.Module):
    """Monotone learnable thresholds ``beta_1 < ... < beta_{K-1}``."""

    def __init__(self, k_levels: int = K_LEVELS):
        super().__init__()
        self.alphas = nn.Parameter(torch.zeros(k_levels - 1))

    def forward(self) -> torch.Tensor:
        betas = [self.alphas[0]]
        for k in range(1, self.alphas.numel()):
            betas.append(betas[-1] + F.softplus(self.alphas[k]))
        return torch.stack(betas)


class OrdinalHead(nn.Module):
    """Two-layer MLP mapping the frozen hidden state to a scalar risk value."""

    def __init__(self, hidden_dim: int, mlp_dim: int = 512, k_levels: int = K_LEVELS):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, 1)
        self.thresholds = OrdinalThresholds(k_levels)

    def risk(self, hidden: torch.Tensor) -> torch.Tensor:
        """Scalar risk projection r_i, shape ``(B,)``."""
        h = F.gelu(self.fc1(self.norm(hidden)))
        return self.fc2(h).squeeze(-1)

    def cumulative_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        """``p^>_{i,k} = sigma(r_i - beta_k)``, shape ``(B, K-1)``."""
        return torch.sigmoid(self.risk(hidden)[:, None] - self.thresholds()[None, :])

    def level_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        return level_probabilities(self.cumulative_probs(hidden))


class CategoryHead(nn.Linear):
    """Linear classifier over harm categories plus ``none``."""


def level_probabilities(cum_probs: torch.Tensor) -> torch.Tensor:
    """Recover the categorical level distribution from cumulative probs."""
    k_minus_1 = cum_probs.shape[1]
    cols = [1.0 - cum_probs[:, 0]]
    for k in range(1, k_minus_1):
        cols.append(cum_probs[:, k - 1] - cum_probs[:, k])
    cols.append(cum_probs[:, k_minus_1 - 1])
    return torch.stack(cols, dim=1).clamp(min=0.0)


def gaussian_soft_targets(level_idx: torch.Tensor, k_levels: int = K_LEVELS,
                          gamma: float = 0.75) -> torch.Tensor:
    """Gaussian-smoothed label distribution over ordered levels, ``(B, K)``."""
    grid = torch.arange(1, k_levels + 1, device=level_idx.device, dtype=torch.float)
    center = (level_idx + 1).float()[:, None]          # 1-based observed level
    logits = -((grid[None, :] - center) ** 2) / (2 * gamma**2)
    return torch.softmax(logits, dim=1)


def cumulative_targets(soft: torch.Tensor) -> torch.Tensor:
    """Convert a ``(B, K)`` level distribution to cumulative targets, ``(B, K-1)``."""
    return torch.flip(torch.cumsum(torch.flip(soft, dims=[1]), dim=1), dims=[1])[:, 1:]
    # equivalent: sum_{l>k} soft[l]


def ordinal_bce_loss(cum_probs: torch.Tensor, level_idx: torch.Tensor,
                     gamma: float = 0.75) -> torch.Tensor:
    """Mean BCE over the K-1 cumulative thresholds against soft targets."""
    tgt = cumulative_targets(gaussian_soft_targets(level_idx, cum_probs.shape[1] + 1, gamma))
    return F.binary_cross_entropy(cum_probs.clamp(1e-6, 1 - 1e-6), tgt)


def expected_risk_score(level_probs: torch.Tensor) -> torch.Tensor:
    """Continuous risk score in [0, 100]; higher means riskier."""
    k = level_probs.shape[1]
    levels = torch.arange(1, k + 1, device=level_probs.device, dtype=torch.float)
    mu = (level_probs * levels[None, :]).sum(dim=1)
    return 100.0 * (mu - 1.0) / (k - 1)
