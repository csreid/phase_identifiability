from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """CNN on channel-stacked frames -> z in R^d (no LSTM, no decoder)."""

    def __init__(self, in_ch: int = 6, dim: int = 2, width: int = 32):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 5, 2, 2), nn.GroupNorm(8, c), nn.GELU(),
            nn.Conv2d(c, 2 * c, 3, 2, 1), nn.GroupNorm(8, 2 * c), nn.GELU(),
            nn.Conv2d(2 * c, 2 * c, 3, 2, 1), nn.GroupNorm(8, 2 * c), nn.GELU(),
            nn.Conv2d(2 * c, 2 * c, 3, 2, 1), nn.GroupNorm(8, 2 * c), nn.GELU(),
            nn.Flatten(),
            nn.LazyLinear(256), nn.GELU(),
            nn.Linear(256, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionPredictor(nn.Module):
    """g(z_t, a_{t:t+k}) -> predicted z_{t+k} (action-conditioned variant)."""

    def __init__(self, dim: int, k: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + k, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))
