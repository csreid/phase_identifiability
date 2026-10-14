"""Windowed (x_t, x_{t+k}) pairs from rollouts. States are eval-only."""

from __future__ import annotations

import torch


class WindowPairs:
    """All valid (window ending at t, window ending at t+k) index pairs.

    Frames are stored as uint8 (N, T+1, 3, H, W). A window of length ``w``
    ending at t is frames[t-w+1 .. t] channel-stacked -> (3w, H, W).
    """

    def __init__(self, rollouts, window: int = 2, k: int = 1):
        self.window, self.k = window, k
        self.frames = torch.stack([(r[0] * 255).round().byte() for r in rollouts])
        self.actions = torch.stack([r[1] for r in rollouts])  # (N, T)
        self.states = torch.stack([r[2] for r in rollouts])  # (N, T+1, 3)
        n, t1 = self.frames.shape[:2]
        # t ranges over [window-1, T-k]; window ending at t needs t-w+1 >= 0.
        ts = torch.arange(window - 1, t1 - k)
        ri = torch.arange(n).repeat_interleave(len(ts))
        ti = ts.repeat(n)
        self.index = torch.stack([ri, ti], dim=1)  # (M, 2)

    def __len__(self) -> int:
        return len(self.index)

    def _window(self, ri: torch.Tensor, ti: torch.Tensor) -> torch.Tensor:
        offs = torch.arange(-self.window + 1, 1)
        fr = self.frames[ri[:, None], ti[:, None] + offs[None]]  # (B, w, 3, H, W)
        b, w, c, h, wd = fr.shape
        return fr.reshape(b, w * c, h, wd).float() / 255.0

    def batch(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        ri, ti = self.index[idx].unbind(1)
        koffs = torch.arange(self.k)
        return {
            "x": self._window(ri, ti),
            "x_next": self._window(ri, ti + self.k),
            "a": self.actions[ri[:, None], ti[:, None] + koffs[None]],  # (B, k)
            "state": self.states[ri, ti],  # (B, 3): cos, sin, theta_dot
        }

    def all_windows(self, bs: int = 1024):
        """Iterate (x, state) over every valid anchor window, in index order."""
        for s in range(0, len(self), bs):
            b = self.batch(torch.arange(s, min(s + bs, len(self))))
            yield b["x"], b["state"]
