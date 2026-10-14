"""Identifiability metrics. All take numpy-compatible (N, d) arrays."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _r2(y: torch.Tensor, yhat: torch.Tensor) -> float:
    """Mean per-column R^2."""
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    return float((1 - ss_res / ss_tot).mean())


def _split(n: int, seed: int = 0, frac: float = 0.8):
    perm = np.random.default_rng(seed).permutation(n)
    c = int(frac * n)
    return perm[:c], perm[c:]


def _standardize(a: np.ndarray, tr: np.ndarray) -> torch.Tensor:
    a = torch.as_tensor(a, dtype=torch.float32)
    mu, sd = a[tr].mean(0), a[tr].std(0).clamp_min(1e-8)
    return (a - mu) / sd


def linear_r2(x, y, seed: int = 0) -> float:
    """Held-out R^2 of the best affine map x -> y."""
    tr, te = _split(len(x), seed)
    x, y = _standardize(x, tr), _standardize(y, tr)
    ones = torch.ones(len(x), 1)
    xa = torch.cat([x, ones], 1)
    w = torch.linalg.lstsq(xa[tr], y[tr]).solution
    return _r2(y[te], xa[te] @ w)


def mlp_r2(x, y, seed: int = 0, epochs: int = 300, hidden: int = 128) -> float:
    """Held-out R^2 of a small MLP x -> y (full-batch Adam)."""
    torch.manual_seed(seed)
    tr, te = _split(len(x), seed)
    x, y = _standardize(x, tr), _standardize(y, tr)
    net = nn.Sequential(
        nn.Linear(x.shape[1], hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, y.shape[1]),
    )
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.mse_loss(net(x[tr]), y[tr]).backward()
        opt.step()
    with torch.no_grad():
        return _r2(y[te], net(x[te]))


def state_targets(states: np.ndarray) -> np.ndarray:
    """(cos, sin, theta_dot) -> same; kept as one place to change targets."""
    return states


def hamiltonian(states: np.ndarray) -> np.ndarray:
    from phase_id.env import H_true

    s = torch.as_tensor(states)
    theta = torch.atan2(s[:, 1], s[:, 0])
    return H_true(theta, s[:, 2])[:, None].numpy()


def identifiability_report(z: np.ndarray, states: np.ndarray, seed: int = 0) -> dict:
    """z <-> state decodability in both directions, plus H(z)."""
    s = state_targets(states)
    return {
        "z->s linear": linear_r2(z, s, seed),
        "z->s mlp": mlp_r2(z, s, seed),
        "s->z linear": linear_r2(s, z, seed),
        "s->z mlp": mlp_r2(s, z, seed),
        "z->theta_dot mlp": mlp_r2(z, s[:, 2:3], seed),
        "z->H mlp": mlp_r2(z, hamiltonian(states), seed),
    }


def cross_seed_consistency(zs: list[np.ndarray]) -> dict:
    """Mean pairwise R^2 of z_i -> z_j (affine and MLP), over ordered pairs."""
    lin, mlp = [], []
    for i in range(len(zs)):
        for j in range(len(zs)):
            if i != j:
                lin.append(linear_r2(zs[i], zs[j]))
                mlp.append(mlp_r2(zs[i], zs[j]))
    return {"affine": float(np.mean(lin)), "mlp": float(np.mean(mlp))}
