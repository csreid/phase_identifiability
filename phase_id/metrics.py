"""Identifiability metrics. All take numpy-compatible (N, d) arrays."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _r2(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
    """Per-column R^2, shape (y.shape[1],)."""
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    return 1 - ss_res / ss_tot


def _split(n: int, seed: int = 0, frac: float = 0.8):
    perm = np.random.default_rng(seed).permutation(n)
    c = int(frac * n)
    return perm[:c], perm[c:]


def _standardize(a: np.ndarray, tr: np.ndarray) -> torch.Tensor:
    a = torch.as_tensor(a, dtype=torch.float32)
    mu, sd = a[tr].mean(0), a[tr].std(0).clamp_min(1e-8)
    return (a - mu) / sd


def linear_r2(x, y, seed: int = 0, per_col: bool = False):
    """Held-out R^2 of the best affine map x -> y (mean, or per column)."""
    tr, te = _split(len(x), seed)
    x, y = _standardize(x, tr), _standardize(y, tr)
    ones = torch.ones(len(x), 1)
    xa = torch.cat([x, ones], 1)
    w = torch.linalg.lstsq(xa[tr], y[tr]).solution
    r = _r2(y[te], xa[te] @ w)
    return r.numpy() if per_col else float(r.mean())


def mlp_r2(
    x, y, seed: int = 0, epochs: int = 300, hidden: int = 128,
    per_col: bool = False,
):
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
        r = _r2(y[te], net(x[te]))
    return r.numpy() if per_col else float(r.mean())


def state_targets(states: np.ndarray) -> np.ndarray:
    """(cos, sin, theta_dot) -> same; kept as one place to change targets."""
    return states


def hamiltonian(states: np.ndarray) -> np.ndarray:
    from phase_id.env import H_true

    s = torch.as_tensor(states)
    theta = torch.atan2(s[:, 1], s[:, 0])
    return H_true(theta, s[:, 2])[:, None].numpy()


def effective_rank(z: np.ndarray) -> float:
    """exp(entropy of normalised singular values of centred z); 1 = collapsed."""
    sv = np.linalg.svd(z - z.mean(0), compute_uv=False)
    p = sv / max(sv.sum(), 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def identifiability_report(
    z: np.ndarray, states: np.ndarray, seed: int = 0, mlp: bool = True
) -> dict:
    """Flat dict of scalars: z<->state decodability, per column, plus H,
    sign(theta_dot), and the effective rank of z.

    Columns of ``states`` are (cos theta, sin theta, theta_dot).
    """
    s = state_targets(states)
    names = ["cos", "sin", "theta_dot"]
    h = hamiltonian(states)
    sign = (s[:, 2:3] > 0).astype(np.float32)
    out = {
        "z->s linear": linear_r2(z, s, seed),
        "s->z linear": linear_r2(s, z, seed),
        "z->H linear": linear_r2(z, h, seed),
        "H->z mlp": mlp_r2(h, z, seed) if mlp else float("nan"),
        "z->sign(theta_dot) linear": linear_r2(z, sign, seed),
        "eff_rank": effective_rank(z),
    }
    for n, v in zip(names, linear_r2(z, s, seed, per_col=True)):
        out[f"z->{n} linear"] = float(v)
    if mlp:
        out["z->s mlp"] = mlp_r2(z, s, seed)
        out["s->z mlp"] = mlp_r2(s, z, seed)
        out["z->H mlp"] = mlp_r2(z, h, seed)
        out["z->sign(theta_dot) mlp"] = mlp_r2(z, sign, seed)
        for n, v in zip(names, mlp_r2(z, s, seed, per_col=True)):
            out[f"z->{n} mlp"] = float(v)
    return out


def cross_seed_consistency(zs: list[np.ndarray]) -> dict:
    """Mean pairwise R^2 of z_i -> z_j (affine and MLP), over ordered pairs."""
    lin, mlp = [], []
    for i in range(len(zs)):
        for j in range(len(zs)):
            if i != j:
                lin.append(linear_r2(zs[i], zs[j]))
                mlp.append(mlp_r2(zs[i], zs[j]))
    return {"affine": float(np.mean(lin)), "mlp": float(np.mean(mlp))}
