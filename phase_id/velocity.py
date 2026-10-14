"""Position-only latents + finite-difference velocity.

q: (N, T+1, d) per-frame embeddings of rollouts; states: (N, T+1, 3) with
columns (cos theta, sin theta, theta_dot).
"""

from __future__ import annotations

import numpy as np

from phase_id.env import _DT
from phase_id.metrics import effective_rank, linear_r2, mlp_r2


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def circle_align(phi: np.ndarray, theta: np.ndarray):
    """Best (sign, offset) with phi ~ sign*theta + offset on the circle.

    Returns (mean abs angular error, sign, offset).
    """
    best = None
    for s in (1.0, -1.0):
        d = phi - s * theta
        c = float(np.angle(np.exp(1j * d).mean()))
        err = float(np.abs(wrap(d - c)).mean())
        if best is None or err < best[0]:
            best = (err, s, c)
    return best


def central_diff(x: np.ndarray, dt: float = _DT) -> np.ndarray:
    """(N, T+1, ...) -> (N, T-1, ...), derivative at t = 1..T-1."""
    return (x[:, 2:] - x[:, :-2]) / (2 * dt)


def angle_velocity(phi: np.ndarray, dt: float = _DT) -> np.ndarray:
    """Central difference of an angle with wrapping, (N, T+1) -> (N, T-1)."""
    return wrap(phi[:, 2:] - phi[:, :-2]) / (2 * dt)


def fd_report(q: np.ndarray, states: np.ndarray, mlp: bool = True) -> dict:
    """Position quality + how well finite differences of q recover theta_dot.

    If q is 2-D (an S^1 embedding) also reports the angular finite
    difference, which is the 'natural' velocity estimate on a circle.
    """
    n, t1, d = q.shape
    theta = np.arctan2(states[..., 1], states[..., 0])
    thd = states[:, 1:-1, 2].reshape(-1, 1)
    q_mid = q[:, 1:-1].reshape(-1, d)
    dq = central_diff(q).reshape(-1, d)
    out = {"eff_rank": effective_rank(q.reshape(-1, d))}

    # Ceiling: central difference of the *true* angle (dt-discretisation error).
    th_fd = angle_velocity(theta).reshape(-1, 1)
    out["vel/ceiling_true_fd->thd linear"] = linear_r2(th_fd, thd)

    # Position decodability.
    cs = states[..., :2].reshape(-1, 2)
    out["pos/q->(cos,sin) linear"] = linear_r2(q.reshape(-1, d), cs)
    if mlp:
        out["pos/q->(cos,sin) mlp"] = mlp_r2(q.reshape(-1, d), cs)

    # Raw finite difference of the latent; position-dependent Jacobian expected.
    out["vel/dq->thd linear"] = linear_r2(dq, thd)
    both = np.concatenate([q_mid, dq], 1)
    out["vel/[q,dq]->thd linear"] = linear_r2(both, thd)
    if mlp:
        out["vel/dq->thd mlp"] = mlp_r2(dq, thd)
        out["vel/[q,dq]->thd mlp"] = mlp_r2(both, thd)

    if d == 2:
        phi = np.arctan2(q[..., 1], q[..., 0])
        err, s, _ = circle_align(phi[:, 1:-1].ravel(), theta[:, 1:-1].ravel())
        out["pos/angle_err_rad"] = err
        out["pos/orientation"] = s
        dphi = s * angle_velocity(phi).reshape(-1, 1)
        out["vel/dphi->thd linear"] = linear_r2(dphi, thd)
        out["vel/dphi_slope"] = float((dphi * thd).sum() / (dphi**2).sum())
        if mlp:
            out["vel/dphi->thd mlp"] = mlp_r2(dphi, thd)
            out["vel/[phi,dphi]->thd mlp"] = mlp_r2(
                np.concatenate(
                    [np.cos(phi[:, 1:-1]).reshape(-1, 1),
                     np.sin(phi[:, 1:-1]).reshape(-1, 1), dphi], 1),
                thd,
            )
    return out


def cross_seed_circle(qs: list[np.ndarray]) -> dict:
    """Mean abs angular error of phi_j vs phi_i (up to rotation/reflection),
    and mean linear R^2 between angular velocities, over ordered pairs."""
    phis = [np.arctan2(q[..., 1], q[..., 0]) for q in qs]
    errs, vr = [], []
    for i in range(len(qs)):
        for j in range(len(qs)):
            if i == j:
                continue
            errs.append(circle_align(phis[j].ravel(), phis[i].ravel())[0])
            vr.append(
                linear_r2(
                    angle_velocity(phis[i]).reshape(-1, 1),
                    angle_velocity(phis[j]).reshape(-1, 1),
                )
            )
    return {"angle_err_rad": float(np.mean(errs)), "dphi_linear_r2": float(np.mean(vr))}
