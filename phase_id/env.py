"""Modified Pendulum-v1 with analytic port-Hamiltonian dynamics and pixel obs.

Ported from hamilton_rl/data/pendulum.py (dynamics, pixel wrapper, seeded
rollout collector only).

Canonical coordinates: q = theta, p = theta_dot, H = p^2/2 + 1.5*g*(1+cos q).
Control enters as p_dot += 3u (u clamped to [-2, 2]); quadratic drag acts on p.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as TF
from gymnasium import spaces
from tqdm import tqdm

_G = 10.0
_DT = 0.1  # 2x Pendulum-v1's 0.05, for easier velocity ID
_MAX_SPEED = 8.0  # nominal velocity scale (grid range / plot range)
_DRAG_COEFF = 0.05  # quadratic drag, p_dot = -c p |p|; bounds speed smoothly
B_TRUE = 3.0


def angle_normalize(theta: torch.Tensor) -> torch.Tensor:
    return ((theta + torch.pi) % (2 * torch.pi)) - torch.pi


def _grad_V(theta: torch.Tensor, g: float = _G) -> torch.Tensor:
    return -1.5 * g * torch.sin(theta)


def _drag_substep(p: torch.Tensor, tau: float, c: float) -> torch.Tensor:
    """Exact flow of dp/dt = -c p |p| over time tau (self-inverting in tau)."""
    if c == 0.0:
        return p
    sign = torch.sign(p)
    inv_abs_p = 1.0 / p.abs().clamp_min(1e-12) + c * tau
    return sign / inv_abs_p.clamp_min(1e-12)


def _dissipation_substep(q, p, tau, damping, drag):
    if damping != 0.0:
        p = p * float(np.exp(-damping * tau))
    if drag != 0.0:
        p = _drag_substep(p, tau, drag)
    return q, p


def _dissipation_substep_inverse(q, p, tau, damping, drag):
    if drag != 0.0:
        p = _drag_substep(p, -tau, drag)
    if damping != 0.0:
        p = p * float(np.exp(damping * tau))
    return q, p


def analytic_pendulum_step(
    theta: torch.Tensor,
    theta_dot: torch.Tensor,
    u: torch.Tensor,
    dt: float = _DT,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
    g: float = _G,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Strang-split leapfrog step (dissipation / kick-drift-kick / dissipation)."""
    u = u.clamp(-2.0, 2.0)
    q, p = theta, theta_dot
    has_dissipation = damping != 0.0 or drag != 0.0

    if has_dissipation:
        q, p = _dissipation_substep(q, p, dt / 2, damping, drag)

    Bu = B_TRUE * u
    p = p - (dt / 2) * _grad_V(q, g) + (dt / 2) * Bu
    q = q + dt * p
    p = p - (dt / 2) * _grad_V(q, g) + (dt / 2) * Bu

    if has_dissipation:
        q, p = _dissipation_substep(q, p, dt / 2, damping, drag)

    return q, p


def analytic_pendulum_step_inverse(
    theta: torch.Tensor,
    theta_dot: torch.Tensor,
    u: torch.Tensor,
    dt: float = _DT,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
    g: float = _G,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact inverse of ``analytic_pendulum_step``."""
    u = u.clamp(-2.0, 2.0)
    q, p = theta, theta_dot
    has_dissipation = damping != 0.0 or drag != 0.0

    if has_dissipation:
        q, p = _dissipation_substep_inverse(q, p, dt / 2, damping, drag)

    Bu = B_TRUE * u
    p = p + (dt / 2) * _grad_V(q, g) - (dt / 2) * Bu
    q = q - dt * p
    p = p + (dt / 2) * _grad_V(q, g) - (dt / 2) * Bu

    if has_dissipation:
        q, p = _dissipation_substep_inverse(q, p, dt / 2, damping, drag)

    return q, p


def H_true(q: torch.Tensor, p: torch.Tensor, g: float = _G) -> torch.Tensor:
    return 0.5 * p**2 + 1.5 * g * (1.0 + torch.cos(q))


def preprocess_frame(frame: np.ndarray, img_size: int = 64) -> torch.Tensor:
    t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    return TF.resize(t, [img_size, img_size], antialias=True)


class PendulumPixelEnv(gym.Wrapper):
    """Pendulum-v1 with (3, img_size, img_size) uint8 pixel observations.

    Physics is driven by ``analytic_pendulum_step``; gym's own step is
    bypassed. ``unwrapped.state`` is [theta, theta_dot].
    """

    def __init__(
        self,
        img_size: int = 64,
        damping: float = 0.0,
        drag: float = _DRAG_COEFF,
    ):
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        super().__init__(env)
        self.img_size = img_size
        self.damping = damping
        self.drag = drag
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, img_size, img_size), dtype=np.uint8
        )

    def _obs(self) -> np.ndarray:
        # Suppress gym's torque arrow: it would leak the action into pixels.
        self.env.unwrapped.last_u = None  # type: ignore[union-attr]
        frame = self.env.render()
        t = preprocess_frame(frame, self.img_size)
        return (t * 255).byte().numpy()

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._obs(), {}

    def set_state(self, theta: float, theta_dot: float) -> np.ndarray:
        self.env.unwrapped.state = np.array(  # type: ignore[union-attr]
            [theta, theta_dot], dtype=np.float64
        )
        return self._obs()

    def step(self, action):
        theta, theta_dot = self.env.unwrapped.state  # type: ignore[union-attr]
        u = torch.as_tensor([float(action[0])], dtype=torch.float64)
        new_theta, new_theta_dot = analytic_pendulum_step(
            torch.as_tensor([theta], dtype=torch.float64),
            torch.as_tensor([theta_dot], dtype=torch.float64),
            u,
            damping=self.damping,
            drag=self.drag,
        )
        self.env.unwrapped.state = np.array(  # type: ignore[union-attr]
            [new_theta.item(), new_theta_dot.item()], dtype=np.float64
        )
        return self._obs(), 0.0, False, False, {}


def _grid_seeds(
    n_points: int,
    rng: np.random.Generator,
    min_theta: float = -np.pi,
    max_theta: float = np.pi,
    min_theta_dot: float = -_MAX_SPEED,
    max_theta_dot: float = _MAX_SPEED,
) -> np.ndarray:
    """Near-square covering grid of (theta, theta_dot), tiled to n_points, shuffled."""
    aspect = (max_theta_dot - min_theta_dot) / (max_theta - min_theta)
    n_theta = max(1, round(np.sqrt(n_points / aspect)))
    n_theta_dot = max(1, round(n_points / n_theta))

    grid_t, grid_td = np.meshgrid(
        np.linspace(min_theta, max_theta, n_theta),
        np.linspace(min_theta_dot, max_theta_dot, n_theta_dot),
        indexing="xy",
    )
    grid = np.stack([grid_t.ravel(), grid_td.ravel()], axis=-1)
    reps = int(np.ceil(n_points / len(grid)))
    seeds = np.tile(grid, (reps, 1))[:n_points]
    rng.shuffle(seeds)
    return seeds


def collect_seeded_random_rollouts(
    n_rollouts: int,
    rollout_len: int,
    img_size: int = 64,
    damping: float = 0.0,
    drag: float = _DRAG_COEFF,
    seed: int = 0,
    progress: bool = True,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Short uniform-random-torque rollouts seeded from a covering grid.

    Returns n_rollouts tuples:
        frames  (rollout_len+1, 3, H, W) float32 in [0, 1]
        actions (rollout_len,)           float32
        states  (rollout_len+1, 3)       float32: (cos th, sin th, th_dot)
    """
    rng = np.random.default_rng(seed)
    seeds = _grid_seeds(n_rollouts, rng)

    env = PendulumPixelEnv(img_size=img_size, damping=damping, drag=drag)
    rollouts = []
    try:
        it = tqdm(seeds, desc="rollouts", disable=not progress)
        for theta0, theta_dot0 in it:
            env.reset()
            obs = env.set_state(float(theta0), float(theta_dot0))
            frames = [torch.from_numpy(obs).float() / 255.0]
            actions = []
            states = [
                np.array(
                    [np.cos(theta0), np.sin(theta0), theta_dot0],
                    dtype=np.float32,
                )
            ]
            for _ in range(rollout_len):
                a = float(rng.uniform(-2.0, 2.0))
                obs, *_ = env.step(np.array([a], dtype=np.float32))
                th, thd = env.unwrapped.state  # type: ignore[union-attr]
                frames.append(torch.from_numpy(obs).float() / 255.0)
                actions.append(a)
                states.append(
                    np.array([np.cos(th), np.sin(th), thd], dtype=np.float32)
                )
            rollouts.append(
                (
                    torch.stack(frames),
                    torch.tensor(actions, dtype=torch.float32),
                    torch.from_numpy(np.stack(states)),
                )
            )
    finally:
        env.close()
    return rollouts
