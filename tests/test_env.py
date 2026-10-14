import numpy as np
import torch

from phase_id.data import WindowPairs
from phase_id.env import (
    PendulumPixelEnv,
    analytic_pendulum_step,
    analytic_pendulum_step_inverse,
    collect_seeded_random_rollouts,
)


def test_step_inverse_roundtrip():
    g = torch.Generator().manual_seed(0)
    th = (torch.rand(100, generator=g, dtype=torch.float64) - 0.5) * 6.28
    thd = (torch.rand(100, generator=g, dtype=torch.float64) - 0.5) * 16
    u = (torch.rand(100, generator=g, dtype=torch.float64) - 0.5) * 4
    th1, thd1 = analytic_pendulum_step(th, thd, u)
    th0, thd0 = analytic_pendulum_step_inverse(th1, thd1, u)
    assert torch.allclose(th, th0, atol=1e-9)
    assert torch.allclose(thd, thd0, atol=1e-9)


def test_env_obs_and_set_state():
    env = PendulumPixelEnv(img_size=64)
    env.reset()
    obs = env.set_state(0.5, 1.0)
    assert obs.shape == (3, 64, 64) and obs.dtype == np.uint8
    np.testing.assert_allclose(env.unwrapped.state, [0.5, 1.0])
    env.step(np.array([0.0], dtype=np.float32))
    assert not np.allclose(env.unwrapped.state, [0.5, 1.0])
    env.close()


def test_rollouts_and_windows():
    r = collect_seeded_random_rollouts(6, 5, 32, progress=False)
    assert r[0][0].shape == (6, 3, 32, 32) and r[0][2].shape == (6, 3)
    ds = WindowPairs(r, window=2, k=1)
    assert len(ds) == 6 * 4  # t in [1, 4]
    b = ds.batch(torch.arange(8))
    assert b["x"].shape == (8, 6, 32, 32) and b["a"].shape == (8, 1)
