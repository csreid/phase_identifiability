"""Exp 03: direct position contrastive learning (no temporal pairs).

Positives are two renderings of the same pendulum angle, jittered by
eps ~ N(0, sigma^2) in *angle* (optionally plus independent pixel noise on each
view). Negatives are all other samples in the batch. Angles are uniform on the
circle, the latent is S^1 (2-D unit sphere), and the positive conditional is a
narrow wrapped Gaussian around the anchor: the setting under which contrastive
learning is expected to recover the latent up to an isometry
(Zimmermann et al. 2021).

Velocity is then tested exactly as in exp02: finite differences of the
per-frame embedding along rollouts (the encoder never sees temporal pairs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp02_fd_velocity import (  # noqa: E402
    embed_rollouts,
    evaluate,
    log_report,
    snapshot,
)
from phase_id.data import WindowPairs  # noqa: E402
from phase_id.env import PendulumPixelEnv, collect_seeded_random_rollouts  # noqa: E402
from phase_id.losses import info_nce  # noqa: E402
from phase_id.models import Encoder  # noqa: E402
from phase_id.velocity import cross_seed_circle, fd_report  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--table-size", type=int, default=2048,
                   help="pre-rendered angles on a uniform grid over the circle")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="std (rad) of the angle jitter between the two views")
    p.add_argument("--pixel-noise", type=float, default=0.0,
                   help="std of independent Gaussian pixel noise per view")
    p.add_argument("--eval-rollouts", type=int, default=300)
    p.add_argument("--rollout-len", type=int, default=20)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--fig-every", type=int, default=1)
    p.add_argument("--out", type=str, default="runs/exp03")
    return p.parse_args()


def render_table(n: int, img_size: int) -> torch.Tensor:
    """(n, 3, H, W) uint8 frames at theta_j = -pi + 2*pi*j/n (theta_dot is
    irrelevant to the render)."""
    env = PendulumPixelEnv(img_size=img_size)
    env.reset()
    th = -np.pi + 2 * np.pi * np.arange(n) / n
    frames = [torch.from_numpy(env.set_state(float(t), 0.0)) for t in th]
    env.close()
    return torch.stack(frames)


def sample_views(table, bs, sigma, pixel_noise, g, dev):
    n = len(table)
    i = torch.randint(n, (bs,), generator=g)
    eps = torch.randn(bs, generator=g) * sigma
    j = (i + torch.round(eps / (2 * np.pi / n)).long()) % n  # wraps on the circle
    views = []
    for idx in (i, j):
        x = table[idx].float().div(255.0)
        if pixel_noise > 0:
            x = x + pixel_noise * torch.randn(x.shape, generator=g)
        views.append(x.to(dev))
    return views


def train(args, table, eval_ds, seed, dev, writer):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    enc = Encoder(3, args.dim).to(dev)
    enc(table[:2].float().div(255.0).to(dev))  # materialise LazyLinear
    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.epochs * args.steps_per_epoch
    )
    step = 0
    if args.fig_every > 0:
        snapshot(writer, enc, eval_ds, dev, 0, args)
    for ep in range(args.epochs):
        enc.train()
        tot = 0.0
        for _ in range(args.steps_per_epoch):
            a, p = sample_views(
                table, args.batch_size, args.sigma, args.pixel_noise, g, dev
            )
            loss = info_nce(enc(a), enc(p), args.tau, args.normalize)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, step = tot + loss.item(), step + 1
            if step % 20 == 0:
                writer.add_scalar("train/loss_step", loss.item(), step)
        done = ep + 1
        writer.add_scalar("train/loss", tot / args.steps_per_epoch, done)
        writer.add_scalar("train/lr", sched.get_last_lr()[0], done)
        if args.fig_every > 0 and (done % args.fig_every == 0 or done == args.epochs):
            snapshot(writer, enc, eval_ds, dev, done, args)
        msg = f"  seed {seed} ep {ep:3d} loss {tot / args.steps_per_epoch:.4f}"
        if args.val_every > 0 and (done % args.val_every == 0 or done == args.epochs):
            _, rep = evaluate(writer, enc, eval_ds, dev, done, args)
            msg += (
                f"  ang_err {rep.get('pos/angle_err_rad', float('nan')):.3f}"
                f"  arc {rep.get('pos/arc_extent_rad', float('nan')):.2f}"
                f"  dphi->thd {rep.get('vel/dphi->thd linear', float('nan')):.3f}"
                f"  [q,dq]->thd mlp {rep['vel/[q,dq]->thd mlp']:.3f}"
            )
        print(msg, flush=True)
    return enc


def main():
    args = parse()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={dev}  args={vars(args)}")

    table = render_table(args.table_size, args.img_size)
    eval_ds = WindowPairs(
        collect_seeded_random_rollouts(
            args.eval_rollouts, args.rollout_len, args.img_size, seed=12345
        ),
        window=1, k=1,
    )

    summary = SummaryWriter(out / "tb" / "summary")
    summary.add_text("args", "```\n" + json.dumps(vars(args), indent=2) + "\n```")
    torch.manual_seed(999)
    rand_enc = Encoder(3, args.dim).to(dev)
    rand_enc(table[:2].float().div(255.0).to(dev))
    floor = fd_report(
        embed_rollouts(rand_enc, eval_ds, dev, args.normalize),
        eval_ds.states.numpy(),
    )
    log_report(summary, floor, 0, "random_init")
    results = {"random_init": floor}
    print("random-init floor:", {k: round(v, 3) for k, v in floor.items()})

    qs = []
    for seed in range(args.seeds):
        writer = SummaryWriter(out / "tb" / f"seed{seed}")
        enc = train(args, table, eval_ds, seed, dev, writer)
        q, rep = evaluate(writer, enc, eval_ds, dev, args.epochs, args, final=True)
        qs.append(q)
        results[f"seed{seed}"] = rep
        print(f"seed {seed}:", {k: round(v, 3) for k, v in rep.items()})
        writer.close()

    if len(qs) > 1 and args.dim == 2:
        cs = cross_seed_circle(qs)
        results["cross_seed"] = cs
        for k, v in cs.items():
            summary.add_scalar(f"cross_seed/{k}", v, 0)
        print("cross-seed:", cs)
    summary.close()
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out / 'results.json'}\ntensorboard --logdir {out / 'tb'}")


if __name__ == "__main__":
    main()
