"""Exp 02: position-only contrastive encoder + finite-difference velocity.

A single-frame encoder f(frame) -> q is trained with InfoNCE on temporal
positives (frame_t, frame_{t+k}); no velocity information is ever given.
theta_dot is then estimated from finite differences of q across consecutive
frames. On a unit-sphere 2-D latent (an S^1 embedding) the natural estimate is
the wrapped angular difference of atan2(q).
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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase_id.data import WindowPairs
from phase_id.env import _DT, collect_seeded_random_rollouts
from phase_id.losses import info_nce
from phase_id.metrics import hamiltonian
from phase_id.models import Encoder
from phase_id.velocity import (
    angle_velocity,
    circle_align,
    cross_seed_circle,
    fd_report,
)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--n-rollouts", type=int, default=2000)
    p.add_argument("--eval-rollouts", type=int, default=300)
    p.add_argument("--rollout-len", type=int, default=20)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--out", type=str, default="runs/exp02")
    return p.parse_args()


@torch.no_grad()
def embed_rollouts(enc, ds, dev, normalize, bs=1024):
    """Encode every frame of every rollout -> (N, T+1, d) numpy."""
    enc.eval()
    n, t1 = ds.frames.shape[:2]
    flat = ds.frames.reshape(n * t1, *ds.frames.shape[2:])
    out = []
    for s in range(0, len(flat), bs):
        z = enc(flat[s : s + bs].float().div(255.0).to(dev))
        out.append(F.normalize(z, dim=-1).cpu() if normalize else z.cpu())
    return torch.cat(out).reshape(n, t1, -1).numpy()


def log_report(writer, rep, step, prefix):
    for k, v in rep.items():
        writer.add_scalar(f"{prefix}/{k}", v, step)


def fd_figure(q, states):
    """q on the circle, estimated vs true theta_dot, reconstructed phase portrait."""
    theta = np.arctan2(states[..., 1], states[..., 0])
    thd = states[:, 1:-1, 2].ravel()
    h = hamiltonian(states[:, 1:-1].reshape(-1, 3))[:, 0]
    phi = np.arctan2(q[..., 1], q[..., 0])
    _, s, c = circle_align(phi[:, 1:-1].ravel(), theta[:, 1:-1].ravel())
    p_hat = s * angle_velocity(phi).ravel()
    phi_al = s * (phi[:, 1:-1].ravel() - c)  # aligned so ~ theta
    phi_al = (phi_al + np.pi) % (2 * np.pi) - np.pi

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    im = ax[0].scatter(
        q[:, 1:-1, 0].ravel(), q[:, 1:-1, 1].ravel(),
        c=theta[:, 1:-1].ravel(), s=2, cmap="twilight",
    )
    ax[0].set_title("q colored by theta")
    ax[0].set_aspect("equal")
    fig.colorbar(im, ax=ax[0])

    lim = float(np.abs(thd).max())
    ax[1].scatter(thd, p_hat, s=2, alpha=0.3, linewidths=0)
    ax[1].plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax[1].set_xlabel("true theta_dot")
    ax[1].set_ylabel("finite-diff estimate")
    ax[1].set_title("velocity from dphi")

    for a, (x, y, ttl) in zip(
        ax[2:],
        [
            (phi_al, p_hat, "reconstructed phase portrait"),
            (theta[:, 1:-1].ravel(), thd, "true phase portrait"),
        ],
    ):
        im = a.scatter(x, y, c=h, s=2, cmap="viridis", vmin=0, vmax=60)
        a.set_xlabel("theta")
        a.set_ylabel("theta_dot")
        a.set_title(ttl + " (color=H)")
        fig.colorbar(im, ax=a)
    fig.tight_layout()
    return fig


def evaluate(writer, enc, eval_ds, dev, step, args, final=False):
    q = embed_rollouts(enc, eval_ds, dev, args.normalize)
    states = eval_ds.states.numpy()
    rep = fd_report(q, states)
    log_report(writer, rep, step, "final" if final else "probe")
    if q.shape[-1] == 2:
        fig = fd_figure(q, states)
        writer.add_figure("val/fd", fig, step)
        plt.close(fig)
    return q, rep


def train(args, ds, eval_ds, seed, dev, writer):
    torch.manual_seed(seed)
    enc = Encoder(3, args.dim).to(dev)
    enc(ds.batch(torch.arange(2))["x"].to(dev))  # materialise LazyLinear
    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.epochs * (len(ds) // args.batch_size)
    )
    g = torch.Generator().manual_seed(seed)
    step = 0
    for ep in range(args.epochs):
        enc.train()
        perm = torch.randperm(len(ds), generator=g)
        tot, nb = 0.0, 0
        for s in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            b = ds.batch(perm[s : s + args.batch_size])
            loss = info_nce(
                enc(b["x"].to(dev)), enc(b["x_next"].to(dev)),
                args.tau, args.normalize,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, nb, step = tot + loss.item(), nb + 1, step + 1
            if step % 20 == 0:
                writer.add_scalar("train/loss_step", loss.item(), step)
        writer.add_scalar("train/loss", tot / nb, ep)
        writer.add_scalar("train/lr", sched.get_last_lr()[0], ep)
        msg = f"  seed {seed} ep {ep:3d} loss {tot / nb:.4f}"
        if args.val_every > 0 and (
            (ep + 1) % args.val_every == 0 or ep == args.epochs - 1
        ):
            _, rep = evaluate(writer, enc, eval_ds, dev, ep, args)
            msg += (
                f"  ang_err {rep.get('pos/angle_err_rad', float('nan')):.3f}"
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
    print(f"device={dev}  dt={_DT}  args={vars(args)}")

    # window=1: the encoder only ever sees one frame, so no velocity info.
    train_ds = WindowPairs(
        collect_seeded_random_rollouts(
            args.n_rollouts, args.rollout_len, args.img_size, seed=0
        ),
        window=1, k=args.k,
    )
    eval_ds = WindowPairs(
        collect_seeded_random_rollouts(
            args.eval_rollouts, args.rollout_len, args.img_size, seed=12345
        ),
        window=1, k=args.k,
    )
    print(f"train pairs {len(train_ds)}  eval frames {eval_ds.frames.shape[:2]}")

    summary = SummaryWriter(out / "tb" / "summary")
    summary.add_text("args", "```\n" + json.dumps(vars(args), indent=2) + "\n```")

    torch.manual_seed(999)
    rand_enc = Encoder(3, args.dim).to(dev)
    rand_enc(train_ds.batch(torch.arange(2))["x"].to(dev))
    q0 = embed_rollouts(rand_enc, eval_ds, dev, args.normalize)
    floor = fd_report(q0, eval_ds.states.numpy())
    results = {"random_init": floor}
    log_report(summary, floor, 0, "random_init")
    print("random-init floor:", {k: round(v, 3) for k, v in floor.items()})

    qs = []
    for seed in range(args.seeds):
        writer = SummaryWriter(out / "tb" / f"seed{seed}")
        enc = train(args, train_ds, eval_ds, seed, dev, writer)
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

    mean_final = {
        k: float(np.mean([results[f"seed{i}"][k] for i in range(args.seeds)]))
        for k in floor
    }
    summary.add_hparams(
        {k: v for k, v in vars(args).items() if k != "out"},
        {f"hparam/{k}": v for k, v in mean_final.items()},
        run_name=".",
    )
    summary.close()
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out / 'results.json'}")
    print(f"tensorboard --logdir {out / 'tb'}")


if __name__ == "__main__":
    main()
