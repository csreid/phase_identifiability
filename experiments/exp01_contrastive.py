"""Exp 01: contrastive recovery of (theta, theta_dot) from pixels.

Trains N seeds of encoder with InfoNCE on temporal positive pairs, then
reports probe R^2 (both directions), cross-seed latent consistency, a
random-encoder floor, and latent scatter plots.
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

from phase_id.data import WindowPairs
from phase_id.env import collect_seeded_random_rollouts
from phase_id.losses import info_nce
from phase_id.metrics import (
    cross_seed_consistency,
    hamiltonian,
    identifiability_report,
)
from phase_id.models import ActionPredictor, Encoder


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--n-rollouts", type=int, default=2000)
    p.add_argument("--eval-rollouts", type=int, default=300)
    p.add_argument("--rollout-len", type=int, default=20)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--action-cond", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--out", type=str, default="runs/exp01")
    return p.parse_args()


@torch.no_grad()
def embed(enc, ds, dev):
    enc.eval()
    zs, ss = [], []
    for x, s in ds.all_windows():
        zs.append(enc(x.to(dev)).cpu())
        ss.append(s)
    return torch.cat(zs).numpy(), torch.cat(ss).numpy()


def log_report(writer, rep, step, prefix="probe"):
    for k, v in rep.items():
        writer.add_scalar(f"{prefix}/{k}", v, step)


def evaluate(writer, enc, eval_ds, dev, step, args, final=False):
    z, states = embed(enc, eval_ds, dev)
    rep = identifiability_report(z, states)
    log_report(writer, rep, step, "final" if final else "probe")
    writer.add_histogram("val/z", z, step)
    for d in range(z.shape[1]):
        writer.add_scalar(f"val/z_std/dim{d}", float(z[:, d].std()), step)
    if z.shape[1] >= 2:
        fig = latent_figure(z, states)
        writer.add_figure("val/latent", fig, step)
        plt.close(fig)
    return z, states, rep


def train(args, ds, eval_ds, seed, dev, writer):
    torch.manual_seed(seed)
    enc = Encoder(3 * args.window, args.dim).to(dev)
    pred = ActionPredictor(args.dim, args.k).to(dev) if args.action_cond else None
    # materialise LazyLinear before building the optimiser
    enc(ds.batch(torch.arange(2))["x"].to(dev))
    params = list(enc.parameters()) + (list(pred.parameters()) if pred else [])
    opt = torch.optim.AdamW(params, lr=args.lr)
    steps = args.epochs * (len(ds) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    g = torch.Generator().manual_seed(seed)
    step = 0
    for ep in range(args.epochs):
        enc.train()
        perm = torch.randperm(len(ds), generator=g)
        tot, nb = 0.0, 0
        for s in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            b = ds.batch(perm[s : s + args.batch_size])
            z = enc(b["x"].to(dev))
            z_next = enc(b["x_next"].to(dev))
            anchor = pred(z, b["a"].to(dev)) if pred else z
            loss = info_nce(anchor, z_next, args.tau, args.normalize)
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
            _, _, rep = evaluate(writer, enc, eval_ds, dev, ep, args)
            msg += (
                f"  z->s {rep['z->s mlp']:.3f}"
                f"  z->H {rep['z->H mlp']:.3f}"
                f"  z->thd {rep['z->theta_dot mlp']:.3f}"
                f"  rank {rep['eff_rank']:.2f}"
            )
        print(msg, flush=True)
    return enc


def latent_figure(z, states):
    """z (first two dims) colored by theta, theta_dot, and H."""
    theta = np.arctan2(states[:, 1], states[:, 0])
    panels = [
        (theta, "theta", "twilight"),
        (states[:, 2], "theta_dot", "coolwarm"),
        (hamiltonian(states)[:, 0], "H", "viridis"),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, (c, name, cmap) in zip(ax, panels):
        im = a.scatter(z[:, 0], z[:, 1], c=c, s=2, cmap=cmap)
        a.set_title(f"z colored by {name}")
        a.set_aspect("equal", "datalim")
        fig.colorbar(im, ax=a)
    fig.tight_layout()
    return fig


def coverage_figure(ds):
    st = ds.states.reshape(-1, 3).numpy()
    theta = np.arctan2(st[:, 1], st[:, 0])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(theta, st[:, 2], s=2, alpha=0.25, linewidths=0)
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("theta_dot (rad/s)")
    ax.set_title(f"Training phase-space coverage (N={len(st):,})")
    fig.tight_layout()
    return fig


def main():
    args = parse()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={dev}  args={vars(args)}")

    train_ds = WindowPairs(
        collect_seeded_random_rollouts(
            args.n_rollouts, args.rollout_len, args.img_size, seed=0
        ),
        args.window, args.k,
    )
    eval_ds = WindowPairs(
        collect_seeded_random_rollouts(
            args.eval_rollouts, args.rollout_len, args.img_size, seed=12345
        ),
        args.window, args.k,
    )
    print(f"train pairs {len(train_ds)}  eval windows {len(eval_ds)}")

    summary = SummaryWriter(out / "tb" / "summary")
    summary.add_text("args", "```\n" + json.dumps(vars(args), indent=2) + "\n```")
    fig = coverage_figure(train_ds)
    summary.add_figure("data/phase_space_coverage", fig, 0)
    plt.close(fig)

    # Floor: untrained encoder.
    torch.manual_seed(999)
    rand_enc = Encoder(3 * args.window, args.dim).to(dev)
    rand_enc(train_ds.batch(torch.arange(2))["x"].to(dev))
    z0, states = embed(rand_enc, eval_ds, dev)
    floor = identifiability_report(z0, states)
    results = {"random_init": floor}
    log_report(summary, floor, 0, "random_init")
    print("random-init floor:", {k: round(v, 3) for k, v in floor.items()})

    zs = []
    for seed in range(args.seeds):
        writer = SummaryWriter(out / "tb" / f"seed{seed}")
        enc = train(args, train_ds, eval_ds, seed, dev, writer)
        z, states, rep = evaluate(
            writer, enc, eval_ds, dev, args.epochs, args, final=True
        )
        zs.append(z)
        results[f"seed{seed}"] = rep
        print(f"seed {seed}:", {k: round(v, 3) for k, v in rep.items()})
        writer.close()

    if len(zs) > 1:
        cs = cross_seed_consistency(zs)
        results["cross_seed"] = cs
        for k, v in cs.items():
            summary.add_scalar(f"cross_seed/{k}", v, 0)
        print("cross-seed R^2:", cs)

    keys = [k for k in floor if k != "H->z mlp"]
    mean_final = {
        k: float(np.mean([results[f"seed{i}"][k] for i in range(args.seeds)]))
        for k in keys
    }
    hp = {k: v for k, v in vars(args).items() if k != "out"}
    summary.add_hparams(
        hp, {f"hparam/{k}": v for k, v in mean_final.items()}, run_name="."
    )
    summary.close()

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out / 'results.json'}")
    print(f"tensorboard --logdir {out / 'tb'}")


if __name__ == "__main__":
    main()
