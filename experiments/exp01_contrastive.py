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

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase_id.data import WindowPairs
from phase_id.env import collect_seeded_random_rollouts
from phase_id.losses import info_nce
from phase_id.metrics import cross_seed_consistency, identifiability_report
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


def train(args, ds, seed, dev):
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
            tot, nb = tot + loss.item(), nb + 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  seed {seed} ep {ep:3d} loss {tot / nb:.4f}", flush=True)
    return enc


def scatter(z, states, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = np.arctan2(states[:, 1], states[:, 0])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, c, name, cmap in [
        (ax[0], theta, "theta", "twilight"),
        (ax[1], states[:, 2], "theta_dot", "coolwarm"),
    ]:
        im = a.scatter(z[:, 0], z[:, 1], c=c, s=2, cmap=cmap)
        a.set_title(f"z colored by {name}")
        a.set_aspect("equal", "datalim")
        fig.colorbar(im, ax=a)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


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

    # Floor: untrained encoder.
    torch.manual_seed(999)
    rand_enc = Encoder(3 * args.window, args.dim).to(dev)
    rand_enc(train_ds.batch(torch.arange(2))["x"].to(dev))
    z0, states = embed(rand_enc, eval_ds, dev)
    results = {"random_init": identifiability_report(z0, states)}
    # Ceiling: raw state -> state is trivially 1.0; report for reference only.
    print("random-init floor:", results["random_init"])

    zs = []
    for seed in range(args.seeds):
        enc = train(args, train_ds, seed, dev)
        z, states = embed(enc, eval_ds, dev)
        zs.append(z)
        rep = identifiability_report(z, states)
        results[f"seed{seed}"] = rep
        print(f"seed {seed}:", {k: round(v, 3) for k, v in rep.items()})
        if args.dim == 2:
            scatter(z, states, out / f"latent_seed{seed}.png")

    if len(zs) > 1:
        results["cross_seed"] = cross_seed_consistency(zs)
        print("cross-seed R^2:", results["cross_seed"])

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
