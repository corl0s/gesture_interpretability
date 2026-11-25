"""K1 analysis: Kang's universal IDs, axes and grid fit (Fig. 5), with uncertainty.

Loads every <ids_dir>/pairs_*.pt chunk from k1_kang_ids.py and computes:
  universal IDs  = mean over pairs of per-pair IDs (average_id.py)        -> universal.pt
  axes           = Kang compute_axes at size 224                          -> axes.pt
  grid R^2       = affine fit from the 2-D projection to true (x, y), per layer and per size,
                   with a 95% bootstrap CI over pairs (resample pairs, recompute universal IDs)
  var_explained  = fraction of ID variance in the 2-D axis plane
  null95         = 95th percentile of grid R^2 when cell labels are shuffled within pairs
Writes grid_stats.json and a grid scatter at the best layer.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from deixis import ids as I
from deixis.kang import GRID, kang_axes


def load(ids_dir):
    chunks = [torch.load(p, weights_only=False) for p in sorted(glob.glob(os.path.join(ids_dir, "pairs_*.pt")))]
    if not chunks:
        raise SystemExit(f"no pairs_*.pt in {ids_dir}")
    locs = chunks[0]["locs"]
    X = torch.cat([c["ids"] for c in chunks]).float()  # [P, n_locs, L+1, d]
    pairs = sum([list(c["pairs"]) for c in chunks], [])
    acc = sum([list(c["acc"]) for c in chunks], [])
    return X, locs, pairs, acc, chunks[0]


def universal(X, locs):
    U = X.mean(0)  # [n_locs, L+1, d]
    return {tuple(l): U[i] for i, l in enumerate(locs)}


def grid_stats(ids, size):
    cells = [(x, y) for x in range(GRID) for y in range(GRID)]
    xa, ya = kang_axes(ids, size=224)
    Q = torch.stack([ids[(x, y, size)] for x, y in cells])  # [16, L+1, d]
    r2, ev = [], []
    for L in range(Q.shape[1]):
        c, e = I.project_onto_plane(Q[:, L].numpy(), xa[L].numpy(), ya[L].numpy())
        r2.append(I.grid_r2(c, cells))
        ev.append(float(e))
    return np.array(r2), np.array(ev)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids_dir", required=True)
    p.add_argument("--n_boot", type=int, default=200)
    p.add_argument("--n_perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args, _ = p.parse_known_args()  # tolerate --model/--out from scc_experiment.qsub
    rng = np.random.default_rng(args.seed)
    X, locs, pairs, acc, meta = load(args.ids_dir)
    P = X.shape[0]
    U = universal(X, locs)
    torch.save({"universal": U, "locs": locs, "pairs": pairs, "model": meta["model"]}, os.path.join(args.ids_dir, "universal.pt"))
    xa, ya = kang_axes(U, 224)
    torch.save({"x": xa, "y": ya}, os.path.join(args.ids_dir, "axes.pt"))

    sizes = sorted({l[2] for l in locs})
    res = {"model": meta["model"], "n_pairs": P, "behaviour_acc_mean": float(np.mean(acc)), "sizes": {}}
    for s in sizes:
        r2, ev = grid_stats(U, s)
        boots = []
        for _ in range(args.n_boot):
            idx = torch.from_numpy(rng.integers(0, P, P))
            boots.append(grid_stats(universal(X[idx], locs), s)[0])
        boots = np.array(boots)
        nulls = []
        for _ in range(args.n_perm):  # shuffle cell labels within each pair (same perm for all sizes)
            perm = rng.permutation(16)
            Xp = X.clone()
            cell_idx = {(l[0], l[1]): [] for l in locs}
            for i, l in enumerate(locs):
                cell_idx[(l[0], l[1])].append((i, l[2]))
            keys = sorted(cell_idx)
            for j in range(P):
                pj = rng.permutation(16)
                src = {keys[k]: keys[pj[k]] for k in range(16)}
                for c, items in cell_idx.items():
                    for i, sz in items:
                        i2 = locs.index((src[c][0], src[c][1], sz))
                        Xp[j, i] = X[j, i2]
            nulls.append(grid_stats(universal(Xp, locs), s)[0])
        nulls = np.array(nulls)
        res["sizes"][str(s)] = {
            "grid_r2": np.round(r2, 4).tolist(),
            "grid_r2_ci_lo": np.round(np.nanpercentile(boots, 2.5, axis=0), 4).tolist(),
            "grid_r2_ci_hi": np.round(np.nanpercentile(boots, 97.5, axis=0), 4).tolist(),
            "null95": np.round(np.nanpercentile(nulls, 95, axis=0), 4).tolist(),
            "var_explained": np.round(ev, 4).tolist(),
        }
        best = int(np.nanargmax(r2))
        print(f"size {s}: best layer {best} R2 {r2[best]:.3f} [{res['sizes'][str(s)]['grid_r2_ci_lo'][best]:.3f}, "
              f"{res['sizes'][str(s)]['grid_r2_ci_hi'][best]:.3f}] null95 {res['sizes'][str(s)]['null95'][best]:.3f}")
    with open(os.path.join(args.ids_dir, "grid_stats.json"), "w") as f:
        json.dump(res, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r2 = np.array(res["sizes"]["224"]["grid_r2"])
    best = int(np.nanargmax(r2))
    cells = [(x, y) for x in range(GRID) for y in range(GRID)]
    Q = torch.stack([U[(x, y, 224)] for x, y in cells])[:, best].numpy()
    c, _ = I.project_onto_plane(Q, xa[best].numpy(), ya[best].numpy())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(c[:, 0], c[:, 1], c=[x for x, _ in cells], cmap="viridis", s=80)
    for (x, y), (u, v) in zip(cells, c):
        ax.annotate(f"{x},{y}", (u, v), fontsize=8)
    ax.set_title(f"{os.path.basename(meta['model'])} universal IDs, layer {best} (size 224)")
    plt.savefig(os.path.join(args.ids_dir, f"grid_L{best}.png"), dpi=140)
    print("saved", args.ids_dir)


if __name__ == "__main__":
    main()
