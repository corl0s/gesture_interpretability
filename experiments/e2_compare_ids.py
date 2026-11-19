"""E2 analysis: do pointing / arrow cues produce the same spatial IDs as named objects?

For each layer:
  object self-fit   Kang Fig. 5 statistic: object IDs projected on their own (x, y) axes;
                    grid R^2 and variance explained.
  cue -> object     cue-slot IDs projected on the *object* axes (shared code test), and
                    nearest-ID accuracy: is the cue ID for cell c most similar to the
                    object ID for cell c? (chance = 1/16)
  cue self-fit      cue-slot IDs on their own axes (is there *any* linear location code?)
Correlational evidence only; causal tests are E3/E4.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deixis import ids as I
from deixis.common import save_json


def load_ids(path, slot, correct_only=False):
    d = torch.load(path, weights_only=False)
    k = d["slots"].index(slot)
    keep = list(range(len(d["locs"])))
    if correct_only and d.get("aggregated"):
        raise SystemExit(f"{path}: --correct_only needs per-image activations (file was saved with --aggregate)")
    if correct_only and d["correct"]:
        keep = [i for i in keep if d["correct"][i]]
    acts = d["acts"][keep, k].float()
    locs = [tuple(d["locs"][i]) for i in keep]
    groups = [d["groups"][i] for i in keep]
    return I.ids_by_loc(acts, locs, groups), d


def permutation_null(path, slot, obj_ids, cells, grid, n_perm, seed=0, correct_only=False):
    """Null for 'cue IDs align with object IDs': shuffle cell labels within each object pair
    (keeps group structure and the number of images per cell), recompute both statistics.
    Returns per-layer 95th percentiles."""
    d = torch.load(path, weights_only=False)
    k = d["slots"].index(slot)
    keep = [i for i in range(len(d["locs"])) if not (correct_only and d["correct"] and not d["correct"][i])]
    acts = d["acts"][keep, k].float()
    locs = [tuple(d["locs"][i]) for i in keep]
    groups = [d["groups"][i] for i in keep]
    O = I.stack_ids(obj_ids, cells)
    rng = np.random.default_rng(seed)
    r2s, nns = [], []
    for _ in range(n_perm):
        perm = list(locs)
        for g in set(groups):
            idx = [i for i, gg in enumerate(groups) if gg == g]
            sh = rng.permutation(idx)
            for i, j in zip(idx, sh):
                perm[i] = locs[j]
        ids = I.ids_by_loc(acts, perm, groups)
        if len(ids) < len(cells):
            continue
        r2s.append(I.per_layer_plane_stats(ids, obj_ids, cells, grid)["grid_r2"])
        nns.append(I.nearest_id_accuracy(I.cross_similarity(I.stack_ids(ids, cells), O)).numpy())
    return np.nanpercentile(np.array(r2s), 95, axis=0), np.nanpercentile(np.array(nns), 95, axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids_dir", required=True, help="e.g. results/qwen3vl-2b/ids")
    p.add_argument("--conditions", default="hand,arrow,hand_empty,arrow_empty")
    p.add_argument("--object_slot", default="target_word")
    p.add_argument("--correct_only", action="store_true")
    p.add_argument("--n_perm", type=int, default=0, help="label-permutation null draws (e.g. 200)")
    args = p.parse_args()

    obj_ids, od = load_ids(os.path.join(args.ids_dir, "object.pt"), args.object_slot)
    grid = od["grid"]
    cells = [(x, y) for y in range(grid) for x in range(grid)]
    res = {"object_self": {k: v.tolist() for k, v in I.per_layer_plane_stats(obj_ids, obj_ids, cells, grid).items()}}
    O = I.stack_ids(obj_ids, cells)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    L = np.arange(O.shape[1])
    axs[0].plot(L, res["object_self"]["grid_r2"], "k-", lw=2, label="object (self)")
    for cond in args.conditions.split(","):
        path = os.path.join(args.ids_dir, f"{cond}.pt")
        if not os.path.exists(path):
            continue
        slots = torch.load(path, weights_only=False)["slots"]
        for slot in slots:
            cid, _ = load_ids(path, slot, args.correct_only)
            if len(cid) < len(cells):
                continue
            to_obj = I.per_layer_plane_stats(cid, obj_ids, cells, grid)
            self_fit = I.per_layer_plane_stats(cid, cid, cells, grid)
            nn_acc = I.nearest_id_accuracy(I.cross_similarity(I.stack_ids(cid, cells), O)).numpy()
            res[f"{cond}/{slot}"] = {"grid_r2_on_object_axes": to_obj["grid_r2"].tolist(),
                                     "grid_r2_self": self_fit["grid_r2"].tolist(),
                                     "nearest_object_id_acc": nn_acc.tolist()}
            if args.n_perm:
                r95, n95 = permutation_null(path, slot, obj_ids, cells, grid, args.n_perm, correct_only=args.correct_only)
                res[f"{cond}/{slot}"].update({"null95_grid_r2_on_object_axes": r95.tolist(),
                                              "null95_nearest_object_id_acc": n95.tolist()})
            lab = f"{cond}/{slot}"
            axs[0].plot(L, to_obj["grid_r2"], label=lab)
            axs[1].plot(L, nn_acc, label=lab)
            axs[2].plot(L, self_fit["grid_r2"], label=lab)
    axs[0].set_title("grid R² on OBJECT axes (shared code)")
    axs[1].set_title("nearest object-ID accuracy (chance 1/16)")
    axs[1].axhline(1 / len(cells), color="gray", ls=":")
    axs[2].set_title("grid R² on OWN axes (any linear code)")
    for a in axs:
        a.set_xlabel("layer (resid_pre)")
        a.axvspan(-0.5, od["deepstack_layers"] - 0.5, color="gray", alpha=0.1)
    axs[0].legend(fontsize=6)
    plt.tight_layout()
    tag = "_correct" if args.correct_only else ""
    plt.savefig(os.path.join(args.ids_dir, f"compare{tag}.png"), dpi=140)
    save_json(res, os.path.join(args.ids_dir, f"compare{tag}.json"))

    # Kang Fig. 5-style scatter at the layer where object IDs are most grid-like
    best = int(np.nanargmax(res["object_self"]["grid_r2"]))
    xa, ya = I.axes(obj_ids, grid)
    fig, ax = plt.subplots(figsize=(5, 5))
    c, _ = I.project_onto_plane(O[:, best].numpy(), xa[best].numpy(), ya[best].numpy())
    ax.scatter(c[:, 0], c[:, 1], c="k", s=30, label="object")
    for i, (x, y) in enumerate(cells):
        ax.annotate(f"{x},{y}", c[i], fontsize=7)
    for cond in args.conditions.split(","):
        path = os.path.join(args.ids_dir, f"{cond}.pt")
        if os.path.exists(path):
            cid, _ = load_ids(path, "last", args.correct_only)
            if len(cid) == len(cells):
                cc, _ = I.project_onto_plane(I.stack_ids(cid, cells)[:, best].numpy(), xa[best].numpy(), ya[best].numpy())
                ax.scatter(cc[:, 0], cc[:, 1], s=18, label=f"{cond}/last")
    ax.set_title(f"projection on object axes, layer {best}")
    ax.legend(fontsize=7)
    plt.savefig(os.path.join(args.ids_dir, f"scatter_L{best}{tag}.png"), dpi=140)
    print("best object layer", best, "R2", res["object_self"]["grid_r2"][best])


if __name__ == "__main__":
    main()
