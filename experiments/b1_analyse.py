"""B1 analysis: gesture IDs (per pointed cell), grid geometry, target-vs-angle tests, link to spatial IDs.

Per cue (photo / cartoon / arrow / empty), read-out position (cue / tgt / ans) and layer:
  gesture IDs   two-way fixed effects: x = g[(scene, origin)] + c[pointed cell] + noise, solved by
                alternating demeaning (= Kang's centring when the design is balanced; our design is not,
                because some (origin, target) combos are invalid). IDs are the cell effects c.
  grid_r2       Kang axes on the 16 cell IDs, affine fit of the 2-D projection to the true cells (as K1),
                with a permutation null (targets shuffled within each (scene, origin) group).
  cross-origin  IDs estimated on origin set A predict the pointed target among each group's candidates
                on origin set B (and B->A), by cosine to the group-centred activation; compared with a
                within-set split by scene. Target code: cross ~ within. Arm-angle code: cross << within.
  var_target / var_angle / var_cell_x_origin
                held-out (2-fold over scenes) fraction of within-(scene, origin) variance explained by
                pointed-cell effects, by arm-angle-bin effects (16 bins of 22.5 deg) and by cell x origin
                effects. Target code: cell ~ cell x origin > angle. Arm-direction code: angle and
                cell x origin > cell (the cell effect depends on where the arm comes from).
  spatial link  (optional --spatial_ids = K1 universal.pt of the same model) cosine between gesture ID
                and spatial ID of the same cell; 16-way nearest-ID accuracy (chance 1/16).
Behaviour: accuracy per cue, per origin, and when the fingertip-nearest object is NOT the target.
Writes gesture_stats.json and gesture_ids.pt in --ids_dir.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from deixis import ids as I
from deixis.kang import GRID, kang_axes

CELLS = [(x, y) for x in range(GRID) for y in range(GRID)]
CIDX = {c: i for i, c in enumerate(CELLS)}
ORIGIN_NAMES = ["bot_l", "bot_m", "bot_r", "left_hi", "left_lo", "right_hi", "right_lo", "top_l", "top_m", "top_r"]


def load(ids_dir):
    files = sorted(glob.glob(os.path.join(ids_dir, "scenes_*.pt")))
    if not files:
        raise SystemExit(f"no scenes_*.pt in {ids_dir}")
    chunks = [torch.load(f, weights_only=False) for f in files]
    rows = sum([c["rows"] for c in chunks], [])
    acts = torch.cat([c["acts"] for c in chunks])  # [N, 3, L+1, d] fp16
    return rows, acts, chunks[0]


def fixed_effects(X, grp, cell, n_iter=15, K=16):
    """X [n, d] float; grp, cell int arrays. Returns label effects [K, d] (NaN for unseen labels)."""
    grp_t, cell_t = torch.as_tensor(grp), torch.as_tensor(cell)
    ng = int(grp_t.max()) + 1
    gcount = torch.bincount(grp_t, minlength=ng).clamp(min=1).float()[:, None]
    ccount = torch.bincount(cell_t, minlength=K).float()[:, None]
    C = torch.zeros(K, X.shape[1])
    for _ in range(n_iter):
        Gm = torch.zeros(ng, X.shape[1]).index_add_(0, grp_t, X - C[cell_t]) / gcount
        C = torch.zeros(K, X.shape[1]).index_add_(0, cell_t, X - Gm[grp_t]) / ccount.clamp(min=1)
        C = C - (C * ccount).sum(0) / ccount.sum()  # weighted sum-to-zero
    C[ccount[:, 0] == 0] = float("nan")
    return C


def heldout_r2(X, grp, lab, train, test, K):
    """Fraction of within-group variance on `test` explained by label effects fitted on `train`
    (two-way fixed effects); prediction = label effects demeaned within each test group."""
    E = torch.nan_to_num(fixed_effects(X[train], grp[train], lab[train], K=K))
    Xc = demean(X[test], grp[test])
    P = demean(E[torch.as_tensor(lab[test])], grp[test])
    return float(1 - ((Xc - P) ** 2).sum() / (Xc ** 2).sum())


def demean(X, grp):
    grp_t = torch.as_tensor(grp)
    ng = int(grp_t.max()) + 1
    cnt = torch.bincount(grp_t, minlength=ng).clamp(min=1).float()[:, None]
    return X - (torch.zeros(ng, X.shape[1]).index_add_(0, grp_t, X) / cnt)[grp_t]


def grid_r2(C):
    if torch.isnan(C).any():
        return float("nan")
    ids = {(x, y, 0): C[CIDX[(x, y)]][None] for (x, y) in CELLS}
    xa, ya = kang_axes(ids, size=0)
    c, _ = I.project_onto_plane(C.numpy(), xa[0].numpy(), ya[0].numpy())
    return float(I.grid_r2(c, CELLS))


def decode(Xc, cand, true, C):
    """Group-centred activations Xc [n, d]; cand: list of candidate cell-index lists; true: cell idx."""
    Cn = torch.nn.functional.normalize(torch.nan_to_num(C), dim=1)
    Xn = torch.nn.functional.normalize(Xc, dim=1)
    S = Xn @ Cn.T
    hit = [int(max(cs, key=lambda k: S[i, k]) == true[i]) for i, cs in enumerate(cand)]
    return float(np.mean(hit)), float(np.mean([1 / len(cs) for cs in cand]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids_dir", required=True)
    p.add_argument("--spatial_ids", default="", help="K1 universal.pt of the same model (optional)")
    p.add_argument("--n_perm", type=int, default=50)
    p.add_argument("--layers", default="", help="comma list; default all")
    p.add_argument("--seed", type=int, default=0)
    args, _ = p.parse_known_args()
    rng = np.random.default_rng(args.seed)
    rows, acts, meta = load(args.ids_dir)
    n_layers = meta["n_layers"]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(1, n_layers + 1))
    positions = list(meta["positions"])
    spatial = None
    if args.spatial_ids:
        U = torch.load(args.spatial_ids, weights_only=False)["universal"]
        spatial = {(x, y): U[(x, y, 224)].float() for (x, y) in CELLS}  # [L+1, d]

    res = {"model": meta["model"], "n_images": len(rows), "behaviour": {}, "ids": {}}
    # ---------------- behaviour
    for cue in meta["cues"]:
        if cue == "empty":
            continue
        r = [x for x in rows if x["cue"] == cue]
        by_o = defaultdict(list)
        for x in r:
            by_o[x["origin"]].append(x["correct"])
        hard = [x["correct"] for x in r if not x["tip_nearest_is_target"]]
        res["behaviour"][cue] = {"acc": float(np.mean([x["correct"] for x in r])), "n": len(r),
                                 "acc_tip_nearest_not_target": float(np.mean(hard)) if hard else None,
                                 "n_tip_nearest_not_target": len(hard),
                                 "acc_by_origin": {o: float(np.mean(v)) for o, v in sorted(by_o.items())}}
        print(f"[behaviour] {cue}: {res['behaviour'][cue]['acc']:.3f}  hard {res['behaviour'][cue]['acc_tip_nearest_not_target']}")

    # opposite sides, so a cell is seen from roughly opposite arm angles in A and B
    side_A = {"bot_l", "bot_m", "bot_r", "left_hi", "left_lo"}
    saved = {}
    for cue in meta["cues"]:
        idx = [i for i, x in enumerate(rows) if x["cue"] == cue]
        R = [rows[i] for i in idx]
        gkey = {}
        grp = np.array([gkey.setdefault((x["scene"], x["origin"]), len(gkey)) for x in R])
        cell = np.array([CIDX[tuple(x["target_cell"])] for x in R])
        ang = np.array([int(((x["angle"] + 11.25) % 360) // 22.5) for x in R])
        scene = np.array([x["scene"] for x in R])
        inA = np.array([x["origin"] in side_A for x in R])
        cxo = np.array([CIDX[tuple(x["target_cell"])] * len(ORIGIN_NAMES) + ORIGIN_NAMES.index(x["origin"]) for x in R])
        cand_by_g = defaultdict(set)
        for g, c in zip(grp, cell):
            cand_by_g[g].add(int(c))
        cand = [sorted(cand_by_g[g]) for g in grp]
        fold = np.array([s % 2 for s in scene])
        res["ids"][cue] = {}
        saved[cue] = {}
        for pi, pos in enumerate(positions):
            out = defaultdict(list)
            ids_pos = []
            for L in layers:
                X = acts[idx, pi, L].float()
                C = fixed_effects(X, grp, cell)
                ids_pos.append(C)
                out["grid_r2"].append(grid_r2(C))
                # permutation null: shuffle targets within groups
                null = []
                for _ in range(args.n_perm if L in layers[:: max(1, len(layers) // 8)] else 0):
                    pc = cell.copy()
                    for g in np.unique(grp):
                        m = np.nonzero(grp == g)[0]
                        pc[m] = pc[m][rng.permutation(len(m))]
                    null.append(grid_r2(fixed_effects(X, grp, pc)))
                out["null95"].append(float(np.nanpercentile(null, 95)) if null else None)
                Xc = demean(X, grp)
                # cross-origin vs within (split by scene) decoding
                cr = []
                for tr, te in ((inA, ~inA), (~inA, inA)):
                    if tr.sum() and te.sum():
                        Ct = fixed_effects(X[tr], grp[tr], cell[tr])
                        cr.append(decode(Xc[te], [cand[i] for i in np.nonzero(te)[0]], cell[te], Ct))
                wi = []
                for f in (0, 1):
                    tr, te = fold != f, fold == f
                    if tr.sum() and te.sum():
                        Ct = fixed_effects(X[tr], grp[tr], cell[tr])
                        wi.append(decode(Xc[te], [cand[i] for i in np.nonzero(te)[0]], cell[te], Ct))
                out["decode_cross_origin"].append(float(np.mean([a for a, _ in cr])) if cr else None)
                out["decode_within"].append(float(np.mean([a for a, _ in wi])) if wi else None)
                out["decode_chance"].append(float(np.mean([c for _, c in wi])) if wi else None)
                # held-out within-group variance explained by target cell vs arm-angle bin (2-fold over scenes)
                vt, va, vx = [], [], []
                for f in (0, 1):
                    tr, te = fold != f, fold == f
                    if tr.sum() and te.sum():
                        vt.append(heldout_r2(X, grp, cell, tr, te, 16))
                        va.append(heldout_r2(X, grp, ang, tr, te, 16))
                        vx.append(heldout_r2(X, grp, cxo, tr, te, 16 * len(ORIGIN_NAMES)))
                out["var_target"].append(float(np.mean(vt)) if vt else None)
                out["var_angle"].append(float(np.mean(va)) if va else None)
                out["var_cell_x_origin"].append(float(np.mean(vx)) if vx else None)
                if spatial is not None:
                    Sp = torch.stack([spatial[c][L] for c in CELLS])
                    ok = ~torch.isnan(C).any(1)
                    sim = torch.nn.functional.normalize(C[ok], dim=1) @ torch.nn.functional.normalize(Sp, dim=1).T
                    tr_i = torch.nonzero(ok)[:, 0]
                    out["spatial_diag_cos"].append(float(sim[torch.arange(len(tr_i)), tr_i].mean()))
                    out["spatial_nearest_acc"].append(float((sim.argmax(1) == tr_i).float().mean()))
            res["ids"][cue][pos] = {"layers": layers, **{k: [None if v is None else round(v, 4) for v in vs]
                                                         for k, vs in out.items()}}
            saved[cue][pos] = torch.stack(ids_pos)  # [n_layers_used, 16, d]
            r2 = np.array([np.nan if v is None else v for v in out["grid_r2"]], float)
            b = int(np.nanargmax(r2)) if np.isfinite(r2).any() else 0
            print(f"[{cue:7s} {pos:3s}] best L{layers[b]} grid R2 {r2[b]:.3f} | decode within "
                  f"{out['decode_within'][b]} cross-origin {out['decode_cross_origin'][b]} chance "
                  f"{out['decode_chance'][b]} | held-out var: cell {out['var_target'][b]} angle {out['var_angle'][b]} "
                  f"cell x origin {out['var_cell_x_origin'][b]}")
    torch.save({"ids": saved, "cells": CELLS, "layers": layers, "model": meta["model"]},
               os.path.join(args.ids_dir, "gesture_ids.pt"))
    with open(os.path.join(args.ids_dir, "gesture_stats.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved", args.ids_dir)


if __name__ == "__main__":
    main()
