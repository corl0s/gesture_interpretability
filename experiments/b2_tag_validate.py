"""Validation of the name tag (B2): does the "pointed at" tag on object names generalise?

The tag: for the same object in the same (scene, origin), its name's last sub-token when it is the target vs
when it is the foil; activations demeaned within (scene, origin, object); direction w = mean(target) -
mean(foil) on a fit set; on a test set, an image is scored correct when w . x(target name) > w . x(foil name)
(chance .5). Every split keeps fit and test scenes disjoint.

  scene        fit on even scenes, test on odd, and the reverse (as b2_analyse)
  prereg       the layer is chosen on one scene half (highest fit-half accuracy, with a nested split inside
               that half) and accuracy reported on the other half: no after-the-fact layer choice
  hands        fit on scenes with one half of the hands, test on scenes with the other half (and reverse):
               does the tag survive a change of hand appearance?
  cross-cue    fit on cue a (even scenes), test on cue b (odd scenes), and the reverse: is it one gesture
               code shared by real hand, cartoon and arrow?
  CI           95% bootstrap over test scenes for the scene split and the prereg accuracy
Writes tag_validation.json in --ids_dir.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from b1_analyse import demean  # noqa: E402


def load(ids_dir):
    files = sorted(glob.glob(os.path.join(ids_dir, "scenes_*.pt")))
    chunks = [torch.load(f, weights_only=False) for f in files]
    return sum([c["rows"] for c in chunks], []), torch.cat([c["acts"] for c in chunks]), chunks[0]


def build(A, R, P, L_i):
    """Rows: two per image (target name, foil name). Groups with both labels only."""
    X, lab, key, img = [], [], [], []
    for i, x in enumerate(R):
        for pos, j, l_ in (("tgt", x["t_idx"], 1), ("foil", x["f_idx"], 0)):
            X.append(A[i, P[pos], L_i].float())
            lab.append(l_)
            key.append((x["scene"], x["origin"], j))
            img.append(i)
    lab, img = np.array(lab), np.array(img)
    labs = {}
    for k_, l_ in zip(key, lab):
        labs.setdefault(k_, set()).add(int(l_))
    keep = np.array([len(labs[k_]) == 2 for k_ in key])
    X = torch.stack(X)[torch.as_tensor(keep)]
    key = [k_ for k_, m in zip(key, keep) if m]
    d = {}
    grp = np.array([d.setdefault(k_, len(d)) for k_ in key])
    img = img[keep]
    return {"X": X, "lab": lab[keep], "grp": grp, "img": img,
            "scene": np.array([R[i]["scene"] for i in img]), "hand": np.array([R[i]["hand"] for i in img])}


def sub(D, m):
    m_t = torch.as_tensor(m)
    g = D["grp"][m]
    d = {}
    return {"X": D["X"][m_t], "lab": D["lab"][m], "grp": np.array([d.setdefault(v, len(d)) for v in g]),
            "img": D["img"][m], "scene": D["scene"][m], "hand": D["hand"][m]}


def fit(D):
    Xc = demean(D["X"], D["grp"])
    return Xc[torch.as_tensor(D["lab"] == 1)].mean(0) - Xc[torch.as_tensor(D["lab"] == 0)].mean(0)


def test(D, w):
    """Per-image hits (target name scores above foil name) and the image's scene."""
    s = (demean(D["X"], D["grp"]) @ w).numpy()
    st, sf, sc = {}, {}, {}
    for r in range(len(s)):
        (st if D["lab"][r] == 1 else sf)[D["img"][r]] = s[r]
        sc[D["img"][r]] = D["scene"][r]
    ids = sorted(set(st) & set(sf))
    return (np.array([float(st[i] > sf[i]) + 0.5 * float(st[i] == sf[i]) for i in ids]),  # ties count half
            np.array([sc[i] for i in ids]))


def boot_ci(hit, scene, rng, n=1000):
    us = np.unique(scene)
    by = {u: hit[scene == u] for u in us}
    vals = [np.mean(np.concatenate([by[u] for u in rng.choice(us, len(us))])) for _ in range(n)]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def two_way(D, ma, mb):
    """Fit on ma test on mb and the reverse; pooled hits and scenes (empty when a side has no rows)."""
    hs, ss = [], []
    for f_, t_ in ((ma, mb), (mb, ma)):
        if f_.sum() and t_.sum():
            h, s_ = test(sub(D, t_), fit(sub(D, f_)))
            hs.append(h)
            ss.append(s_)
    return (np.concatenate(hs), np.concatenate(ss)) if hs else (np.array([np.nan]), np.array([-1]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    args, _ = p.parse_known_args()
    rng = np.random.default_rng(args.seed)
    rows, acts, meta = load(args.ids_dir)
    P = {s: i for i, s in enumerate(meta["positions"])}
    layers = meta["layers"]
    cues = [c for c in meta["cues"]]
    by_cue = {c: [i for i, x in enumerate(rows) if x["cue"] == c] for c in cues}
    hands = sorted({x["hand"] for x in rows})
    half = set(hands[::2])
    res = {"model": meta["model"], "layers": layers, "hands_half_A": sorted(half), "cues": {}, "cross_cue": {}}

    data = {}  # (cue, L_i) -> built rows
    for c in cues:
        R, A = [rows[i] for i in by_cue[c]], acts[by_cue[c]]
        out = {"scene_acc": [], "hands_acc": [], "fold_acc": []}
        for L_i in range(len(layers)):
            D = build(A, R, P, L_i)
            data[(c, L_i)] = D
            ev = D["scene"] % 2 == 0
            h, _ = two_way(D, ev, ~ev)
            out["scene_acc"].append(round(float(h.mean()), 4))
            inA = np.array([hh in half for hh in D["hand"]])
            h, _ = two_way(D, inA, ~inA)
            out["hands_acc"].append(round(float(h.mean()), 4))
            # nested accuracy inside each scene half (for the pre-registered layer choice)
            fa = []
            for f in (0, 1):
                m = D["scene"] % 2 == f
                q = (D["scene"] // 2) % 2 == 0
                hh, _ = two_way(D, m & q, m & ~q)
                fa.append(float(hh.mean()))
            out["fold_acc"].append([round(v, 4) for v in fa])
        # pre-registered layer: choose on half f (nested), report on half 1-f (fit on f, test on 1-f)
        pre = []
        for f in (0, 1):
            v = np.array([fa[f] for fa in out["fold_acc"]], float)
            L_i = int(np.nanargmax(v)) if np.isfinite(v).any() else len(layers) // 2
            D = data[(c, L_i)]
            mf = D["scene"] % 2 == f
            hit, sc = test(sub(D, ~mf), fit(sub(D, mf)))
            pre.append({"chosen_on_half": f, "layer": layers[L_i], "acc": round(float(hit.mean()), 4),
                        "ci95": boot_ci(hit, sc, rng), "n": len(hit)})
        b = int(np.nanargmax(out["scene_acc"]))
        D = data[(c, b)]
        ev = D["scene"] % 2 == 0
        hit, sc = two_way(D, ev, ~ev)
        out["best_scene"] = {"layer": layers[b], "acc": out["scene_acc"][b], "ci95": boot_ci(hit, sc, rng)}
        out["prereg"] = pre
        bh = int(np.nanargmax(out["hands_acc"])) if np.isfinite(out["hands_acc"]).any() else b
        out["best_hands"] = {"layer": layers[bh], "acc": out["hands_acc"][bh]}
        res["cues"][c] = out
        print(f"[{c:7s}] scene split best L{layers[b]} {out['best_scene']} | prereg {pre} | "
              f"held-out hands best L{layers[bh]} {out['hands_acc'][bh]} (at scene-best layer {out['hands_acc'][b]})")

    for a in cues:
        for b_ in cues:
            if a == b_:
                continue
            accs = []
            for L_i in range(len(layers)):
                Da, Db = data[(a, L_i)], data[(b_, L_i)]
                h1, _ = test(sub(Db, Db["scene"] % 2 == 1), fit(sub(Da, Da["scene"] % 2 == 0)))
                h2, _ = test(sub(Db, Db["scene"] % 2 == 0), fit(sub(Da, Da["scene"] % 2 == 1)))
                accs.append(round(float(np.concatenate([h1, h2]).mean()), 4))
            bi = int(np.argmax(accs))
            res["cross_cue"][f"{a}->{b_}"] = {"acc": accs, "best_layer": layers[bi], "best": accs[bi]}
            print(f"[cross-cue {a}->{b_}] best L{layers[bi]} {accs[bi]}  " +
                  " ".join(f"L{layers[i]}:{accs[i]}" for i in range(len(layers)) if layers[i] in (16, 20, 22, 24, 26)))
    with open(os.path.join(args.ids_dir, "tag_validation.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved", os.path.join(args.ids_dir, "tag_validation.json"))


if __name__ == "__main__":
    main()
