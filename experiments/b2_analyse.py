"""B2 analysis: is the pointed-at object tagged, and where does the gesture's information live?

Input: <out>/<model>/gesture_b2/scenes_*.pt from b1_gesture_ids.py --b2 (positions cue, tgt, foil, ans,
obj0..obj{k-1}, img_cue, img_tip; every --layer_stride-th layer). Per cue and saved layer:

1. Pointed tag on the object's image tokens (obj_tag_*). For object j of a scene seen from one origin, the
   pixels of its tokens are identical whichever object the cue points at (cues never touch object boxes),
   so (scene, origin, object) groups differ only in where the cue points. Activations are demeaned within
   those groups; the tag direction w = mean(pointed) - mean(not pointed) is fitted on half the scenes and
   tested on the other half (2-fold by scene parity):
     obj_tag_acc        the pointed object has the highest score w.x among the group's candidate targets
                        (chance obj_tag_chance)
     obj_tag_over_tipn  on images where another candidate is nearest the fingertip: score(pointed) >
                        score(fingertip-nearest) (0.5 = no preference)
     obj_tag_behav_r    Pearson r between score(target) - score(foil) and the model's LD (target - foil)
     obj_tag_behav_agree  sign(score diff) == sign(LD): the tag predicts the model's choice, right or wrong
2. Pointed tag on the name tokens (name_tag_*): the same object's name token when it is the target vs the
   foil, demeaned within (scene, origin, object); held-out score(tgt token) > score(foil token)
   (chance .5) and the same behaviour link.
3. B1 metrics (decode within / cross-origin, held-out variance for cell / angle / cell x origin /
   fingertip-nearest cell, pointed over fingertip-nearest) at the text and cue-image positions.
   For img_tip / img_cue, note the tokens themselves move with the pointing direction, so a direction
   code there can come from token location alone; cross-origin target decoding is the informative number.
Writes b2_stats.json in --ids_dir.
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from b1_analyse import CIDX, ORIGIN_NAMES, decode, demean, fixed_effects, heldout_r2  # noqa: E402

B1_POS = ("cue", "tgt", "ans", "img_cue", "img_tip")
SIDE_A = {"bot_l", "bot_m", "bot_r", "left_hi", "left_lo"}


def load(ids_dir):
    files = sorted(glob.glob(os.path.join(ids_dir, "scenes_*.pt")))
    if not files:
        raise SystemExit(f"no scenes_*.pt in {ids_dir}")
    chunks = [torch.load(f, weights_only=False) for f in files]
    return sum([c["rows"] for c in chunks], []), torch.cat([c["acts"] for c in chunks]), chunks[0]


def groups(keys):
    d = {}
    return np.array([d.setdefault(k, len(d)) for k in keys])


def tag_scores(X, grp, lab, train):
    """Demean within groups, fit w = mean(lab=1) - mean(lab=0) on train rows; return scores for all rows."""
    Xc = demean(X, grp)
    tr = torch.as_tensor(train)
    l1, l0 = torch.as_tensor(lab == 1), torch.as_tensor(lab == 0)
    w = Xc[tr & l1].mean(0) - Xc[tr & l0].mean(0)
    return (Xc @ (w / w.norm().clamp(min=1e-8))).numpy()


def object_tag(acts, R, P, L_i, k):
    """Section 1 for one cue / layer. R: rows of this cue, acts: [n, P, nL, d]."""
    feats, img, obj, lab, gk = [], [], [], [], []
    cand = defaultdict(set)  # candidate targets per (scene, origin)
    for x in R:
        cand[(x["scene"], x["origin"])].add(x["t_idx"])
    for i, x in enumerate(R):
        for j in range(k):
            if j not in cand[(x["scene"], x["origin"])]:
                continue
            feats.append(acts[i, P[f"obj{j}"], L_i].float())
            img.append(i)
            obj.append(j)
            lab.append(int(j == x["t_idx"]))
            gk.append((x["scene"], x["origin"], j))
    X, img, obj, lab = torch.stack(feats), np.array(img), np.array(obj), np.array(lab)
    grp = groups(gk)
    scene = np.array([R[i]["scene"] for i in img])
    out = defaultdict(list)
    for f in (0, 1):
        s = tag_scores(X, grp, lab, scene % 2 != f)
        test_imgs = sorted(set(img[scene % 2 == f]))
        by_img = defaultdict(dict)
        for r in np.nonzero(scene % 2 == f)[0]:
            by_img[img[r]][obj[r]] = s[r]
        for i in test_imgs:
            x, sc = R[i], by_img[i]
            out["acc"].append(max(sc, key=sc.get) == x["t_idx"])
            out["chance"].append(1 / len(sc))
            tn = [j for j, (n, c) in enumerate(x["objects"]) if list(c) == x["tip_nearest_cell"]]
            if tn and tn[0] != x["t_idx"] and tn[0] in sc:
                out["over_tipn"].append(sc[x["t_idx"]] > sc[tn[0]])
            if x.get("f_idx") in sc:
                out["diff"].append(sc[x["t_idx"]] - sc[x["f_idx"]])
                out["ld"].append(x["ld"])
    return summarise(out)


def name_tag(acts, R, P, L_i):
    feats, lab, gk, img = [], [], [], []
    for i, x in enumerate(R):
        for pos, j, l_ in (("tgt", x["t_idx"], 1), ("foil", x["f_idx"], 0)):
            feats.append(acts[i, P[pos], L_i].float())
            lab.append(l_)
            gk.append((x["scene"], x["origin"], j))
            img.append(i)
    X, lab, img = torch.stack(feats), np.array(lab), np.array(img)
    grp = groups(gk)
    # keep groups where the object is named both as target and as foil (else demeaning leaves nothing)
    both = {g for g in np.unique(grp) if len(set(lab[grp == g])) == 2}
    keep = np.array([g in both for g in grp])
    X, lab, img, grp = X[torch.as_tensor(keep)], lab[keep], img[keep], groups(grp[keep])
    scene = np.array([R[i]["scene"] for i in img])
    out = defaultdict(list)
    for f in (0, 1):
        s = tag_scores(X, grp, lab, scene % 2 != f)
        te = np.nonzero(scene % 2 == f)[0]
        st = {img[r]: s[r] for r in te if lab[r] == 1}
        sf = {img[r]: s[r] for r in te if lab[r] == 0}
        for i in set(st) & set(sf):
            out["acc"].append(st[i] > sf[i])
            out["diff"].append(st[i] - sf[i])
            out["ld"].append(R[i]["ld"])
    out["chance"] = [0.5]
    return summarise(out)


def summarise(out):
    r = {"acc": float(np.mean(out["acc"])), "chance": float(np.mean(out["chance"])), "n": len(out["acc"])}
    if out.get("over_tipn"):
        r["over_tipn"] = float(np.mean(out["over_tipn"]))
        r["n_over_tipn"] = len(out["over_tipn"])
    if len(out.get("diff", [])) > 2:
        d, ld = np.array(out["diff"]), np.array(out["ld"])
        r["behav_r"] = float(np.corrcoef(d, ld)[0, 1])
        r["behav_agree"] = float(np.mean(np.sign(d) == np.sign(ld)))
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}


def b1_metrics(X, R):
    grp = groups([(x["scene"], x["origin"]) for x in R])
    cell = np.array([CIDX[tuple(x["target_cell"])] for x in R])
    ang = np.array([int(((x["angle"] + 11.25) % 360) // 22.5) for x in R])
    scene = np.array([x["scene"] for x in R])
    inA = np.array([x["origin"] in SIDE_A for x in R])
    cxo = np.array([CIDX[tuple(x["target_cell"])] * len(ORIGIN_NAMES) + ORIGIN_NAMES.index(x["origin"]) for x in R])
    tipc = np.array([CIDX[tuple(x["tip_nearest_cell"])] for x in R])
    differs = np.array([not x["tip_nearest_is_target"] for x in R])
    cand_by_g = defaultdict(set)
    for g, c in zip(grp, cell):
        cand_by_g[g].add(int(c))
    cand = [sorted(cand_by_g[g]) for g in grp]
    Xc = demean(X, grp)
    fold = scene % 2
    cr, wi, pt, v = [], [], [], defaultdict(list)
    for tr, te in ((inA, ~inA), (~inA, inA)):
        cr.append(decode(Xc[te], [cand[i] for i in np.nonzero(te)[0]], cell[te], fixed_effects(X[tr], grp[tr], cell[tr])))
    for f in (0, 1):
        tr, te = fold != f, fold == f
        Ct = fixed_effects(X[tr], grp[tr], cell[tr])
        wi.append(decode(Xc[te], [cand[i] for i in np.nonzero(te)[0]], cell[te], Ct))
        m = te & differs
        if m.sum():
            Cn = torch.nn.functional.normalize(torch.nan_to_num(Ct), dim=1)
            S = torch.nn.functional.normalize(Xc[m], dim=1) @ Cn.T
            ii = torch.arange(int(m.sum()))
            pt.append(float((S[ii, torch.as_tensor(cell[m])] > S[ii, torch.as_tensor(tipc[m])]).float().mean()))
        v["var_cell"].append(heldout_r2(X, grp, cell, tr, te, 16))
        v["var_angle"].append(heldout_r2(X, grp, ang, tr, te, 16))
        v["var_cell_x_origin"].append(heldout_r2(X, grp, cxo, tr, te, 16 * len(ORIGIN_NAMES)))
        v["var_tip_nearest"].append(heldout_r2(X, grp, tipc, tr, te, 16))
    r = {"decode_within": np.mean([a for a, _ in wi]), "decode_cross_origin": np.mean([a for a, _ in cr]),
         "decode_chance": np.mean([c for _, c in wi]), "pointed_over_tipnearest": np.mean(pt) if pt else None,
         **{k: np.mean(x) for k, x in v.items()}}
    return {k: None if x is None else round(float(x), 4) for k, x in r.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids_dir", required=True)
    p.add_argument("--report_layers", default="10,16,20,24,30", help="also printed besides each best layer")
    args, _ = p.parse_known_args()
    rows, acts, meta = load(args.ids_dir)
    P = {s: i for i, s in enumerate(meta["positions"])}
    layers, k = meta["layers"], meta["k"]
    res = {"model": meta["model"], "n_images": len(rows), "layers": layers, "cues": {}}
    show = [int(x) for x in args.report_layers.split(",") if x]
    for cue in meta["cues"]:
        idx = [i for i, x in enumerate(rows) if x["cue"] == cue]
        R, A = [rows[i] for i in idx], acts[idx]
        out = {"obj_tag": [], "name_tag": [], **{pos: [] for pos in B1_POS}}
        for L_i, L in enumerate(layers):
            out["obj_tag"].append(object_tag(A, R, P, L_i, k))
            out["name_tag"].append(name_tag(A, R, P, L_i))
            for pos in B1_POS:
                out[pos].append(b1_metrics(A[:, P[pos], L_i].float(), R))
        res["cues"][cue] = out
        for key in ("obj_tag", "name_tag"):
            accs = [d["acc"] for d in out[key]]
            b = int(np.argmax(accs))
            for L_i in sorted({b} | {layers.index(L) for L in show if L in layers}):
                print(f"[{cue:7s} {key:8s}] L{layers[L_i]:2d}{'*' if L_i == b else ' '} {out[key][L_i]}")
        for pos in B1_POS:
            dec = [d["decode_cross_origin"] for d in out[pos]]
            b = int(np.argmax(dec))
            for L_i in sorted({b} | {layers.index(L) for L in show if L in layers}):
                print(f"[{cue:7s} {pos:8s}] L{layers[L_i]:2d}{'*' if L_i == b else ' '} {out[pos][L_i]}")
    import json
    with open(os.path.join(args.ids_dir, "b2_stats.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved", os.path.join(args.ids_dir, "b2_stats.json"))


if __name__ == "__main__":
    main()
