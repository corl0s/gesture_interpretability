"""Analysis of b3_attention: which heads route the pointed-at object, and from where?

For each comparison (query q, key a vs key b, or query a vs query b on one key) the per-image score is
att(.., a) - att(.., b) for one (layer, head); "acc" is the order-balanced fraction of images with score > 0
(mean over target-named-first and target-named-second), so a head that just prefers the first-named object
scores .5. Comparisons:
  <q>: tgt_name>foil_name   for q in at2, is, last  (does the prefix read the target's name?)
  <q>: tgt_obj>foil_obj     for q in at1, at2, last, cue_q (does it look at the target object in the image?)
  name->img_cue             tgt_name->img_cue vs foil_name->img_cue (does the target's name look at the hand?)
  name->at1                 tgt_name->at1 vs foil_name->at1
  name->own_obj             tgt_name->tgt_obj vs tgt_name->foil_obj (name-to-own-image binding, sanity)
Head selection without peeking: the best (layer, head) is picked on even scenes and scored on odd scenes, and
the reverse (mean reported as "sel"); "top" is the best head on all scenes (descriptive). Also: first-named
preference of the selected head, Spearman of its score with the answer LD on the held-out half, and the mean
attention mass per layer for a few (query, key) pairs. Writes attn_stats.json next to the inputs.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def bal_acc(s, tf):
    """s [N, ...] scores; order-balanced accuracy over axis 0 (ties count half)."""
    hit = (s > 0).astype(float) + 0.5 * (s == 0)
    return (hit[tf].mean(0) + hit[~tf].mean(0)) / 2


def comparisons(a, Q, K):
    q, k = Q.index, K.index
    c = {}
    for qq in ("at2", "is", "last"):
        c[f"{qq}: tgt_name>foil_name"] = a[:, :, :, q(qq), k("tgt_name")] - a[:, :, :, q(qq), k("foil_name")]
    for qq in ("at1", "cue_q", "at2", "last"):
        c[f"{qq}: tgt_obj>foil_obj"] = a[:, :, :, q(qq), k("tgt_obj")] - a[:, :, :, q(qq), k("foil_obj")]
    c["name->img_cue"] = a[:, :, :, q("tgt_name"), k("img_cue")] - a[:, :, :, q("foil_name"), k("img_cue")]
    c["name->at1"] = a[:, :, :, q("tgt_name"), k("at1")] - a[:, :, :, q("foil_name"), k("at1")]
    c["name->cue_q"] = a[:, :, :, q("tgt_name"), k("cue_q")] - a[:, :, :, q("foil_name"), k("cue_q")]
    c["name->own_obj"] = a[:, :, :, q("tgt_name"), k("tgt_obj")] - a[:, :, :, q("tgt_name"), k("foil_obj")]
    return c


def analyse(d):
    a = d["att"].float().numpy()  # [N, nL, H, nQ, nK]
    Q, K, R = list(d["queries"]), list(d["keys"]), d["rows"]
    tf = np.array([r["target_first"] for r in R])
    ev = np.array([r["scene"] % 2 == 0 for r in R])
    ld = np.array([r["ld"] for r in R])
    nL, H = a.shape[1], a.shape[2]
    out = {"n": len(R), "n_layers": nL, "n_heads": H, "comparisons": {}, "mass": {}}
    for name, s in comparisons(a, Q, K).items():
        acc_all = bal_acc(s, tf)  # [nL, H]
        sel, rho, pfn = [], [], []
        for fit, tst in ((ev, ~ev), (~ev, ev)):
            acc_f = bal_acc(s[fit], tf[fit])
            L, h = np.unravel_index(int(np.argmax(acc_f)), acc_f.shape)
            st = s[tst][:, L, h]
            sel.append({"layer": int(L), "head": int(h), "acc_test": round(float(bal_acc(st, tf[tst])), 4)})
            rho.append(spearman(st, ld[tst]))
            # first-named preference: score of first-named minus second-named
            first_minus_second = np.where(tf[tst], st, -st)
            pfn.append(float((first_minus_second > 0).mean()))
        L, h = np.unravel_index(int(np.argmax(acc_all)), acc_all.shape)
        out["comparisons"][name] = {
            "sel": sel, "sel_acc": round(float(np.mean([x["acc_test"] for x in sel])), 4),
            "sel_spearman_ld": round(float(np.nanmean(rho)), 4), "sel_first_named": round(float(np.mean(pfn)), 4),
            "top": {"layer": int(L), "head": int(h), "acc": round(float(acc_all[L, h]), 4)},
            "head_mean_acc_by_layer": [round(float(x), 4) for x in bal_acc(s.mean(2), tf)],
            "best_head_acc_by_layer": [round(float(x), 4) for x in acc_all.max(1)],
            "n_heads_over_.65_by_layer": [int(x) for x in (acc_all > 0.65).sum(1)]}
    for qq, kk in (("at1", "img_cue"), ("at1", "image"), ("at2", "img_cue"), ("at2", "tgt_name"),
                   ("at2", "foil_name"), ("at2", "at1"), ("last", "at1"), ("last", "img_cue"),
                   ("tgt_name", "img_cue"), ("tgt_name", "at1"), ("cue_q", "img_cue")):
        out["mass"][f"{qq}->{kk}"] = [round(float(x), 4) for x in a[:, :, :, Q.index(qq), K.index(kk)].mean((0, 2))]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attn_dir", required=True)
    p.add_argument("--show_layers", default="")
    args = p.parse_args()
    res = {}
    for f in sorted(glob.glob(os.path.join(args.attn_dir, "*.pt"))):
        d = torch.load(f, weights_only=False)
        r = analyse(d)
        res[d["cue"]] = r
        nL = r["n_layers"]
        show = [int(x) for x in args.show_layers.split(",") if x] or list(range(0, nL, 2))
        print(f"\n[{d['cue']}] n={r['n']} {d['model']}  (layers shown: {show})")
        print(f"  {'comparison':26s} {'sel acc':>7s} {'(L,h)':>14s} {'rho_ld':>6s} {'1st':>5s} | best-head acc by layer")
        for name, c in r["comparisons"].items():
            lh = "/".join(f"{x['layer']},{x['head']}" for x in c["sel"])
            print(f"  {name:26s} {c['sel_acc']:7.3f} {lh:>14s} {c['sel_spearman_ld']:6.2f} {c['sel_first_named']:5.2f} | "
                  + " ".join(f"{c['best_head_acc_by_layer'][L]:.2f}" for L in show))
        print("  mean mass by layer:")
        for k_, v in r["mass"].items():
            print(f"    {k_:22s} " + " ".join(f"{v[L]:.3f}" for L in show))
    with open(os.path.join(args.attn_dir, "attn_stats.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved", os.path.join(args.attn_dir, "attn_stats.json"))


if __name__ == "__main__":
    main()
