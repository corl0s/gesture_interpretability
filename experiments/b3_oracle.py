"""B3 oracle injection (Kang App. A.8): does adding the correct "pointed at" tag fix the answers?

On single images of the B1 scenes (use scenes the tag directions were not fitted on; b2 used 0-99), add
+s*w at the target name's last sub-token and -s*w at the foil name's (w = the cue's name-tag direction at
layer L, from b2_analyse's name_tag_dirs.pt) and score the forced-prefix answer, LD = lp(target) - lp(foil):
  oracle_L<L>_s<s>   accuracy with the correct tag injected
  noise_L<L>_s<s>    random vectors of the same norm at the same two positions (control)
  anti_L<L>_s<s>     the tag injected the wrong way round (target -w, foil +w): it should break the answer
Also reported: order-balanced accuracy, accuracy when the fingertip-nearest object is not the target, and
pick-first-named, for the baseline and each condition. Output: <out>/<model>/gesture_oracle/<cue>.json.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from deixis import gesture as G
from deixis.common import base_parser, load_vlm, out_dir, save_json


def summary(rs, key):
    ld = np.array([r[key] for r in rs])
    tf = np.array([r["target_first"] for r in rs])
    hard = np.array([not r["tip_nearest_is_target"] for r in rs])
    c = ld > 0
    return {"acc": round(float(c.mean()), 4),
            "acc_order_balanced": round(float((c[tf].mean() + c[~tf].mean()) / 2), 4) if tf.any() and (~tf).any() else None,
            "acc_tip_nearest_not_target": round(float(c[hard].mean()), 4) if hard.any() else None,
            "pick_first_named": round(float((c == tf).mean()), 4), "median_ld": round(float(np.median(ld)), 4)}


def main():
    p = base_parser(__doc__)
    p.add_argument("--kang_obj_dir", required=True)
    p.add_argument("--hand_dir", required=True)
    p.add_argument("--exclude_hands", default="")
    p.add_argument("--cue", default="arrow", choices=["photo", "cartoon", "arrow"])
    p.add_argument("--scene_start", type=int, default=100)
    p.add_argument("--scene_end", type=int, default=200)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--steer_dirs", required=True, help="name_tag_dirs.pt from b2_analyse")
    p.add_argument("--layers", default="20,22,24")
    p.add_argument("--scales", default="1,3,5")
    p.add_argument("--save_every", type=int, default=10, help="checkpoint every n scenes")
    args = p.parse_args()

    vlm = load_vlm(args)
    vlm.answer_aliases = {}
    od = out_dir(args, "gesture_oracle")
    path = os.path.join(od, f"{args.cue}.json")
    with open(os.path.join(args.hand_dir, "hands.json")) as f:
        hands = [h for h in sorted(json.load(f)) if h not in set(args.exclude_hands.split(","))]
    names = G.pool(args.kang_obj_dir)
    first_tok = {n: vlm.tok.encode(" " + n, add_special_tokens=False)[0] for n in names}
    last_sub = {n: vlm.tok.tokenize(" " + n)[-1] for n in names}

    def ok_pair(a, b):
        return first_tok[a] != first_tok[b] and last_sub[a] != last_sub[b]

    sd = torch.load(args.steer_dirs, weights_only=False)
    W = {L: sd["dirs"][args.cue][i].float() for i, L in enumerate(sd["layers"])}
    layers = [int(x) for x in args.layers.split(",") if x]
    missing = [L for L in layers if L not in W]
    if missing:
        raise SystemExit(f"no tag direction at layers {missing}; available {sorted(W)}")
    scales = [float(x) for x in args.scales.split(",") if x]
    cw = G.CUE_WORD[args.cue]
    gen = torch.Generator().manual_seed(args.seed)
    rows = []
    for n_sc, si in enumerate(tqdm(range(args.scene_start, args.scene_end), desc=f"oracle {args.cue}")):
        sc = G.make_scene(si, names, hands, args.kang_obj_dir, args.hand_dir, k=args.k, seed=args.seed)
        rng = random.Random(f"{args.seed}-q-{si}")  # same target/foil/order draws as b1_gesture_ids
        for origin, targets in sc.valid.items():
            for t in targets:
                f = G.foil_index(sc, t, origin, rng, ok_pair)
                T, F = sc.objects[t][0], sc.objects[f][0]
                swap = rng.random() < 0.5
                first, second = (F, T) if swap else (T, F)
                geo = G.geometry(sc, origin, t)
                enc = vlm.encode(G.render(sc, origin, t, args.cue, args.kang_obj_dir, args.hand_dir),
                                 G.QUERY.format(cue=cw, first=first, second=second), prefix=G.PREFIX.format(cue=cw))
                px = torch.tensor([vlm.find_token(enc, T, 0), vlm.find_token(enc, F, 0)])

                def ld(logits):
                    lp = torch.log_softmax(logits.float(), -1)
                    return round(float(lp[first_tok[T]] - lp[first_tok[F]]), 4)

                r = {"scene": si, "origin": origin, "target": T, "foil": F, "target_first": not swap,
                     "tip_nearest_is_target": geo["tip_nearest"] == t, "hand": sc.hand, "base": ld(vlm.forward(enc))}
                for L in layers:
                    w = W[L]
                    for s in scales:
                        vec = torch.stack([s * w, -s * w])
                        r[f"oracle_L{L}_s{s:g}"] = ld(vlm.forward(enc, add={L: (px, vec)}))
                        r[f"anti_L{L}_s{s:g}"] = ld(vlm.forward(enc, add={L: (px, -vec)}))
                        nz = torch.randn(vec.shape, generator=gen)
                        nz = nz / nz.norm(dim=-1, keepdim=True) * vec.norm(dim=-1, keepdim=True)
                        r[f"noise_L{L}_s{s:g}"] = ld(vlm.forward(enc, add={L: (px, nz)}))
                rows.append(r)
        if args.save_every and (n_sc + 1) % args.save_every == 0:
            save_json({"rows": rows}, path)
    keys = [k for k in rows[0] if k == "base" or k.startswith(("oracle_", "anti_", "noise_"))]
    summ = {k: summary(rows, k) for k in keys}
    save_json({"rows": rows, "summary": summ, "cue": args.cue, "model": vlm.model_id, "layers": layers,
               "scales": scales, "n": len(rows)}, path)
    print(f"{args.cue}: n={len(rows)}")
    for k in keys:
        print(f"  {k:18s} {summ[k]}")


if __name__ == "__main__":
    main()
