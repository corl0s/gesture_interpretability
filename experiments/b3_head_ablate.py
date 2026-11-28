"""B3 head ablation: are the heads that look at the target (b3_attention) needed for the answer?

Heads are chosen from a b3_attention file (default: the arrow) on scenes 100-160 and ablated on fresh scenes
(default 200-260). A head is ablated by zeroing its output (the input of o_proj) at all text positions after
the image. Sets:
  name_readers  top --n_heads heads by order-balanced acc for at2/is/last attending target name > foil name
  obj_lookers   top --n_heads heads for at1/at2/last attending the target's picture > the foil's
  both          union
  rand_<set>_<i> controls: the same number of heads per layer, drawn at random from the other heads (--n_rand)
Reports acc, order-balanced acc, acc when tip-nearest != target, pick-first-named and median LD per condition.
Output: <out>/<model>/gesture_ablate/<cue>.json.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm

from b3_attn_analyse import bal_acc, comparisons  # noqa: E402
from b3_oracle import summary  # noqa: E402
from deixis import gesture as G  # noqa: E402
from deixis.common import base_parser, load_vlm, out_dir, save_json  # noqa: E402

NAME_CMP = ("at2: tgt_name>foil_name", "is: tgt_name>foil_name", "last: tgt_name>foil_name")
OBJ_CMP = ("at1: tgt_obj>foil_obj", "at2: tgt_obj>foil_obj", "last: tgt_obj>foil_obj")


def top_heads(attn_file, cmps, n):
    d = torch.load(attn_file, weights_only=False)
    a = d["att"].float().numpy()
    tf = np.array([r["target_first"] for r in d["rows"]])
    c = comparisons(a, list(d["queries"]), list(d["keys"]))
    best = np.max(np.stack([bal_acc(c[k], tf) for k in cmps]), 0)  # [nL, H]: best over the comparisons
    order = np.argsort(-best, axis=None)[:n]
    return sorted((int(L), int(h)) for L, h in zip(*np.unravel_index(order, best.shape))), best


def main():
    p = base_parser(__doc__)
    p.add_argument("--kang_obj_dir", required=True)
    p.add_argument("--hand_dir", required=True)
    p.add_argument("--exclude_hands", default="")
    p.add_argument("--cue", default="arrow", choices=["photo", "cartoon", "arrow"])
    p.add_argument("--scene_start", type=int, default=200)
    p.add_argument("--scene_end", type=int, default=260)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--attn_file", required=True, help="b3_attention output used to pick the heads")
    p.add_argument("--n_heads", type=int, default=10)
    p.add_argument("--n_rand", type=int, default=3)
    p.add_argument("--save_every", type=int, default=20)
    args = p.parse_args()

    sets = {}
    sets["name_readers"], best_n = top_heads(args.attn_file, NAME_CMP, args.n_heads)
    sets["obj_lookers"], best_o = top_heads(args.attn_file, OBJ_CMP, args.n_heads)
    sets["both"] = sorted(set(sets["name_readers"]) | set(sets["obj_lookers"]))
    nL, H = best_n.shape
    rr = random.Random(args.seed)
    for s in ("name_readers", "obj_lookers"):
        chosen = set(sets["both"])
        for i in range(args.n_rand):
            rs = []
            for L, _ in sets[s]:  # same layer, random other head
                h = rr.choice([h for h in range(H) if (L, h) not in chosen and (L, h) not in rs])
                rs.append((L, h))
            sets[f"rand_{s}_{i}"] = sorted(rs)
    print({k: v for k, v in sets.items()})

    vlm = load_vlm(args)
    lm = vlm.model.model.language_model
    n_heads_cfg = lm.config.num_attention_heads
    cur = {"heads": [], "pos": None}

    def make(L):
        def pre(module, a):
            hs = [h for LL, h in cur["heads"] if LL == L]
            if not hs:
                return None
            x = a[0].clone()
            dh = x.shape[-1] // n_heads_cfg
            for h in hs:
                x[0, cur["pos"], h * dh:(h + 1) * dh] = 0
            return (x,) + tuple(a[1:])
        return pre

    handles = [lm.layers[L].self_attn.o_proj.register_forward_pre_hook(make(L)) for L in range(vlm.n_layers)]
    od = out_dir(args, "gesture_ablate")
    path = os.path.join(od, f"{args.cue}.json")
    with open(os.path.join(args.hand_dir, "hands.json")) as f:
        hands = [h for h in sorted(json.load(f)) if h not in set(args.exclude_hands.split(","))]
    names = G.pool(args.kang_obj_dir)
    first_tok = {n: vlm.tok.encode(" " + n, add_special_tokens=False)[0] for n in names}
    last_sub = {n: vlm.tok.tokenize(" " + n)[-1] for n in names}

    def ok_pair(a, b):
        return first_tok[a] != first_tok[b] and last_sub[a] != last_sub[b]

    cw = G.CUE_WORD[args.cue]
    rows = []
    meta = {"cue": args.cue, "model": vlm.model_id, "attn_file": args.attn_file,
            "sets": {k: [list(x) for x in v] for k, v in sets.items()}}
    for n_sc, si in enumerate(tqdm(range(args.scene_start, args.scene_end), desc=f"ablate {args.cue}")):
        sc = G.make_scene(si, names, hands, args.kang_obj_dir, args.hand_dir, k=args.k, seed=args.seed)
        rng = random.Random(f"{args.seed}-q-{si}")
        for origin, targets in sc.valid.items():
            for t in targets:
                f = G.foil_index(sc, t, origin, rng, ok_pair)
                T, F = sc.objects[t][0], sc.objects[f][0]
                swap = rng.random() < 0.5
                first, second = (F, T) if swap else (T, F)
                geo = G.geometry(sc, origin, t)
                enc = vlm.encode(G.render(sc, origin, t, args.cue, args.kang_obj_dir, args.hand_dir),
                                 G.QUERY.format(cue=cw, first=first, second=second), prefix=G.PREFIX.format(cue=cw))
                cur["pos"] = vlm.text_positions(enc).to(vlm.device)
                r = {"scene": si, "origin": origin, "target": T, "foil": F, "target_first": not swap,
                     "tip_nearest_is_target": geo["tip_nearest"] == t, "hand": sc.hand}
                for cond, hs in [("base", [])] + list(sets.items()):
                    cur["heads"] = hs
                    lp = torch.log_softmax(vlm.forward(enc).float(), -1)
                    r[cond] = round(float(lp[first_tok[T]] - lp[first_tok[F]]), 4)
                cur["heads"] = []
                rows.append(r)
        if args.save_every and (n_sc + 1) % args.save_every == 0:
            save_json({"rows": rows, **meta}, path)
    for h in handles:
        h.remove()
    keys = ["base"] + list(sets)
    summ = {k: summary(rows, k) for k in keys}
    base = np.array([r["base"] for r in rows])
    for k in keys:  # mean drop of LD (target - foil) vs no ablation: > 0 = the ablation hurts the target
        summ[k]["mean_ld_drop"] = round(float(np.mean(base - np.array([r[k] for r in rows]))), 4)
    save_json({"rows": rows, "summary": summ, "n": len(rows), **meta}, path)
    print(f"{args.cue}: n={len(rows)}")
    for k in keys:
        print(f"  {k:22s} {summ[k]}")


if __name__ == "__main__":
    main()
