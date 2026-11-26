"""B2 mirror swap (Kang Fig 4, Alg 1) on the B1 gesture scenes: is the cue's code causal?

Minimal pair inside one (scene, origin): x = the cue points at A, y = the same scene with the cue pointing at
B. Both use the same question text ("Is the {cue} pointing at the {first} or the {second}?", order
randomised once per pair) and forced prefix, so the token sequences are identical. At layer L, x's
resid_pre at a token subset is replaced by y's and we measure how far x's answer moves towards y's:
  ld_shift = (LD_x - LD_patched) / (LD_x - LD_y),  LD = lp(A) - lp(B) after the prefix
1 = the subset carries the whole effect of the pointing change, 0 = none.

Subsets: cue_word ("hand"/"arrow"), img_cue (image tokens covered >= 10% by either cue), img_objects (image
tokens wholly inside A's and B's boxes; their pixels are identical in x and y), image_all, text_all.
Controls (same layer, same positions):
  noise_cue_word, noise_img_cue  x + random vectors with the per-position norm of (y - x)  [Kang Fig 2 control]
  hand_cue_word, hand_img_cue    (photo only) source = x re-rendered with a different hand, still pointing
                                 at A: a change of hand appearance of the size the swap brings, which should
                                 move nothing
Text subsets (B2 found the pointing reaches the text at L16-24 and tags the object NAMES):
  name_A, name_B (all sub-tokens of each name in the question), names (both), prefix (the forced answer
  prefix "The {cue} is pointing at the"), last (its final token).
Steering (--steer_dirs = name_tag_dirs.pt from b2_analyse): at layer L, add s*w at B's name and -s*w at A's
  name (last sub-tokens; w = the cue's name-tag direction at L, target minus foil), for s in --steer_scales:
  steer_s<s>; control steer_noise_s<s> = random vectors of the same norm at the same positions. Use scenes the
  directions were not fitted on (b2 used 0-99).
Only pairs whose answer follows the cue (LD_x - LD_y > --min_gap) are patched; each row records whether
the model is right on x and y. Every candidate pair's LD_x, LD_y is logged ("candidates"), so the share of
pairs that move the wrong way is known. Output: <out>/<model>/gesture_swap<--tag>/<cue>_rows.json, <cue>_swap.png.
"""
import dataclasses
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from deixis import gesture as G
from deixis.common import base_parser, load_vlm, out_dir, save_json

SUBSETS = ["cue_word", "img_cue", "img_objects", "image_all", "text_all"]
TEXT_SUBSETS = ["name_A", "name_B", "names", "prefix", "last"]


def name_span(vlm, enc, name):
    """Sequence positions of all sub-tokens of `name` (first occurrence after the image)."""
    sub = vlm.tok.tokenize(" " + name)
    p = vlm.find_token(enc, name, 0)
    span = list(range(p - len(sub) + 1, p + 1))
    assert [enc.tokens[i] for i in span] == sub, (name, [enc.tokens[i] for i in span], sub)
    return span


def token_cover(vlm, enc, mask, thr=0.10):
    rows, cols = enc.grid_hw
    sx, sy = enc.scale
    m = vlm.merged_px
    out = []
    for r in range(rows):
        for c in range(cols):
            blk = mask[int(r * m / sy):int((r + 1) * m / sy), int(c * m / sx):int((c + 1) * m / sx)]
            if blk.size and blk.mean() >= thr:
                out.append(int(enc.image_pos[r * cols + c]))
    return out


def tokens_inside(vlm, enc, box):
    rows, cols = enc.grid_hw
    sx, sy = enc.scale
    m = vlm.merged_px
    x0, y0, x1, y1 = box
    return [int(enc.image_pos[r * cols + c]) for r in range(rows) for c in range(cols)
            if c * m / sx >= x0 and r * m / sy >= y0 and (c + 1) * m / sx <= x1 and (r + 1) * m / sy <= y1]


def main():
    p = base_parser(__doc__)
    p.add_argument("--kang_obj_dir", required=True)
    p.add_argument("--hand_dir", required=True)
    p.add_argument("--exclude_hands", default="")
    p.add_argument("--cue", default="arrow", choices=["photo", "cartoon", "arrow"])
    p.add_argument("--scene_start", type=int, default=0)
    p.add_argument("--scene_end", type=int, default=300)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n_pairs", type=int, default=150, help="stop after this many patched pairs")
    p.add_argument("--min_gap", type=float, default=0.5, help="min LD_x - LD_y (nats) for a pair to be patched")
    p.add_argument("--layer_stride", type=int, default=2)
    p.add_argument("--subsets", default=",".join(SUBSETS), help=f"from {SUBSETS + TEXT_SUBSETS}")
    p.add_argument("--steer_dirs", default="", help="name_tag_dirs.pt from b2_analyse (enables steering)")
    p.add_argument("--steer_scales", default="1,3")
    p.add_argument("--tag", default="", help="output dir suffix: gesture_swap<tag>")
    p.add_argument("--save_every", type=int, default=10)
    args = p.parse_args()

    vlm = load_vlm(args)
    vlm.answer_aliases = {}
    od = out_dir(args, "gesture_swap" + args.tag)
    rows_path = os.path.join(od, f"{args.cue}_rows.json")
    with open(os.path.join(args.hand_dir, "hands.json")) as f:
        hands = [h for h in sorted(json.load(f)) if h not in set(args.exclude_hands.split(","))]
    names = G.pool(args.kang_obj_dir)
    first_tok = {n: vlm.tok.encode(" " + n, add_special_tokens=False)[0] for n in names}
    last_sub = {n: vlm.tok.tokenize(" " + n)[-1] for n in names}
    layers = list(range(0, vlm.n_layers, args.layer_stride))  # resid_pre of the final norm is not patched
    subsets = args.subsets.split(",")
    cw = G.CUE_WORD[args.cue]
    gen = torch.Generator().manual_seed(args.seed)
    steer, scales = None, [float(x) for x in args.steer_scales.split(",") if x]
    if args.steer_dirs:
        sd = torch.load(args.steer_dirs, weights_only=False)
        steer = {L: sd["dirs"][args.cue][i].float() for i, L in enumerate(sd["layers"])}
    cands = []

    rows, n_pairs, n_cand = [], 0, 0
    bar = tqdm(total=args.n_pairs, desc=f"swap {args.cue}")
    for si in range(args.scene_start, args.scene_end):
        if n_pairs >= args.n_pairs:
            break
        sc = G.make_scene(si, names, hands, args.kang_obj_dir, args.hand_dir, k=args.k, seed=args.seed)
        rng = random.Random(f"{args.seed}-swap-{si}")
        other = hands[(hands.index(sc.hand) + 1) % len(hands)]
        boxes = [G.obj_box(args.kang_obj_dir, n, c) for n, c in sc.objects]
        for origin, targets in sc.valid.items():
            pairs = [(a, b) for a in targets for b in targets if a != b
                     and first_tok[sc.objects[a][0]] != first_tok[sc.objects[b][0]]
                     and last_sub[sc.objects[a][0]] != last_sub[sc.objects[b][0]]]
            if not pairs or n_pairs >= args.n_pairs:
                continue
            a, b = rng.choice(pairs)  # one pair per (scene, origin): pairs stay independent
            A, B = sc.objects[a][0], sc.objects[b][0]
            first, second = (A, B) if rng.random() < 0.5 else (B, A)
            q, pre = G.QUERY.format(cue=cw, first=first, second=second), G.PREFIX.format(cue=cw)
            ex = vlm.encode(G.render(sc, origin, a, args.cue, args.kang_obj_dir, args.hand_dir), q, prefix=pre)
            ey = vlm.encode(G.render(sc, origin, b, args.cue, args.kang_obj_dir, args.hand_dir), q, prefix=pre)
            assert ex.seq_len == ey.seq_len and torch.equal(ex.image_pos, ey.image_pos)
            n_cand += 1

            def ld(logits):
                lp = torch.log_softmax(logits.float(), -1)
                return float(lp[first_tok[A]] - lp[first_tok[B]])

            lx, cx = vlm.run_cache(ex, None, layers)  # full-sequence resid_pre [nL, seq, d]
            ly, cy = vlm.run_cache(ey, None, layers)
            ld_x, ld_y = ld(lx), ld(ly)
            cands.append({"scene": si, "origin": origin, "A": A, "B": B, "ld_x": round(ld_x, 4), "ld_y": round(ld_y, 4)})
            if not ld_x - ld_y > args.min_gap:
                continue
            o = G.ORIGINS[origin]
            ta, tb = G.cell_center(sc.objects[a][1]), G.cell_center(sc.objects[b][1])
            mask = G.cue_mask(args.cue, o, ta, sc.hand, args.hand_dir) | G.cue_mask(args.cue, o, tb, sc.hand, args.hand_dir)
            img_start = int(ex.image_pos.max()) + 1
            pos = {"cue_word": [vlm.find_token(ex, cw, 0)],
                   "img_cue": token_cover(vlm, ex, mask),
                   "img_objects": tokens_inside(vlm, ex, boxes[a]) + tokens_inside(vlm, ex, boxes[b]),
                   "image_all": ex.image_pos.tolist(),
                   "text_all": list(range(img_start, ex.seq_len))}
            nA, nB = name_span(vlm, ex, A), name_span(vlm, ex, B)
            n_pre = len(vlm.tok.tokenize(pre))
            assert ex.tokens[-n_pre:] == vlm.tok.tokenize(pre), (ex.tokens[-n_pre - 2:], pre)
            pos.update({"name_A": nA, "name_B": nB, "names": nA + nB,
                        "prefix": list(range(ex.seq_len - n_pre, ex.seq_len)), "last": [ex.last]})
            ch = None  # hand-appearance control source: x with another hand (skip if it touches an object)
            if args.cue == "photo":
                sh = dataclasses.replace(sc, hand=other)
                mh = G.cue_mask("photo", o, ta, other, args.hand_dir)
                if not any(mh[y0:y1, x0:x1].any() for (x0, y0, x1, y1) in boxes):
                    eh = vlm.encode(G.render(sh, origin, a, "photo", args.kang_obj_dir, args.hand_dir), q, prefix=pre)
                    if eh.seq_len == ex.seq_len:
                        _, ch = vlm.run_cache(eh, None, layers)
            base = {"scene": si, "origin": origin, "cue": args.cue, "A": A, "B": B, "A_cell": list(sc.objects[a][1]),
                    "B_cell": list(sc.objects[b][1]), "A_first": first == A, "ld_x": round(ld_x, 4),
                    "ld_y": round(ld_y, 4), "x_correct": ld_x > 0, "y_correct": ld_y < 0, "hand": sc.hand}

            def shift(logits):
                return round((ld_x - ld(logits)) / (ld_x - ld_y), 4)

            for li, L in enumerate(layers):
                res = {}
                for s in subsets:
                    px = torch.tensor(sorted(set(pos[s])))
                    if len(px):
                        res[s] = shift(vlm.forward(ex, patch={L: (px, cy[li][px])}))
                for s in [s_ for s_ in ("cue_word", "img_cue", "names") if s_ in subsets]:
                    px = torch.tensor(sorted(set(pos[s])))
                    if not len(px):
                        continue
                    d = cy[li][px] - cx[li][px]
                    nz = torch.randn(d.shape, generator=gen)
                    nz = nz / nz.norm(dim=-1, keepdim=True) * d.norm(dim=-1, keepdim=True)
                    res[f"noise_{s}"] = shift(vlm.forward(ex, patch={L: (px, cx[li][px] + nz)}))
                    if ch is not None:
                        res[f"hand_{s}"] = shift(vlm.forward(ex, patch={L: (px, ch[li][px])}))
                if steer is not None and L in steer:
                    px = torch.tensor([nA[-1], nB[-1]])
                    w = steer[L]
                    for sc_ in scales:
                        vec = torch.stack([-sc_ * w, sc_ * w])
                        res[f"steer_s{sc_:g}"] = shift(vlm.forward(ex, add={L: (px, vec)}))
                        nz = torch.randn(vec.shape, generator=gen)
                        nz = nz / nz.norm(dim=-1, keepdim=True) * vec.norm(dim=-1, keepdim=True)
                        res[f"steer_noise_s{sc_:g}"] = shift(vlm.forward(ex, add={L: (px, nz)}))
                rows.append({**base, "layer": L, **{f"n_{s}": len(set(v)) for s, v in pos.items()}, **res})
            n_pairs += 1
            bar.update(1)
            if args.save_every and n_pairs % args.save_every == 0:
                save_json({"rows": rows, "n_candidates": n_cand, "candidates": cands}, rows_path)
    bar.close()
    save_json({"rows": rows, "n_candidates": n_cand, "n_pairs": n_pairs, "layers": layers, "model": vlm.model_id,
               "min_gap": args.min_gap, "candidates": cands}, rows_path)
    g = np.array([c["ld_x"] - c["ld_y"] for c in cands])
    print(f"{args.cue}: {n_pairs} pairs follow the cue out of {n_cand} candidates; LD_x - LD_y > {args.min_gap}: "
          f"{np.mean(g > args.min_gap):.3f}, < -{args.min_gap} (moves the wrong way): {np.mean(g < -args.min_gap):.3f}, "
          f"median {np.median(g):.3f}")
    if not rows:
        return
    keys = [k for k in rows[0] if k in SUBSETS + TEXT_SUBSETS or k.startswith(("noise_", "hand_", "steer_"))]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in keys:
        m = [np.nanmedian([r.get(k, np.nan) for r in rows if r["layer"] == L]) for L in layers]
        ax.plot(layers, m, marker=".", ls="--" if k.startswith(("noise_", "hand_", "steer_noise")) else "-", label=k)
    ax.axhline(0, color="gray", lw=0.5)
    # Qwen3-VL DeepStack adds visual features at image positions in the first layers: image swaps there
    # leave x's injected features in place, so they understate what the image tokens carry
    ax.axvspan(-0.5, vlm.deepstack_layers - 0.5, color="gray", alpha=0.1)
    ax.set_xlabel("layer (resid_pre)")
    ax.set_ylabel("median ld_shift")
    ax.set_title(f"{args.cue}: swap pointing A -> B ({n_pairs}/{n_cand} pairs follow the cue)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(od, f"{args.cue}_swap.png"), dpi=140)
    print("saved", rows_path)


if __name__ == "__main__":
    main()
