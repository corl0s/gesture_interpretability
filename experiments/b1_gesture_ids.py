"""B1: gesture-ID extraction on Kang-style scenes with real (YouRefIt) hand cut-outs.

For each scene (K objects at fixed cells, one hand), every valid (origin, target) combination is rendered
for each cue (photo = real hand, cartoon, arrow, empty = real hand with no objects). Query
"Is the {cue} pointing at the {first} or the {second}?" (target/foil order randomised), with the answer
started as "The {cue} is pointing at the" (forced prefix), so behaviour = lp(target) - lp(foil) at the
last position. Residual stream (resid_pre, all layers) is saved at three positions:
  cue   : the cue word ("hand"/"arrow"), first occurrence; it precedes the object names, so it sees
          only the image and "Is the"
  tgt   : the target name's last sub-token in the question (first occurrence)
  ans   : the last prefix token (where the answer is produced)
Output: <out>/<model>/gesture_ids/scenes_<start>_<end>.pt with
  {"acts": fp16 [N, 3, L+1, d] (omitted with --behaviour_only), "rows": [dict per image], ...}
Analyse with b1_analyse.py.
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import torch
from tqdm import tqdm

from deixis import gesture as G
from deixis.common import base_parser, load_vlm, out_dir

POS = ("cue", "tgt", "ans")


def main():
    p = base_parser(__doc__)
    p.add_argument("--kang_obj_dir", required=True)
    p.add_argument("--hand_dir", required=True, help="dir with <hand>.png + hands.json (base/tip)")
    p.add_argument("--hands", default="", help="comma-separated hand names (default: all in hands.json)")
    p.add_argument("--exclude_hands", default="")
    p.add_argument("--scene_start", type=int, default=0)
    p.add_argument("--scene_end", type=int, default=150)
    p.add_argument("--k", type=int, default=5, help="objects per scene")
    p.add_argument("--cues", default="photo,cartoon,arrow,empty")
    p.add_argument("--behaviour_only", action="store_true", help="pilot: no activations saved")
    args = p.parse_args()

    vlm = load_vlm(args)
    vlm.answer_aliases = {}
    od = out_dir(args, "gesture_ids")
    with open(os.path.join(args.hand_dir, "hands.json")) as f:
        all_hands = sorted(json.load(f))
    hands = [h for h in (args.hands.split(",") if args.hands else all_hands) if h]
    hands = [h for h in hands if h not in set(args.exclude_hands.split(","))]
    names = G.pool(args.kang_obj_dir)
    cues = [c for c in args.cues.split(",") if c]
    geo_cues = tuple(c for c in cues if c != "empty") or ("photo",)
    if "empty" in cues and "photo" not in geo_cues:
        geo_cues = geo_cues + ("photo",)
    first_tok = {n: vlm.tok.encode(" " + n, add_special_tokens=False)[0] for n in names}
    last_sub = {n: vlm.tok.tokenize(" " + n)[-1] for n in names}

    def ok_pair(a, b):  # distinct answer tokens (behaviour) and distinct last sub-tokens (read-out position)
        return first_tok[a] != first_tok[b] and last_sub[a] != last_sub[b]

    rows, acts = [], []
    t0 = time.time()
    for si in tqdm(range(args.scene_start, args.scene_end), desc="scenes"):
        sc = G.make_scene(si, names, hands, args.kang_obj_dir, args.hand_dir, k=args.k, seed=args.seed,
                          cues=geo_cues)
        rng = random.Random(f"{args.seed}-q-{si}")
        for origin, targets in sc.valid.items():
            for t in targets:
                f = G.foil_index(sc, t, origin, rng, ok_pair)
                tname, fname = sc.objects[t][0], sc.objects[f][0]
                swap = rng.random() < 0.5
                geo = G.geometry(sc, origin, t)
                for cue in cues:
                    cw = G.CUE_WORD[cue]
                    first, second = (fname, tname) if swap else (tname, fname)
                    q = G.QUERY.format(cue=cw, first=first, second=second)
                    img = G.render(sc, origin, t, cue, args.kang_obj_dir, args.hand_dir)
                    enc = vlm.encode(img, q, prefix=G.PREFIX.format(cue=cw))
                    start = int(enc.image_pos.max()) + 1
                    pos = []
                    for word in (cw, tname):
                        tok = vlm.tok.tokenize(" " + word)[-1]
                        hits = [i for i in range(start, enc.seq_len) if enc.tokens[i] == tok]
                        if not hits:
                            raise ValueError(f"{word!r} ({tok!r}) not in prompt")
                        pos.append(hits[0])
                    pos.append(enc.last)
                    if args.behaviour_only:
                        logits = vlm.forward(enc)
                    else:
                        logits, a_ = vlm.run_cache(enc, torch.tensor(pos))  # [L+1, 3, d]
                        acts.append(a_.transpose(0, 1).half())
                    lp = torch.log_softmax(logits.float(), -1)
                    ld = float(lp[first_tok[tname]] - lp[first_tok[fname]])
                    rows.append({"scene": si, "origin": origin, "cue": cue, "hand": sc.hand,
                                 "target": tname, "target_cell": list(sc.objects[t][1]), "t_idx": t,
                                 "foil": fname, "foil_cell": list(sc.objects[f][1]), "target_first": not swap,
                                 "angle": round(geo["angle"], 2), "tip_gap": round(geo["tip_gap"], 1),
                                 "tip_nearest_is_target": geo["tip_nearest"] == t,
                                 "tip_nearest_cell": list(sc.objects[geo["tip_nearest"]][1]),
                                 "objects": [[n, list(c)] for n, c in sc.objects], "ld": round(ld, 4),
                                 "correct": ld > 0})
    out = {"rows": rows, "positions": POS, "n_layers": vlm.n_layers, "model": vlm.model_id,
           "cues": cues, "k": args.k, "seed": args.seed, "hands": hands, "len": G.LEN, "obj_px": G.OBJ_PX,
           "query": G.QUERY, "prefix": G.PREFIX, "origins": G.ORIGINS}
    if acts:
        out["acts"] = torch.stack(acts)
    tag = "pilot" if args.behaviour_only else "scenes"
    path = os.path.join(od, f"{tag}_{args.scene_start:04d}_{args.scene_end:04d}.pt")
    torch.save(out, path)
    n = len(rows)
    print(f"saved {path}: {n} images in {time.time() - t0:.0f}s")
    for cue in cues:
        r = [x for x in rows if x["cue"] == cue]
        if r and cue != "empty":
            acc = sum(x["correct"] for x in r) / len(r)
            hard = [x for x in r if not x["tip_nearest_is_target"]]
            hacc = sum(x["correct"] for x in hard) / max(1, len(hard))
            print(f"  {cue:8s} acc {acc:.3f} (n={len(r)}); tip-nearest != target: {hacc:.3f} (n={len(hard)})")


if __name__ == "__main__":
    main()
