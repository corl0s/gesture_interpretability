"""B3 attention: where do the tokens that carry the pointing (the token map: " at"#1, " at"#2, last) look?

Single images of the B1 scenes (same target/foil/order draws as b1_gesture_ids and b3_oracle). Text layers run
with eager attention; at every layer, for every head, the attention mass from each query position to each key
group is stored (sum of the softmax weights over the group's positions).
  queries: cue_q (cue word in the question), at1 (question " at"), tgt_name / foil_name (last sub-token),
           is (prefix " is"), at2 (prefix " at"), last
  keys:    tgt_name / foil_name (all sub-tokens), at1, cue_q, tgt_obj / foil_obj (image tokens inside the
           object boxes), other_obj, img_cue (tokens >= 10% covered by the cue), img_tip, image (all image
           tokens), pre (everything before the image: system prompt, attention sink)
Note at1 comes before the names, so it can only see the image. Output: <out>/<model>/gesture_attn/<cue>.pt
with att [N, n_layers, n_heads, n_queries, n_keys] fp16 and rows. Analysis: b3_attn_analyse.py.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from tqdm import tqdm

from b1_gesture_ids import b2_slots  # noqa: E402
from deixis import gesture as G  # noqa: E402
from deixis.common import base_parser, load_vlm, out_dir  # noqa: E402

QUERIES = ("cue_q", "at1", "tgt_name", "foil_name", "is", "at2", "last")
KEYS = ("tgt_name", "foil_name", "at1", "cue_q", "tgt_obj", "foil_obj", "other_obj", "img_cue", "img_tip",
        "image", "pre")


def name_span(vlm, enc, name):
    sub = vlm.tok.tokenize(" " + name)
    p = vlm.find_token(enc, name, 0)
    span = list(range(p - len(sub) + 1, p + 1))
    assert [enc.tokens[i] for i in span] == sub, (name, [enc.tokens[i] for i in span], sub)
    return span


def main():
    p = base_parser(__doc__)
    p.add_argument("--kang_obj_dir", required=True)
    p.add_argument("--hand_dir", required=True)
    p.add_argument("--exclude_hands", default="")
    p.add_argument("--cue", default="arrow", choices=["photo", "cartoon", "arrow"])
    p.add_argument("--scene_start", type=int, default=100)
    p.add_argument("--scene_end", type=int, default=160)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--save_every", type=int, default=20)
    args = p.parse_args()

    vlm = load_vlm(args)
    lm = vlm.model.model.language_model
    lm.config._attn_implementation = "eager"  # text layers only (the vision tower keeps its own setting)
    store = {}

    def make(L):
        def hook(module, inp, out):
            w = out[1]
            assert w is not None, "attention weights not returned (eager not active)"
            store[L] = w[0, :, store["q"], :].float() @ store["M"]  # [H, nQ, nK]
        return hook

    handles = [lm.layers[L].self_attn.register_forward_hook(make(L)) for L in range(vlm.n_layers)]
    od = out_dir(args, "gesture_attn")
    path = os.path.join(od, f"{args.cue}.pt")
    with open(os.path.join(args.hand_dir, "hands.json")) as f:
        hands = [h for h in sorted(json.load(f)) if h not in set(args.exclude_hands.split(","))]
    names = G.pool(args.kang_obj_dir)
    first_tok = {n: vlm.tok.encode(" " + n, add_special_tokens=False)[0] for n in names}
    last_sub = {n: vlm.tok.tokenize(" " + n)[-1] for n in names}

    def ok_pair(a, b):
        return first_tok[a] != first_tok[b] and last_sub[a] != last_sub[b]

    cw = G.CUE_WORD[args.cue]
    rows, att = [], []

    def save(final=False):
        torch.save({"rows": rows, "att": torch.stack(att), "queries": QUERIES, "keys": KEYS, "cue": args.cue,
                    "model": vlm.model_id, "n_layers": vlm.n_layers, "final": final}, path)

    for n_sc, si in enumerate(tqdm(range(args.scene_start, args.scene_end), desc=f"attn {args.cue}")):
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
                qpos = {"cue_q": vlm.find_token(enc, cw, 0), "at1": vlm.find_token(enc, "at", 0),
                        "tgt_name": vlm.find_token(enc, T, 0), "foil_name": vlm.find_token(enc, F, 0),
                        "is": vlm.find_token(enc, "is", 0), "at2": vlm.find_token(enc, "at", 1), "last": enc.last}
                assert qpos["at1"] < min(qpos["tgt_name"], qpos["foil_name"]) < qpos["is"] < qpos["at2"] < qpos["last"]
                slots = b2_slots(vlm, enc, sc, origin, t, args.cue, args, [qpos["cue_q"], qpos["tgt_name"], qpos["foil_name"]])
                objs = {j: slots[f"obj{j}"] for j in range(len(sc.objects))}
                kpos = {"tgt_name": name_span(vlm, enc, T), "foil_name": name_span(vlm, enc, F),
                        "at1": [qpos["at1"]], "cue_q": [qpos["cue_q"]], "tgt_obj": objs[t], "foil_obj": objs[f],
                        "other_obj": [q for j, v in objs.items() if j not in (t, f) for q in v],
                        "img_cue": slots["img_cue"], "img_tip": slots["img_tip"], "image": enc.image_pos.tolist(),
                        "pre": list(range(int(enc.image_pos.min())))}
                M = torch.zeros(enc.seq_len, len(KEYS), device=vlm.device)
                for j, k_ in enumerate(KEYS):
                    M[kpos[k_], j] = 1.0
                store.clear()
                store["q"] = torch.tensor([qpos[q] for q in QUERIES], device=vlm.device)
                store["M"] = M
                logits = vlm.forward(enc)
                att.append(torch.stack([store[L] for L in range(vlm.n_layers)]).half().cpu())  # [nL, H, nQ, nK]
                lp = torch.log_softmax(logits.float(), -1)
                rows.append({"scene": si, "origin": origin, "target": T, "foil": F, "target_first": not swap,
                             "tip_nearest_is_target": geo["tip_nearest"] == t, "hand": sc.hand,
                             "ld": round(float(lp[first_tok[T]] - lp[first_tok[F]]), 4),
                             "n_img_cue": len(slots["img_cue"])})
        if args.save_every and (n_sc + 1) % args.save_every == 0:
            save()
    for h in handles:
        h.remove()
    save(final=True)
    print(f"{args.cue}: n={len(rows)} saved {path}")


if __name__ == "__main__":
    main()
