"""K1: exact replication of Kang et al.'s spatial-ID extraction (their Fig. 5 / universal IDs).

Per pair (a, b): 960 images (16 cells of a x 15 cells of b x 4 sizes), query "Is {b} to the left or
right of {a}?", residual stream at b's last sub-token at every layer (resid_pre[L] = HF
hidden_states[L]; L = n_layers is the pre-final-norm stream). Per-pair IDs are computed exactly as in
their compute_id.py and saved, so universal IDs (average_id.py) and bootstrap CIs over pairs can be
formed later (k1_analyse.py). Chunk the 90 pairs over jobs with --pair_start/--pair_end.

Output: <out>/<model>/kang_ids/pairs_<start>_<end>.pt with
  {"pairs": [(a, b)], "ids": tensor [n_pairs, 64 locs, n_layers+1, d] fp16, "locs": [(x, y, size)],
   "acc": per-pair fraction answering the correct side, "n_layers", "model"}
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from deixis.common import base_parser, load_vlm, out_dir
from deixis.kang import CELL, KANG_SIZES, center_and_average, kang_pairs, kang_query, pair_specs, render_pair


def main():
    p = base_parser(__doc__)
    p.add_argument("--kang_obj_dir", required=True, help="dir with Kang's Objaverse renders <name>.png")
    p.add_argument("--pair_start", type=int, default=0)
    p.add_argument("--pair_end", type=int, default=90)
    p.add_argument("--max_images", type=int, default=0, help="debug: cap images per pair")
    args = p.parse_args()

    vlm = load_vlm(args)
    vlm.answer_aliases = {}
    od = out_dir(args, "kang_ids")
    pairs = kang_pairs()[args.pair_start:args.pair_end]
    specs = pair_specs()
    if args.max_images:
        specs = specs[: args.max_images]
    locs = None
    all_ids, accs = [], []
    for a, b in tqdm(pairs, desc="pairs"):
        q = kang_query(a, b)
        tok = vlm.tok.tokenize(" " + b)[-1]  # Kang: tokenizer.tokenize(" " + word)[-1]
        acts, correct = {}, []
        for ac, bc, s in specs:
            img = render_pair(args.kang_obj_dir, a, b, ac, bc, s)
            enc = vlm.encode(img, q)
            start = int(enc.image_pos.max()) + 1
            hits = [i for i in range(start, enc.seq_len) if enc.tokens[i] == tok]
            if not hits:
                raise ValueError(f"token {tok!r} for {b!r} not found in prompt")
            logits, a_ = vlm.run_cache(enc, torch.tensor([hits[0]]))  # first occurrence, as Kang
            acts[(ac, bc, s)] = a_[:, 0]  # [L+1, d]
            # behavioural check: b is left of a iff bx < ax (same column is ambiguous -> skipped)
            if bc[0] != ac[0]:
                truth = "left" if bc[0] < ac[0] else "right"
                other = "right" if truth == "left" else "left"
                correct.append(vlm.answer_logprob(logits, truth) > vlm.answer_logprob(logits, other))
        ids = center_and_average(acts)
        if locs is None:
            locs = sorted(ids)
        all_ids.append(torch.stack([ids[l] for l in locs]).half())
        accs.append(sum(correct) / max(1, len(correct)))
    out = {"pairs": pairs, "ids": torch.stack(all_ids), "locs": locs, "acc": accs,
           "n_layers": vlm.n_layers, "model": vlm.model_id, "cell": CELL, "sizes": KANG_SIZES,
           "query": "Is {b} to the left or right of {a}?"}
    path = os.path.join(od, f"pairs_{args.pair_start:02d}_{args.pair_end:02d}.pt")
    torch.save(out, path)
    print("saved", path, tuple(out["ids"].shape), "mean acc", sum(accs) / len(accs))


if __name__ == "__main__":
    main()
