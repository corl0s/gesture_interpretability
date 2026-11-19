"""E3: cue swapping = Kang et al. mirror swapping (Fig. 4, Alg. 1) for deictic cues.

Kang swap activations between an image and its mirror. Our minimal pair is cleaner:
x = cue points at A, y = identical scene with the cue pointing at B. At layer L, replace
x's resid_pre at a token subset with y's and measure how far x's belief moves to y's.

Metrics per (pair, layer, subset), with GT = x's answer (A):
  belief_shift = (logP_x(A) - logP_patched(A)) / (logP_x(A) - logP_y(A))   [Kang]
  ld_shift     = (LD_x - LD_patched) / (LD_x - LD_y),  LD = logP(A) - logP(B)  [more stable]
1 = patching the subset fully transfers the cue's effect, 0 = no effect.
--pair_filter correct (default, Kang-style): only pairs answered right in both directions.
--pair_filter follows: every pair whose LD moves with the cue by > --min_gap nats (LD_x - LD_y),
  i.e. also pairs the model gets wrong in one or both images. Each row records the pair's
  outcome ("both" / "x_only" / "y_only" / "neither") and LD_x, LD_y, so relay strength can be
  compared between right and wrong pairs. Use ld_shift for those (belief_shift's denominator
  can be tiny when the model is wrong).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from deixis.common import base_parser, encode_spec, layout_from, load_vlm, names_from, out_dir, save_json, slot_positions
from deixis.stimuli import specs_minimal_pairs

SUBSETS = ["image_all", "image_cue", "image_objects", "text_all", "cue_word", "last"]


def main():
    p = base_parser(__doc__)
    p.add_argument("--cue", default="hand", choices=["hand", "arrow"])
    p.add_argument("--n", type=int, default=40, help="candidate minimal pairs (filtered to correct ones)")
    p.add_argument("--layer_stride", type=int, default=1)
    p.add_argument("--subsets", default=",".join(SUBSETS))
    p.add_argument("--pair_filter", default="correct", choices=["correct", "follows"])
    p.add_argument("--min_gap", type=float, default=0.5, help="follows mode: min LD_x - LD_y (nats)")
    args = p.parse_args()
    layout, names = layout_from(args), names_from(args)
    vlm = load_vlm(args)
    od = out_dir(args, "e3_swap")
    subsets = args.subsets.split(",")
    layers = list(range(0, vlm.n_layers + 1, args.layer_stride))

    rows = []
    pairs = specs_minimal_pairs(layout, names, args.cue, args.n, args.seed, args.min_sep, args.origin_jitter)
    for k, (x, y) in enumerate(tqdm(pairs, desc="E3")):
        ex, mx = encode_spec(vlm, x, layout, args.obj_dir)
        ey, my = encode_spec(vlm, y, layout, args.obj_dir)
        assert ex.seq_len == ey.seq_len and torch.equal(ex.image_pos, ey.image_pos)
        A, B = x.target, x.foil
        ly, cache_y = vlm.run_cache(ey, None, layers)  # y: full-sequence resid at every layer
        lx = vlm.forward(ex)
        lpA_x, lpB_x = vlm.answer_logprob(lx, A), vlm.answer_logprob(lx, B)
        lpA_y, lpB_y = vlm.answer_logprob(ly, A), vlm.answer_logprob(ly, B)
        ld_x, ld_y = lpA_x - lpB_x, lpA_y - lpB_y
        ok_x, ok_y = ld_x > 0, ld_y < 0
        outcome = {(True, True): "both", (True, False): "x_only", (False, True): "y_only", (False, False): "neither"}[(ok_x, ok_y)]
        if args.pair_filter == "correct" and outcome != "both":
            continue
        if args.pair_filter == "follows" and not (ld_x - ld_y > args.min_gap):
            continue
        for s in subsets:
            if s == "image_cue":  # union of both cues' footprints
                px = torch.cat([slot_positions(vlm, ex, x, mx, s, layout), slot_positions(vlm, ey, y, my, s, layout)])
            elif s == "image_objects":  # both object cells (pixels identical in x and y)
                px = torch.cat([slot_positions(vlm, ex, x, mx, "image_target", layout),
                                slot_positions(vlm, ex, x, mx, "image_foil", layout)])
            else:
                px = slot_positions(vlm, ex, x, mx, s, layout)
            px = torch.unique(px)
            for li, L in enumerate(layers):
                lp = vlm.forward(ex, patch={L: (px, cache_y[li][px])})
                pA, pB = vlm.answer_logprob(lp, A), vlm.answer_logprob(lp, B)
                rows.append({"pair": k, "outcome": outcome, "ld_x": ld_x, "ld_y": ld_y,
                             "subset": s, "layer": L, "n_pos": len(px),
                             "belief_shift": (lpA_x - pA) / (lpA_x - lpA_y),
                             "ld_shift": (ld_x - (pA - pB)) / (ld_x - ld_y)})
    tag = "" if args.pair_filter == "correct" else f"_{args.pair_filter}"
    save_json(rows, os.path.join(od, f"{args.cue}{tag}_rows.json"))
    if not rows:
        print("no pair was answered correctly in both directions; nothing to patch")
        return

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for metric, ax in zip(("ld_shift", "belief_shift"), axs):
        for s in sorted({r["subset"] for r in rows}):
            m = [np.median([r[metric] for r in rows if r["subset"] == s and r["layer"] == L]) for L in layers]
            ax.plot(layers, m, marker=".", label=s)
        ax.set_title(f"{args.cue}: median {metric} (n pairs={len({r['pair'] for r in rows})})")
        ax.set_xlabel("layer (resid_pre)")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvspan(-0.5, vlm.deepstack_layers - 0.5, color="gray", alpha=0.1)
    axs[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(od, f"{args.cue}{tag}_swap.png"), dpi=140)
    print("saved", od)


if __name__ == "__main__":
    main()
