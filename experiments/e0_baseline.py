"""E0: behavioural baseline on minimal pairs.

For each cue (hand / arrow) and each minimal pair (cue -> A vs cue -> B, same scene):
  - two-way choice accuracy: logP(target) > logP(foil) at the first answer token
  - the same scene with the cue removed (prior / saliency preference)
  - free-form greedy answers for the first --n_generate samples
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from deixis.common import base_parser, encode_spec, layout_from, load_vlm, names_from, out_dir, save_json
from deixis.stimuli import no_cue, render, specs_minimal_pairs


def main():
    p = base_parser(__doc__)
    p.add_argument("--cues", default="hand,arrow")
    p.add_argument("--n", type=int, default=100, help="minimal pairs per cue")
    p.add_argument("--n_generate", type=int, default=10)
    p.add_argument("--save_images", type=int, default=6)
    args = p.parse_args()
    layout, names = layout_from(args), names_from(args)
    vlm = load_vlm(args)
    od = out_dir(args, "e0_baseline")

    summary = {}
    for cue in args.cues.split(","):
        rows = []
        pairs = specs_minimal_pairs(layout, names, cue, args.n, args.seed, args.min_sep, args.origin_jitter)
        for k, (x, y) in enumerate(tqdm(pairs, desc=f"E0 {cue}")):
            row = {"pair": k, "cue": cue}
            for tag, spec in (("x", x), ("y", y)):
                enc, _ = encode_spec(vlm, spec, layout, args.obj_dir)
                lg = vlm.forward(enc)
                lt, lf = vlm.answer_logprob(lg, spec.target), vlm.answer_logprob(lg, spec.foil)
                row[f"{tag}_lp_target"], row[f"{tag}_lp_foil"] = lt, lf
                row[f"{tag}_correct"] = bool(lt > lf)
                if k < args.n_generate:
                    row[f"{tag}_generated"] = vlm.generate(enc)
                if k < args.save_images:
                    render(spec, layout, args.obj_dir)[0].save(os.path.join(od, f"{cue}_{k}_{tag}.png"))
            # no-cue prior: preference for x.target (A) over B in the identical scene without the cue
            enc, _ = encode_spec(vlm, no_cue(x), layout, args.obj_dir)
            lg = vlm.forward(enc)
            row["nocue_lp_A"], row["nocue_lp_B"] = vlm.answer_logprob(lg, x.target), vlm.answer_logprob(lg, x.foil)
            rows.append(row)
        acc = np.mean([r["x_correct"] for r in rows] + [r["y_correct"] for r in rows])
        both = np.mean([r["x_correct"] and r["y_correct"] for r in rows])
        # does the answer follow the cue at all? compare logit diffs across the pair
        follow = np.mean([(r["x_lp_target"] - r["x_lp_foil"]) + (r["y_lp_target"] - r["y_lp_foil"]) > 0 for r in rows])
        summary[cue] = {"acc": float(acc), "both_correct": float(both), "pair_follows_cue": float(follow), "n_pairs": len(rows)}
        save_json(rows, os.path.join(od, f"{cue}_rows.json"))
        print(cue, summary[cue])
    save_json(summary, os.path.join(od, "summary.json"))


if __name__ == "__main__":
    main()
