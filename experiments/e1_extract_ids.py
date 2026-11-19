"""E1/E2: extract location-indexed activations (Kang et al. Fig. 5 / compute_id.py).

--condition object      Kang replication: b swept over the 4x4 grid, query
                        "Is the b to the left or right of the a?", slot = b's word token.
--condition hand|arrow  The cue points at every cell (target b sits there, distractor a
                        elsewhere); query "Which object is the <cue> pointing at?".
                        Slots: cue word, 'pointing', 'at', last token.
--condition hand_empty|arrow_empty
                        Same cue geometry, no objects: does direction alone create a
                        location code at the referent slots?

Saves resid_pre at every layer (incl. final-norm input) for every slot, float16.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from deixis.common import base_parser, encode_spec, layout_from, load_vlm, names_from, out_dir, slot_positions
from deixis.stimuli import specs_cue_ids, specs_object_ids

SLOTS = {
    "object": ["target_word", "last"],
    "hand": ["cue_word", "pointing", "at", "last"],
    "arrow": ["cue_word", "pointing", "at", "last"],
}


def main():
    p = base_parser(__doc__)
    p.add_argument("--condition", required=True, choices=["object", "hand", "arrow", "hand_empty", "arrow_empty"])
    p.add_argument("--n_pairs", type=int, default=20, help="object pairs (Kang: 90); 16 images each")
    args = p.parse_args()
    layout, names = layout_from(args), names_from(args)

    cond = args.condition
    if cond == "object":
        specs = specs_object_ids(layout, names, args.n_pairs, args.seed)
    else:
        cue, empty = cond.split("_")[0], cond.endswith("_empty")
        specs = specs_cue_ids(layout, names, cue, args.n_pairs, args.seed, args.min_sep, empty, args.origin_jitter)
    slots = SLOTS[cond.split("_")[0]]

    vlm = load_vlm(args)
    all_acts, correct, lds = [], [], []
    for spec in tqdm(specs, desc=f"extract {cond}"):
        enc, meta = encode_spec(vlm, spec, layout, args.obj_dir)
        pos = torch.cat([slot_positions(vlm, enc, spec, meta, s, layout) for s in slots])
        logits, acts = vlm.run_cache(enc, pos)  # [L+1, n_slots, d]
        all_acts.append(acts.permute(1, 0, 2).half())  # [n_slots, L+1, d]
        if spec.target and spec.foil:
            ld = vlm.answer_logprob(logits, spec.target) - vlm.answer_logprob(logits, spec.foil)
            lds.append(ld)
            correct.append(ld > 0)
    out = {
        "condition": cond, "slots": slots, "acts": torch.stack(all_acts),  # [N, n_slots, L+1, d]
        "locs": [s.loc for s in specs], "groups": [s.group for s in specs],
        "specs": [s.to_dict() for s in specs], "correct": correct, "logit_diff": lds,
        "n_layers": vlm.n_layers, "deepstack_layers": vlm.deepstack_layers, "grid": layout.grid,
    }
    path = os.path.join(out_dir(args, "ids"), f"{cond}.pt")
    torch.save(out, path)
    if correct:
        print(f"{cond}: accuracy (target vs distractor) = {sum(correct) / len(correct):.3f}")
    print("saved", path, tuple(out["acts"].shape))


if __name__ == "__main__":
    main()
