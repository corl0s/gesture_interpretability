"""E4: spatial-ID steering of deictic answers (Kang et al. Fig. 6, Alg. 2).

On cue images the model answers correctly (cue -> A at cell a, distractor B at cell b), add
at layer L and slot S:
    v = Delta_L(b) - Delta_L(a), rescaled to alpha * ||x_L[S]||      (Kang: alpha = 5)
where Delta comes from --id_source (default: object-word IDs from E1, i.e. a *transfer* test:
does the referent slot of the gesture question read the same spatial code as object words?).
Control: a random direction with the same norm.
Metric: shift in LD = logP(B) - logP(A) relative to the unsteered run; flip = answer becomes B.
A positive, above-control shift is causal evidence that the slot carries a referent location in
the spatial-ID code (sufficiency at that layer/slot).
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

from deixis import ids as I
from deixis.common import base_parser, encode_spec, layout_from, load_vlm, names_from, out_dir, save_json, slot_positions
from deixis.stimuli import specs_minimal_pairs


def main():
    p = base_parser(__doc__)
    p.add_argument("--cue", default="hand", choices=["hand", "arrow"])
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--id_source", required=True, help="path to an E1 .pt file (e.g. results/<model>/ids/object.pt)")
    p.add_argument("--id_slot", default=None, help="slot in id_source (default: first slot)")
    p.add_argument("--slots", default="last,at,image_cue")
    p.add_argument("--alpha", type=float, default=5.0)
    p.add_argument("--layer_stride", type=int, default=1)
    args = p.parse_args()
    layout, names = layout_from(args), names_from(args)

    d = torch.load(args.id_source, weights_only=False)
    k = d["slots"].index(args.id_slot) if args.id_slot else 0
    ids = I.ids_by_loc(d["acts"][:, k].float(), [tuple(l) for l in d["locs"]], d["groups"])

    vlm = load_vlm(args)
    od = out_dir(args, "e4_steer")
    layers = list(range(0, vlm.n_layers + 1, args.layer_stride))
    g = torch.Generator().manual_seed(args.seed)

    rows = []
    for k, (x, _) in enumerate(tqdm(specs_minimal_pairs(layout, names, args.cue, args.n, args.seed, args.min_sep, args.origin_jitter), desc="E4")):
        enc, meta = encode_spec(vlm, x, layout, args.obj_dir)
        A, B = x.target, x.foil
        a_cell, b_cell = dict(x.objects)[A], dict(x.objects)[B]
        for slot in args.slots.split(","):
            pos = slot_positions(vlm, enc, x, meta, slot, layout)
            lg0, acts = vlm.run_cache(enc, pos, layers)  # [L, n_pos, d]
            ld0 = vlm.answer_logprob(lg0, B) - vlm.answer_logprob(lg0, A)
            if ld0 > 0:
                break  # model already wrong on this sample; skip it
            for li, L in enumerate(layers):
                norm = acts[li].norm(dim=-1, keepdim=True)  # per-position ||x||
                v = ids[b_cell][L] - ids[a_cell][L]
                r = torch.randn(v.shape, generator=g)
                for kind, vec in (("id", v), ("random", r)):
                    add = args.alpha * norm * (vec / vec.norm())[None]
                    lg = vlm.forward(enc, add={L: (pos, add)})
                    ld = vlm.answer_logprob(lg, B) - vlm.answer_logprob(lg, A)
                    rows.append({"sample": k, "slot": slot, "layer": L, "kind": kind,
                                 "ld_shift": ld - ld0, "flip": bool(ld > 0)})
    save_json(rows, os.path.join(od, f"{args.cue}_a{args.alpha:g}_rows.json"))
    if not rows:
        print("no correctly answered samples")
        return
    slots = args.slots.split(",")
    fig, axs = plt.subplots(1, len(slots), figsize=(5 * len(slots), 3.8), squeeze=False)
    for ax, slot in zip(axs[0], slots):
        for kind, st in (("id", "-"), ("random", ":")):
            m = [np.mean([r["ld_shift"] for r in rows if r["slot"] == slot and r["layer"] == L and r["kind"] == kind] or [np.nan]) for L in layers]
            f = [np.mean([r["flip"] for r in rows if r["slot"] == slot and r["layer"] == L and r["kind"] == kind] or [np.nan]) for L in layers]
            ax.plot(layers, m, st, label=f"{kind}: mean ΔLD")
            ax2 = ax.twinx() if kind == "id" else ax2
            ax2.plot(layers, f, st, color="C3", alpha=0.6)
        ax.set_title(f"{args.cue} / steer at {slot} (α={args.alpha})")
        ax.set_xlabel("layer")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(od, f"{args.cue}_a{args.alpha:g}_steer.png"), dpi=140)
    print("saved", od)


if __name__ == "__main__":
    main()
