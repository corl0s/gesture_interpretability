"""R1: two-choice pointing on real same-scene pairs (YouRefIt), the real-image analogue of E0.

For a pair (frame x: person points at a; frame y: same scene, points at b), both frames get the
SAME question:
  "Which object is the person pointing at? A) <target_1>  B) <target_2>. Answer with A or B."
(the A/B order is randomised per pair, fixed within the pair). Score LD = logP(correct letter) -
logP(other letter). Metrics as in E0: acc, both_correct, pair_follows_cue (LD moves toward the new
target when only the pointing changes). With people_<split>.json, adds a person-masked prior.
Rows keep everything needed to reuse the pairs for patching (R3) and steering (R4).
Evidence: behavioural.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from deixis.common import base_parser, load_vlm, out_dir, save_json
from deixis.yourefit import DEFAULT_ZIP, YouRefIt, mask_box, same_scene_pairs, sample

Q = ("Which object is the person pointing at?\nA) {A}\nB) {B}\nAnswer with A or B.")


def main():
    p = base_parser(__doc__)
    p.add_argument("--zip", default=DEFAULT_ZIP)
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=1000, help="pairs to sample (0 = all)")
    p.add_argument("--max_side", type=int, default=1024)
    p.add_argument("--people", default="")
    args = p.parse_args()
    people = json.load(open(args.people)) if args.people and os.path.exists(args.people) else {}

    vlm = load_vlm(args)
    vlm.answer_aliases = {}
    vlm.check_answer_vocab(["A", "B"])
    od = out_dir(args, "r1_yourefit")
    yr = YouRefIt(args.zip)
    pairs = sample(same_scene_pairs(yr.refs(args.split)), args.n or None, args.seed)
    rng = random.Random(args.seed + 7)

    def ld(img, q, correct):
        lg = vlm.forward(vlm.encode(img, q))
        other = "B" if correct == "A" else "A"
        return vlm.answer_logprob(lg, correct) - vlm.answer_logprob(lg, other)

    rows = []
    for k, (a, b) in enumerate(tqdm(pairs, desc="R1")):
        a_first = rng.random() < 0.5
        q = Q.format(A=a.target if a_first else b.target, B=b.target if a_first else a.target)
        let_a, let_b = ("A", "B") if a_first else ("B", "A")
        ix, sx = yr.image(a, args.max_side)
        iy, sy = yr.image(b, args.max_side)
        row = {"pair": k, "scene": a.scene, "id_x": a.id, "id_y": b.id, "target_x": a.target, "target_y": b.target,
               "letter_x": let_a, "question": q,
               "ld_x": ld(ix, q, let_a),   # >0: frame x answered correctly (its own target a)
               "ld_y": ld(iy, q, let_b)}   # >0: frame y answered correctly (b)
        if a.id in people and b.id in people and people[a.id] and people[b.id]:
            mx = mask_box(ix, [v * sx for v in people[a.id]["box"]])
            my = mask_box(iy, [v * sy for v in people[b.id]["box"]])
            row["prior_ld_x_masked"] = ld(mx, q, let_a)
            row["prior_ld_y_masked"] = ld(my, q, let_b)
        rows.append(row)
    save_json(rows, os.path.join(od, f"{args.split}_rows.json"))
    n = len(rows)
    # follows cue: logP(a)-logP(b) is higher in frame x (points at a) than in frame y (points at b)
    summ = {"n_pairs": n,
            "acc": sum((r["ld_x"] > 0) + (r["ld_y"] > 0) for r in rows) / (2 * n),
            "both_correct": sum(r["ld_x"] > 0 and r["ld_y"] > 0 for r in rows) / n,
            "pair_follows_cue": sum(r["ld_x"] + r["ld_y"] > 0 for r in rows) / n}
    m = [r for r in rows if "prior_ld_x_masked" in r]
    if m:
        summ["masked_acc"] = sum((r["prior_ld_x_masked"] > 0) + (r["prior_ld_y_masked"] > 0) for r in m) / (2 * len(m))
        summ["masked_pair_follows_cue"] = sum(r["prior_ld_x_masked"] + r["prior_ld_y_masked"] > 0 for r in m) / len(m)
        summ["n_masked"] = len(m)
    save_json(summ, os.path.join(od, f"{args.split}_summary.json"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
