"""R0: does the VLM use real pointing? YouRefIt test frames, centre-point grounding.

Prompt = the user's YouRefIt adapter prompt (centre point, 0-1000 scale), with the referring text:
  lang          : the annotated sentence                     (language only)
  gesture       : "the object the person is pointing at"     (gesture only)
  both          : "<sentence>, which the person is pointing at"
  gesture_mask  : gesture prompt, pointing person hidden     (can only guess: saliency/prior)
  lang_mask     : sentence, person hidden                    (cost of masking itself)
Score: predicted centre inside the target box widened by 6% per side (the adapter's metric).
Also logs distance of the prediction to the pointing wrist (proximity bias) when people_*.json exists.
Evidence: behavioural.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from deixis.common import base_parser, load_vlm, out_dir, save_json
from deixis.yourefit import DEFAULT_ZIP, YouRefIt, center, mask_box, point_in_box, sample

PROMPT = ("You are a visual grounding assistant. Given an image and a referring expression, identify the "
          "center location of the referred object.\n\nReferring expression: {text}\n\n"
          "Reply with the center coordinates as (x, y) where x and y are integers between 0 and 1000.")
GESTURE = "the object the person is pointing at"
CONDS = ["lang", "gesture", "both", "gesture_mask", "lang_mask"]


def parse_point(s):
    m = re.search(r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    x, y = float(m.group(1)), float(m.group(2))
    return (x, y) if 0 <= x <= 1000 and 0 <= y <= 1000 else None


def main():
    p = base_parser(__doc__)
    p.add_argument("--zip", default=DEFAULT_ZIP)
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=0, help="0 = all frames")
    p.add_argument("--max_side", type=int, default=1024)
    p.add_argument("--conds", default=",".join(CONDS))
    p.add_argument("--people", default="", help="people_<split>.json from r_people.py (needed for *_mask)")
    args = p.parse_args()
    conds = args.conds.split(",")
    people = json.load(open(args.people)) if args.people and os.path.exists(args.people) else {}
    if any(c.endswith("_mask") for c in conds) and not people:
        print("no people file: dropping *_mask conditions")
        conds = [c for c in conds if not c.endswith("_mask")]

    vlm = load_vlm(args)
    od = out_dir(args, "r0_yourefit")
    yr = YouRefIt(args.zip)
    refs = sample(yr.refs(args.split), args.n or None, args.seed)
    rows = []
    for r in tqdm(refs, desc="R0"):
        img, s = yr.image(r, args.max_side)
        W, H = img.size
        box = [v * s for v in r.bbox]
        pers = people.get(r.id)
        pbox = [v * s for v in pers["box"]] if pers else None
        wrist = [v * s for v in pers["ray"][1]] if pers else None
        for c in conds:
            if c.endswith("_mask") and pbox is None:
                continue
            text = {"lang": r.sentence, "gesture": GESTURE, "both": f"{r.sentence}, which the person is pointing at",
                    "gesture_mask": GESTURE, "lang_mask": r.sentence}[c]
            im = mask_box(img, pbox) if c.endswith("_mask") else img
            enc = vlm.encode(im, PROMPT.format(text=text))
            out = vlm.generate(enc, max_new_tokens=24)
            pt = parse_point(out)
            px = (pt[0] / 1000 * W, pt[1] / 1000 * H) if pt else None
            row = {"id": r.id, "scene": r.scene, "cond": c, "output": out, "point_px": px,
                   "hit": bool(px and point_in_box(px, box)), "parsed": pt is not None,
                   "target_in_person_box": bool(pbox and point_in_box(center(box), pbox, 0.0))}
            if px and wrist:
                cx, cy = center(box)
                row["dist_to_wrist"] = ((px[0] - wrist[0]) ** 2 + (px[1] - wrist[1]) ** 2) ** 0.5 / W
                row["target_dist_to_wrist"] = ((cx - wrist[0]) ** 2 + (cy - wrist[1]) ** 2) ** 0.5 / W
            rows.append(row)
    save_json(rows, os.path.join(od, f"{args.split}_rows.json"))
    summ = {}
    for c in conds:
        rc = [x for x in rows if x["cond"] == c]
        if rc:
            summ[c] = {"n": len(rc), "acc": sum(x["hit"] for x in rc) / len(rc),
                       "parse_rate": sum(x["parsed"] for x in rc) / len(rc)}
    save_json(summ, os.path.join(od, f"{args.split}_summary.json"))
    for c, v in summ.items():
        print(f"{c:13s} acc {v['acc']:.3f}  parsed {v['parse_rate']:.3f}  n {v['n']}")


if __name__ == "__main__":
    main()
