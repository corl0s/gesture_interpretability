"""List usable YouRefIt same-scene pairs (two frames, same video, different targets).

Writes <out>/yourefit_pairs_{split}.csv and .json with one row per unordered pair:
  scene, id_a, id_b, target_a, target_b, sentence_a, sentence_b, bbox_a, bbox_b, W, H,
  center_dist_px, iou
Filters (same as the SCC count): distinct normalised target and sentence, IoU < 0.1, box centres
more than 10% of the image width apart. Reads the zip only (no extraction). CPU only.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deixis.yourefit import DEFAULT_ZIP, YouRefIt, center, iou, same_scene_pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", default=DEFAULT_ZIP)
    p.add_argument("--out", default="results_scaled")
    p.add_argument("--splits", default="test,train")
    args, _ = p.parse_known_args()  # tolerate --model/--out passthrough from scc_experiment.qsub
    args.out = os.path.join(args.out, "yourefit")
    os.makedirs(args.out, exist_ok=True)
    yr = YouRefIt(args.zip)
    for split in args.splits.split(","):
        refs = yr.refs(split)
        sizes = {}
        for r in refs:  # image sizes from the JPEG header only (cheap)
            from PIL import Image
            import io
            with yr._zf.open(f"yourefit/images/{r.id}.jpg") as f:
                sizes[r.id] = Image.open(io.BytesIO(f.read())).size
        pairs = same_scene_pairs(refs, img_width={k: v[0] for k, v in sizes.items()})
        rows = []
        for a, b in pairs:
            (ax, ay), (bx, by) = center(a.bbox), center(b.bbox)
            W, H = sizes[a.id]
            rows.append({"scene": a.scene, "id_a": a.id, "id_b": b.id,
                         "target_a": a.target, "target_b": b.target,
                         "sentence_a": a.sentence, "sentence_b": b.sentence,
                         "bbox_a": [round(v, 1) for v in a.bbox], "bbox_b": [round(v, 1) for v in b.bbox],
                         "W": W, "H": H, "same_size": sizes[a.id] == sizes[b.id],
                         "center_dist_px": round(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5, 1),
                         "iou": round(iou(a.bbox, b.bbox), 4)})
        base = os.path.join(args.out, f"yourefit_pairs_{split}")
        with open(base + ".json", "w") as f:
            json.dump(rows, f, indent=1)
        with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        scenes = len({r["scene"] for r in rows})
        print(f"{split}: {len(refs)} refs, {len(rows)} usable pairs in {scenes} scenes, "
              f"same_size {sum(r['same_size'] for r in rows)} -> {base}.csv/.json")


if __name__ == "__main__":
    main()
