"""Detect the pointing person in YouRefIt frames (torchvision Keypoint R-CNN, COCO 17 keypoints).

Saves <out>/people_{split}.json: {id: {"box": xyxy, "score", "kpts": [[x,y,v]x17], "arm": "left|right",
"ray": [[ex,ey],[wx,wy]]}} in ORIGINAL image pixels. The pointing arm is the one whose wrist is
farthest from its shoulder (most extended); ray = elbow -> wrist.
Used by R0/R1 (person-masked controls) and later for pointing-direction probes.

Weights: torchvision downloads keypointrcnn_resnet50_fpn (COCO) on first use. Set TORCH_HOME to a
project dir (SCC home is at quota) and run once on a login node to pre-download.
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from tqdm import tqdm

from deixis.yourefit import DEFAULT_ZIP, YouRefIt

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", default=DEFAULT_ZIP)
    p.add_argument("--out", default="results_scaled")
    p.add_argument("--splits", default="test,train")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_side", type=int, default=1333)
    args, _ = p.parse_known_args()  # tolerate --model/--out passthrough from scc_experiment.qsub
    args.out = os.path.join(args.out, "yourefit")
    os.makedirs(args.out, exist_ok=True)
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
    from torchvision.transforms.functional import to_tensor
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1).eval().to(args.device)
    yr = YouRefIt(args.zip)
    for split in args.splits.split(","):
        res = {}
        for r in tqdm(yr.refs(split), desc=f"people {split}"):
            img, s = yr.image(r, max_side=args.max_side)
            with torch.no_grad():
                det = model([to_tensor(img).to(args.device)])[0]
            keep = [i for i in range(len(det["scores"])) if det["labels"][i] == 1]
            if not keep:
                res[r.id] = None
                continue
            # the pointing person = highest-scoring person (YouRefIt frames have one actor)
            i = max(keep, key=lambda k: float(det["scores"][k]))
            box = (det["boxes"][i] / s).tolist()
            k = det["keypoints"][i].clone()
            k[:, :2] /= s
            k = k.tolist()
            def ext(sh, wr):
                return ((k[wr][0] - k[sh][0]) ** 2 + (k[wr][1] - k[sh][1]) ** 2) ** 0.5
            arm = "left" if ext(L_SH, L_WR) >= ext(R_SH, R_WR) else "right"
            el, wr = (L_EL, L_WR) if arm == "left" else (R_EL, R_WR)
            res[r.id] = {"box": [round(v, 1) for v in box], "score": round(float(det["scores"][i]), 3),
                         "kpts": [[round(a, 1), round(b, 1), round(c, 2)] for a, b, c in k],
                         "arm": arm, "ray": [k[el][:2], k[wr][:2]]}
        path = os.path.join(args.out, f"people_{split}.json")
        with open(path, "w") as f:
            json.dump(res, f)
        found = sum(v is not None for v in res.values())
        print(f"{split}: person found in {found}/{len(res)} frames -> {path}")


if __name__ == "__main__":
    main()
