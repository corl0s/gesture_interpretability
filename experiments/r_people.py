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
L_HIP, R_HIP = 11, 12


def pointing_arm(det, i, s, min_kp_score: float = 1.0):
    """GT-free pointing-arm choice for person i (no target information used).

    A hanging arm is the most 'extended' (shoulder-wrist distance), so extension is the wrong cue.
    Score per arm (in torso lengths):
      raise = (shoulder_y - wrist_y) / torso + 1      wrist height relative to the shoulder
      away  = |wrist_x - torso_cx| / torso              lateral reach away from the body
      bend  = angle(elbow->wrist, straight down) / 90   0 = hanging, 1 = horizontal, 2 = straight up
    score = raise + away + bend; arms whose elbow/wrist keypoint logits are < min_kp_score are skipped.
    Returns (arm, score, {arm: score})."""
    import math
    k = (det["keypoints"][i][:, :2] / s).tolist()
    ks = det.get("keypoints_scores")
    ks = ks[i].tolist() if ks is not None else [99.0] * 17
    sh_y = (k[L_SH][1] + k[R_SH][1]) / 2
    hip_y = (k[L_HIP][1] + k[R_HIP][1]) / 2
    torso = max(abs(hip_y - sh_y), 1.0)
    cx = (k[L_SH][0] + k[R_SH][0] + k[L_HIP][0] + k[R_HIP][0]) / 4
    scores = {}
    for arm, (sh, el, wr) in {"left": (L_SH, L_EL, L_WR), "right": (R_SH, R_EL, R_WR)}.items():
        if ks[el] < min_kp_score or ks[wr] < min_kp_score:
            continue
        dx, dy = k[wr][0] - k[el][0], k[wr][1] - k[el][1]
        bend = math.degrees(math.atan2(abs(dx), dy)) / 90.0 if (dx or dy) else 0.0  # dy>0 is down
        raise_ = (k[sh][1] - k[wr][1]) / torso + 1.0
        away = abs(k[wr][0] - cx) / torso
        scores[arm] = raise_ + away + bend
    if not scores:  # no confident arm: fall back to the higher wrist
        arm = "left" if k[L_WR][1] <= k[R_WR][1] else "right"
        return arm, float("-inf"), {}
    arm = max(scores, key=scores.get)
    return arm, scores[arm], scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", default=DEFAULT_ZIP)
    p.add_argument("--out", default="results_scaled")
    p.add_argument("--splits", default="test,train")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_side", type=int, default=1333)
    p.add_argument("--min_person_score", type=float, default=0.7)
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
            confident = [i for i in keep if float(det["scores"][i]) >= args.min_person_score] or \
                [max(keep, key=lambda q: float(det["scores"][q]))]
            # GT-free choice: among confident people, the one with the most "pointing-like" arm
            cands = [(pointing_arm(det, i, s), i) for i in confident]
            (arm, arm_score, arm_scores), i = max(cands, key=lambda c: c[0][1])
            box = (det["boxes"][i] / s).tolist()
            k = det["keypoints"][i].clone()
            k[:, :2] /= s
            k = k.tolist()
            el, wr = (L_EL, L_WR) if arm == "left" else (R_EL, R_WR)
            res[r.id] = {"box": [round(v, 1) for v in box], "score": round(float(det["scores"][i]), 3),
                         "kpts": [[round(a, 1), round(b, 1), round(c, 2)] for a, b, c in k],
                         "arm": arm, "arm_score": round(arm_score, 3),
                         "arm_scores": {a: round(v, 3) for a, v in arm_scores.items()},
                         "ray": [k[el][:2], k[wr][:2]], "n_people": len(confident)}
        path = os.path.join(args.out, f"people_{split}.json")
        with open(path, "w") as f:
            json.dump(res, f)
        found = sum(v is not None for v in res.values())
        print(f"{split}: person found in {found}/{len(res)} frames -> {path}")


if __name__ == "__main__":
    main()
