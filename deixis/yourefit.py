"""YouRefIt (Chen et al., ICCV 2021) loader, read-only, straight from the authors' zip.

Layout inside `yourefit.zip` (checked on SCC 2026-11-19):
  yourefit/images/<id>.jpg      4,221 frames (train 2,970 / test 1,251)
  yourefit/pickle/<id>.p        dict: bbox (xyxy, px, correct; the ERU index .pth has 561
                                mis-scaled train boxes, so we never use it), anno_sentence,
                                anno_target, frame, ...
  yourefit/{train,test}_id.txt  image ids
Scene = video prefix of the id, e.g. cs_p_290_2_0_422 -> cs_p_290 (ref 2, frame 422).

Nothing is extracted; images are decoded from the zip on demand.
"""
from __future__ import annotations

import io
import os
import pickle
import random
import re
import zipfile
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from PIL import Image

DEFAULT_ZIP = "/projectnb/mlresearch/vishnuav/yourefit/YouRefIt Data/yourefit.zip"
_ROOT = "yourefit/"


@dataclass
class Ref:
    id: str
    scene: str
    bbox: Tuple[float, float, float, float]  # xyxy in original pixels
    sentence: str
    target: str  # anno_target, e.g. "a computer monitor"
    size: Tuple[int, int] = (0, 0)  # original (W, H), filled on first image load

    def to_dict(self):
        return asdict(self)


def scene_of(img_id: str) -> str:
    m = re.match(r"(cs_[a-z]+_\d+)_", img_id)
    return m.group(1) if m else img_id.rsplit("_", 3)[0]


class YouRefIt:
    def __init__(self, zip_path: str = DEFAULT_ZIP):
        if not os.path.exists(zip_path):
            raise FileNotFoundError(zip_path)
        self.zip_path = zip_path
        self._zf = zipfile.ZipFile(zip_path, "r")
        self._names = set(self._zf.namelist())

    # -------------------------------------------------------------- annotations
    def ids(self, split: str) -> List[str]:
        raw = self._zf.read(f"{_ROOT}{split}_id.txt").decode("utf-8").split()
        return [os.path.splitext(os.path.basename(x))[0] for x in raw if x.strip()]

    def ref(self, img_id: str) -> Ref:
        with self._zf.open(f"{_ROOT}pickle/{img_id}.p") as f:
            d = pickle.load(f, encoding="latin1")
        x0, y0, x1, y1 = [float(v) for v in d["bbox"]]
        return Ref(id=img_id, scene=scene_of(img_id), bbox=(x0, y0, x1, y1),
                   sentence=str(d.get("anno_sentence", "")).strip(),
                   target=str(d.get("anno_target", "")).strip())

    def refs(self, split: str) -> List[Ref]:
        return [self.ref(i) for i in self.ids(split)]

    # ------------------------------------------------------------------ images
    def image(self, ref: Ref, max_side: Optional[int] = 1024) -> Tuple[Image.Image, float]:
        """Returns (RGB image, scale) with the long side resized to `max_side` (None = original).
        Multiply original-pixel boxes by `scale` to get boxes in the returned image."""
        with self._zf.open(f"{_ROOT}images/{ref.id}.jpg") as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
        ref.size = img.size
        s = 1.0
        if max_side and max(img.size) > max_side:
            s = max_side / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
        return img, s


# --------------------------------------------------------------------- geometry
def iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def point_in_box(p, box, widen: float = 0.06) -> bool:
    """Hit if point p lies in box widened by `widen` x box size on each side
    (same scoring as the user's YouRefIt Qwen3.5-9B adapter: 6%)."""
    w, h = box[2] - box[0], box[3] - box[1]
    return (box[0] - widen * w <= p[0] <= box[2] + widen * w) and (box[1] - widen * h <= p[1] <= box[3] + widen * h)


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"^(a|an|the|this|that)\s+", "", s)
    return re.sub(r"\s+", " ", s)


def same_scene_pairs(refs: List[Ref], min_dist_frac: float = 0.10, max_iou: float = 0.10,
                     img_width: Optional[Dict[str, int]] = None) -> List[Tuple[Ref, Ref]]:
    """Unordered pairs of refs in the same scene with different targets (normalised anno_target),
    non-overlapping boxes (IoU < max_iou) and centres > min_dist_frac * image width apart.
    Image width defaults to 1920 if unknown (the filter is only coarse)."""
    by_scene: Dict[str, List[Ref]] = {}
    for r in refs:
        by_scene.setdefault(r.scene, []).append(r)
    out = []
    for scene, rs in sorted(by_scene.items()):
        for a, b in combinations(sorted(rs, key=lambda r: r.id), 2):
            if norm(a.target) == norm(b.target) or norm(a.sentence) == norm(b.sentence):
                continue
            if iou(a.bbox, b.bbox) >= max_iou:
                continue
            W = (img_width or {}).get(a.id, 1920)
            (ax, ay), (bx, by) = center(a.bbox), center(b.bbox)
            if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= min_dist_frac * W:
                continue
            out.append((a, b))
    return out


def sample(items, n: Optional[int], seed: int):
    items = list(items)
    random.Random(seed).shuffle(items)
    return items[:n] if n else items


# ------------------------------------------------------------------ person masks
def mask_box(img: Image.Image, box, mode: str = "mean") -> Image.Image:
    """Hide a region (e.g. the pointing person) by filling it with the image mean colour."""
    from PIL import ImageDraw, ImageStat
    out = img.copy()
    fill = tuple(int(v) for v in ImageStat.Stat(img).mean[:3])
    ImageDraw.Draw(out).rectangle([round(v) for v in box], fill=fill)
    return out
