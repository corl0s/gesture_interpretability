"""Synthetic stimuli for deictic-grounding experiments.

Geometry follows Kang et al. (2026) `tile_two_objects.py`: a 4x4 grid of cells, two
objects (a, b), object b swept over every cell. We add a *cue strip* below the grid
from which a stylised pointing hand (or a drawn arrow with identical geometry)
points at one cell. Only the cue direction carries the referent; the cue length is
fixed, so fingertip proximity is not a usable shortcut.

Objects are procedural coloured shapes with single-token names in the Qwen3 tokenizer
(verified: ' star', ' heart', ... are one token each). Kang et al. use Objaverse
renders; drop PNGs (RGBA, name = file stem) into data/objects/ to use those instead.

NOTE: the hand is a 2D cartoon (arm, fist, index finger). It is a placeholder for
EgoPoint-Bench-style rendered or real hands; results on it are a lower bar.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, asdict
from itertools import permutations
from typing import Optional

from PIL import Image, ImageDraw

SHAPE_COLORS = {
    "star": (225, 175, 0),
    "heart": (215, 30, 60),
    "circle": (30, 110, 215),
    "square": (35, 160, 65),
    "triangle": (235, 115, 20),
    "diamond": (145, 55, 195),
    "cross": (70, 70, 70),
    "moon": (0, 160, 165),
    "ring": (235, 100, 170),
    "house": (130, 80, 40),
}
# Colour words the model may answer with instead of the shape name (each colour is unique
# to one shape, so a colour word identifies the object). Used as extra answer variants.
COLOR_ALIASES = {
    "star": ["yellow"], "heart": ["red"], "circle": ["blue"], "square": ["green"],
    "triangle": ["orange"], "diamond": ["purple"], "cross": ["gray", "grey", "black"], "moon": ["teal", "cyan"],
    "ring": ["pink"], "house": ["brown"],
}
SKIN = (224, 172, 135)
SKIN_EDGE = (150, 100, 70)
SLEEVE = (60, 80, 120)


@dataclass
class Layout:
    grid: int = 4
    cell: int = 112  # px per cell; 112 = 3.5 merged Qwen3-VL tokens (32 px each)
    strip: int = 128  # cue strip height below the grid
    obj_size: int = 72
    cue_length: int = 150  # origin -> fingertip / arrow tip, px
    cue_scale: float = 1.4  # thickness scale of hand / arrow drawing
    bg: tuple = (245, 245, 240)

    @property
    def W(self) -> int:
        return self.grid * self.cell

    @property
    def H(self) -> int:
        return self.grid * self.cell + self.strip

    def cell_center(self, x: int, y: int) -> tuple:
        """(x, y) = (column, row), as in Kang et al."""
        return (x * self.cell + self.cell / 2, y * self.cell + self.cell / 2)

    def cell_box(self, x: int, y: int) -> tuple:
        return (x * self.cell, y * self.cell, (x + 1) * self.cell, (y + 1) * self.cell)

    def origin(self, dx: float = 0.0) -> tuple:
        return (self.W / 2 + dx, self.H - 8)

    def cells(self):
        return [(x, y) for y in range(self.grid) for x in range(self.grid)]


# ----------------------------------------------------------------------------- shapes
def _load_object_png(name: str, size: int, obj_dir: str) -> Optional[Image.Image]:
    p = os.path.join(obj_dir, f"{name}.png")
    if obj_dir and os.path.exists(p):
        return Image.open(p).convert("RGBA").resize((size, size), Image.LANCZOS)
    return None


def draw_object(img: Image.Image, name: str, center: tuple, size: int, obj_dir: str = "data/objects"):
    png = _load_object_png(name, size, obj_dir)
    cx, cy = center
    if png is not None:
        img.paste(png, (int(cx - size / 2), int(cy - size / 2)), png)
        return
    d = ImageDraw.Draw(img)
    r = size / 2
    col = SHAPE_COLORS[name]
    if name == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif name == "square":
        s = r * 0.85
        d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=col)
    elif name == "triangle":
        d.polygon([(cx, cy - r), (cx + r, cy + r * 0.8), (cx - r, cy + r * 0.8)], fill=col)
    elif name == "diamond":
        d.polygon([(cx, cy - r), (cx + r * 0.75, cy), (cx, cy + r), (cx - r * 0.75, cy)], fill=col)
    elif name == "star":
        pts = []
        for k in range(10):
            rr = r if k % 2 == 0 else r * 0.42
            a = -math.pi / 2 + k * math.pi / 5
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        d.polygon(pts, fill=col)
    elif name == "heart":
        q = r * 0.5
        d.ellipse([cx - r, cy - r * 0.75, cx, cy + r * 0.25], fill=col)
        d.ellipse([cx, cy - r * 0.75, cx + r, cy + r * 0.25], fill=col)
        d.polygon([(cx - r * 0.97, cy - q * 0.1), (cx + r * 0.97, cy - q * 0.1), (cx, cy + r)], fill=col)
    elif name == "cross":
        w = r * 0.32
        d.rectangle([cx - w, cy - r, cx + w, cy + r], fill=col)
        d.rectangle([cx - r, cy - w, cx + r, cy + w], fill=col)
    elif name == "moon":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        off = r * 0.45
        d.ellipse([cx - r + off, cy - r - off * 0.2, cx + r + off, cy + r - off * 0.2], fill=img.getpixel((0, 0)))
    elif name == "ring":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        ri = r * 0.55
        d.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], fill=img.getpixel((0, 0)))
    elif name == "house":
        d.polygon([(cx, cy - r), (cx + r, cy - r * 0.1), (cx - r, cy - r * 0.1)], fill=col)
        d.rectangle([cx - r * 0.72, cy - r * 0.1, cx + r * 0.72, cy + r * 0.9], fill=col)
    else:
        raise KeyError(name)


# ------------------------------------------------------------------------------- cues
def _unit(o, t):
    dx, dy = t[0] - o[0], t[1] - o[1]
    n = math.hypot(dx, dy)
    return dx / n, dy / n


def _thick_line(d: ImageDraw.ImageDraw, p0, p1, width, fill):
    d.line([p0, p1], fill=fill, width=int(width))
    r = width / 2
    for p in (p0, p1):
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=fill)


def draw_hand(img: Image.Image, origin: tuple, target: tuple, length: float, k: float = 1.0) -> tuple:
    """Arm from `origin` with index finger pointing at `target`. Returns bbox."""
    d = ImageDraw.Draw(img)
    ux, uy = _unit(origin, target)
    px, py = -uy, ux  # perpendicular
    tip = (origin[0] + ux * length, origin[1] + uy * length)
    finger = 34 * k
    knuckle = (tip[0] - ux * finger, tip[1] - uy * finger)
    wrist = (knuckle[0] - ux * 22 * k, knuckle[1] - uy * 22 * k)
    elbow = origin
    _thick_line(d, elbow, wrist, 26 * k, SLEEVE)
    # fist
    fc = ((knuckle[0] + wrist[0]) / 2, (knuckle[1] + wrist[1]) / 2)
    fr = 19 * k
    d.ellipse([fc[0] - fr, fc[1] - fr, fc[0] + fr, fc[1] + fr], fill=SKIN, outline=SKIN_EDGE, width=2)
    # thumb
    th = (fc[0] + px * 15 * k + ux * 6 * k, fc[1] + py * 15 * k + uy * 6 * k)
    tr = 7 * k
    d.ellipse([th[0] - tr, th[1] - tr, th[0] + tr, th[1] + tr], fill=SKIN, outline=SKIN_EDGE, width=2)
    # index finger
    _thick_line(d, knuckle, tip, 11 * k, SKIN_EDGE)
    _thick_line(d, knuckle, tip, 8 * k, SKIN)
    m = 22 * k
    xs = [elbow[0], tip[0], fc[0] - m, fc[0] + m]
    ys = [elbow[1], tip[1], fc[1] - m, fc[1] + m]
    return (min(xs) - 14 * k, min(ys) - 14 * k, max(xs) + 14 * k, max(ys) + 14 * k)


def draw_arrow(img: Image.Image, origin: tuple, target: tuple, length: float, k: float = 1.0) -> tuple:
    """Arrow with the same origin, direction and tip as the hand."""
    d = ImageDraw.Draw(img)
    ux, uy = _unit(origin, target)
    px, py = -uy, ux
    tip = (origin[0] + ux * length, origin[1] + uy * length)
    head, hw = 26 * k, 14 * k
    base = (tip[0] - ux * head, tip[1] - uy * head)
    _thick_line(d, origin, base, 9 * k, (20, 20, 20))
    d.polygon([tip, (base[0] + px * hw, base[1] + py * hw), (base[0] - px * hw, base[1] - py * hw)], fill=(20, 20, 20))
    xs = [origin[0], tip[0], base[0] + px * hw, base[0] - px * hw]
    ys = [origin[1], tip[1], base[1] + py * hw, base[1] - py * hw]
    return (min(xs) - 6, min(ys) - 6, max(xs) + 6, max(ys) + 6)


CUES = {"hand": draw_hand, "arrow": draw_arrow}


def angle_from(origin, p) -> float:
    return math.degrees(math.atan2(p[1] - origin[1], p[0] - origin[0]))


def angular_sep(layout: Layout, origin, c1, c2) -> float:
    a = abs(angle_from(origin, layout.cell_center(*c1)) - angle_from(origin, layout.cell_center(*c2)))
    return min(a, 360 - a)


# ---------------------------------------------------------------------------- scenes
@dataclass
class Spec:
    """One stimulus. objects: list of (name, (x, y)). cue points at cue_cell."""
    objects: list
    cue: Optional[str] = None  # None | 'hand' | 'arrow'
    cue_cell: Optional[tuple] = None
    origin_dx: float = 0.0
    query: str = ""
    target: Optional[str] = None  # correct answer (object name), if any
    foil: Optional[str] = None  # the competing answer
    group: int = 0  # mean-centering group (object pair), as in Kang et al.
    loc: Optional[tuple] = None  # grid cell this sample is indexed by
    obj_size: Optional[int] = None  # per-sample object size (px); None = layout.obj_size

    def to_dict(self):
        return asdict(self)


def render(spec: Spec, layout: Layout, obj_dir: str = "data/objects") -> tuple:
    img = Image.new("RGB", (layout.W, layout.H), layout.bg)
    d = ImageDraw.Draw(img)
    d.line([(0, layout.grid * layout.cell), (layout.W, layout.grid * layout.cell)], fill=(215, 215, 210), width=2)
    for name, cell in spec.objects:
        draw_object(img, name, layout.cell_center(*cell), spec.obj_size or layout.obj_size, obj_dir)
    meta = {"cue_bbox": None}
    if spec.cue is not None:
        origin = layout.origin(spec.origin_dx)
        meta["cue_bbox"] = CUES[spec.cue](img, origin, layout.cell_center(*spec.cue_cell), layout.cue_length, layout.cue_scale)
    return img, meta


# --------------------------------------------------------------------------- queries
Q_KANG = "Is the {b} to the left or right of the {a}?"  # Kang et al. extraction query (with 'the')
Q_CUE = {
    "hand": "Which shape is the hand pointing at? Answer with only the name of the shape.",
    "arrow": "Which shape is the arrow pointing at? Answer with only the name of the shape.",
}
# A text-only reference control: the referent is named, no cue in the image.
Q_TEXT = "Is there a {b} in the image? Answer with one word."


def object_pairs(names, n_pairs: Optional[int], seed: int):
    pairs = list(permutations(names, 2))
    random.Random(seed).shuffle(pairs)
    return pairs[:n_pairs] if n_pairs else pairs


def specs_object_ids(layout: Layout, names, n_pairs=None, seed=0, sizes=None):
    """E1 (Kang Fig. 5): object b swept over all cells, a at a random other cell.
    sizes: list of object sizes (Kang: 4 sizes per pair and cell); None = layout.obj_size."""
    rng = random.Random(seed)
    out = []
    for g, (a, b) in enumerate(object_pairs(names, n_pairs, seed)):
        for size in (sizes or [None]):
            for cell in layout.cells():
                a_cell = rng.choice([c for c in layout.cells() if c != cell])
                out.append(Spec(objects=[(a, a_cell), (b, cell)], query=Q_KANG.format(a=a, b=b),
                                target=b, group=g, loc=cell, obj_size=size))
    return out


def specs_cue_ids(layout: Layout, names, cue: str, n_pairs=None, seed=0, min_sep=25.0,
                  empty=False, origin_jitter=0.0, sizes=None):
    """E2: cue points at every cell; target b sits there (unless empty), distractor a elsewhere.

    empty=True renders no objects at all, isolating the direction signal from referent binding.
    """
    rng = random.Random(seed + 1)
    out = []
    for g, (a, b) in enumerate(object_pairs(names, n_pairs, seed)):
        for size in (sizes or [None]):
            for cell in layout.cells():
                dx = rng.uniform(-origin_jitter, origin_jitter)
                origin = layout.origin(dx)
                if empty:
                    objs = []
                else:
                    cands = [c for c in layout.cells() if c != cell and angular_sep(layout, origin, c, cell) >= min_sep]
                    objs = [(a, rng.choice(cands)), (b, cell)]
                out.append(Spec(objects=objs, cue=cue, cue_cell=cell, origin_dx=dx, query=Q_CUE[cue],
                                target=None if empty else b, foil=None if empty else a, group=g, loc=cell,
                                obj_size=size))
    return out


def specs_minimal_pairs(layout: Layout, names, cue: str, n: int, seed=0, min_sep=25.0, origin_jitter=0.0):
    """Minimal pairs for patching/steering: same scene, cue -> A (x run) vs cue -> B (y run)."""
    rng = random.Random(seed + 2)
    out = []
    while len(out) < n:
        a, b = rng.sample(list(names), 2)
        ca, cb = rng.sample(layout.cells(), 2)
        dx = rng.uniform(-origin_jitter, origin_jitter)
        if angular_sep(layout, layout.origin(dx), ca, cb) < min_sep:
            continue
        objs = [(a, ca), (b, cb)]
        x = Spec(objects=objs, cue=cue, cue_cell=ca, origin_dx=dx, query=Q_CUE[cue], target=a, foil=b, group=len(out), loc=ca)
        y = Spec(objects=objs, cue=cue, cue_cell=cb, origin_dx=dx, query=Q_CUE[cue], target=b, foil=a, group=len(out), loc=cb)
        out.append((x, y))
    return out


def no_cue(spec: Spec) -> Spec:
    """Same scene with the cue removed (prior / saliency baseline)."""
    d = spec.to_dict()
    d["cue"] = None
    return Spec(**d)
