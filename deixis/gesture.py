"""Kang-style gesture scenes for gesture-ID extraction (Part B).

Canvas, grid and objects follow deixis/kang.py (896 x 896 black canvas, 4 x 4 cells of 224 px, Kang's
Objaverse renders), with two stated differences: objects are cropped to their visible outline and
resized so the longest side is OBJ_PX (Kang's PNGs are padded: the object spans ~48% of its tile), and
objects that are hard to see on black are excluded (POOL).

Design (targets x origins, one fixed scene):
  scene   = K objects at K distinct cells (same objects, cells and hand in every condition of the scene)
  target  = each of the K objects in turn (the hand points at its cell centre)
  origin  = the arm enters from one of ORIGINS (bottom, sides, top edges)
  cue     = photo (a real YouRefIt hand cut-out), cartoon (drawn hand), arrow, or empty (photo hand, no objects)
A (target, origin) combination is valid only if origin->target distance >= LEN + MIN_GAP and, for every
cue type, the drawn cue does not touch any object's box (dilated by MARGIN). All cue types share the
same valid set, so conditions are matched. Gesture IDs are then estimated as per-cell effects after
removing (scene, origin) means: Kang's "fix everything, vary one factor, subtract the mean", with the
origin averaged out so the ID is keyed by the pointed cell, not the arm angle.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .stimuli import _unit, draw_arrow, draw_hand, draw_photo_hand

CANVAS, CELL, GRID = 896, 224, 4
OBJ_PX = 160
LEN = 340  # base -> fingertip length (px); 340 keeps ~30% of (target, origin) combos valid with 5 objects
MIN_GAP = 100  # origin->target distance must be >= LEN + MIN_GAP (fingertip stops short)
MARGIN = 8  # px dilation of object boxes for the overlap check
CARTOON_K = 2.8
ARROW_COLOR = (200, 200, 200)
ORIGINS = {
    "bot_l": (150, 896), "bot_m": (448, 896), "bot_r": (746, 896),
    "left_hi": (0, 600), "left_lo": (0, 750), "right_hi": (896, 600), "right_lo": (896, 750),
    "top_l": (150, 0), "top_m": (448, 0), "top_r": (746, 0),
}
# Kang's 55 renders minus those hard to see on black (dark: lum < 60; spindly: alpha fill < 0.25; listed)
# and "glove" (a hand-shaped object, confusable with the pointing hand). 41 objects.
EXCLUDE = {"handsaw", "hook", "scissors", "tire", "camcorder", "cat", "chime", "monitor", "person", "handcuff",
           "axe", "clippers", "spoon", "glove"}
CUE_WORD = {"photo": "hand", "cartoon": "hand", "arrow": "arrow", "empty": "hand"}
QUERY = "Is the {cue} pointing at the {first} or the {second}?"
PREFIX = "The {cue} is pointing at the"


def pool(obj_dir: str) -> List[str]:
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(obj_dir) if f.endswith(".png"))
    return [n for n in names if n not in EXCLUDE]


def cell_center(c):
    return (c[0] * CELL + CELL / 2, c[1] * CELL + CELL / 2)


_obj_cache: Dict[Tuple[str, str], Image.Image] = {}


def trimmed(obj_dir: str, name: str) -> Image.Image:
    key = (obj_dir, name)
    if key not in _obj_cache:
        im = Image.open(os.path.join(obj_dir, f"{name}.png")).convert("RGBA")
        bb = im.getchannel("A").point(lambda v: 255 if v > 20 else 0).getbbox()
        im = im.crop(bb)
        s = OBJ_PX / max(im.size)
        _obj_cache[key] = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    return _obj_cache[key]


def obj_box(obj_dir: str, name: str, c):
    im = trimmed(obj_dir, name)
    cx, cy = cell_center(c)
    x0, y0 = int(cx - im.width / 2), int(cy - im.height / 2)
    return (x0, y0, x0 + im.width, y0 + im.height)


def draw_cue(layer: Image.Image, cue: str, origin, target, hand: str, hand_dir: str):
    if cue in ("photo", "empty"):
        draw_photo_hand(layer, origin, target, LEN, hand=hand, hand_dir=hand_dir)
    elif cue == "cartoon":
        draw_hand(layer, origin, target, LEN, CARTOON_K)
    elif cue == "arrow":
        draw_arrow(layer, origin, target, LEN, CARTOON_K, color=ARROW_COLOR)
    else:
        raise KeyError(cue)


def cue_mask(cue: str, origin, target, hand: str, hand_dir: str) -> np.ndarray:
    layer = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw_cue(layer, cue, origin, target, hand, hand_dir)
    return np.asarray(layer.getchannel("A")) > 20


@dataclass
class Scene:
    idx: int
    objects: List[Tuple[str, Tuple[int, int]]]  # (name, cell)
    hand: str
    valid: Dict[str, List[int]] = field(default_factory=dict)  # origin -> indices of valid targets


def make_scene(idx: int, names: List[str], hands: List[str], obj_dir: str, hand_dir: str, k: int = 5,
               seed: int = 0, cues=("photo", "cartoon", "arrow")) -> Scene:
    rng = random.Random(f"{seed}-{idx}")
    cells = rng.sample([(x, y) for y in range(GRID) for x in range(GRID)], k)
    objs = list(zip(rng.sample(names, k), cells))
    sc = Scene(idx, objs, hands[idx % len(hands)])
    boxes = []
    for n, c in objs:
        x0, y0, x1, y1 = obj_box(obj_dir, n, c)
        boxes.append((max(0, x0 - MARGIN), max(0, y0 - MARGIN), min(CANVAS, x1 + MARGIN), min(CANVAS, y1 + MARGIN)))
    for oname, o in ORIGINS.items():
        ok = []
        for t, (_, c) in enumerate(objs):
            tc = cell_center(c)
            if math.dist(o, tc) < LEN + MIN_GAP:
                continue
            clear = True
            for cue in cues:
                m = cue_mask(cue, o, tc, sc.hand, hand_dir)
                if any(m[y0:y1, x0:x1].any() for (x0, y0, x1, y1) in boxes):
                    clear = False
                    break
            if clear:
                ok.append(t)
        if len(ok) >= 2:  # need >= 2 targets per (scene, origin) to subtract a mean
            sc.valid[oname] = ok
    return sc


def render(sc: Scene, origin: str, t: int, cue: str, obj_dir: str, hand_dir: str) -> Image.Image:
    img = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 255))
    if cue != "empty":
        for n, c in sc.objects:
            im = trimmed(obj_dir, n)
            x0, y0, _, _ = obj_box(obj_dir, n, c)
            img.paste(im, (x0, y0), im)
    layer = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw_cue(layer, cue, ORIGINS[origin], cell_center(sc.objects[t][1]), sc.hand, hand_dir)
    img.alpha_composite(layer)
    return img.convert("RGB")


def foil_index(sc: Scene, t: int, origin: str, rng: random.Random, ok_pair=None) -> int:
    """Foil = another object in the scene, preferring one at >= 25 deg from the target as seen from the
    origin (so the two-choice question has a clear answer); ok_pair(a, b) can veto token collisions."""
    o = ORIGINS[origin]
    tc = cell_center(sc.objects[t][1])
    cand = []
    for f, (n, c) in enumerate(sc.objects):
        if f == t or (ok_pair and not ok_pair(sc.objects[t][0], n)):
            continue
        a1 = math.degrees(math.atan2(tc[1] - o[1], tc[0] - o[0]))
        fc = cell_center(c)
        a2 = math.degrees(math.atan2(fc[1] - o[1], fc[0] - o[0]))
        d = abs(a1 - a2) % 360
        cand.append((min(d, 360 - d) >= 25, f))
    good = [f for g, f in cand if g] or [f for _, f in cand]
    return rng.choice(good)


def geometry(sc: Scene, origin: str, t: int) -> dict:
    o = ORIGINS[origin]
    tc = cell_center(sc.objects[t][1])
    ux, uy = _unit(o, tc)
    tip = (o[0] + ux * LEN, o[1] + uy * LEN)
    d = [math.dist(tip, cell_center(c)) for _, c in sc.objects]
    return {"angle": math.degrees(math.atan2(-uy, ux)) % 360,  # 0 = right, 90 = up
            "tip": tip, "tip_gap": d[t], "tip_nearest": int(np.argmin(d))}
