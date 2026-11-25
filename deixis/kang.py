"""Exact re-implementation of Kang et al.'s spatial-ID data and averaging (arXiv 2601.12626),
following their released code (external/linear-mech-vlms/spatial_id_derivation):

  dataengine/tile_two_objects.py : 4x4 grid, cell 224 px (896 x 896 canvas), object a at one cell,
                                   object b at every other cell, both resized to `size` and centred,
                                   transparent canvas (their extraction_data.sh passes no --bg_dir,
                                   so the RGB image has a black background).
  dataengine/extraction_data.sh  : 10 objects -> 90 ordered pairs, sizes 74/124/174/224
                                   (16 x 15 x 4 = 960 images per pair, 86,400 in total).
  batch_extraction.sh            : query "Is {b} to the left or right of {a}?", read at b's last
                                   sub-token (first occurrence), every layer.
  compute_id.py                  : mean-centre within (pair, cell of a) over all (cell of b, size),
                                   then average over cells of a -> per-pair ID keyed (x, y, size).
  average_id.py                  : average per-pair IDs over pairs -> universal ID[(x, y, size)].
  compute_axes.py                : x/y axes from mean grid differences at size 224 (x2 >= x1, as in
                                   their loop, i.e. including zero differences).
Cells are (x, y) = (column, row), as in their grid_coords.
"""
from __future__ import annotations

import os
from itertools import permutations
from typing import Dict, List, Tuple

import torch
from PIL import Image

KANG_OBJECTS = ["backpack", "fan", "camel", "frog", "dog", "cake", "log", "robot", "person", "teapot"]
KANG_SIZES = [74, 124, 174, 224]
CELL = 224
GRID = 4


def kang_pairs() -> List[Tuple[str, str]]:
    """Ordered (a, b) pairs in the same order as extraction_data.sh's nested loop."""
    return [(a, b) for a in KANG_OBJECTS for b in KANG_OBJECTS if a != b]


def kang_query(a: str, b: str) -> str:
    return f"Is {b} to the left or right of {a}?"


_cache: Dict[Tuple[str, int], Image.Image] = {}


def _obj(path: str, size: int) -> Image.Image:
    key = (path, size)
    if key not in _cache:
        _cache[key] = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
    return _cache[key]


def render_pair(obj_dir: str, a: str, b: str, a_cell, b_cell, size: int, bg: Image.Image | None = None) -> Image.Image:
    """One image of tile_two_objects.py (no local variation); returns RGB like their PIL .convert('RGB')."""
    canvas = bg.copy().convert("RGBA").resize((CELL * GRID, CELL * GRID)) if bg is not None \
        else Image.new("RGBA", (CELL * GRID, CELL * GRID), (0, 0, 0, 0))
    for name, (cx, cy) in ((a, a_cell), (b, b_cell)):
        im = _obj(os.path.join(obj_dir, f"{name}.png"), size)
        x = int(cx * CELL + (CELL - size) / 2)
        y = int(cy * CELL + (CELL - size) / 2)
        canvas.paste(im, (x, y), im)
    return canvas.convert("RGB")


def cells():
    # their grid_coords order: x outer, y inner
    return [(x, y) for x in range(GRID) for y in range(GRID)]


def pair_specs():
    """For one pair: list of (a_cell, b_cell, size) in tile_two_objects.py order."""
    return [(ac, bc, s) for s in KANG_SIZES for ac in cells() for bc in cells() if ac != bc]


def center_and_average(acts: Dict[Tuple, torch.Tensor]) -> Dict[Tuple, torch.Tensor]:
    """acts[(a_cell, b_cell, size)] = tensor[..., d] for ONE pair ->
    per-pair ID[(bx, by, size)]: centre within each a_cell over its (b_cell, size), then average
    over a_cells (compute_id.py's subdirectory branch)."""
    by_a: Dict[Tuple, Dict[Tuple, torch.Tensor]] = {}
    for (ac, bc, s), v in acts.items():
        by_a.setdefault(ac, {})[(bc[0], bc[1], s)] = v
    out: Dict[Tuple, List[torch.Tensor]] = {}
    for ac, d in by_a.items():
        mu = torch.stack(list(d.values())).mean(0)
        for loc, v in d.items():
            out.setdefault(loc, []).append(v - mu)
    return {loc: torch.stack(vs).mean(0) for loc, vs in out.items()}


def kang_axes(ids: Dict[Tuple, torch.Tensor], size: int = 224):
    """compute_axes.py: returns (x_axis, y_axis) from IDs keyed (x, y, size)."""
    ys, xs = [], []
    for x in range(GRID):
        for y1 in range(GRID - 1):
            for y2 in range(y1, GRID):
                ys.append(ids[(x, y2, size)] - ids[(x, y1, size)])
    for y in range(GRID):
        for x1 in range(GRID - 1):
            for x2 in range(x1, GRID):
                xs.append(ids[(x2, y, size)] - ids[(x1, y, size)])
    return torch.stack(xs).mean(0), torch.stack(ys).mean(0)
