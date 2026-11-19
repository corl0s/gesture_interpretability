"""Spatial-ID algebra, following Kang et al. (2026) eqs. 2-5 and their code
(spatial_id_derivation/compute_id.py, average_id.py, compute_axes.py; utils/linalg.py).

acts: tensor [N, n_layers, d] for one token slot; locs: list of N (x, y) cells;
groups: list of N group ids (object pair). All functions work on all layers at once.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import torch


def ids_by_loc(acts: torch.Tensor, locs: Sequence[tuple], groups: Sequence[int]) -> Dict[tuple, torch.Tensor]:
    """Eq. 3 then eq. 4: mean-centre within each group over locations, then average
    the centred vectors per location across groups -> universal ID per cell [n_layers, d]."""
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    per_loc = defaultdict(list)
    for g, idx in by_group.items():
        # average repeated locations within a group first (Kang: one image per loc per pair)
        loc_mean = {}
        tmp = defaultdict(list)
        for i in idx:
            tmp[tuple(locs[i])].append(acts[i])
        for loc, v in tmp.items():
            loc_mean[loc] = torch.stack(v).mean(0)
        mu = torch.stack(list(loc_mean.values())).mean(0)
        for loc, v in loc_mean.items():
            per_loc[loc].append(v - mu)
    return {loc: torch.stack(v).mean(0) for loc, v in per_loc.items()}


def axes(ids: Dict[tuple, torch.Tensor], grid: int = 4):
    """Kang compute_axes.py: mean difference over same-row (x axis) / same-column (y axis)
    pairs with x2 >= x1 (their loop includes x2 == x1, i.e. zero vectors; kept for fidelity)."""
    xs, ys = [], []
    for y in range(grid):
        for x1 in range(grid - 1):
            for x2 in range(x1, grid):
                xs.append(ids[(x2, y)] - ids[(x1, y)])
    for x in range(grid):
        for y1 in range(grid - 1):
            for y2 in range(y1, grid):
                ys.append(ids[(x, y2)] - ids[(x, y1)])
    return torch.stack(xs).mean(0), torch.stack(ys).mean(0)


def project_onto_plane(emb: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray):
    """Kang utils/linalg.py: Gram-Schmidt the two axes, project; returns coords [N,2] and
    the fraction of variance captured by the plane."""
    nx = np.linalg.norm(x_axis)
    if nx < 1e-8:  # e.g. text token at layer 0: identical in every image, so the ID is exactly 0
        return np.full((len(emb), 2), np.nan), float("nan")
    u = x_axis / nx
    v = y_axis - y_axis.dot(u) * u
    nv = np.linalg.norm(v)
    if nv < 1e-8:
        return np.full((len(emb), 2), np.nan), float("nan")
    v = v / nv
    c = np.stack([emb @ u, emb @ v], 1)
    tv = emb.var(0).sum()
    ev = (c[:, 0].var() + c[:, 1].var()) / tv if tv > 0 else float("nan")
    return c, ev


def grid_r2(coords: np.ndarray, locs: Sequence[tuple]) -> float:
    """R^2 of an affine map from projected 2D coords to true (x, y) grid coordinates.
    A quantitative stand-in for 'the IDs arrange in an approximate grid' (Kang Fig. 5)."""
    if not np.isfinite(coords).all():
        return float("nan")
    X = np.c_[coords, np.ones(len(coords))]
    Y = np.asarray(locs, dtype=float)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = ((Y - X @ beta) ** 2).sum()
    tot = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - res / tot)


def stack_ids(ids: Dict[tuple, torch.Tensor], cells: Sequence[tuple]) -> torch.Tensor:
    return torch.stack([ids[c] for c in cells])  # [n_cells, n_layers, d]


def cross_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two ID sets per layer: a,b [n_cells, n_layers, d] -> [n_layers, n_cells, n_cells]."""
    a = torch.nn.functional.normalize(a.permute(1, 0, 2), dim=-1)
    b = torch.nn.functional.normalize(b.permute(1, 0, 2), dim=-1)
    return a @ b.transpose(1, 2)


def nearest_id_accuracy(sim: torch.Tensor) -> torch.Tensor:
    """Fraction of cells whose most similar reference ID is the same cell. sim [L, n, n] -> [L]. Chance = 1/n."""
    n = sim.shape[-1]
    return (sim.argmax(-1) == torch.arange(n)).float().mean(-1)


def per_layer_plane_stats(ids_query: Dict[tuple, torch.Tensor], ids_ref: Dict[tuple, torch.Tensor],
                          cells: Sequence[tuple], grid: int = 4):
    """For each layer: project query IDs onto the reference (object) axes; report grid R^2 and
    variance explained. Query == ref gives Kang's own Fig. 5 statistic."""
    xa, ya = axes(ids_ref, grid)
    Q = stack_ids(ids_query, cells)
    out = {"grid_r2": [], "var_explained": []}
    for L in range(Q.shape[1]):
        c, ev = project_onto_plane(Q[:, L].numpy(), xa[L].numpy(), ya[L].numpy())
        out["grid_r2"].append(grid_r2(c, cells))
        out["var_explained"].append(float(ev))
    return {k: np.array(v) for k, v in out.items()}
