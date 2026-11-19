"""Shared CLI plumbing: arguments, model loading, token-slot lookup."""
from __future__ import annotations

import argparse
import json
import os

import torch

from .stimuli import Layout, SHAPE_COLORS, Spec, render


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--model", default="qwen3vl-2b", help="alias (qwen3vl-2b/4b/8b/32b) or HF id")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto", help="bfloat16 | float16 | float32")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--objects", default=",".join(SHAPE_COLORS), help="comma-separated object names")
    p.add_argument("--obj_dir", default="data/objects", help="optional RGBA PNGs named <object>.png")
    p.add_argument("--cell", type=int, default=112)
    p.add_argument("--strip", type=int, default=128)
    p.add_argument("--obj_size", type=int, default=72)
    p.add_argument("--cue_length", type=int, default=150)
    p.add_argument("--cue_scale", type=float, default=1.4)
    p.add_argument("--min_sep", type=float, default=25.0, help="min angular separation (deg) target vs distractor")
    p.add_argument("--origin_jitter", type=float, default=0.0, help="+- px horizontal jitter of the cue origin")
    return p


def layout_from(args) -> Layout:
    return Layout(cell=args.cell, strip=args.strip, obj_size=args.obj_size, cue_length=args.cue_length,
                  cue_scale=args.cue_scale)


def names_from(args):
    return [s.strip() for s in args.objects.split(",") if s.strip()]


def load_vlm(args):
    from .model import VLM
    vlm = VLM(args.model, device=args.device, dtype=args.dtype)
    if not os.path.isdir(args.obj_dir) or not os.listdir(args.obj_dir):  # procedural shapes
        from .stimuli import COLOR_ALIASES
        vlm.answer_aliases = COLOR_ALIASES
    vlm.check_answer_vocab(names_from(args))
    return vlm


def model_tag(args) -> str:
    return args.model.replace("/", "_")


def out_dir(args, *parts) -> str:
    d = os.path.join(args.out, model_tag(args), *parts)
    os.makedirs(d, exist_ok=True)
    return d


def encode_spec(vlm, spec: Spec, layout: Layout, obj_dir: str):
    img, meta = render(spec, layout, obj_dir)
    return vlm.encode(img, spec.query), meta


CUE_WORDS = {"hand": "hand", "arrow": "arrow"}


def slot_positions(vlm, enc, spec: Spec, meta: dict, slot: str, layout: Layout) -> torch.Tensor:
    """Token positions for a named slot.

    text slots : 'last' | 'target_word' | 'cue_word' | 'pointing' | 'at' | 'text_all'
    image slots: 'image_all' | 'image_cue' | 'image_target' | 'image_foil'
    """
    if slot == "last":
        return torch.tensor([enc.last])
    if slot == "target_word":
        return torch.tensor([vlm.find_token(enc, spec.target)])
    if slot == "cue_word":
        return torch.tensor([vlm.find_token(enc, CUE_WORDS[spec.cue])])
    if slot in ("pointing", "at"):
        return torch.tensor([vlm.find_token(enc, slot)])
    if slot == "text_all":
        return vlm.text_positions(enc)
    if slot == "image_all":
        return enc.image_pos
    if slot == "image_cue":
        return vlm.image_tokens_in_box(enc, meta["cue_bbox"])
    if slot in ("image_target", "image_foil"):
        name = spec.target if slot == "image_target" else spec.foil
        cell = dict(spec.objects)[name]
        return vlm.image_tokens_in_box(enc, layout.cell_box(*cell))
    raise KeyError(slot)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
