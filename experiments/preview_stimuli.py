"""Render a contact sheet of stimuli for visual inspection (no model needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from deixis.common import base_parser, layout_from, names_from
from deixis.stimuli import render, specs_cue_ids, specs_minimal_pairs, specs_object_ids


def main():
    args = base_parser(__doc__).parse_args()
    layout, names = layout_from(args), names_from(args)
    rows = [
        specs_object_ids(layout, names, 1, args.seed)[:8],
        specs_cue_ids(layout, names, "hand", 1, args.seed, args.min_sep)[:8],
        specs_cue_ids(layout, names, "arrow", 1, args.seed, args.min_sep)[:8],
        [s for p in specs_minimal_pairs(layout, names, "hand", 4, args.seed, args.min_sep) for s in p],
    ]
    W, H = layout.W // 2, layout.H // 2
    sheet = Image.new("RGB", (W * 8, H * len(rows)), "white")
    for r, specs in enumerate(rows):
        for c, s in enumerate(specs):
            sheet.paste(render(s, layout, args.obj_dir)[0].resize((W, H)), (c * W, r * H))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "stimuli_preview.png")
    sheet.save(path)
    print(path)


if __name__ == "__main__":
    main()
