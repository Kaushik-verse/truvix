"""Local showcase: score a folder of photos and save one contact sheet with heatmaps.

  python scripts/demo.py                       # the example photos shipped in docs/examples
  python scripts/demo.py --images my_photos/   # your own photos
  open demo_output.png

Runs on a laptop CPU or Apple GPU in under a minute; no dataset or Colab needed.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from truvix.cli import load_model, overlay
from truvix.patchnet import heatmap

SKIP = ("_mask", "_original", "_heatmap")
BG, INK, RISK = (247, 246, 242), (43, 43, 43), (180, 85, 45)
TILE = 300


def font(size: int):
    for name in ("Georgia.ttf", "/System/Library/Fonts/Supplemental/Georgia.ttf", "DejaVuSerif.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="docs/examples")
    ap.add_argument("--out", default="demo_output.png")
    ap.add_argument("--max", type=int, default=12)
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg") and not any(s in p.stem for s in SKIP)
                   and p.stem not in ("splice", "authentic"))
    if args.images == "docs/examples":  # balanced default: two photos of each kind
        picked, seen = [], {}
        for p in paths:
            kind = p.stem.rsplit("_", 1)[0]
            if seen.get(kind, 0) < 2:
                seen[kind] = seen.get(kind, 0) + 1
                picked.append(p)
        paths = picked
    paths = paths[: args.max]
    if not paths:
        raise SystemExit(f"no photos found in {args.images}")
    model, thr = load_model()
    cols = 4
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (TILE + 16) + 16, rows * (TILE + 64) + 16), BG)
    draw = ImageDraw.Draw(sheet)
    f_name, f_score = font(15), font(13)
    print(f"threshold {thr:.3f}  (scores above it are sent to a reviewer)")
    for i, p in enumerate(paths):
        img = np.asarray(Image.open(p).convert("RGB"))
        hm, score = heatmap(model, img)
        flagged = score > thr
        print(f"{p.name:28s} score {score:.3f}  {'FLAG for review' if flagged else 'looks unedited'}")
        tile = Image.fromarray(overlay(img, hm)).resize((TILE, TILE))
        x, y = 16 + (i % cols) * (TILE + 16), 16 + (i // cols) * (TILE + 64)
        sheet.paste(tile, (x, y))
        draw.text((x, y + TILE + 6), p.stem, fill=INK, font=f_name)
        draw.text((x, y + TILE + 28), f"score {score:.2f}  {'FLAG' if flagged else 'ok'}",
                  fill=RISK if flagged else INK, font=f_score)
    sheet.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
