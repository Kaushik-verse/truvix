"""Build the Truvix dataset from the downloaded base photos.

For each real room photo we write one authentic claim photo and one tampered claim
photo (same scene, so the detector cannot cheat on scene content), both pushed
through the same delivery pipeline. Splits are by base photo: a room in the test
set never appears in training.

  python scripts/make_dataset.py                 # paint, splice
  python scripts/make_dataset.py --diffusion     # + Stable Diffusion inpainting (needs a 2 GB download)
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

import cv2

from truvix.damage import DAMAGE_PROMPTS, paint_damage, random_blob_mask, splice_damage, visible_mask
from truvix.postprocess import benign_global_edit, delivery, square_crop

FAMILIES = ["paint", "splice", "diffusion"]
OOD_PROMPTS = [
    "a flooded bathroom with water all over the floor",
    "a living room with a broken coffee table and stained carpet",
    "a bedroom with a torn mattress and a hole in the wall",
    "a kitchen with burnt cabinets and a cracked countertop",
    "a dining room with a smashed glass table",
]


def load(p: Path) -> np.ndarray:
    return square_crop(np.asarray(Image.open(p).convert("RGB")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="data/bases")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--diffusion", action="store_true")
    ap.add_argument("--ood", type=int, default=60, help="fully generated 'damage' photos, test only (needs --diffusion)")
    ap.add_argument("--guidance", type=float, default=1.0, help="SD guidance; >1 follows the damage prompt more strongly")
    ap.add_argument("--steps", type=int, default=5, help="SD (LCM) denoising steps")
    ap.add_argument("--min-visible", type=float, default=0.0,
                    help="v3: an edit must visibly change at least this share of the image, else retry with a new seed")
    ap.add_argument("--max-tries", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    bases = sorted(Path(args.bases).glob("*.png"))

    # split by base photo, stratified by room: 70 / 10 / 20
    split_of = {}
    by_room: dict[str, list[Path]] = {}
    for b in bases:
        by_room.setdefault(b.stem.rsplit("_", 1)[0], []).append(b)
    for room, items in by_room.items():
        perm = rng.permutation(len(items))
        n = len(items)
        for rank, i in enumerate(perm):
            split_of[items[i].stem] = "train" if rank < 0.7 * n else "val" if rank < 0.8 * n else "test"

    families = [f for f in FAMILIES if args.diffusion or f != "diffusion"]
    inpaint = None
    if args.diffusion:
        from truvix.damage import DiffusionInpainter
        inpaint = DiffusionInpainter()
    rows = []
    stats = {"rejected": {f: 0 for f in families}, "kept_after_max_tries": {f: 0 for f in families},
             "tries": {f: [] for f in families}}
    t0 = time.time()
    for k, b in enumerate(bases):
        bid = b.stem
        img = benign_global_edit(load(b), rng)
        family = families[k % len(families)]

        def make_edit():
            if family == "paint":
                fake, mask = paint_damage(img, rng)
                return fake, mask, ""
            if family == "splice":
                donor = img if rng.random() < 0.5 else load(bases[int(rng.integers(len(bases)))])
                fake, mask = splice_damage(img, donor, rng)
                return fake, mask, ""
            mask = random_blob_mask(*img.shape[:2], rng)
            prompt = DAMAGE_PROMPTS[int(rng.integers(len(DAMAGE_PROMPTS)))]
            fake, mask = inpaint(img, mask, prompt, seed=int(rng.integers(1 << 30)),
                                 steps=args.steps, guidance=args.guidance)
            return fake, mask, prompt

        for tries in range(1, args.max_tries + 1):
            fake, mask, prompt = make_edit()
            if args.min_visible <= 0:
                break
            vis = visible_mask(img, fake, mask)
            if vis.mean() >= args.min_visible:
                mask = vis  # label = pixels that visibly changed
                break
            stats["rejected"][family] += 1
        else:
            mask = visible_mask(img, fake, mask) if visible_mask(img, fake, mask).any() else mask
            stats["kept_after_max_tries"][family] += 1
        stats["tries"][family].append(tries)

        for label, im, m in [(0, img, None), (1, fake, mask)]:
            name = f"{bid}_{'fake' if label else 'real'}"
            im2, m2, pp = delivery(im, m, rng)
            Image.fromarray(im2).save(out / "images" / f"{name}.png")  # lossless store of the delivered JPEG pixels
            if m2 is not None:
                Image.fromarray(m2 * 255).save(out / "masks" / f"{name}.png")
            rows.append({"image": f"images/{name}.png", "mask": f"masks/{name}.png" if label else "",
                         "label": label, "family": family if label else "authentic", "base_id": bid,
                         "split": split_of[bid], "prompt": prompt,
                         "edited_frac": round(float(m2.mean()), 4) if m2 is not None else 0.0, **pp})
        if k % 25 == 0:
            print(f"{k}/{len(bases)}  {time.time() - t0:.0f}s", flush=True)

    # unseen tool: OpenCV Telea inpainting (classical, never seen in training), test rooms only
    for b in bases:
        if split_of[b.stem] != "test":
            continue
        img = benign_global_edit(load(b), rng)
        mask = random_blob_mask(*img.shape[:2], rng, (0.03, 0.12))
        fake = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        if args.min_visible > 0 and visible_mask(img, fake, mask).any():
            mask = visible_mask(img, fake, mask)
        im2, m2, pp = delivery(fake, mask, rng)
        name = f"{b.stem}_telea"
        Image.fromarray(im2).save(out / "images" / f"{name}.png")
        Image.fromarray(m2 * 255).save(out / "masks" / f"{name}.png")
        rows.append({"image": f"images/{name}.png", "mask": f"masks/{name}.png", "label": 1,
                     "family": "telea_unseen", "base_id": b.stem, "split": "ood", "prompt": "",
                     "edited_frac": round(float(m2.mean()), 4), **pp})

    # unseen attack: whole photo generated from a prompt (no real photo underneath)
    if inpaint is not None and args.ood:
        full = np.ones((512, 512), np.uint8)
        blank = np.full((512, 512, 3), 127, np.uint8)
        for i in range(args.ood):
            prompt = OOD_PROMPTS[i % len(OOD_PROMPTS)]
            gen, _ = inpaint(blank, full, prompt, seed=10_000 + i, steps=max(args.steps, 6),
                             paste_back=False, guidance=args.guidance)
            im2, _, pp = delivery(gen, None, rng)
            name = f"ood_{i:03d}_fake"
            Image.fromarray(im2).save(out / "images" / f"{name}.png")
            rows.append({"image": f"images/{name}.png", "mask": "", "label": 1, "family": "fullgen",
                         "base_id": name, "split": "ood", "prompt": prompt, "edited_frac": 1.0, **pp})

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import json
    summary = {"guidance": args.guidance, "steps": args.steps, "min_visible": args.min_visible,
               "rejected_edits": stats["rejected"], "kept_after_max_tries": stats["kept_after_max_tries"],
               "mean_tries": {f: round(float(np.mean(v)), 3) for f, v in stats["tries"].items() if v}}
    (out / "generation_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    print("done", len(rows), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
