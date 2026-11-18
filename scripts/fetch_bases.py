"""Download a seeded sample of real room photos (MIT Indoor-67) to use as authentic bases.

Source: Quattoni & Torralba, "Recognizing Indoor Scenes", CVPR 2009, mirrored on the
Hugging Face Hub (NuclearUnicorn-UWC/mit-indoor-outdoor). Only rooms a short-term
rental actually has are kept.
"""
import argparse
import csv
import io
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

REPO = "NuclearUnicorn-UWC/mit-indoor-outdoor"
ROOMS = ["bedroom", "livingroom", "bathroom", "kitchen", "dining_room"]
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/indoorCVPR_09/Images/{{room}}?limit=1000"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{path}}"
MIN_SIDE = 256


def list_room(room: str) -> list[str]:
    with urllib.request.urlopen(API.format(room=room), timeout=60) as r:
        return sorted(x["path"] for x in json.load(r) if x["path"].lower().endswith(".jpg"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-room", type=int, default=190)
    ap.add_argument("--out", default="data/bases")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    for room in ROOMS:
        paths = list_room(room)
        rng.shuffle(paths)
        cand = paths[: args.per_room + 60]  # spare candidates for tiny or broken files

        def get(p):
            try:
                with urllib.request.urlopen(RAW.format(path=p), timeout=60) as r:
                    return Image.open(io.BytesIO(r.read())).convert("RGB")
            except Exception as e:  # broken files exist in the original set
                print("skip", p, e)
                return None

        with ThreadPoolExecutor(16) as ex:
            imgs = list(ex.map(get, cand))
        kept = 0
        for p, img in zip(cand, imgs):  # keep the seeded order, so the sample is reproducible
            if kept >= args.per_room:
                break
            if img is None or min(img.size) < MIN_SIDE:
                continue
            base_id = f"{room}_{kept:04d}"
            img.save(out / f"{base_id}.png")  # lossless copy of the decoded photo
            rows.append({"base_id": base_id, "room": room, "source_path": p,
                         "width": img.size[0], "height": img.size[1]})
            kept += 1
        print(room, kept)

    with open(Path(args.out).parent / "bases_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
