"""Check a claim photo.

  python -m truvix.cli photo.jpg --heatmap out.png
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .patchnet import heatmap, load

ROOT = Path(__file__).resolve().parent.parent


def load_model(path: Path | None = None):
    path = path or ROOT / "models" / "patchnet.pt"
    m = load(path)
    meta = json.loads((ROOT / "results" / "metrics.json").read_text()) if (ROOT / "results" / "metrics.json").exists() else {}
    thr = meta.get("detectors", {}).get("patchnet", {}).get("threshold", 0.5)
    return m, thr


def overlay(img: np.ndarray, hm: np.ndarray) -> np.ndarray:
    """Terracotta tint where the detector thinks pixels were edited."""
    a = (np.clip((hm - 0.3) / 0.7, 0, 1) * 0.65)[..., None]
    tint = np.array([180, 85, 45], np.float32)
    return (img * (1 - a) + tint * a).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Flag AI-edited or doctored damage-claim photos.")
    ap.add_argument("image")
    ap.add_argument("--heatmap", help="write an overlay PNG here")
    args = ap.parse_args()
    img = np.asarray(Image.open(args.image).convert("RGB"))
    if max(img.shape[:2]) > 1024:  # keep inference fast; training photos were <= 512 px
        s = 1024 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    model, thr = load_model()
    hm, score = heatmap(model, img)
    verdict = "REVIEW: likely edited" if score > thr else "no edit found"
    print(json.dumps({"image": args.image, "score": round(score, 4), "threshold": round(thr, 4),
                      "verdict": verdict, "flagged_area": round(float((hm > 0.5).mean()), 4)}))
    if args.heatmap:
        Image.fromarray(overlay(img, hm)).save(args.heatmap)


if __name__ == "__main__":
    main()
