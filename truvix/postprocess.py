"""What happens to every claim photo on its way to a reviewer, honest or not.

Applied identically to authentic and edited images so the detector cannot learn
"was saved once vs twice" or "was resized vs not" as a shortcut.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

SIZE = 512


def square_crop(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y:y + s, x:x + s], (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_CUBIC)


def benign_global_edit(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brightness / contrast / gamma tweaks a phone applies. Not tampering."""
    out = img.astype(np.float32) / 255.0
    if rng.random() < 0.5:
        out = out ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.5:
        out = (out - 0.5) * rng.uniform(0.85, 1.15) + 0.5 + rng.uniform(-0.06, 0.06)
    return (out.clip(0, 1) * 255).round().astype(np.uint8)


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=int(quality))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def delivery(img: np.ndarray, mask: np.ndarray | None, rng: np.random.Generator):
    """Random downscale + JPEG, sometimes a second messaging-app JPEG. Returns (img, mask, params)."""
    scale = float(rng.uniform(0.6, 1.0))
    q1 = int(rng.integers(65, 96))
    second = bool(rng.random() < 0.3)
    h, w = img.shape[:2]
    nh, nw = int(round(h * scale)), int(round(w * scale))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = jpeg(out, q1)
    q2 = None
    if second:
        q2 = int(rng.integers(60, 86))
        out = jpeg(out, q2)
    m = None if mask is None else cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return out, m, {"scale": round(scale, 3), "jpeg_q1": q1, "jpeg_q2": q2}


def attack(img: np.ndarray, kind: str) -> np.ndarray:
    """Extra laundering used only in the robustness test."""
    if kind == "jpeg50":
        return jpeg(img, 50)
    if kind == "half":
        h, w = img.shape[:2]
        return cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    if kind == "screenshot":  # resize up, blur slightly, PNG-like lossless
        h, w = img.shape[:2]
        up = cv2.resize(img, (int(w * 1.3), int(h * 1.3)), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(up, (3, 3), 0)
    raise ValueError(kind)
