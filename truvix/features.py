"""Hand-crafted forensic features for the classical baselines.

ELA  - error level analysis: re-save at a known JPEG quality and look at how unevenly
       the image changes. Edited regions often have a different compression history.
SRM  - spatial rich-model high-pass residuals (Fridrich & Kodovsky, 2012). We measure how
       uneven the residual noise energy is across the image: a pasted or generated region
       usually carries a different noise level than the camera noise around it.
"""
from __future__ import annotations

import cv2
import numpy as np

from .postprocess import jpeg

# three classic SRM kernels: first-order, second-order, and the 5x5 "KV" kernel
_SRM = [
    np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float32),
    np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], np.float32) / 2,
    np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], np.float32) / 12,
]
BLOCK = 32
_PCTS = [5, 25, 50, 75, 95, 100]


def _block_map(x: np.ndarray, block: int = BLOCK) -> np.ndarray:
    h, w = x.shape[:2]
    h2, w2 = h // block * block, w // block * block
    x = x[:h2, :w2]
    return x.reshape(h2 // block, block, w2 // block, block).mean(axis=(1, 3))


def _spread(blocks: np.ndarray) -> list[float]:
    v = blocks.ravel()
    p = np.percentile(v, _PCTS)
    med = p[2] + 1e-6
    return [*p, float(v.std()), float(p[-1] / med), float(p[4] / med)]


def ela_features(img: np.ndarray, quality: int = 90) -> np.ndarray:
    diff = np.abs(img.astype(np.float32) - jpeg(img, quality).astype(np.float32)).mean(axis=2)
    return np.array(_spread(_block_map(diff)), np.float32)


def srm_residuals(img: np.ndarray) -> list[np.ndarray]:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return [np.clip(cv2.filter2D(g, -1, k), -3, 3) for k in _SRM]


def srm_features(img: np.ndarray) -> np.ndarray:
    feats = []
    for r in srm_residuals(img):
        feats += _spread(_block_map(r ** 2))
        hist, _ = np.histogram(r, bins=7, range=(-3.5, 3.5), density=True)
        feats += hist.tolist()
    return np.array(feats, np.float32)


ELA_DIM = len(_PCTS) + 3
SRM_DIM = 3 * (len(_PCTS) + 3 + 7)
