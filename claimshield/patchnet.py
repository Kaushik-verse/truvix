"""Patch-level detector: a small ImageNet ResNet-18 fine-tuned to answer
"were the pixels in this 128x128 window edited?". Sliding it over a photo gives a
heatmap (where) and an image score (whether), which is what a claims reviewer needs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

PATCH = 128
STRIDE = 64
POS_FRAC = 0.25  # a patch is "edited" if at least this share of its pixels were edited
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def build(pretrained: bool = True) -> nn.Module:
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def to_tensor(patches: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(patches)).permute(0, 3, 1, 2).contiguous().float() / 255.0
    return (t - _MEAN) / _STD


def grid(h: int, w: int, patch: int = PATCH, stride: int = STRIDE) -> list[tuple[int, int]]:
    ys = list(range(0, max(h - patch, 0) + 1, stride))
    xs = list(range(0, max(w - patch, 0) + 1, stride))
    if ys[-1] != h - patch:
        ys.append(max(h - patch, 0))
    if xs[-1] != w - patch:
        xs.append(max(w - patch, 0))
    return [(y, x) for y in ys for x in xs]


def patch_labels(mask: np.ndarray | None, coords, patch: int = PATCH) -> np.ndarray:
    """1 = edited, 0 = clean, -1 = ambiguous (thin overlap; skipped in training)."""
    if mask is None:
        return np.zeros(len(coords), np.int64)
    out = []
    for y, x in coords:
        f = float(mask[y:y + patch, x:x + patch].mean())
        out.append(1 if f >= POS_FRAC else 0 if f == 0 else -1)
    return np.array(out, np.int64)


@torch.no_grad()
def heatmap(model: nn.Module, img: np.ndarray, dev: str | None = None, batch: int = 64) -> tuple[np.ndarray, float]:
    """Returns (per-pixel edit probability, image score = max window probability)."""
    dev = dev or next(model.parameters()).device
    h, w = img.shape[:2]
    coords = grid(h, w)
    probs = []
    for i in range(0, len(coords), batch):
        c = coords[i:i + batch]
        x = to_tensor(np.stack([img[y:y + PATCH, xx:xx + PATCH] for y, xx in c])).to(dev)
        probs.append(torch.sigmoid(model(x)).squeeze(1).cpu().numpy())
    probs = np.concatenate(probs)
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for (y, x), p in zip(coords, probs):
        acc[y:y + PATCH, x:x + PATCH] += p
        cnt[y:y + PATCH, x:x + PATCH] += 1
    return acc / np.maximum(cnt, 1), float(probs.max())
