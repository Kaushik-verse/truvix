"""Fine-tune a patch detector on the training split.

  python scripts/train_patchnet.py                                   # v2 setup: ResNet-18, 6 epochs
  python scripts/train_patchnet.py --arch resnet50 --epochs 20 --launder --out models/cand_resnet50.pt
  python scripts/train_patchnet.py --exclude diffusion --out models/lofo_diffusion.pt

The checkpoint kept is the epoch with the best *image-level* validation ROC-AUC
(the number a reviewer cares about); the test split is never touched here.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score

from truvix.patchnet import ARCHS, PATCH, build, device, grid, heatmap, patch_labels, to_tensor
from truvix.postprocess import jpeg

import cv2


def launder(img: np.ndarray, mask, rng: np.random.Generator):
    """One extra random trip through a messaging app / screenshot, applied to a copy.

    Teaches the detector that re-compression, resizing and blur are not edits.
    """
    h, w = img.shape[:2]
    kind = rng.choice(["jpeg", "down", "screenshot"])
    m = mask
    if kind == "down":
        s = rng.uniform(0.5, 0.8)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        m = None if mask is None else cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    elif kind == "screenshot":
        s = rng.uniform(1.1, 1.4)
        img = cv2.GaussianBlur(cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR), (3, 3), 0)
        m = None if mask is None else cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return jpeg(img, int(rng.integers(45, 90))), m


def load_patches(rows, root: Path, rng: np.random.Generator, neg_keep: float, launder_copies: int = 0):
    pos, neg = [], []
    for r in rows:
        img = np.asarray(Image.open(root / r["image"]).convert("RGB"))
        mask = (np.asarray(Image.open(root / r["mask"])) > 127).astype(np.float32) if r["mask"] else None
        versions = [(img, mask)] + [launder(img, mask, rng) for _ in range(launder_copies)]
        for im, mk in versions:
            if min(im.shape[:2]) < PATCH:
                continue
            coords = grid(*im.shape[:2])
            for (y, x), l in zip(coords, patch_labels(mk, coords)):
                if l == 1:
                    pos.append(im[y:y + PATCH, x:x + PATCH])
                elif l == 0 and rng.random() < neg_keep:
                    neg.append(im[y:y + PATCH, x:x + PATCH])
    return np.stack(pos), np.stack(neg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--exclude", default="", help="manipulation family left out of training")
    ap.add_argument("--arch", default="resnet18", choices=ARCHS)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--launder", action="store_true", help="add one laundered copy of every training photo")
    ap.add_argument("--neg-keep", type=float, default=0.35)
    ap.add_argument("--out", default="models/patchnet.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    root = Path(args.data)
    rows = list(csv.DictReader(open(root / "manifest.csv")))
    keep = lambda r, s: r["split"] == s and r["family"] != args.exclude
    tr_rows = [r for r in rows if keep(r, "train")]
    va_rows = [r for r in rows if keep(r, "val")]
    tr_pos, tr_neg = load_patches(tr_rows, root, rng, args.neg_keep, 1 if args.launder else 0)
    print(f"{args.arch}: train patches {len(tr_pos)} edited / {len(tr_neg)} clean; val photos {len(va_rows)}", flush=True)
    va_imgs = [np.asarray(Image.open(root / r["image"]).convert("RGB")) for r in va_rows]
    va_y = np.array([int(r["label"]) for r in va_rows])

    dev = device()
    model = build(arch=args.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = (2 * len(tr_pos) + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss()
    best, log = -1.0, []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        # balanced epoch: every edited patch + an equal fresh draw of clean ones
        neg_idx = rng.choice(len(tr_neg), size=min(len(tr_neg), len(tr_pos)), replace=False)
        x = np.concatenate([tr_pos, tr_neg[neg_idx]])
        y = np.r_[np.ones(len(tr_pos)), np.zeros(len(neg_idx))].astype(np.float32)
        order = rng.permutation(len(x))
        t0, tot, n_steps = time.time(), 0.0, 0
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            xb = x[b]
            if rng.random() < 0.5:
                xb = xb[:, :, ::-1]
            if rng.random() < 0.5:
                xb = xb[:, ::-1]
            if rng.random() < 0.5:
                xb = np.rot90(xb, k=int(rng.integers(1, 4)), axes=(1, 2))
            xb = to_tensor(np.ascontiguousarray(xb)).to(dev)
            loss = lossf(model(xb).squeeze(1), torch.from_numpy(y[b]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if n_steps < args.epochs * steps_per_epoch - 1:
                sched.step()
            n_steps += 1
            tot += loss.item() * len(b)
        model.eval()
        s = np.array([heatmap(model, im, dev)[1] for im in va_imgs])
        auc = float(roc_auc_score(va_y, s))
        log.append({"epoch": ep + 1, "train_loss": round(tot / len(order), 4), "val_image_auc": round(auc, 4),
                    "sec": round(time.time() - t0, 1)})
        print(log[-1], flush=True)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), args.out)
    json.dump({"arch": args.arch, "exclude": args.exclude, "launder": args.launder, "epochs": args.epochs,
               "best_val_image_auc": best, "log": log},
              open(Path(args.out).with_suffix(".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
