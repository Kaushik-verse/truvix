"""Fine-tune the ResNet-18 patch detector on the training split.

  python scripts/train_patchnet.py                      # main model -> models/patchnet.pt
  python scripts/train_patchnet.py --exclude diffusion --out models/lofo_diffusion.pt
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

from claimshield.patchnet import PATCH, build, device, grid, patch_labels, to_tensor


def load_patches(rows, root: Path, rng: np.random.Generator):
    pos, neg = [], []
    for r in rows:
        img = np.asarray(Image.open(root / r["image"]).convert("RGB"))
        mask = (np.asarray(Image.open(root / r["mask"])) > 127).astype(np.float32) if r["mask"] else None
        coords = grid(*img.shape[:2])
        labels = patch_labels(mask, coords)
        for (y, x), l in zip(coords, labels):
            if l == 1:
                pos.append(img[y:y + PATCH, x:x + PATCH])
            elif l == 0 and rng.random() < 0.35:  # thin out the many clean patches
                neg.append(img[y:y + PATCH, x:x + PATCH])
    return np.stack(pos), np.stack(neg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--exclude", default="", help="manipulation family left out of training")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out", default="models/patchnet.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    root = Path(args.data)
    rows = list(csv.DictReader(open(root / "manifest.csv")))
    keep = lambda r, s: r["split"] == s and r["family"] != args.exclude
    tr_pos, tr_neg = load_patches([r for r in rows if keep(r, "train")], root, rng)
    va_pos, va_neg = load_patches([r for r in rows if keep(r, "val")], root, rng)
    print(f"train patches: {len(tr_pos)} edited / {len(tr_neg)} clean; val {len(va_pos)} / {len(va_neg)}")

    dev = device()
    model = build().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.BCEWithLogitsLoss()
    best, log = -1.0, []

    def evaluate():
        model.eval()
        x = np.concatenate([va_pos, va_neg])
        y = np.r_[np.ones(len(va_pos)), np.zeros(len(va_neg))]
        p = []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                p.append(torch.sigmoid(model(to_tensor(x[i:i + 256]).to(dev))).squeeze(1).cpu().numpy())
        p = np.concatenate(p)
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, p))

    for ep in range(args.epochs):
        model.train()
        # balanced epoch: every edited patch + an equal random draw of clean ones
        neg_idx = rng.choice(len(tr_neg), size=min(len(tr_neg), len(tr_pos)), replace=False)
        x = np.concatenate([tr_pos, tr_neg[neg_idx]])
        y = np.r_[np.ones(len(tr_pos)), np.zeros(len(neg_idx))].astype(np.float32)
        order = rng.permutation(len(x))
        t0, tot = time.time(), 0.0
        for i in range(0, len(order), 64):
            b = order[i:i + 64]
            xb = x[b]
            if rng.random() < 0.5:
                xb = xb[:, :, ::-1]
            if rng.random() < 0.5:
                xb = xb[:, ::-1]
            xb = to_tensor(np.ascontiguousarray(xb)).to(dev)
            loss = lossf(model(xb).squeeze(1), torch.from_numpy(y[b]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        auc = evaluate()
        log.append({"epoch": ep + 1, "train_loss": tot / len(order), "val_patch_auc": auc, "sec": time.time() - t0})
        print(log[-1], flush=True)
        if auc > best:
            best = auc
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
    json.dump({"exclude": args.exclude, "best_val_patch_auc": best, "log": log},
              open(Path(args.out).with_suffix(".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
