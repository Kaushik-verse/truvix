"""Held-out evaluation of every detector. Writes results/metrics.json and results/*.csv.

Operating point: the decision threshold is chosen on the validation split so that at
most 5% of honest photos are flagged (a false flag accuses an honest host or guest),
then frozen and applied to test.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from claimshield.features import ela_features, srm_features
from claimshield.patchnet import build, device, heatmap
from claimshield.postprocess import attack

FAMILIES = ["paint", "splice", "diffusion"]
ATTACKS = ["jpeg50", "half", "screenshot"]
MAX_FPR = 0.05


def threshold_at_fpr(y, s, max_fpr=MAX_FPR):
    neg = np.sort(np.asarray(s)[np.asarray(y) == 0])
    return float(neg[min(len(neg) - 1, int(np.ceil((1 - max_fpr) * len(neg))))]) + 1e-9


def summary(y, s, thr):
    y, s = np.asarray(y), np.asarray(s)
    pred = s > thr
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    fpr = float(pred[y == 0].mean()) if (y == 0).any() else float("nan")
    auc = float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")
    return {"auc": round(auc, 4), "tpr": round(tpr, 4), "fpr": round(fpr, 4),
            "bal_acc": round((tpr + 1 - fpr) / 2, 4), "n": int(len(y))}


def load(root, r):
    return np.asarray(Image.open(root / r["image"]).convert("RGB"))


def localisation(hm, mask, thr=0.5):
    pred = hm > thr
    m = mask > 0
    inter = (pred & m).sum()
    union = (pred | m).sum()
    f1 = 2 * inter / max(pred.sum() + m.sum(), 1)
    return float(f1), float(inter / max(union, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    root, out = Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(root / "manifest.csv")))
    for r in rows:
        r["label"] = int(r["label"])
    split = lambda s: [r for r in rows if r["split"] == s]
    tr, va, te, ood = split("train"), split("val"), split("test"), split("ood")
    families = [f for f in FAMILIES if any(r["family"] == f for r in te)]
    res = {"counts": {s: {"authentic": sum(r["label"] == 0 for r in rows if r["split"] == s),
                          "edited": sum(r["label"] == 1 for r in rows if r["split"] == s)}
                      for s in ["train", "val", "test", "ood"]}}

    # ---------- features for the classical baselines
    print("features ...", flush=True)
    feat = {}
    for r in rows:
        im = load(root, r)
        feat[r["image"]] = (ela_features(im), srm_features(im))
    att_imgs = {a: {r["image"]: attack(load(root, r), a) for r in te} for a in ATTACKS}

    def fit_lr(train_rows, k):
        X = np.stack([feat[r["image"]][k] for r in train_rows])
        y = [r["label"] for r in train_rows]
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X, y)

    scorers = {}
    for name, k, fn in [("ela_lr", 0, ela_features), ("srm_lr", 1, srm_features)]:
        clf = fit_lr(tr, k)
        scorers[name] = (lambda rs, clf=clf, k=k: clf.predict_proba(np.stack([feat[r["image"]][k] for r in rs]))[:, 1],
                         lambda ims, clf=clf, fn=fn: clf.predict_proba(np.stack([fn(i) for i in ims]))[:, 1])

    # ---------- the patch detector
    dev = device()
    model = build(pretrained=False).to(dev).eval()
    model.load_state_dict(torch.load(Path(args.models) / "patchnet.pt", map_location=dev))
    cache = {}

    def pn_scores(rs):
        s = []
        for r in rs:
            if r["image"] not in cache:
                cache[r["image"]] = heatmap(model, load(root, r), dev)
            s.append(cache[r["image"]][1])
        return np.array(s)

    scorers["patchnet"] = (pn_scores, lambda ims: np.array([heatmap(model, i, dev)[1] for i in ims]))

    # ---------- main table, per family, OOD, robustness
    res["detectors"] = {}
    for name, (score_rows, score_imgs) in scorers.items():
        print(name, flush=True)
        thr = threshold_at_fpr([r["label"] for r in va], score_rows(va))
        y_te, s_te = [r["label"] for r in te], score_rows(te)
        d = {"threshold": round(thr, 6), "test": summary(y_te, s_te, thr), "per_family": {}}
        auth = [r for r in te if r["label"] == 0]
        s_auth = score_rows(auth)
        for f in families:
            fr = [r for r in te if r["family"] == f]
            d["per_family"][f] = summary([0] * len(auth) + [1] * len(fr), np.r_[s_auth, score_rows(fr)], thr)
        d["unseen_tools"] = {}
        for f in sorted({r["family"] for r in ood}):
            fr = [r for r in ood if r["family"] == f]
            d["unseen_tools"][f] = summary([0] * len(auth) + [1] * len(fr), np.r_[s_auth, score_rows(fr)], thr)
        d["robustness"] = {}
        for a in ATTACKS:
            ims = [att_imgs[a][r["image"]] for r in te]
            d["robustness"][a] = summary(y_te, score_imgs(ims), thr)
        res["detectors"][name] = d
        with open(out / f"scores_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "family", "label", "score"])
            for r, s in zip(te + ood, np.r_[s_te, score_rows(ood)] if ood else s_te):
                w.writerow([r["image"], r["family"], r["label"], round(float(s), 6)])

    # ---------- localisation (patch detector only, edited test photos)
    loc = {f: [] for f in families}
    for r in te:
        if r["label"] == 1:
            hm = cache[r["image"]][0]
            mask = np.asarray(Image.open(root / r["mask"])) > 127
            loc[r["family"]].append(localisation(hm, mask))
    res["localisation"] = {f: {"pixel_f1": round(float(np.mean([a for a, _ in v])), 4),
                               "iou": round(float(np.mean([b for _, b in v])), 4), "n": len(v)}
                           for f, v in loc.items() if v}

    # ---------- leave-one-family-out (generalisation to an unseen editing tool)
    res["lofo"] = {}
    for f in families:
        p = Path(args.models) / f"lofo_{f}.pt"
        auth = [r for r in te if r["label"] == 0]
        fr = [r for r in te if r["family"] == f]
        y = [0] * len(auth) + [1] * len(fr)
        entry = {}
        for name, k in [("ela_lr", 0), ("srm_lr", 1)]:
            clf = fit_lr([r for r in tr if r["family"] != f], k)
            vthr = threshold_at_fpr([r["label"] for r in va if r["family"] != f],
                                    clf.predict_proba(np.stack([feat[r["image"]][k] for r in va if r["family"] != f]))[:, 1])
            entry[name] = summary(y, clf.predict_proba(np.stack([feat[r["image"]][k] for r in auth + fr]))[:, 1], vthr)
        if p.exists():
            m = build(pretrained=False).to(dev).eval()
            m.load_state_dict(torch.load(p, map_location=dev))
            sc = lambda rs: np.array([heatmap(m, load(root, r), dev)[1] for r in rs])
            vr = [r for r in va if r["family"] != f]
            vthr = threshold_at_fpr([r["label"] for r in vr], sc(vr))
            entry["patchnet"] = summary(y, sc(auth + fr), vthr)
        res["lofo"][f] = entry

    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
