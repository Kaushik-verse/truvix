#!/usr/bin/env bash
# Full pipeline on a MacBook Air M2 (8 GB): ~30 minutes after downloads.
set -euo pipefail
PY=${PY:-.venv/bin/python}
$PY scripts/fetch_bases.py                       # 942 real room photos (MIT Indoor-67)
$PY scripts/make_dataset.py ${DIFFUSION:+--diffusion}   # authentic + paint / splice (+ diffusion), unseen Telea
$PY scripts/train_patchnet.py --epochs 5
for f in paint splice ${DIFFUSION:+diffusion}; do
  $PY scripts/train_patchnet.py --epochs 3 --exclude $f --out models/lofo_$f.pt
done
$PY scripts/evaluate.py
$PY -m pytest -q tests
