# Next steps

All of these run offline: the base photos, the ResNet-18 weights, SD 1.5 inpainting (fp16 safetensors) and LCM-LoRA are already on disk.

## Where things stand (2026-11-18, 01:35)

- v1 dataset in `data/`: 942 real rooms → 942 honest + 942 edited (paint / splice), plus 188 unseen OpenCV-Telea edits.
- v1 results in `results/metrics.json`. Patch net test AUC 0.82; ELA and SRM baselines are near chance. Leave-one-family-out drops to about 0.53–0.55 AUC.
- The Stable Diffusion family isn't in v1. On the M2, full fp16 gave NaN (black) patches at 40–75 s per image, and full fp32 swapped to more than 5 min per image.
- **Untested change (committed):** `DiffusionInpainter` now runs the UNet in fp16 and decodes with an fp32 VAE copy, which is the usual NaN fix. Test it on 3 images first, check speed and that `black` is about 0. If it is about 40 s per image, the 314 diffusion edits take roughly 3.5 h overnight. Otherwise, generate on a Colab T4.
- The README draft has the real v1 numbers. Its references were checked against arXiv and Crossref, and the article links were checked live.
- Paused on request: one project at a time while RAM is short.

## Resume commands

```bash
cd ~/Documents/claimshield
source .venv/bin/activate

# v2: add the diffusion family (slow on M2, roughly 1 min per image; ~10 min on a Colab T4)
python scripts/make_dataset.py --diffusion --ood 60

python scripts/train_patchnet.py --epochs 5
for f in paint splice diffusion; do python scripts/train_patchnet.py --epochs 3 --exclude $f --out models/lofo_$f.pt; done
python scripts/evaluate.py
python scripts/make_figures.py
python -m pytest -q tests
```

## Before pushing

- Put the real numbers from `results/metrics.json` into the README.
- Push only when Kaushik says so: the author is the user only, with no co-author trailer.
