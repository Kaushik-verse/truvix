# Next steps

All of these run offline: the base photos, the ResNet-18 weights, SD 1.5 inpainting (fp16 safetensors) and LCM-LoRA are already on disk.

## Where things stand (2026-11-18, 01:35)

- v1 dataset in `data/`: 942 real rooms → 942 honest + 942 edited (paint / splice), plus 188 unseen OpenCV-Telea edits.
- v1 results in `results/metrics.json`. Patch net test AUC 0.82; ELA and SRM baselines are near chance. Leave-one-family-out drops to about 0.53–0.55 AUC.
- The Stable Diffusion family isn't in v1. On the M2 in fp16 it produced NaN (black) patches and took 40–75 s per image. The code now uses fp32 on MPS.

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
