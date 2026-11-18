#!/usr/bin/env bash
# ClaimShield v3 on a Colab T4, about 1.5 h. Log: run_colab.log
set -euo pipefail
cd /content/claimshield
nvidia-smi --query-gpu=name,memory.total --format=csv
pip -q install diffusers==0.35.1 peft==0.17.1 transformers==4.57.1 accelerate opencv-python-headless 2>&1 | grep -v "dependency conflicts\|gradio\|^ERROR: pip's" || true
pip -q install -e . --no-deps
mkdir -p results models

echo "== 1/6 room photos"
python scripts/fetch_bases.py --per-room 400

echo "== 2/6 dataset (stronger, visibility-checked diffusion edits)"
python scripts/make_dataset.py --diffusion --guidance 1.8 --steps 8 --min-visible 0.004 --ood 80

echo "== 3/6 candidate detectors (validation picks the winner)"
python scripts/train_patchnet.py --arch resnet18        --epochs 20 --launder --neg-keep 0.25 --out models/cand_resnet18.pt
python scripts/train_patchnet.py --arch efficientnet_b0 --epochs 20 --launder --neg-keep 0.25 --lr 1e-3 --out models/cand_effb0.pt
python scripts/train_patchnet.py --arch resnet50        --epochs 20 --launder --neg-keep 0.25 --out models/cand_resnet50.pt
python scripts/select_model.py models/cand_*.pt

echo "== 4/6 leave-one-family-out with the selected backbone"
ARCH=$(python -c "import json; print(json.load(open('models/patchnet.json'))['arch'])")
LR=$([ "$ARCH" = efficientnet_b0 ] && echo 1e-3 || echo 3e-4)
for f in paint splice diffusion; do
  python scripts/train_patchnet.py --arch $ARCH --lr $LR --epochs 10 --launder --neg-keep 0.25 --exclude $f --out models/lofo_$f.pt
done

echo "== 5/6 evaluation + tests"
python scripts/evaluate.py > results/eval_stdout.json
(python -m pytest -q tests || true) | tail -3 > results/pytest.txt
cat results/pytest.txt

echo "== 6/6 packaging"
mkdir -p examples
python - <<'PY'
import csv, shutil
seen = {}
for r in csv.DictReader(open("data/manifest.csv")):
    if r["split"] not in ("test", "ood"):
        continue
    k = r["family"]
    if seen.get(k, 0) >= 3:
        continue
    seen[k] = seen.get(k, 0) + 1
    tag = f"{k}_{seen[k]}"
    shutil.copy("data/" + r["image"], f"examples/{tag}.png")
    if r["mask"]:
        shutil.copy("data/" + r["mask"], f"examples/{tag}_mask.png")
    if k != "authentic" and r["base_id"] and not r["base_id"].startswith("ood"):
        real = f"data/images/{r['base_id']}_real.png"
        try:
            shutil.copy(real, f"examples/{tag}_original.png")
        except FileNotFoundError:
            pass
PY
cp run_colab.log results/ 2>/dev/null || true
tar czf claimshield_results.tar.gz results models/*.pt models/*.json data/manifest.csv data/bases_manifest.csv data/generation_stats.json examples
ls -la claimshield_results.tar.gz
echo ALL_DONE
