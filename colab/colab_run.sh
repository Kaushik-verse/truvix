#!/usr/bin/env bash
# Runs the full v2 pipeline on a Colab T4. Log: run_colab.log
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv
pip -q install diffusers==0.35.1 peft==0.17.1 transformers==4.57.1 accelerate opencv-python-headless
pip -q install -e . --no-deps
python scripts/fetch_bases.py
python scripts/make_dataset.py --diffusion --ood 60
python scripts/train_patchnet.py --epochs 5
for f in paint splice diffusion; do python scripts/train_patchnet.py --epochs 3 --exclude $f --out models/lofo_$f.pt; done
mkdir -p results
python scripts/evaluate.py > results/eval_stdout.json
python -m pytest -q tests | tail -3 > results/pytest.txt
mkdir -p examples
python - <<'PY'
import csv, shutil
seen = set()
for r in csv.DictReader(open("data/manifest.csv")):
    if r["split"] not in ("test", "ood") or r["family"] in seen:
        continue
    seen.add(r["family"])
    shutil.copy("data/" + r["image"], f"examples/{r['family']}.png")
    if r["mask"]:
        shutil.copy("data/" + r["mask"], f"examples/{r['family']}_mask.png")
PY
tar czf claimshield_results.tar.gz results models/*.pt models/*.json data/manifest.csv data/bases_manifest.csv examples
echo ALL_DONE
