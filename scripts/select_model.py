"""Pick the patch-detector candidate with the best validation image ROC-AUC.

Selection uses validation only; the test split is first touched by evaluate.py.
  python scripts/select_model.py models/cand_*.pt
"""
import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    cands = [Path(p) for p in sys.argv[1:]]
    table = []
    for c in cands:
        meta = json.loads(c.with_suffix(".json").read_text())
        table.append({"checkpoint": c.name, "arch": meta["arch"], "launder": meta.get("launder", False),
                      "epochs": meta["epochs"], "best_val_image_auc": round(meta["best_val_image_auc"], 4)})
    best = max(zip(table, cands), key=lambda t: t[0]["best_val_image_auc"])
    d = best[1].parent  # winner is copied next to the candidates
    shutil.copy(best[1], d / "patchnet.pt")
    shutil.copy(best[1].with_suffix(".json"), d / "patchnet.json")
    (d / "model_selection.json").write_text(json.dumps({"selected": best[0], "candidates": table}, indent=2))
    print(json.dumps({"selected": best[0], "candidates": table}, indent=2))


if __name__ == "__main__":
    main()
