# ClaimShield showcase figures. Paste into a new cell in the SAME Colab session that ran v3.
# Uses the trained model and dataset already on the runtime; no retraining (about 1 minute).
import os, csv, zipfile, random
import numpy as np, matplotlib.pyplot as plt
from PIL import Image

os.chdir("/content/claimshield")
if not (os.path.exists("models/patchnet.pt") and os.path.isdir("data/images")):
    raise SystemExit("Runtime was reset: the v3 data is gone. Tell Claude.")

from claimshield.cli import load_model, overlay
from claimshield.patchnet import heatmap

BG = "#F7F6F2"
plt.rcParams.update({"font.family": "serif", "figure.facecolor": BG, "savefig.facecolor": BG})
rows = list(csv.DictReader(open("data/manifest.csv")))
model, thr = load_model()
os.makedirs("showcase", exist_ok=True)

# 1. What each edit looks like: honest original | edited claim photo | edited pixels.
#    Shows the test example with the largest edit per type, so the edit is easy to see.
fig, axes = plt.subplots(3, 3, figsize=(9, 9.4))
for ax_row, fam in zip(axes, ["paint", "splice", "diffusion"]):
    r = max((r for r in rows if r["family"] == fam and r["split"] == "test"), key=lambda r: float(r["edited_frac"]))
    panels = [(f"data/images/{r['base_id']}_real.png", "honest original"),
              ("data/" + r["image"], f"{fam}: {r['prompt']}" if r["prompt"] else fam),
              ("data/" + r["mask"], "edited pixels")]
    for ax, (path, title) in zip(ax_row, panels):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
plt.tight_layout()
plt.savefig("showcase/dataset_examples.png", dpi=110)
plt.show()

# 2. Heatmaps on randomly drawn test photos (seeded, not hand-picked): 2 per type.
random.seed(7)
fams = ["authentic", "paint", "splice", "diffusion", "telea_unseen", "fullgen"]
names = {"authentic": "honest", "paint": "paint", "splice": "splice", "diffusion": "diffusion",
         "telea_unseen": "OpenCV inpaint (unseen)", "fullgen": "fully generated (unseen)"}
fig, axes = plt.subplots(3, 4, figsize=(13, 10.4))
for i, fam in enumerate(fams):
    pool = [r for r in rows if r["family"] == fam and r["split"] in ("test", "ood")]
    for j, r in enumerate(random.sample(pool, 2)):
        ax = axes.flat[2 * i + j]
        img = np.asarray(Image.open("data/" + r["image"]).convert("RGB"))
        hm, score = heatmap(model, img)
        flagged = score > thr
        correct = flagged == (fam != "authentic")
        ax.imshow(overlay(img, hm))
        ax.set_title(f"{names[fam]}\nscore {score:.2f}: {'FLAGGED' if flagged else 'passed'} "
                     f"({'correct' if correct else 'wrong'})", fontsize=9,
                     color="#2B2B2B" if correct else "#B4552D")
        ax.axis("off")
plt.tight_layout()
plt.savefig("showcase/heatmaps_random.png", dpi=110)
plt.show()

with zipfile.ZipFile("claimshield_showcase.zip", "w") as z:
    for f in os.listdir("showcase"):
        z.write("showcase/" + f, f)
from google.colab import files
files.download("claimshield_showcase.zip")
print("done: claimshield_showcase.zip")
