# Next steps

## Where things stand (2026-11-18)

- **v3 is current.** It used 1,653 rooms and visibility-checked SD edits (guidance 1.8, 8 steps). Three backbones were trained with laundering augmentation; EfficientNet-B0 was picked on validation (0.814).
- **Test results:**
  - Overall AUC 0.76 [0.73–0.80], with 51% of edits caught and 8.5% of honest photos flagged.
  - By edit type: splice 0.95, diffusion 0.68, paint 0.68.
  - Unseen sources: OpenCV inpaint 0.95, fully generated 0.91.
  - Leave-one-family-out: 0.49–0.64, so it doesn't generalise to unseen editing tools.
- **Where results live:** `results/` holds v3; earlier versions are in `results/v2/` and `results/v1_paint_splice/`.
- **What's committed:** `models/patchnet.pt` (EfficientNet-B0, 16 MB). Candidate and leave-one-family-out weights are not committed; their JSON logs are.
- **Images:** the v3 images exist only on Colab. `data/manifest.csv` describes them, and `docs/examples/` keeps 3 examples per edit type with masks and originals.
- **Local room photos:** `data/bases` holds the first 190 rooms per type (v2). v3's list (`data/bases_manifest.csv`) extends it to 400.

## Ideas, if continuing

1. **Real damage.** Add genuinely damaged-room photos to the honest class. This is the most important gap for real use.
2. **Better diffusion damage.** SD 1.5 inpainting rarely draws damage; it swaps objects. Damage-specific edits could come from an instruction-following editor (e.g. InstructPix2Pix) or a damage LoRA.
3. **Smarter image scoring.** Learn the aggregation on validation, e.g. mean of the top-k windows, instead of the single maximum window. That should reduce false flags.
4. **Larger validation set.** The ≤5% false-flag target became 8.5% on test.

## Reproduce

Open `colab/truvix_colab.ipynb` in Colab (it clones the public repo), choose T4, Run all (about 1.5 h). The last cell downloads `truvix_results.tar.gz`.
