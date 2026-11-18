# Next steps

## Where things stand (2026-11-18)

- **v2 done.** It has paint, splice and Stable Diffusion edits, plus two edit sources the model never saw. It was generated and trained on a Colab T4 with `colab/claimshield_colab.ipynb` (about 25 min), and the results were copied back here.
  - Patch net test AUC is 0.73. By edit type: splice 0.91, paint 0.69, diffusion 0.59.
  - The ELA and SRM baselines score about 0.5.
  - Full numbers are in `results/metrics.json`, and v1 (paint and splice only) is in `results/v1_paint_splice/`.
- `models/patchnet.pt` is committed (43 MB) so the CLI works from a clone. The leave-one-family-out models are not committed.
- The v2 images exist only on Colab. Locally, `data/manifest.csv` describes them, and `docs/examples/` holds one example of each edit type.

## Worth doing next

1. **Stronger diffusion edits.** At guidance 1.0, many SD edits barely change the room (see the bottom row of `docs/examples/heatmaps.png`).
   - Fix: in `DiffusionInpainter.__call__`, set guidance to 1.5–2.0 and steps to 6–8, then re-run the notebook.
   - Expect a fairer test: higher diffusion recall, or a clearer negative result.
2. **Train longer.** Validation AUC was still rising at epoch 5. Try 12–15 epochs; that's about 2 minutes on a T4.
3. **Honest real damage.** Add real damaged-room photos to the honest class, so the model can't learn "damage = fake".

## Reproduce

```bash
# Colab T4: upload colab/claimshield_colab.ipynb → Runtime → Run all → download claimshield_results.tar.gz
# Laptop, without diffusion:
./run_all.sh
```

Before pushing: push only when Kaushik says so, with the author set to the user only and no co-author trailer.
