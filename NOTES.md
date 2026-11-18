# Working notes (resume here after a shutdown)

State on 2026-11-18 ~01:20:
- `data/bases/` 942 real room photos (MIT Indoor-67) — downloaded, done.
- Stable Diffusion 1.5 inpainting (fp16 safetensors) + LCM-LoRA are in `~/.cache/huggingface` — fully offline-capable.
- `data/` currently holds the v1 dataset: authentic + paint + splice, plus 190 unseen OpenCV-Telea edits (split `ood`).
- v1 patch net training: `models/patchnet.pt`, LOFO models `models/lofo_*.pt`.

Next steps (all offline):
1. `.venv/bin/python scripts/make_dataset.py --diffusion --ood 60` — adds the diffusion family (SD inpainting of damage prompts) and 60 fully generated photos.
2. `DIFFUSION=1 ./run_all.sh` minus fetch, or run train / lofo / evaluate steps by hand.
3. Fill README numbers from `results/metrics.json`.

Dropped: LaMa (only a third-party TorchScript mirror was available; loading it would run untrusted code).
