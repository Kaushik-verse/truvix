# Notices, licences and attribution

ClaimShield's **source code** is released under the MIT License (see `LICENSE`), © 2026 Ch Kaushik.
The MIT License covers only the code and the original diagrams (`docs/banner.svg`, `docs/pipeline.svg`, `docs/results.svg`). Everything listed below keeps its own terms.

## Images

**MIT Indoor-67** (Quattoni & Torralba, *Recognizing Indoor Scenes*, CVPR 2009)
- Source: <https://web.mit.edu/torralba/www/indoor.html>. Downloaded through the Hugging Face mirror [`NuclearUnicorn-UWC/mit-indoor-outdoor`](https://huggingface.co/datasets/NuclearUnicorn-UWC/mit-indoor-outdoor).
- The dataset page states: "The images provided here are for research purposes only." Copyright in each photo stays with its original owner.
- **This repository does not redistribute the dataset.** `scripts/fetch_bases.py` downloads it at run time, and `data/*.csv` stores only file paths and generated metadata.
- The example figures in `docs/examples/`, `docs/dataset_examples.jpg` and `docs/heatmaps_random.jpg` contain a small number of these photos, some edited. They are included only to illustrate the research results. They are **not** covered by the MIT License and may be used **for research and education only**.
- If you hold the rights to one of these photos and want it removed, please open an issue and it will be taken down.

**Synthetic images**
- The edited and fully generated images were made with Stable Diffusion (see below) and OpenCV.
- Their only purpose is to train and test a *detector* of edited photos. They show no real event, place or claim and must not be used to deceive anyone.

## Models

| Model | Used for | Licence |
|---|---|---|
| [`stable-diffusion-v1-5/stable-diffusion-inpainting`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) (originally RunwayML / Stability AI) | generating the "diffusion" and "fully generated" test images | CreativeML OpenRAIL-M |
| [`latent-consistency/lcm-lora-sdv1-5`](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) | fast 8-step sampling | OpenRAIL++ |
| torchvision ImageNet weights: ResNet-18, ResNet-50, EfficientNet-B0 | starting point for the detector | torchvision code BSD-3-Clause; weights trained on ImageNet, whose terms allow non-commercial research use |

Notes on the Stable Diffusion licence:
- No model weights are redistributed here. They are downloaded at run time from the Hugging Face Hub, under their own licences.
- Under CreativeML OpenRAIL-M §6 the licensor claims no rights in generated outputs. The user stays accountable for how outputs are used.
- The licence's use restrictions (Attachment A) forbid, among other things, generating "verifiably false information and/or content with the purpose of harming others". This project generates edits only to *detect* that kind of content, which is the opposite purpose.
- The ClaimShield detector is an image classifier. It does not generate images or imitate Stable Diffusion.

**`models/patchnet.pt` (the trained detector)**
- It was fine-tuned from ImageNet weights on images derived from MIT Indoor-67, so it is provided **for research and education only**.

## Software

| Library | Licence |
|---|---|
| PyTorch, torchvision | BSD-3-Clause |
| NumPy, scikit-learn, pandas | BSD-3-Clause |
| OpenCV (`opencv-python-headless`) | Apache-2.0 |
| Pillow | MIT-CMU (HPND) |
| matplotlib | Matplotlib License (PSF-based) |
| Hugging Face diffusers, transformers, peft, accelerate | Apache-2.0 |
| pytest | MIT |

All of these are installed from PyPI; none is vendored in this repository.

## Papers

The methods and baselines build on the papers listed under *References* in `README.md`. Please cite them if you build on this work.

## News sources and trademarks

- The motivating cases are described from public reporting, with links: Your Mileage May Vary (21 Oct 2026), The Guardian (2 Aug 2025), AI Incident Database #1161, and the Airbnb Newsroom (17 Jun 2026). One short sentence from the Airbnb Newsroom is quoted with attribution.
- **ClaimShield is an independent research project. It is not affiliated with, endorsed by or sponsored by Airbnb, Inc. or any other company named here.** "Airbnb" is a trademark of Airbnb, Inc. and is used only to refer to public news reports.

## No warranty, not a verdict

The detector makes mistakes; the README reports its error rates. Its output is meant to send a photo to a human reviewer. It must not be used on its own to accuse anyone or to accept or reject a claim.
