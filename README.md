<p align="center"><img src="docs/banner.svg" width="100%" alt="ClaimShield"></p>

A host says the guest broke the table and sends a photo. Was the crack really there, or was it painted in, pasted in, or generated? ClaimShield scores a damage-claim photo for signs of local editing and draws a heatmap of where the edit is, so a reviewer looks at the right part of the picture before money moves.

## Why this exists

In October 2026 an Airbnb guest in Australia was billed about AU$1,700 for a "flooded bathroom". The photo had water that didn't splash, dry toilet paper, and reportedly a SynthID watermark. The claim was rejected after the guest posted it publicly ([Your Mileage May Vary, 21 Oct 2026](https://yourmileagemayvary.com/2026/10/21/airbnb-ai-damage-claim/)). In 2025 a London guest was charged for £12,000 of damage backed by allegedly doctored photos, and was refunded only on appeal ([The Guardian, 2 Aug 2025](https://www.theguardian.com/technology/2025/aug/02/airbnb-guest-damage-claim-refund-photos); [AI Incident Database #1161](https://incidentdatabase.ai/cite/1161/)). Airbnb's own June 2026 scam advisory reports that "62 percent of Americans believe AI-generated content is making scams more convincing" ([Airbnb Newsroom, 17 Jun 2026](https://news.airbnb.com/partnering-with-experts-on-tips-to-help-avoid-summer-travel-scams-in-u-s/)).

Both of those cases were caught by people squinting at pixels. This project asks how far a small model, trained on a laptop, can get on that job, and where it fails.

## How it works

<p align="center"><img src="docs/pipeline.svg" width="100%" alt="pipeline"></p>

**Own dataset, built locally.** The project starts from 942 real photos of bedrooms, living rooms, bathrooms, kitchens and dining rooms (MIT Indoor-67). Every room is used twice: once left honest, and once with fake damage added. Both copies go through the same delivery pipeline: phone-style tone tweaks, a random downscale, JPEG at quality 65–95, and in 30% of cases a second messaging-app JPEG. The detector therefore can't learn "saved twice" or "different size" as a shortcut. Splits are made by room, so a room in the test set was never seen in training.

| Edit family | What it imitates | How it is made |
|---|---|---|
| paint | a phone photo editor: drawn cracks, colour stains | procedural cracks and tinted stains blended into a random region |
| splice | pasting damage from elsewhere | copy-move or a splice from another photo, darkened and feathered |
| unseen tool (test only) | an editor the model never saw | OpenCV Telea inpainting |
| diffusion (planned, v2) | the news case | Stable Diffusion 1.5 inpainting + LCM-LoRA; code is in place, see *Limits* |

| split | honest | edited |
|---|---|---|
| train | 660 | 660 |
| validation | 94 | 94 |
| test | 188 | 188 |
| unseen tool | — | 188 (on the test rooms) |

**Detector.** An ImageNet ResNet-18 fine-tuned on 128 px windows, answering "were these pixels edited?". Sliding it over the photo gives a heatmap; the image score is the most suspicious window. Training takes about 2.5 minutes on a MacBook Air M2.

**Baselines.** Error-level analysis (ELA) statistics + logistic regression, and SRM noise-residual statistics (Fridrich & Kodovsky) + logistic regression.

**Operating point.** A false flag accuses an honest host or guest, so the threshold is set on the validation split to flag at most 5% of honest photos. It is then frozen for the test set.

## Results

<p align="center"><img src="docs/results.svg" width="100%" alt="results"></p>

Held-out test set, threshold frozen from validation.

| detector | ROC-AUC | edited photos caught | honest photos flagged |
|---|---|---|---|
| ELA + logistic | 0.52 | 9.0% | 7.4% |
| SRM noise + logistic | 0.55 | 8.0% | 4.8% |
| **ResNet-18 patch net** | **0.82** | **54.3%** | **1.6%** |

Patch net by edit family, and under extra laundering of every test photo:

| condition | ROC-AUC | caught | honest flagged |
|---|---|---|---|
| splice | 0.95 | 84.1% | 1.6% |
| paint (cracks, stains) | 0.71 | 28.0% | 1.6% |
| unseen tool (OpenCV inpaint) | 0.91 | 56.4% | 1.6% |
| + re-saved at JPEG 50 | 0.83 | 53.7% | 3.7% |
| + downscaled to half size | 0.72 | 18.6% | 0.5% |
| + screenshot (upscale, blur) | 0.83 | 64.9% | 12.8% |

Localisation (pixel F1 of the heatmap at 0.5 against the true edit mask): splice 0.37, paint 0.18.

## What does not work (yet)

- **It does not transfer between editing styles.** Trained without paint edits, the patch net scores 0.55 AUC on paint; trained without splices, it scores 0.53 on splices. That is close to chance. Its result on the unseen OpenCV tool (0.91) is probably helped by the blurry fill that tool leaves, which resembles the feathered splices. The model has learned these two editors, not editing in general.
- **Classical forensics fail here.** After a downscale and a JPEG re-save, ELA and SRM statistics carry almost no signal (AUC 0.52–0.55). That matches how claim photos actually travel.
- **Small, thin edits are hard.** Painted cracks are a few pixels wide, so only 28% are caught. Halving the image size drops overall recall to 19%.
- **Screenshots raise false alarms.** Upscaling and blur push the honest-photo flag rate from 1.6% to 12.8%.
- **No diffusion edits in v1.** On the M2, SD 1.5 inpainting in fp16 returned NaN (black) patches and took 40–75 s per image. The generator now runs in fp32 off-CUDA. The diffusion family is the next experiment (`NEXT_STEPS.md`), and a Colab T4 is the realistic place to generate it.
- **Synthetic damage is not real damage.** No genuinely damaged rooms are in the honest class. A real deployment would need honest damage photos so the model doesn't learn "damage = fake".
- **A score is not a verdict.** The output is meant to send a claim to a human reviewer, never to auto-reject it.

## Use it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ".[generate]" for the diffusion family
./run_all.sh                       # fetch photos, build data, train, evaluate, test
python -m claimshield.cli claim.jpg --heatmap claim_heatmap.png
```

`results/metrics.json` holds every number above. `data/manifest.csv` lists each image with its family, split, edit fraction and JPEG history.

## References

- Quattoni & Torralba, *Recognizing Indoor Scenes*, CVPR 2009 — base photos (MIT Indoor-67).
- Fridrich & Kodovsky, *Rich Models for Steganalysis of Digital Images*, IEEE TIFS 2012 — SRM residual filters.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016, [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
- Wang et al., *CNN-generated images are surprisingly easy to spot... for now*, CVPR 2020, [arXiv:1912.11035](https://arxiv.org/abs/1912.11035).
- Ojha et al., *Towards Universal Fake Image Detectors that Generalize Across Generative Models*, CVPR 2023, [arXiv:2302.10174](https://arxiv.org/abs/2302.10174) — why cross-generator generalisation is the hard part.
- Corvi et al., *On the detection of synthetic images generated by diffusion models*, ICASSP 2023, [arXiv:2211.00680](https://arxiv.org/abs/2211.00680).
- Guillaro et al., *TruFor: trustworthy image forgery detection and localization*, CVPR 2023, [arXiv:2212.10957](https://arxiv.org/abs/2212.10957).
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022, [arXiv:2112.10752](https://arxiv.org/abs/2112.10752); Luo et al., *LCM-LoRA*, [arXiv:2311.05556](https://arxiv.org/abs/2311.05556) — the v2 generator.
- Zhu et al., *GenImage: a million-scale benchmark for detecting AI-generated images*, NeurIPS 2023, [arXiv:2306.08571](https://arxiv.org/abs/2306.08571) — public benchmark for future comparison.

## Licence

MIT © Ch Kaushik
