"""Ways a fraudulent claimant can add "damage" to an honest photo.

Three families, from the cheapest to the most convincing:
  paint     - hand-drawn cracks and stains blended into the photo (a phone photo editor)
  splice    - a region copied from this or another photo and pasted over the scene
  diffusion - a generative model inpaints a damage prompt inside a mask (the case in the news)

Every function takes and returns uint8 RGB arrays and a binary mask (1 = edited pixel).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

DAMAGE_PROMPTS = [
    "a large crack in the surface",
    "a dark brown water stain",
    "a black burn mark",
    "shattered broken glass",
    "a red wine stain",
    "a deep scratch and gouge",
    "a hole punched in the wall",
    "black mold stain",
    "a broken chipped edge",
    "a dirty grease stain",
]
NEGATIVE_PROMPT = "cartoon, drawing, text, watermark, blurry, low quality"


def random_blob_mask(h: int, w: int, rng: np.random.Generator,
                     area: tuple[float, float] = (0.02, 0.12)) -> np.ndarray:
    """An irregular blob (union of rotated ellipses) covering `area` of the image."""
    target = rng.uniform(*area) * h * w
    mask = np.zeros((h, w), np.uint8)
    cy = rng.uniform(0.3, 0.85) * h  # damage tends to sit on floors, tables, lower walls
    cx = rng.uniform(0.15, 0.85) * w
    r = math.sqrt(target / math.pi)
    for _ in range(int(rng.integers(2, 5))):
        ax = (int(r * rng.uniform(0.5, 1.1)), int(r * rng.uniform(0.3, 0.9)))
        c = (int(cx + rng.normal(0, r * 0.4)), int(cy + rng.normal(0, r * 0.4)))
        cv2.ellipse(mask, c, ax, float(rng.uniform(0, 180)), 0, 360, 1, -1)
    return mask


def feather(mask: np.ndarray, k: int = 7) -> np.ndarray:
    m = cv2.GaussianBlur(mask.astype(np.float32), (k | 1, k | 1), 0)
    return np.clip(m, 0, 1)[..., None]


def blend(base: np.ndarray, patch: np.ndarray, mask: np.ndarray, k: int = 7) -> np.ndarray:
    a = feather(mask, k)
    return (base.astype(np.float32) * (1 - a) + patch.astype(np.float32) * a).round().clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------- paint
def _crack_layer(h: int, w: int, rng: np.random.Generator, region: np.ndarray) -> np.ndarray:
    layer = np.zeros((h, w), np.uint8)
    ys, xs = np.nonzero(region)
    for _ in range(int(rng.integers(1, 4))):
        i = rng.integers(len(ys))
        y, x = float(ys[i]), float(xs[i])
        ang = rng.uniform(0, 2 * math.pi)
        for _ in range(int(rng.integers(25, 70))):
            ang += rng.normal(0, 0.45)
            ny, nx = y + 4 * math.sin(ang), x + 4 * math.cos(ang)
            cv2.line(layer, (int(x), int(y)), (int(nx), int(ny)), 1, int(rng.integers(1, 4)))
            if rng.random() < 0.08:  # branch
                by, bx, ba = ny, nx, ang + rng.choice([-1, 1]) * rng.uniform(0.6, 1.2)
                for _ in range(int(rng.integers(5, 20))):
                    ba += rng.normal(0, 0.4)
                    nby, nbx = by + 3 * math.sin(ba), bx + 3 * math.cos(ba)
                    cv2.line(layer, (int(bx), int(by)), (int(nbx), int(nby)), 1, 1)
                    by, bx = nby, nbx
            y, x = ny, nx
    return layer


def paint_damage(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    region = random_blob_mask(h, w, rng, (0.01, 0.06))
    out = img.astype(np.float32)
    if rng.random() < 0.5:  # crack
        mask = _crack_layer(h, w, rng, region)
        dark = rng.uniform(0.15, 0.45)
        a = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)[..., None]
        out = out * (1 - a * (1 - dark))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    else:  # stain: soft irregular colour multiply
        mask = region
        tint = np.array(rng.choice([[0.55, 0.42, 0.30], [0.35, 0.30, 0.28], [0.60, 0.25, 0.25], [0.45, 0.50, 0.35]]))
        a = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0)
        noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 6)
        a = (a * (0.6 + 0.8 * noise)).clip(0, 1)[..., None] * rng.uniform(0.5, 0.9)
        out = out * (1 - a) + out * tint * a
        mask = (a[..., 0] > 0.1).astype(np.uint8)
    return out.round().clip(0, 255).astype(np.uint8), mask


# ---------------------------------------------------------------- splice
def splice_damage(img: np.ndarray, donor: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Copy-move (donor is the same photo) or splice (donor is another photo), darkened like grime."""
    h, w = img.shape[:2]
    mask = random_blob_mask(h, w, rng, (0.02, 0.10))
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bh, bw = y1 - y0, x1 - x0
    dh, dw = donor.shape[:2]
    sy, sx = int(rng.integers(0, dh - bh + 1)), int(rng.integers(0, dw - bw + 1))
    patch = img.copy()
    src = donor[sy:sy + bh, sx:sx + bw].astype(np.float32)
    src = src * rng.uniform(0.55, 1.0) + rng.normal(0, 4)  # make it look worn / dirty
    if rng.random() < 0.5:
        src = src[:, ::-1]
    patch[y0:y1, x0:x1] = src.clip(0, 255).astype(np.uint8)
    return blend(img, patch, mask, k=int(rng.choice([3, 7, 11]))), mask


# ---------------------------------------------------------------- diffusion
class DiffusionInpainter:
    """Stable Diffusion 1.5 inpainting + LCM-LoRA (4-6 steps) so it runs on an 8 GB MacBook."""

    def __init__(self, device: str | None = None, fp16: bool | None = None):
        import torch
        from diffusers import LCMScheduler, StableDiffusionInpaintPipeline

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else
                                 "cuda" if torch.cuda.is_available() else "cpu")
        # fp16 only on CUDA. On Apple MPS the fp16 UNet returns NaNs (black patches),
        # even with an fp32 VAE, so the Mac runs in fp32 (fits once nothing else is loaded).
        fp16 = (self.device == "cuda") if fp16 is None else fp16
        dtype = torch.float16 if fp16 else torch.float32
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting", variant="fp16",  # fp16 files, upcast on load
            torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.fuse_lora()
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_attention_slicing()  # 8 GB: the full 4096x4096 attention map does not fit
        self.pipe = pipe.to(self.device)
        self.vae32 = pipe.vae if not fp16 else __import__("copy").deepcopy(pipe.vae).to(self.device, torch.float32)

    def __call__(self, img: np.ndarray, mask: np.ndarray, prompt: str, seed: int,
                 steps: int = 5, paste_back: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Returns (edited image, mask of pixels the model changed)."""
        from PIL import Image

        h, w = img.shape[:2]
        g = self.torch.Generator("cpu").manual_seed(seed)
        dil = cv2.dilate(mask, np.ones((9, 9), np.uint8))
        res = self.pipe(prompt=f"a photo of {prompt}, realistic, indoor",
                        negative_prompt=NEGATIVE_PROMPT,
                        image=Image.fromarray(img), mask_image=Image.fromarray(dil * 255),
                        height=h, width=w, num_inference_steps=steps, guidance_scale=1.0,
                        generator=g, output_type="latent").images
        with self.torch.no_grad():
            dec = self.vae32.decode(res.float() / self.vae32.config.scaling_factor).sample
        gen = ((dec[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
        if not np.isfinite(dec.cpu().numpy()).all():
            raise RuntimeError("generator returned NaN")
        if not paste_back:
            return gen, np.ones_like(mask)
        return blend(img, gen, dil, k=9), dil

