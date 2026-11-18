import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from claimshield.damage import blend, paint_damage, random_blob_mask, splice_damage
from claimshield.features import ELA_DIM, SRM_DIM, ela_features, srm_features
from claimshield.patchnet import PATCH, build, grid, heatmap, patch_labels
from claimshield.postprocess import attack, benign_global_edit, delivery, square_crop

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    base = np.linspace(40, 200, 512, dtype=np.float32)
    im = np.stack([np.add.outer(base, base) / 2] * 3, -1) + rng.normal(0, 6, (512, 512, 3))
    return im.clip(0, 255).astype(np.uint8)


def test_blob_mask_area_in_range():
    rng = np.random.default_rng(1)
    fr = [random_blob_mask(512, 512, rng).mean() for _ in range(50)]
    assert 0.005 < np.median(fr) < 0.2
    assert all(f > 0 for f in fr)


def test_paint_only_changes_masked_region(img):
    rng = np.random.default_rng(2)
    out, mask = paint_damage(img, rng)
    changed = np.abs(out.astype(int) - img.astype(int)).max(-1) > 2
    # almost every visibly changed pixel lies inside the reported mask (dilated for blur)
    import cv2
    m = cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0
    assert changed.sum() > 0
    assert (changed & ~m).sum() / changed.sum() < 0.02


def test_splice_and_blend_shapes(img):
    rng = np.random.default_rng(3)
    out, mask = splice_damage(img, img, rng)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert mask.shape == img.shape[:2] and set(np.unique(mask)) <= {0, 1}
    assert np.array_equal(blend(img, img, mask), img)


def test_delivery_keeps_mask_aligned(img):
    rng = np.random.default_rng(4)
    mask = np.zeros((512, 512), np.uint8)
    mask[100:200, 300:400] = 1
    out, m, pp = delivery(img, mask, rng)
    assert out.shape[:2] == m.shape
    assert 0.6 <= pp["scale"] <= 1.0
    assert 65 <= pp["jpeg_q1"] <= 95


def test_benign_edit_and_crop(img):
    rng = np.random.default_rng(5)
    assert benign_global_edit(img, rng).shape == img.shape
    assert square_crop(np.zeros((300, 450, 3), np.uint8)).shape == (512, 512, 3)
    for a in ["jpeg50", "half", "screenshot"]:
        assert attack(img, a).ndim == 3


def test_feature_dims_and_determinism(img):
    assert ela_features(img).shape == (ELA_DIM,)
    assert srm_features(img).shape == (SRM_DIM,)
    assert np.allclose(srm_features(img), srm_features(img))


def test_srm_sees_a_smooth_pasted_region(img):
    """A noiseless pasted patch should change the spread of residual energy."""
    fake = img.copy()
    fake[100:260, 100:260] = 128
    assert not np.allclose(srm_features(img), srm_features(fake))


def test_grid_covers_image_edges():
    c = grid(300, 410)
    assert (300 - PATCH, 410 - PATCH) in c
    assert all(y + PATCH <= 300 and x + PATCH <= 410 for y, x in c)


def test_patch_labels():
    mask = np.zeros((256, 256), np.float32)
    mask[:128, :128] = 1
    coords = [(0, 0), (128, 128), (0, 100)]
    assert list(patch_labels(mask, coords)) == [1, 0, -1]
    assert list(patch_labels(None, coords)) == [0, 0, 0]


def test_heatmap_shape_and_range(img):
    m = build(pretrained=False).eval()
    hm, score = heatmap(m, img[:300, :400], "cpu")
    assert hm.shape == (300, 400)
    assert 0 <= hm.min() and hm.max() <= 1 and 0 <= score <= 1


def test_manifest_has_no_scene_leakage():
    path = ROOT / "data" / "manifest.csv"
    if not path.exists():
        pytest.skip("dataset not built")
    rows = list(csv.DictReader(open(path)))
    splits = {}
    for r in rows:
        if r["split"] == "ood":  # unseen-tool photos reuse test rooms on purpose
            continue
        splits.setdefault(r["base_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in splits.values()), "a room photo appears in two splits"
    test = [r for r in rows if r["split"] == "test"]
    assert sum(r["label"] == "0" for r in test) == sum(r["label"] == "1" for r in test)


def test_trained_model_flags_the_shipped_examples():
    """The released model separates a pasted-damage photo from an honest one (docs/examples)."""
    p = ROOT / "models" / "patchnet.pt"
    ex = ROOT / "docs" / "examples"
    if not (p.exists() and (ex / "splice.png").exists()):
        pytest.skip("model or examples missing")
    from PIL import Image
    from claimshield.cli import load_model
    model, thr = load_model(p)
    score = lambda n: heatmap(model, np.asarray(Image.open(ex / f"{n}.png").convert("RGB")))[1]
    assert score("splice") > thr > score("authentic")
