"""Cross-check core.ap against pycocotools, the reference COCO implementation.

Why this test matters: AP has many near-identical variants (11-point VOC,
101-point COCO, area-under-raw-curve). Matching pycocotools to ~1e-12 on
random scenes is the evidence that our numbers mean what COCO numbers mean.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from vva_contracts.core.ap import ImageDetections, evaluate_class

COCOeval = pytest.importorskip("pycocotools.cocoeval").COCOeval
COCO = pytest.importorskip("pycocotools.coco").COCO

W, H = 640.0, 480.0  # pixel size used for the COCO side (our side is normalised)


def _random_scene(rng: np.random.Generator, n_images: int):
    """Ground truth plus noisy, sometimes duplicated or spurious predictions."""
    scenes = []
    for _ in range(n_images):
        n_gt = int(rng.integers(0, 6))
        xy = rng.uniform(0.0, 0.7, size=(n_gt, 2))
        wh = rng.uniform(0.05, 0.3, size=(n_gt, 2))
        gt = np.hstack([xy, xy + wh])
        preds, scores = [], []
        for box in gt:
            if rng.random() < 0.85:  # detected, with localisation noise
                preds.append(box + rng.normal(0, 0.02, 4))
                scores.append(rng.uniform(0.3, 1.0))
            if rng.random() < 0.15:  # duplicate detection -> must become an FP
                preds.append(box + rng.normal(0, 0.03, 4))
                scores.append(rng.uniform(0.1, 0.9))
        for _ in range(int(rng.integers(0, 3))):  # pure false positives
            xy2 = rng.uniform(0.0, 0.8, 2)
            preds.append(np.concatenate([xy2, xy2 + rng.uniform(0.05, 0.2, 2)]))
            scores.append(rng.uniform(0.0, 0.8))
        p = np.clip(np.asarray(preds, dtype=np.float64).reshape(-1, 4), 0.0, 1.0)
        p[:, 2:] = np.maximum(p[:, 2:], p[:, :2] + 1e-3)  # keep x2 > x1 after clipping
        scenes.append(ImageDetections(gt.astype(np.float64), p, np.asarray(scores, dtype=np.float64)))
    return scenes


def _coco_ap(scenes) -> list[float]:
    """Run pycocotools on the same scenes (one category, pixel coordinates)."""
    images, annotations, results = [], [], []
    ann_id = 1
    for img_id, scene in enumerate(scenes, start=1):
        images.append({"id": img_id, "width": W, "height": H})
        for box in scene.gt_boxes:
            x1, y1, x2, y2 = box * [W, H, W, H]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        for box, score in zip(scene.pred_boxes, scene.pred_scores, strict=True):
            x1, y1, x2, y2 = box * [W, H, W, H]
            results.append(
                {"image_id": img_id, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)}
            )
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints a lot
        gt = COCO()
        gt.dataset = {"images": images, "annotations": annotations, "categories": [{"id": 1}]}
        gt.createIndex()
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
    # precision shape: [T, R, K, A, M]; area 'all' = 0, maxDets 100 = index 2.
    precision = ev.eval["precision"][:, :, 0, 0, 2]
    return [float(np.mean(row)) for row in precision]


@pytest.mark.parametrize("seed", range(8))
def test_ap_matches_pycocotools(seed: int) -> None:
    rng = np.random.default_rng(seed)
    scenes = _random_scene(rng, n_images=25)
    ours = evaluate_class(scenes, operating_conf=0.5, max_dets=100)
    assert ours.ap_per_iou is not None
    np.testing.assert_allclose(ours.ap_per_iou, _coco_ap(scenes), rtol=0, atol=1e-9)
