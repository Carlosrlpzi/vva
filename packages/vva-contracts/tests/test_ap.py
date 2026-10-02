"""AP correctness: a hand-computed case, edge cases, and a pycocotools cross-check."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from vva_contracts.geometry import iou_matrix
from vva_contracts.metrics.ap import (
    COCO_IOU_THRESHOLDS,
    ImageDetections,
    counts_at_confidence,
    interpolated_precision,
    ranked_outcomes,
)


def _ap(gt: dict[str, np.ndarray], dets: dict[str, ImageDetections], t: float = 0.5) -> float:
    n_gt = sum(len(b) for b in gt.values())
    _, is_tp = ranked_outcomes(gt, dets, t)
    return float(interpolated_precision(is_tp, n_gt).mean())


def test_iou_known_values() -> None:
    a = np.array([[0.0, 0.0, 2.0, 2.0]])
    b = np.array([[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 6.0, 6.0]])
    # Overlap 1x1 = 1, union 4 + 4 - 1 = 7.
    np.testing.assert_allclose(iou_matrix(a, b), [[1 / 7, 1.0, 0.0]])


def test_hand_computed_ap() -> None:
    """Ranked outcomes TP, FP, TP over 2 ground truths.

    recall    = [0.5, 0.5, 1.0]; precision = [1, 0.5, 2/3]; envelope = [1, 2/3, 2/3]
    Recall points 0.00..0.50 (51 points) take precision 1, the 50 points in
    (0.50, 1.00] take 2/3. AP = (51 * 1 + 50 * 2/3) / 101.
    """
    gt = {"img": np.array([[0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 0.7, 0.7]])}
    dets = {
        "img": ImageDetections(
            scores=np.array([0.9, 0.8, 0.7]),
            boxes=np.array([[0.0, 0.0, 0.2, 0.2], [0.8, 0.8, 0.9, 0.9], [0.5, 0.5, 0.7, 0.7]]),
        )
    }
    assert _ap(gt, dets) == pytest.approx((51 + 50 * 2 / 3) / 101, abs=1e-12)


def test_duplicate_detection_is_false_positive() -> None:
    gt = {"img": np.array([[0.0, 0.0, 0.5, 0.5]])}
    dets = {"img": ImageDetections(np.array([0.9, 0.8]), np.array([[0.0, 0.0, 0.5, 0.5]] * 2))}
    assert counts_at_confidence(gt, dets, 0.0) == (1, 1, 0)


def test_no_detections_gives_zero_ap() -> None:
    gt = {"img": np.array([[0.0, 0.0, 0.5, 0.5]])}
    assert _ap(gt, {}) == 0.0


def test_operating_threshold_filters_low_confidence() -> None:
    gt = {"img": np.array([[0.0, 0.0, 0.5, 0.5]])}
    dets = {"img": ImageDetections(np.array([0.4]), np.array([[0.0, 0.0, 0.5, 0.5]]))}
    assert counts_at_confidence(gt, dets, 0.6) == (0, 0, 1)


def _random_case(rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict[str, ImageDetections]]:
    gt: dict[str, np.ndarray] = {}
    dets: dict[str, ImageDetections] = {}
    for i in range(12):
        n_gt = int(rng.integers(0, 4))
        centers = rng.uniform(0.2, 0.8, size=(n_gt, 2))
        sizes = rng.uniform(0.05, 0.3, size=(n_gt, 2))
        boxes = np.hstack([centers - sizes / 2, centers + sizes / 2])
        gt[f"im{i:02d}"] = boxes
        # Detections: jittered copies of the ground truth plus pure noise boxes.
        jitter = boxes + rng.normal(0, 0.02, size=boxes.shape)
        noise_c = rng.uniform(0.1, 0.9, size=(int(rng.integers(0, 3)), 2))
        noise = np.hstack([noise_c - 0.05, noise_c + 0.05])
        all_boxes = np.vstack([jitter, noise]) if len(noise) else jitter
        all_boxes[:, 2:] = np.maximum(all_boxes[:, 2:], all_boxes[:, :2] + 1e-3)
        scores = np.round(rng.uniform(0, 1, size=len(all_boxes)), 2)  # rounding creates ties on purpose
        dets[f"im{i:02d}"] = ImageDetections(scores, all_boxes)
    return gt, dets


@pytest.mark.parametrize("seed", range(10))
def test_matches_pycocotools(seed: int) -> None:
    """Our AP equals pycocotools' AP (no crowd regions, all areas) for every IoU threshold."""
    coco_mod = pytest.importorskip("pycocotools.coco")
    eval_mod = pytest.importorskip("pycocotools.cocoeval")
    rng = np.random.default_rng(seed)
    gt, dets = _random_case(rng)
    if sum(len(b) for b in gt.values()) == 0:
        pytest.skip("no ground truth in this random case")

    # pycocotools works in pixel xywh; use a 1000x1000 canvas (IoU is scale-invariant).
    image_ids = sorted(gt)  # sorted ids -> same image order as our implementation
    scale = 1000.0
    images = [{"id": k + 1, "width": 1000, "height": 1000} for k in range(len(image_ids))]
    annotations, results = [], []
    for k, image_id in enumerate(image_ids):
        for box in gt[image_id] * scale:
            x1, y1, x2, y2 = box.tolist()
            annotations.append(
                {"id": len(annotations) + 1, "image_id": k + 1, "category_id": 1,
                 "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0}
            )  # fmt: skip
        for score, box in zip(dets[image_id].scores, dets[image_id].boxes * scale, strict=True):
            x1, y1, x2, y2 = box.tolist()
            results.append(
                {"image_id": k + 1, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)}
            )

    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints progress
        coco_gt = coco_mod.COCO()
        coco_gt.dataset = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "x"}]}
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results) if results else None
        if coco_dt is None:
            pytest.skip("no detections in this random case")
        evaluator = eval_mod.COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()

    # precision array: [T, R, K, A, M]; area index 0 = all, maxDets index 2 = 100.
    reference = evaluator.eval["precision"][:, :, 0, 0, 2].mean(axis=1)
    ours = [_ap(gt, dets, float(t)) for t in COCO_IOU_THRESHOLDS]
    np.testing.assert_allclose(ours, reference, atol=1e-12)
