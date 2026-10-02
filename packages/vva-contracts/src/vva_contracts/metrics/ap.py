"""Average Precision (AP) for object detection, COCO-compatible.

The problem AP solves
---------------------
A detector does not output a yes/no answer; it outputs boxes with confidence
scores. Every confidence cut-off gives a different (precision, recall) pair,
so comparing two detectors at a single threshold is arbitrary. AP summarizes
the *whole* precision-recall trade-off in one number.

Algorithm (per class, per IoU threshold t), as in pycocotools
---------------------------------------------------------------
1. Per image, sort detections by score (descending, stable) and keep at most
   ``MAX_DETECTIONS_PER_IMAGE``.
2. Greedy matching: each detection, in score order, takes the *still
   unmatched* ground-truth box with the highest IoU, if that IoU >= t. It is
   then a true positive (TP); otherwise a false positive (FP). Each ground
   truth can be matched at most once, so duplicates of one object are FPs.
3. Merge all images and sort detections by score (stable). Walking down the
   list, cumulative counts give
       precision_k = TP_k / (TP_k + FP_k)      recall_k = TP_k / N_gt
4. Monotone envelope: p_interp(r) = max_{r' >= r} p(r'). The raw curve is
   saw-toothed (each FP lowers precision, the next TP raises it); the envelope
   removes that noise and makes the area well-defined. This is what PASCAL VOC
   and then COCO standardized so numbers were comparable across papers.
5. Sample the envelope at 101 recall points r ∈ {0, 0.01, ..., 1}; recall
   values the detector never reaches get precision 0. AP is the mean of those
   101 samples, i.e. a Riemann approximation of the area under the curve.

mAP@0.5 uses t = 0.5; mAP@[.5:.95] averages AP over t ∈ {0.50, 0.55, ..., 0.95},
which rewards tight localization, not just "roughly in the right place".

Deliberate differences from pycocotools
---------------------------------------
No ``iscrowd``/ignore regions and no area ranges (YOLO labels have neither).
With those absent, results match pycocotools exactly; the test suite
cross-checks this when pycocotools is installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vva_contracts.geometry import FloatArray, iou_matrix

# Exactly the expression pycocotools uses, so threshold floats are bit-identical
# (0.65 here is 0.6500000000000001, as in pycocotools).
COCO_IOU_THRESHOLDS: FloatArray = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
COCO_RECALL_THRESHOLDS: FloatArray = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
MAX_DETECTIONS_PER_IMAGE = 100


@dataclass(frozen=True)
class ImageDetections:
    """Detections of one class in one image, in normalized xyxy coordinates."""

    scores: FloatArray  # shape (k,)
    boxes: FloatArray  # shape (k, 4)


def _sorted_capped(dets: ImageDetections) -> ImageDetections:
    """Step 1: stable sort by descending score, then cap the count."""
    # kind="mergesort" is stable: equal scores keep their input order, which
    # makes the result reproducible and identical to pycocotools.
    order = np.argsort(-dets.scores, kind="mergesort")[:MAX_DETECTIONS_PER_IMAGE]
    return ImageDetections(scores=dets.scores[order], boxes=dets.boxes[order])


def match_image(gt_boxes: FloatArray, dets: ImageDetections, iou_threshold: float) -> NDArray[np.bool_]:
    """Step 2: greedy matching in one image.

    Args:
        gt_boxes: ground truth of this class, shape ``(g, 4)``.
        dets: detections already sorted and capped by ``_sorted_capped``.
        iou_threshold: minimum IoU for a match.

    Returns:
        Boolean array, True where the detection at that position is a TP.
    """
    is_tp = np.zeros(len(dets.scores), dtype=np.bool_)
    if len(gt_boxes) == 0 or len(dets.scores) == 0:
        return is_tp

    ious = iou_matrix(dets.boxes, gt_boxes)  # (k, g)
    gt_taken = np.zeros(len(gt_boxes), dtype=np.bool_)
    # pycocotools caps the starting threshold just below 1 so a perfect IoU of
    # exactly 1.0 still satisfies "iou >= best" on the first comparison.
    floor = min(iou_threshold, 1.0 - 1e-10)

    for d in range(len(dets.scores)):
        best = floor
        match = -1
        for g in range(len(gt_boxes)):
            if gt_taken[g]:
                continue
            # "< best -> skip" keeps the LAST ground truth among equal IoUs,
            # which is pycocotools' tie-breaking; it rarely matters but keeps
            # the cross-check exact.
            if ious[d, g] < best:
                continue
            best = float(ious[d, g])
            match = g
        if match >= 0:
            gt_taken[match] = True
            is_tp[d] = True
    return is_tp


def ranked_outcomes(
    gt_by_image: dict[str, FloatArray],
    dets_by_image: dict[str, ImageDetections],
    iou_threshold: float,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Step 3 input: all detections of one class, globally ranked by score.

    Images are processed in sorted ``image_id`` order before the global stable
    sort, so ties between images resolve deterministically.

    Returns:
        ``(scores, is_tp)`` in descending score order.
    """
    scores_parts: list[FloatArray] = []
    tp_parts: list[NDArray[np.bool_]] = []
    empty_gt = np.zeros((0, 4), dtype=np.float64)
    for image_id in sorted(set(gt_by_image) | set(dets_by_image)):
        dets = dets_by_image.get(image_id)
        if dets is None or len(dets.scores) == 0:
            continue  # an image without detections contributes only to N_gt
        ranked = _sorted_capped(dets)
        scores_parts.append(ranked.scores)
        tp_parts.append(match_image(gt_by_image.get(image_id, empty_gt), ranked, iou_threshold))

    if not scores_parts:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.bool_)
    scores = np.concatenate(scores_parts)
    is_tp = np.concatenate(tp_parts)
    order = np.argsort(-scores, kind="mergesort")
    return scores[order], is_tp[order]


def interpolated_precision(is_tp_ranked: NDArray[np.bool_], n_gt: int) -> FloatArray:
    """Steps 3-5: the 101-point interpolated precision curve.

    Args:
        is_tp_ranked: TP flags in descending score order.
        n_gt: number of ground-truth boxes of this class (must be > 0).

    Returns:
        Array of 101 precision values, non-increasing in recall.
    """
    if n_gt <= 0:
        raise ValueError("interpolated precision is undefined when there is no ground truth")

    tp_cum = np.cumsum(is_tp_ranked, dtype=np.float64)
    fp_cum = np.cumsum(~is_tp_ranked, dtype=np.float64)
    recall = tp_cum / n_gt
    # np.spacing(1) (~2.2e-16) mirrors pycocotools; it only matters when the
    # denominator is 0, which cannot happen after the first detection anyway.
    precision = tp_cum / (tp_cum + fp_cum + np.spacing(1))

    # Step 4, monotone envelope, written as a reverse running maximum.
    # np.maximum.accumulate on the reversed array computes max over the suffix.
    envelope = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision

    # Step 5: for each recall threshold r, the first ranked position whose
    # recall >= r. side="left" is exactly pycocotools' choice.
    positions = np.searchsorted(recall, COCO_RECALL_THRESHOLDS, side="left")
    sampled = np.zeros(len(COCO_RECALL_THRESHOLDS), dtype=np.float64)
    reachable = positions < len(envelope)  # recall levels the detector reaches
    sampled[reachable] = envelope[positions[reachable]]
    return sampled


def counts_at_confidence(
    gt_by_image: dict[str, FloatArray],
    dets_by_image: dict[str, ImageDetections],
    min_confidence: float,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """TP/FP/FN at one operating point (the threshold the alert rules use).

    Returns:
        ``(tp, fp, fn)`` where ``fn = N_gt - tp``.
    """
    tp = fp = 0
    for image_id in sorted(set(gt_by_image) | set(dets_by_image)):
        dets = dets_by_image.get(image_id)
        if dets is None:
            continue
        keep = dets.scores >= min_confidence
        kept = _sorted_capped(ImageDetections(scores=dets.scores[keep], boxes=dets.boxes[keep]))
        matched = match_image(gt_by_image.get(image_id, np.zeros((0, 4))), kept, iou_threshold)
        tp += int(matched.sum())
        fp += int((~matched).sum())
    n_gt = sum(len(boxes) for boxes in gt_by_image.values())
    return tp, fp, n_gt - tp
