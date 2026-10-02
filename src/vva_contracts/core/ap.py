"""COCO-style detection matching and Average Precision (AP).

Why this module exists
----------------------
"Does the detector work on *my* cameras?" needs one number per class that is
comparable across models and quantisation levels. AP is that number. This
module re-implements the COCO evaluation protocol (Lin et al., 2014) so that
our results can be cross-checked against ``pycocotools`` (see
``tests/test_ap_crosscheck.py``), while staying small enough to audit.

The algorithm, step by step
---------------------------
1. **Matching** (per image, per class, per IoU threshold t): predictions are
   visited in descending confidence. Each one is matched to the unmatched
   ground-truth box with the highest IoU, provided IoU >= t. A matched
   prediction is a true positive (TP); an unmatched one a false positive (FP).
   Each ground truth can be matched at most once, so duplicates become FPs.
2. **Precision/recall curve** (per class, all images pooled): sort every
   prediction by confidence and take cumulative sums. After k predictions,
   precision_k = TP_k / k and recall_k = TP_k / n_gt.
3. **Monotone envelope**: replace each precision by the maximum precision at
   any equal-or-higher recall, p_interp(r) = max_{r' >= r} p(r'). The raw
   curve is jagged; without the envelope its area depends on tiny ranking
   changes. PASCAL VOC introduced it and COCO kept it for comparability.
4. **101-point AP**: average p_interp at recall = 0.00, 0.01, ..., 1.00.
   Recall levels the detector never reaches contribute precision 0.
5. **AP@[.5:.95]**: repeat for t = 0.50, 0.55, ..., 0.95 and average. This
   rewards tight localisation, not just "roughly the right place".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vva_contracts.core.geometry import FloatArray, iou_matrix

# Exactly the thresholds COCO uses (pycocotools Params.setDetParams).
IOU_THRESHOLDS: FloatArray = np.linspace(0.5, 0.95, 10)
RECALL_LEVELS: FloatArray = np.linspace(0.0, 1.0, 101)
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ImageDetections:
    """Ground truth and predictions of ONE class in ONE image (xyxy, normalised)."""

    gt_boxes: FloatArray  # shape (G, 4)
    pred_boxes: FloatArray  # shape (P, 4)
    pred_scores: FloatArray  # shape (P,)


@dataclass(frozen=True)
class ClassResult:
    """Measured evidence for one class; ``ap_per_iou`` is None when n_gt == 0."""

    n_gt: int
    n_pred: int
    ap_per_iou: list[float] | None
    tp_at_operating_point: int
    fp_at_operating_point: int


def match_image(image: ImageDetections, max_dets: int) -> tuple[FloatArray, BoolArray]:
    """Greedy COCO matching for one image.

    Returns:
        (scores, tp) where ``scores`` are the kept prediction scores sorted
        descending and ``tp`` is a (T x P) boolean matrix: tp[t, k] is True if
        the k-th prediction is a true positive at IOU_THRESHOLDS[t].

    """
    # Stable sort ("mergesort") keeps the input order among equal scores,
    # which is what pycocotools does; an unstable sort could flip ties.
    order = np.argsort(-image.pred_scores, kind="mergesort")[:max_dets]
    scores = image.pred_scores[order]
    ious = iou_matrix(image.pred_boxes[order], image.gt_boxes)
    n_thr, n_pred, n_gt = len(IOU_THRESHOLDS), len(order), image.gt_boxes.shape[0]
    tp = np.zeros((n_thr, n_pred), dtype=bool)
    for t_idx, threshold in enumerate(IOU_THRESHOLDS):
        gt_taken = np.zeros(n_gt, dtype=bool)
        for k in range(n_pred):
            # Candidates: ground truths not yet taken whose IoU clears t.
            candidates = (~gt_taken) & (ious[k] >= threshold)
            if not candidates.any():
                continue  # nothing to match -> this prediction stays an FP
            masked = np.where(candidates, ious[k], -1.0)
            # pycocotools keeps the LAST index among equal maxima (it uses
            # '>=' while scanning), so we search the reversed array.
            best = n_gt - 1 - int(np.argmax(masked[::-1]))
            gt_taken[best] = True
            tp[t_idx, k] = True
    return scores, tp


def average_precision(scores: FloatArray, tp: BoolArray, n_gt: int) -> float:
    """101-point interpolated AP for one IoU threshold (see module docstring)."""
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    tp_sorted = tp[order]
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(~tp_sorted)
    recall = tp_cum / n_gt
    # np.spacing(1) (~2.2e-16) avoids 0/0 exactly as pycocotools does.
    precision = tp_cum / (tp_cum + fp_cum + np.spacing(1))
    # Monotone envelope: a reversed running maximum, then reversed back.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    # For each recall level r, the first index whose recall reaches r.
    idx = np.searchsorted(recall, RECALL_LEVELS, side="left")
    reached = idx < len(recall)
    sampled = np.zeros(len(RECALL_LEVELS))
    sampled[reached] = envelope[idx[reached]]
    return float(sampled.mean())


def evaluate_class(images: list[ImageDetections], operating_conf: float, max_dets: int) -> ClassResult:
    """Evaluate one class over all images (pooled, as COCO does)."""
    n_gt = sum(img.gt_boxes.shape[0] for img in images)
    per_image = [match_image(img, max_dets) for img in images]
    if per_image:
        scores = np.concatenate([s for s, _ in per_image])
        tp = np.concatenate([m for _, m in per_image], axis=1)
    else:
        scores, tp = np.zeros(0), np.zeros((len(IOU_THRESHOLDS), 0), dtype=bool)

    # Operating point at IoU 0.5 (row 0). Filtering by confidence keeps a
    # *prefix* of each image's sorted predictions, and greedy matching of a
    # prefix is identical to the prefix of the full matching, so we can reuse
    # the full matrix instead of re-matching.
    above = scores >= operating_conf
    tp_op = int(tp[0, above].sum())
    fp_op = int(above.sum()) - tp_op

    ap_list: list[float] | None = None
    if n_gt > 0:  # AP is undefined without ground truth (COCO reports -1)
        ap_list = [average_precision(scores, tp[t], n_gt) for t in range(len(IOU_THRESHOLDS))]
    return ClassResult(
        n_gt=n_gt,
        n_pred=int(scores.size),
        ap_per_iou=ap_list,
        tp_at_operating_point=tp_op,
        fp_at_operating_point=fp_op,
    )


def agreement_pairs(ref: FloatArray, cand: FloatArray, iou_threshold: float = 0.5) -> list[tuple[int, int, float]]:
    """Greedily pair reference and candidate boxes (both sorted by score).

    Used by QUANT_PARITY to compare FP32 and INT8 outputs box by box. Returns
    (ref_index, cand_index, iou) triples; each box is used at most once.
    """
    ious = iou_matrix(ref, cand)
    taken = np.zeros(cand.shape[0], dtype=bool)
    pairs: list[tuple[int, int, float]] = []
    for r in range(ref.shape[0]):
        masked = np.where(~taken & (ious[r] >= iou_threshold), ious[r], -1.0)
        if masked.size == 0 or masked.max() < 0:
            continue
        c = int(np.argmax(masked))
        taken[c] = True
        pairs.append((r, c, float(ious[r, c])))
    return pairs
