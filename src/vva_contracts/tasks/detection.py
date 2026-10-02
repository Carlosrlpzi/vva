"""DETECTION_EVAL and QUANT_PARITY runners.

Flow: data.yaml -> ground truth of one split -> predictions JSONL -> per
image/class arrays -> core.ap -> contract. The runners only *measure*; every
derived number is re-checked by the contract validators on construction.

Predictions are produced elsewhere (Hailo on the Pi, ONNX on the laptop) and
written in the PredictionRow format. Scoring them here, instead of trusting
an exporter's own mAP printout, keeps one evaluation definition for all
models and quantisation levels.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np

from vva_contracts.contracts.detection import (
    AgreementStats,
    ClassMetrics,
    DetectionEvalReport,
    DetectionMetrics,
    QuantParityReport,
    derive_parity_checks,
    mean_ap,
)
from vva_contracts.contracts.logs import PredictionRow
from vva_contracts.contracts.requests import BoxFormat, DetectionEvalRequest, QuantParityRequest
from vva_contracts.core.ap import IOU_THRESHOLDS, ImageDetections, agreement_pairs, evaluate_class
from vva_contracts.core.geometry import FloatArray, xywhn_to_xyxyn
from vva_contracts.core.stats import percentile, safe_ratio
from vva_contracts.errors import InvalidInput
from vva_contracts.paths import resolve_in_workspace
from vva_contracts.readers.jsonl import read_all
from vva_contracts.readers.yolo import DatasetSpec, list_images, load_dataset, parse_label_file
from vva_contracts.tasks.common import build_meta

# image key -> class id -> (boxes N x 4 xyxyn, scores N)
Predictions = dict[str, dict[int, tuple[FloatArray, FloatArray]]]
GroundTruth = dict[str, dict[int, FloatArray]]


def load_ground_truth(spec: DatasetSpec, split: str) -> tuple[GroundTruth, Path]:
    """Load valid boxes of a split; ANY label issue aborts the evaluation.

    Scoring against labels with errors would produce a number that looks
    precise and is wrong. The message points to DATASET_AUDIT for details.
    """
    if split not in spec.splits:
        raise InvalidInput(f"split {split!r} not in data.yaml (have {sorted(spec.splits)})")
    dirs = spec.splits[split]
    n_classes = len(spec.class_names)
    gt: GroundTruth = {}
    for key in list_images(dirs.images):
        per_class: dict[int, list[list[float]]] = defaultdict(list)
        label_file = dirs.labels / f"{key}.txt"
        if label_file.is_file():  # no label file = background image (no objects)
            parsed = parse_label_file(label_file, n_classes, f"{key}.txt")
            if parsed.issues:
                first = parsed.issues[0]
                raise InvalidInput(
                    f"ground truth has label errors (first: {first.file}:{first.line} {first.kind}); "
                    "run DATASET_AUDIT and fix the labels before evaluating"
                )
            for b in parsed.boxes:
                per_class[b.class_id].append([b.cx, b.cy, b.w, b.h])
        gt[key] = {c: xywhn_to_xyxyn(np.asarray(rows, dtype=np.float64)) for c, rows in per_class.items()}
    if not gt:
        raise InvalidInput(f"split {split!r} contains no images")
    return gt, dirs.labels


def load_predictions(path: Path, box_format: BoxFormat, keys: set[str], n_classes: int) -> tuple[Predictions, int]:
    """Read PredictionRow lines, convert to xyxyn and group by image/class."""
    grouped: dict[str, dict[int, tuple[list[list[float]], list[float]]]] = defaultdict(dict)
    rows = read_all(path, PredictionRow)
    for line_no, row in rows:
        if row.image not in keys:
            # Fail fast: a prediction for an unknown image means the files do
            # not belong together (wrong split, renamed frames, ...).
            raise InvalidInput(f"{path.name}:{line_no}: unknown image key {row.image!r}")
        if row.class_id >= n_classes:
            raise InvalidInput(f"{path.name}:{line_no}: class_id {row.class_id} >= {n_classes}")
        boxes, scores = grouped[row.image].setdefault(row.class_id, ([], []))
        boxes.append(list(row.box))
        scores.append(row.confidence)
    result: Predictions = {}
    for key, per_class in grouped.items():
        result[key] = {}
        for class_id, (boxes, scores) in per_class.items():
            arr = np.asarray(boxes, dtype=np.float64)
            if box_format == "xywhn":
                arr = xywhn_to_xyxyn(arr)
            if np.any(arr[:, 2] < arr[:, 0]) or np.any(arr[:, 3] < arr[:, 1]):
                raise InvalidInput(f"{path.name}: boxes for {key!r} are not valid {box_format}")
            result[key][class_id] = (arr, np.asarray(scores, dtype=np.float64))
    return result, len(rows)


def compute_metrics(
    gt: GroundTruth,
    preds: Predictions,
    *,
    class_names: list[str],
    operating_conf: float,
    max_dets: int,
    n_rows: int,
) -> DetectionMetrics:
    """Score predictions against ground truth for every class."""
    empty_boxes, empty_scores = np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)
    classes: list[ClassMetrics] = []
    for class_id, name in enumerate(class_names):
        images: list[ImageDetections] = []
        # Sorted image order mirrors pycocotools (sorted image ids), which
        # matters only for exact ties in confidence across images.
        for key in sorted(gt):
            boxes, scores = preds.get(key, {}).get(class_id, (empty_boxes, empty_scores))
            images.append(ImageDetections(gt[key].get(class_id, empty_boxes), boxes, scores))
        r = evaluate_class(images, operating_conf, max_dets)
        tp, fp, fn = r.tp_at_operating_point, r.fp_at_operating_point, r.n_gt - r.tp_at_operating_point
        precision, recall = safe_ratio(tp, tp + fp), safe_ratio(tp, r.n_gt)
        aps = r.ap_per_iou
        classes.append(
            ClassMetrics(
                class_id=class_id,
                class_name=name,
                n_gt=r.n_gt,
                n_pred=r.n_pred,
                ap_per_iou=aps,
                ap50=aps[0] if aps else None,
                # Same reduction as the validator (statistics.fmean) so the
                # recomputed value matches bit for bit.
                ap50_95=fmean(aps) if aps else None,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=safe_ratio(2 * precision * recall, precision + recall),
            )
        )
    return DetectionMetrics(
        operating_confidence=operating_conf,
        max_dets_per_image=max_dets,
        iou_thresholds=[float(t) for t in IOU_THRESHOLDS],
        n_images=len(gt),
        n_prediction_rows=n_rows,
        classes=classes,
        map50=mean_ap(classes, "ap50"),
        map50_95=mean_ap(classes, "ap50_95"),
    )


def run_detection_eval(req: DetectionEvalRequest, workspace: Path) -> DetectionEvalReport:
    """Entry point for task DETECTION_EVAL."""
    spec = load_dataset(workspace, req.data_yaml)
    gt, labels_dir = load_ground_truth(spec, req.split)
    pred_path = resolve_in_workspace(workspace, req.predictions)
    preds, n_rows = load_predictions(pred_path, req.box_format, set(gt), len(spec.class_names))
    metrics = compute_metrics(
        gt,
        preds,
        class_names=spec.class_names,
        operating_conf=req.operating_confidence,
        max_dets=req.max_dets_per_image,
        n_rows=n_rows,
    )
    meta = build_meta(workspace, [("data_yaml", spec.yaml_path), ("labels", labels_dir), ("predictions", pred_path)])
    return DetectionEvalReport(meta=meta, split=req.split, box_format=req.box_format, metrics=metrics)


def _agreement(ref: Predictions, cand: Predictions, conf: float, iou: float) -> AgreementStats:
    """Pair FP32 and INT8 boxes per image/class above the operating confidence."""
    n_ref = n_cand = 0
    ious: list[float] = []
    conf_diffs: list[float] = []
    for key in sorted(set(ref) | set(cand)):
        for class_id in sorted(set(ref.get(key, {})) | set(cand.get(key, {}))):
            r_boxes, r_scores = ref.get(key, {}).get(class_id, (np.zeros((0, 4)), np.zeros(0)))
            c_boxes, c_scores = cand.get(key, {}).get(class_id, (np.zeros((0, 4)), np.zeros(0)))
            # Keep only boxes the alert rule would see, sorted by confidence
            # (agreement_pairs pairs greedily, so order decides who wins).
            r_order = np.argsort(-r_scores, kind="mergesort")
            c_order = np.argsort(-c_scores, kind="mergesort")
            r_keep = r_order[r_scores[r_order] >= conf]
            c_keep = c_order[c_scores[c_order] >= conf]
            n_ref, n_cand = n_ref + len(r_keep), n_cand + len(c_keep)
            for ri, ci, value in agreement_pairs(r_boxes[r_keep], c_boxes[c_keep], iou):
                ious.append(value)
                conf_diffs.append(abs(float(r_scores[r_keep][ri]) - float(c_scores[c_keep][ci])))
    if not ious:
        return AgreementStats(
            n_reference=n_ref,
            n_candidate=n_cand,
            n_matched=0,
            mean_iou=None,
            mean_abs_conf_diff=None,
            p95_abs_conf_diff=None,
        )
    return AgreementStats(
        n_reference=n_ref,
        n_candidate=n_cand,
        n_matched=len(ious),
        mean_iou=float(np.mean(ious)),
        mean_abs_conf_diff=float(np.mean(conf_diffs)),
        p95_abs_conf_diff=percentile(conf_diffs, 95),
    )


def run_quant_parity(req: QuantParityRequest, workspace: Path) -> QuantParityReport:
    """Entry point for task QUANT_PARITY."""
    spec = load_dataset(workspace, req.data_yaml)
    gt, labels_dir = load_ground_truth(spec, req.split)
    keys, n_classes = set(gt), len(spec.class_names)
    ref_path = resolve_in_workspace(workspace, req.reference_predictions)
    cand_path = resolve_in_workspace(workspace, req.candidate_predictions)
    ref_preds, ref_rows = load_predictions(ref_path, req.box_format, keys, n_classes)
    cand_preds, cand_rows = load_predictions(cand_path, req.box_format, keys, n_classes)
    # Both sides are scored with identical settings; only the predictions differ.
    reference = compute_metrics(
        gt,
        ref_preds,
        class_names=spec.class_names,
        operating_conf=req.operating_confidence,
        max_dets=req.max_dets_per_image,
        n_rows=ref_rows,
    )
    candidate = compute_metrics(
        gt,
        cand_preds,
        class_names=spec.class_names,
        operating_conf=req.operating_confidence,
        max_dets=req.max_dets_per_image,
        n_rows=cand_rows,
    )

    def delta(r: float | None, c: float | None) -> float | None:
        return None if r is None or c is None else c - r

    failed = derive_parity_checks(reference, candidate, req.max_map50_drop, req.max_map50_95_drop)
    meta = build_meta(
        workspace,
        [("data_yaml", spec.yaml_path), ("labels", labels_dir), ("reference", ref_path), ("candidate", cand_path)],
    )
    return QuantParityReport(
        meta=meta,
        split=req.split,
        box_format=req.box_format,
        reference=reference,
        candidate=candidate,
        delta_map50=delta(reference.map50, candidate.map50),
        delta_map50_95=delta(reference.map50_95, candidate.map50_95),
        max_map50_drop=req.max_map50_drop,
        max_map50_95_drop=req.max_map50_95_drop,
        agreement_iou=req.agreement_iou,
        agreement=_agreement(ref_preds, cand_preds, req.operating_confidence, req.agreement_iou),
        failed_checks=failed,
        verdict="FAIL" if failed else "PASS",
    )
