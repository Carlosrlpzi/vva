"""DETECTION_EVAL and QUANT_PARITY tasks.

Ground truth comes from YOLO labels; predictions come from a JSONL file of
``PredictionRecord`` written by whatever ran the model (Ultralytics/ONNX on
the laptop, or the Hailo HEF on the Pi). Both are converted to normalized
xyxy coordinates, where IoU is unchanged (see ``geometry.py``).

Fail-fast checks before any metric is computed:
* each prediction's width/height must equal the real image size, otherwise
  its boxes are in another coordinate frame (typically the letterboxed model
  input) and every IoU would be silently wrong;
* every prediction must refer to an image of the split;
* an image may appear at most once in the predictions file.
Images with no prediction line count as "no detections" and are reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vva_contracts.contracts.base import InputFingerprint, safe_ratio
from vva_contracts.contracts.detection_eval import (
    BoxAgreement,
    ClassEval,
    DetectionEvalReport,
    OperatingPoint,
    QuantParityReport,
    derive_parity_verdict,
)
from vva_contracts.contracts.records import PredictionRecord
from vva_contracts.contracts.requests import DetectionEvalRequest, QuantParityRequest
from vva_contracts.errors import InputError
from vva_contracts.geometry import FloatArray, iou_matrix, yolo_to_xyxy
from vva_contracts.io import fingerprint, iter_jsonl, manifest_digest
from vva_contracts.metrics.ap import (
    COCO_IOU_THRESHOLDS,
    MAX_DETECTIONS_PER_IMAGE,
    ImageDetections,
    counts_at_confidence,
    interpolated_precision,
    ranked_outcomes,
)
from vva_contracts.tasks._common import resolve_data_yaml, tool_version
from vva_contracts.workspace import display_path, resolve_in_workspace
from vva_contracts.yolo import image_size, list_split, load_class_names, parse_label_file

# class index -> image_id -> boxes
GroundTruth = dict[int, dict[str, FloatArray]]
Predictions = dict[int, dict[str, ImageDetections]]


@dataclass(frozen=True)
class LoadedSplit:
    class_names: list[str]
    eval_class_ids: list[int]
    image_ids: list[str]
    image_sizes: dict[str, tuple[int, int]]
    ground_truth: GroundTruth
    dataset_inputs: list[InputFingerprint]


def load_split(
    workspace: Path, dataset_root: str, data_yaml: str | None, split: str, classes: list[str] | None
) -> LoadedSplit:
    """Read labels of one split; label errors abort (audit the dataset first)."""
    root = resolve_in_workspace(workspace, dataset_root, kind="dir")
    yaml_path = resolve_data_yaml(workspace, root, data_yaml)
    class_names = load_class_names(yaml_path, shown_as=display_path(workspace, yaml_path))
    if classes is not None:
        unknown = sorted(set(classes) - set(class_names))
        if unknown:
            raise InputError(f"classes not in data.yaml: {unknown}")
    eval_ids = [i for i, n in enumerate(class_names) if classes is None or n in classes]

    listing = list_split(workspace, root, split, max_images=200_000)
    if not listing.items:
        raise InputError(f"split {split!r} has no images")

    ground_truth: GroundTruth = {i: {} for i in eval_ids}
    sizes: dict[str, tuple[int, int]] = {}
    for item in listing.items:
        sizes[item.image_id] = image_size(item.image_path)
        if item.label_path is None:
            continue
        parsed = parse_label_file(item.label_path, len(class_names), shown_as=display_path(workspace, item.label_path))
        if parsed.issues:
            first = parsed.issues[0]
            raise InputError(f"{first.file}:{first.line}: {first.code}; run DATASET_AUDIT and fix labels first")
        xyxy = yolo_to_xyxy(*parsed.boxes_cxcywh.T)
        for cls in eval_ids:
            mask = parsed.classes == cls
            if mask.any():
                ground_truth[cls][item.image_id] = xyxy[mask]

    files = [i.image_path for i in listing.items] + [i.label_path for i in listing.items if i.label_path]
    digest, total = manifest_digest(workspace, files)
    manifest = InputFingerprint(
        role="split_manifest",
        path=display_path(workspace, root / "images" / split),
        sha256=digest,
        size_bytes=total,
    )
    return LoadedSplit(
        class_names=class_names,
        eval_class_ids=eval_ids,
        image_ids=[i.image_id for i in listing.items],
        image_sizes=sizes,
        ground_truth=ground_truth,
        dataset_inputs=[fingerprint(workspace, "data_yaml", yaml_path), manifest],
    )


@dataclass(frozen=True)
class LoadedPredictions:
    predictions: Predictions
    n_other_classes: int
    images_seen: set[str]
    fingerprint: InputFingerprint


def load_predictions(workspace: Path, relative: str, split: LoadedSplit) -> LoadedPredictions:
    path = resolve_in_workspace(workspace, relative, kind="file")
    name_to_id = {split.class_names[i]: i for i in split.eval_class_ids}
    raw: dict[int, dict[str, tuple[list[float], list[list[float]]]]] = {i: {} for i in split.eval_class_ids}
    seen: set[str] = set()
    other = 0

    for lineno, record in iter_jsonl(path, PredictionRecord, shown_as=relative):
        where = f"{relative}:{lineno}"
        if record.image_id not in split.image_sizes:
            raise InputError(f"{where}: image_id {record.image_id!r} is not in the split")
        if record.image_id in seen:
            raise InputError(f"{where}: image_id {record.image_id!r} appears twice")
        seen.add(record.image_id)
        if (record.width, record.height) != split.image_sizes[record.image_id]:
            w, h = split.image_sizes[record.image_id]
            raise InputError(
                f"{where}: prediction frame {record.width}x{record.height} != image {w}x{h}; "
                "boxes must be logged in original-image pixels, not the model input"
            )
        for det in record.detections:
            cls = name_to_id.get(det.class_name)
            if cls is None:
                other += 1  # e.g. COCO 'dog' when evaluating only 'person': ignored but counted
                continue
            scores, boxes = raw[cls].setdefault(record.image_id, ([], []))
            scores.append(det.confidence)
            # Pixels -> normalized: IoU-preserving (see geometry.py docstring).
            x1, y1, x2, y2 = det.bbox_xyxy
            boxes.append([x1 / record.width, y1 / record.height, x2 / record.width, y2 / record.height])

    predictions: Predictions = {
        cls: {
            image_id: ImageDetections(
                scores=np.asarray(s, dtype=np.float64), boxes=np.asarray(b, dtype=np.float64).reshape(-1, 4)
            )
            for image_id, (s, b) in per_image.items()
        }
        for cls, per_image in raw.items()
    }
    return LoadedPredictions(predictions, other, seen, fingerprint(workspace, "predictions", path))


def evaluate(
    split_name: str, split: LoadedSplit, preds: LoadedPredictions, operating_confidence: float
) -> DetectionEvalReport:
    per_class: list[ClassEval] = []
    for cls in split.eval_class_ids:
        gt = split.ground_truth[cls]
        dets = preds.predictions[cls]
        n_gt = sum(len(b) for b in gt.values())
        n_pred = sum(len(d.scores) for d in dets.values())

        ap_per_iou: list[float] | None = None
        curve: list[float] | None = None
        if n_gt > 0:
            ap_per_iou = []
            for t in COCO_IOU_THRESHOLDS:
                _, is_tp = ranked_outcomes(gt, dets, float(t))
                sampled = interpolated_precision(is_tp, n_gt)
                if curve is None:
                    curve = sampled.tolist()  # first threshold is 0.50 -> the AP50 curve
                ap_per_iou.append(float(sampled.mean()))

        tp, fp, fn = counts_at_confidence(gt, dets, operating_confidence)
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, n_gt)
        f1 = None
        if precision is not None and recall is not None:
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        per_class.append(
            ClassEval(
                class_name=split.class_names[cls],
                n_gt=n_gt,
                n_pred=n_pred,
                ap50=None if curve is None else sum(curve) / len(curve),
                ap50_95=None if ap_per_iou is None else sum(ap_per_iou) / len(ap_per_iou),
                ap_per_iou=ap_per_iou,
                pr_curve_ap50=curve,
                operating_point=OperatingPoint(
                    confidence=operating_confidence, tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1
                ),
            )
        )

    defined = [c for c in per_class if c.n_gt > 0]
    return DetectionEvalReport(
        tool_version=tool_version(),
        inputs=[*split.dataset_inputs, preds.fingerprint],
        split=split_name,
        iou_thresholds=[float(t) for t in COCO_IOU_THRESHOLDS],
        max_detections_per_image=MAX_DETECTIONS_PER_IMAGE,
        n_images=len(split.image_ids),
        n_images_without_predictions=len(set(split.image_ids) - preds.images_seen),
        n_predictions_other_classes=preds.n_other_classes,
        per_class=per_class,
        map50=sum(c.ap50 or 0.0 for c in defined) / len(defined) if defined else None,
        map50_95=sum(c.ap50_95 or 0.0 for c in defined) / len(defined) if defined else None,
    )


def run(request: DetectionEvalRequest, workspace: Path) -> DetectionEvalReport:
    split = load_split(workspace, request.dataset_root, request.data_yaml, request.split, request.classes)
    preds = load_predictions(workspace, request.predictions, split)
    return evaluate(request.split, split, preds, request.operating_confidence)


def run_parity(request: QuantParityRequest, workspace: Path) -> QuantParityReport:
    split = load_split(workspace, request.dataset_root, request.data_yaml, request.split, request.classes)
    ref = load_predictions(workspace, request.reference_predictions, split)
    cand = load_predictions(workspace, request.candidate_predictions, split)
    ref_report = evaluate(request.split, split, ref, request.operating_confidence)
    cand_report = evaluate(request.split, split, cand, request.operating_confidence)

    delta50 = None if ref_report.map50 is None or cand_report.map50 is None else cand_report.map50 - ref_report.map50
    delta5095 = (
        None
        if ref_report.map50_95 is None or cand_report.map50_95 is None
        else cand_report.map50_95 - ref_report.map50_95
    )
    return QuantParityReport(
        tool_version=tool_version(),
        reference=ref_report,
        candidate=cand_report,
        delta_map50=delta50,
        delta_map50_95=delta5095,
        box_agreement=_box_agreement(ref, cand, request.operating_confidence),
        max_map50_drop=request.max_map50_drop,
        verdict=derive_parity_verdict(delta50, request.max_map50_drop),
    )


def _box_agreement(ref: LoadedPredictions, cand: LoadedPredictions, min_confidence: float) -> BoxAgreement:
    """Greedy one-to-one matching of reference vs candidate boxes (same class, IoU >= 0.5).

    Pairs are taken in descending IoU order, so each box joins its best
    available partner; this is the usual greedy approximation of a maximum
    matching and is exact whenever boxes do not compete for partners.
    """
    n_ref = n_cand = matched = 0
    conf_deltas: list[float] = []
    for cls, ref_images in ref.predictions.items():
        cand_images = cand.predictions.get(cls, {})
        for image_id in sorted(set(ref_images) | set(cand_images)):
            empty = ImageDetections(np.zeros(0), np.zeros((0, 4)))
            r = ref_images.get(image_id, empty)
            c = cand_images.get(image_id, empty)
            r_keep, c_keep = r.scores >= min_confidence, c.scores >= min_confidence
            r_boxes, r_scores = r.boxes[r_keep], r.scores[r_keep]
            c_boxes, c_scores = c.boxes[c_keep], c.scores[c_keep]
            n_ref += len(r_scores)
            n_cand += len(c_scores)
            if len(r_scores) == 0 or len(c_scores) == 0:
                continue
            ious = iou_matrix(r_boxes, c_boxes)
            used_r: set[int] = set()
            used_c: set[int] = set()
            # Flattened argsort over the IoU matrix: best pairs first (stable for ties).
            for flat in np.argsort(-ious, axis=None, kind="mergesort"):
                i, j = divmod(int(flat), ious.shape[1])
                if ious[i, j] < 0.5:
                    break  # sorted descending: nothing further can match
                if i in used_r or j in used_c:
                    continue
                used_r.add(i)
                used_c.add(j)
                matched += 1
                conf_deltas.append(abs(float(r_scores[i]) - float(c_scores[j])))
    total = n_ref + n_cand
    return BoxAgreement(
        n_reference_boxes=n_ref,
        n_candidate_boxes=n_cand,
        n_matched=matched,
        dice=None if total == 0 else 2 * matched / total,
        mean_abs_confidence_delta=sum(conf_deltas) / len(conf_deltas) if conf_deltas else None,
    )
