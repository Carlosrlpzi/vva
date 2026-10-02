"""DETECTION_EVAL and QUANT_PARITY reports.

Every summary number in these reports can be recomputed from other fields of
the same report. The validators do exactly that, which turns the report into
a self-checking document:

* ``ap50 == mean(pr_curve_ap50)``     (AP is the mean of the 101 samples)
* ``ap_per_iou[0] == ap50``           (the first COCO threshold is 0.50)
* ``ap50_95 == mean(ap_per_iou)``
* the curve is non-increasing         (the monotone-envelope property)
* ``tp + fn == n_gt``, ``precision == tp / (tp + fp)``, ``recall == tp / n_gt``,
  ``f1 == 2pr / (p + r)``
* ``map50 == mean(ap50 over classes with ground truth)``

For QUANT_PARITY the deltas, the box-agreement Dice coefficient and the verdict
are recomputed as well. A hand-edited number breaks at least one equation.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from vva_contracts.contracts.base import (
    FLOAT_TOLERANCE,
    SCHEMA_VERSION,
    ClassName,
    InputFingerprint,
    NonNegativeFloat,
    StrictModel,
    UnitInterval,
    Verdict,
    require_close,
    safe_ratio,
)

N_RECALL_POINTS = 101
N_IOU_THRESHOLDS = 10


class OperatingPoint(StrictModel):
    """Counts at the confidence threshold the alert rules will actually use (IoU 0.5)."""

    confidence: UnitInterval
    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    precision: UnitInterval | None
    recall: UnitInterval | None
    f1: UnitInterval | None


class ClassEval(StrictModel):
    class_name: ClassName
    n_gt: int = Field(ge=0)
    n_pred: int = Field(ge=0)
    ap50: UnitInterval | None  # null when n_gt == 0: AP is undefined without positives
    ap50_95: UnitInterval | None
    ap_per_iou: list[UnitInterval] | None
    pr_curve_ap50: list[UnitInterval] | None
    operating_point: OperatingPoint

    @model_validator(mode="after")
    def _self_consistent(self) -> ClassEval:
        undefined = self.n_gt == 0
        nulls = [f is None for f in (self.ap50, self.ap50_95, self.ap_per_iou, self.pr_curve_ap50)]
        # All four AP fields are null together (no ground truth) or all set.
        if (undefined and not all(nulls)) or (not undefined and any(nulls)):
            raise ValueError(f"{self.class_name}: AP fields must be null exactly when n_gt == 0")

        if self.ap_per_iou is not None and self.pr_curve_ap50 is not None:
            curve = self.pr_curve_ap50
            if len(curve) != N_RECALL_POINTS or len(self.ap_per_iou) != N_IOU_THRESHOLDS:
                raise ValueError(f"{self.class_name}: expected 101 curve points and 10 IoU thresholds")
            # Envelope property: interpolated precision never increases with recall.
            if any(b > a + FLOAT_TOLERANCE for a, b in pairwise(curve)):
                raise ValueError(f"{self.class_name}: interpolated PR curve must be non-increasing")
            require_close(f"{self.class_name}.ap50", self.ap50, sum(curve) / N_RECALL_POINTS)
            require_close(f"{self.class_name}.ap_per_iou[0]", self.ap_per_iou[0], self.ap50)
            require_close(f"{self.class_name}.ap50_95", self.ap50_95, sum(self.ap_per_iou) / N_IOU_THRESHOLDS)

        op = self.operating_point
        if op.tp + op.fn != self.n_gt:
            raise ValueError(f"{self.class_name}: tp + fn must equal n_gt")
        if op.tp + op.fp > self.n_pred:
            raise ValueError(f"{self.class_name}: tp + fp cannot exceed n_pred")
        precision = safe_ratio(op.tp, op.tp + op.fp)
        recall = safe_ratio(op.tp, self.n_gt)
        require_close(f"{self.class_name}.precision", op.precision, precision)
        require_close(f"{self.class_name}.recall", op.recall, recall)
        f1 = None if precision is None or recall is None else safe_ratio(2 * precision * recall, precision + recall)
        # F1 with p = r = 0 is conventionally 0, not undefined.
        if precision is not None and recall is not None and f1 is None:
            f1 = 0.0
        require_close(f"{self.class_name}.f1", op.f1, f1)
        return self


class DetectionEvalReport(StrictModel):
    contract_id: Literal["DetectionEvalReport"] = "DetectionEvalReport"
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    tool_version: str
    inputs: list[InputFingerprint]
    split: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    iou_thresholds: list[float] = Field(min_length=N_IOU_THRESHOLDS, max_length=N_IOU_THRESHOLDS)
    max_detections_per_image: int = Field(ge=1)
    n_images: int = Field(ge=1)
    n_images_without_predictions: int = Field(ge=0)
    n_predictions_other_classes: int = Field(ge=0)  # predictions of classes not evaluated (ignored, counted)
    per_class: list[ClassEval] = Field(min_length=1)
    map50: UnitInterval | None
    map50_95: UnitInterval | None

    @model_validator(mode="after")
    def _aggregates(self) -> DetectionEvalReport:
        from vva_contracts.metrics.ap import COCO_IOU_THRESHOLDS  # local import keeps numpy optional for schema export

        if self.iou_thresholds != [float(t) for t in COCO_IOU_THRESHOLDS]:
            raise ValueError("iou_thresholds must be the COCO grid 0.50:0.05:0.95")
        if self.n_images_without_predictions > self.n_images:
            raise ValueError("n_images_without_predictions cannot exceed n_images")
        defined = [c for c in self.per_class if c.n_gt > 0]
        expected50 = sum(c.ap50 or 0.0 for c in defined) / len(defined) if defined else None
        expected50_95 = sum(c.ap50_95 or 0.0 for c in defined) / len(defined) if defined else None
        require_close("map50", self.map50, expected50)
        require_close("map50_95", self.map50_95, expected50_95)
        return self


class BoxAgreement(StrictModel):
    """Do the two models fire on the same boxes at the operating threshold?

    ``dice = 2 * matched / (n_reference + n_candidate)``: 1.0 means every box
    of each model has a same-class counterpart (IoU >= 0.5) in the other.
    It complements Δ mAP, which can hide swaps (one model gains a box where
    the other loses one) that matter for alert behaviour.
    """

    iou_threshold: float = Field(default=0.5, ge=0.5, le=0.5)  # fixed by design
    n_reference_boxes: int = Field(ge=0)
    n_candidate_boxes: int = Field(ge=0)
    n_matched: int = Field(ge=0)
    dice: UnitInterval | None
    mean_abs_confidence_delta: NonNegativeFloat | None  # over matched pairs

    @model_validator(mode="after")
    def _recompute(self) -> BoxAgreement:
        if self.n_matched > min(self.n_reference_boxes, self.n_candidate_boxes):
            raise ValueError("n_matched cannot exceed either box count")
        require_close(
            "dice", self.dice, safe_ratio(2 * self.n_matched, self.n_reference_boxes + self.n_candidate_boxes)
        )
        if (self.n_matched == 0) != (self.mean_abs_confidence_delta is None):
            raise ValueError("mean_abs_confidence_delta must be null exactly when n_matched == 0")
        return self


class QuantParityReport(StrictModel):
    contract_id: Literal["QuantParityReport"] = "QuantParityReport"
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    tool_version: str
    reference: DetectionEvalReport
    candidate: DetectionEvalReport
    delta_map50: float | None  # candidate - reference; negative means the candidate is worse
    delta_map50_95: float | None
    box_agreement: BoxAgreement
    max_map50_drop: UnitInterval
    verdict: Verdict

    @model_validator(mode="after")
    def _recompute(self) -> QuantParityReport:
        ref, cand = self.reference, self.candidate
        if ref.split != cand.split or ref.n_images != cand.n_images:
            raise ValueError("reference and candidate must be evaluated on the same split")
        dataset_ref = [i for i in ref.inputs if i.role != "predictions"]
        dataset_cand = [i for i in cand.inputs if i.role != "predictions"]
        if dataset_ref != dataset_cand:
            raise ValueError("reference and candidate were evaluated on different dataset bytes")

        def delta(a: float | None, b: float | None) -> float | None:
            return None if a is None or b is None else b - a

        require_close("delta_map50", self.delta_map50, delta(ref.map50, cand.map50))
        require_close("delta_map50_95", self.delta_map50_95, delta(ref.map50_95, cand.map50_95))
        if self.verdict != derive_parity_verdict(self.delta_map50, self.max_map50_drop):
            raise ValueError("verdict does not follow from delta_map50 and max_map50_drop")
        return self


def derive_parity_verdict(delta_map50: float | None, max_drop: float) -> Verdict:
    """FAIL when the candidate loses more than ``max_drop`` absolute mAP@0.5."""
    if delta_map50 is None:
        return "FAIL"  # nothing comparable: never certify parity on no evidence
    return "PASS" if delta_map50 >= -max_drop - FLOAT_TOLERANCE else "FAIL"
