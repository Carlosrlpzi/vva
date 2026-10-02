"""DETECTION_EVAL and QUANT_PARITY report contracts.

Why these contracts exist
-------------------------
DETECTION_EVAL turns "the detector seems fine" into per-class numbers on
*your* footage: AP at IoU 0.50, AP averaged over IoU 0.50:0.95 (COCO), and
precision/recall at the confidence the alert rule actually uses.

QUANT_PARITY answers the question created by the Hailo toolchain: compiling
a model to an INT8 HEF quantises its weights and activations. How much did
that cost? The same predictions are scored before (reference, e.g. FP32 ONNX
on the laptop) and after (candidate, HEF on the Pi), and the drop must stay
inside an explicit tolerance.

Invariants enforced (recomputed, not trusted)
---------------------------------------------
* fn = n_gt - tp; precision = tp/(tp+fp); recall = tp/n_gt; F1 = 2PR/(P+R)
* ap50 is the first entry of ap_per_iou; ap50_95 is its mean
* mAP is the mean of per-class AP over classes that have ground truth
* parity deltas are candidate - reference; the verdict follows the deltas
"""

from __future__ import annotations

from statistics import fmean
from typing import Literal

from pydantic import Field, model_validator

from vva_contracts.contracts.base import (
    SCHEMA_VERSION,
    Identifier,
    NonNegInt,
    Ratio,
    ReportMeta,
    StrictModel,
    close,
    require,
)
from vva_contracts.core.stats import safe_ratio
from vva_contracts.errors import ContractViolation

N_IOU_THRESHOLDS = 10
ParityCheck = Literal["map50_drop", "map50_95_drop", "no_ground_truth"]


class ClassMetrics(StrictModel):
    """Measured metrics for one class."""

    class_id: NonNegInt
    class_name: str
    n_gt: NonNegInt
    n_pred: NonNegInt
    ap_per_iou: list[Ratio] | None
    ap50: Ratio | None
    ap50_95: Ratio | None
    tp: NonNegInt
    fp: NonNegInt
    fn: NonNegInt
    precision: Ratio
    recall: Ratio
    f1: Ratio

    @model_validator(mode="after")
    def _recompute(self) -> ClassMetrics:
        """Every derived number must follow from the counts."""
        tag = f"class {self.class_name!r}"
        if self.n_gt == 0:
            # COCO reports -1 for classes absent from the ground truth; we use
            # null so a consumer cannot average it in by mistake.
            require(self.ap_per_iou is None and self.ap50 is None and self.ap50_95 is None, f"{tag}: AP must be null")
        else:
            aps = self.ap_per_iou
            # Explicit raise instead of assert: asserts vanish under python -O.
            if aps is None or len(aps) != N_IOU_THRESHOLDS:
                raise ContractViolation(f"{tag}: need {N_IOU_THRESHOLDS} AP values")
            require(self.ap50 is not None and close(self.ap50, aps[0]), f"{tag}: ap50 != ap_per_iou[0]")
            require(self.ap50_95 is not None and close(self.ap50_95, fmean(aps)), f"{tag}: ap50_95 != mean")
        require(self.tp <= self.n_gt, f"{tag}: tp > n_gt (a ground truth matched twice)")
        require(self.tp + self.fp <= self.n_pred, f"{tag}: tp + fp > n_pred")
        require(self.fn == self.n_gt - self.tp, f"{tag}: fn != n_gt - tp")
        require(close(self.precision, safe_ratio(self.tp, self.tp + self.fp)), f"{tag}: precision mismatch")
        require(close(self.recall, safe_ratio(self.tp, self.n_gt)), f"{tag}: recall mismatch")
        expected_f1 = safe_ratio(2 * self.precision * self.recall, self.precision + self.recall)
        require(close(self.f1, expected_f1), f"{tag}: f1 mismatch")
        return self


def mean_ap(classes: list[ClassMetrics], attr: Literal["ap50", "ap50_95"]) -> float | None:
    """Mean of per-class AP over classes with ground truth (None if there are none)."""
    values = [getattr(c, attr) for c in classes if c.n_gt > 0]
    return fmean(values) if values else None


class DetectionMetrics(StrictModel):
    """All metrics of one predictions file against one split."""

    operating_confidence: Ratio
    max_dets_per_image: int = Field(ge=1)
    iou_thresholds: list[float]
    n_images: NonNegInt
    n_prediction_rows: NonNegInt
    classes: list[ClassMetrics] = Field(min_length=1)
    map50: Ratio | None
    map50_95: Ratio | None

    @model_validator(mode="after")
    def _recompute(self) -> DetectionMetrics:
        """MAP values are recomputed from the per-class entries."""
        require(len(self.iou_thresholds) == N_IOU_THRESHOLDS, "iou_thresholds must list 10 values")
        require([c.class_id for c in self.classes] == list(range(len(self.classes))), "class ids must be 0..K-1")
        for attr, reported in (("ap50", self.map50), ("ap50_95", self.map50_95)):
            expected = mean_ap(self.classes, attr)  # type: ignore[arg-type]
            ok = reported is None if expected is None else reported is not None and close(reported, expected)
            require(ok, f"m{attr} does not equal the mean of per-class {attr}")
        # Truncation to max_dets can only remove rows, never add them.
        require(sum(c.n_pred for c in self.classes) <= self.n_prediction_rows, "more scored predictions than rows")
        return self


class DetectionEvalReport(StrictModel):
    """Accepted DETECTION_EVAL artifact."""

    contract_id: Literal["DetectionEvalReport"] = "DetectionEvalReport"
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    meta: ReportMeta
    split: Identifier
    box_format: Literal["xyxyn", "xywhn"]
    metrics: DetectionMetrics


class AgreementStats(StrictModel):
    """Box-level agreement between reference and candidate (above operating conf)."""

    n_reference: NonNegInt
    n_candidate: NonNegInt
    n_matched: NonNegInt
    mean_iou: Ratio | None
    mean_abs_conf_diff: Ratio | None
    p95_abs_conf_diff: Ratio | None

    @model_validator(mode="after")
    def _consistent(self) -> AgreementStats:
        """Matched pairs cannot exceed either side; stats exist iff pairs exist."""
        require(self.n_matched <= min(self.n_reference, self.n_candidate), "n_matched exceeds a side")
        stats = (self.mean_iou, self.mean_abs_conf_diff, self.p95_abs_conf_diff)
        require(all(s is None for s in stats) == (self.n_matched == 0), "agreement stats null iff no matches")
        return self


def derive_parity_checks(
    reference: DetectionMetrics, candidate: DetectionMetrics, max_map50_drop: float, max_map50_95_drop: float
) -> list[ParityCheck]:
    """Which parity checks fail. Shared by the task and the validator."""
    r50, c50, r95, c95 = reference.map50, candidate.map50, reference.map50_95, candidate.map50_95
    if r50 is None or c50 is None or r95 is None or c95 is None:
        return ["no_ground_truth"]
    failed: list[ParityCheck] = []
    # A *drop* is reference - candidate; a gain never fails the check.
    if r50 - c50 > max_map50_drop:
        failed.append("map50_drop")
    if r95 - c95 > max_map50_95_drop:
        failed.append("map50_95_drop")
    return failed


class QuantParityReport(StrictModel):
    """Accepted QUANT_PARITY artifact."""

    contract_id: Literal["QuantParityReport"] = "QuantParityReport"
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    meta: ReportMeta
    split: Identifier
    box_format: Literal["xyxyn", "xywhn"]
    reference: DetectionMetrics
    candidate: DetectionMetrics
    delta_map50: float | None
    delta_map50_95: float | None
    max_map50_drop: Ratio
    max_map50_95_drop: Ratio
    agreement_iou: Ratio
    agreement: AgreementStats
    failed_checks: list[ParityCheck]
    verdict: Literal["PASS", "FAIL"]

    @model_validator(mode="after")
    def _evidence_derived(self) -> QuantParityReport:
        """Deltas, failed checks and verdict are recomputed from the metrics."""
        ref, cand = self.reference, self.candidate
        require(ref.operating_confidence == cand.operating_confidence, "both sides need the same operating point")
        require(ref.n_images == cand.n_images, "both sides must be scored on the same images")
        for name, r, c, d in (
            ("delta_map50", ref.map50, cand.map50, self.delta_map50),
            ("delta_map50_95", ref.map50_95, cand.map50_95, self.delta_map50_95),
        ):
            if r is None or c is None:
                require(d is None, f"{name} must be null without ground truth")
            else:
                require(d is not None and close(d, c - r), f"{name} != candidate - reference")
        expected = derive_parity_checks(ref, cand, self.max_map50_drop, self.max_map50_95_drop)
        require(self.failed_checks == expected, "failed_checks do not match the metrics")
        require(self.verdict == ("FAIL" if expected else "PASS"), "verdict does not match failed_checks")
        return self
