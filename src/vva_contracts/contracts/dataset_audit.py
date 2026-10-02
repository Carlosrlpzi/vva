"""DATASET_AUDIT report contract.

Why this contract exists
------------------------
A detector is only as good as its labels and its train/val split. This report
answers, with counts that can be re-derived: are the labels valid, is any
class missing, are objects too small *at inference resolution*, and does
validation leak training images (near-duplicate frames or shared recordings)?

Box size classes follow COCO's area bins, measured in letterboxed inference
pixels: small < 32^2, medium < 96^2, large otherwise. A person that is
"large" in the 2560x1440 main stream can be "small" in a 640x640 input.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from vva_contracts.contracts.base import (
    SCHEMA_VERSION,
    NonNegFloat,
    NonNegInt,
    Ratio,
    ReportMeta,
    StrictModel,
    close,
    require,
)
from vva_contracts.errors import ContractViolation

IssueKind = Literal["malformed_line", "class_out_of_range", "coord_out_of_range", "zero_area", "box_exceeds_image"]
DatasetFlag = Literal[
    "label_errors",
    "cross_split_near_duplicates",
    "group_leakage",
    "orphan_labels",
    "class_missing_in_split",
    "severe_class_imbalance",
    "many_small_objects",
    "unmatched_group_pattern",
]
Verdict = Literal["PASS", "WARN", "FAIL"]

# Flags that make the dataset unusable for a trustworthy evaluation.
FAIL_FLAGS: frozenset[str] = frozenset({"label_errors", "cross_split_near_duplicates", "group_leakage"})
IMBALANCE_RATIO_LIMIT = 10.0
SMALL_OBJECT_FRACTION_LIMIT = 0.5


class IssueExample(StrictModel):
    """One concrete label problem, so a human can open the file and look."""

    file: str
    line: NonNegInt
    kind: IssueKind
    detail: str


class BoxSizeSummary(StrictModel):
    """Distribution of box sizes in letterboxed inference pixels."""

    n_boxes: NonNegInt
    frac_small: Ratio
    frac_medium: Ratio
    frac_large: Ratio
    short_side_px_p05: NonNegFloat | None
    short_side_px_p50: NonNegFloat | None
    short_side_px_p95: NonNegFloat | None

    @model_validator(mode="after")
    def _consistent(self) -> BoxSizeSummary:
        """Fractions partition the boxes; percentiles exist iff boxes exist."""
        quantiles = (self.short_side_px_p05, self.short_side_px_p50, self.short_side_px_p95)
        if self.n_boxes == 0:
            require(all(q is None for q in quantiles), "no boxes -> size percentiles must be null")
            require(self.frac_small == self.frac_medium == self.frac_large == 0.0, "no boxes -> fractions 0")
            return self
        require(close(self.frac_small + self.frac_medium + self.frac_large, 1.0), "size fractions must sum to 1")
        p05, p50, p95 = quantiles
        if p05 is None or p50 is None or p95 is None:
            raise ContractViolation("percentiles required when boxes exist")
        require(p05 <= p50 <= p95, "percentiles must be ordered p05 <= p50 <= p95")
        return self


class SplitAudit(StrictModel):
    """Counts for one split. Only *valid* boxes are counted in class_counts."""

    split: str
    n_images: NonNegInt
    n_label_files: NonNegInt
    n_background_images: NonNegInt
    n_orphan_labels: NonNegInt
    n_boxes: NonNegInt
    class_counts: dict[str, NonNegInt]
    issue_counts: dict[IssueKind, NonNegInt]
    box_sizes: BoxSizeSummary

    @model_validator(mode="after")
    def _consistent(self) -> SplitAudit:
        """Class counts add up to n_boxes; sizes describe the same boxes."""
        require(sum(self.class_counts.values()) == self.n_boxes, f"{self.split}: class counts != n_boxes")
        require(self.box_sizes.n_boxes == self.n_boxes, f"{self.split}: box_sizes.n_boxes != n_boxes")
        require(self.n_background_images <= self.n_images, f"{self.split}: more background images than images")
        return self


class NearDuplicatePair(StrictModel):
    """Two visually near-identical images found in different splits."""

    split_a: str
    image_a: str
    split_b: str
    image_b: str
    hamming: NonNegInt


class LeakageSummary(StrictModel):
    """Evidence of train/validation contamination."""

    near_duplicate_pairs: NonNegInt
    near_duplicate_examples: list[NearDuplicatePair]
    group_pattern: str | None
    n_groups_spanning_splits: NonNegInt | None
    group_examples: list[str]
    n_images_unmatched_pattern: NonNegInt | None

    @model_validator(mode="after")
    def _consistent(self) -> LeakageSummary:
        """Group statistics exist exactly when a group pattern was given."""
        has_pattern = self.group_pattern is not None
        require(
            (self.n_groups_spanning_splits is not None) == has_pattern
            and (self.n_images_unmatched_pattern is not None) == has_pattern,
            "group statistics must be present iff group_pattern is set",
        )
        require(len(self.near_duplicate_examples) <= self.near_duplicate_pairs, "more examples than pairs")
        return self


def _severely_imbalanced(splits: list[SplitAudit]) -> bool:
    """Pooled majority/minority ratio above the limit (classes with 0 boxes excluded).

    Classes with zero boxes are handled by ``class_missing_in_split``; including
    them here would divide by zero.
    """
    pooled: dict[str, int] = {}
    for s in splits:
        for name, count in s.class_counts.items():
            pooled[name] = pooled.get(name, 0) + count
    present = [c for c in pooled.values() if c > 0]
    return len(present) >= 2 and max(present) / min(present) > IMBALANCE_RATIO_LIMIT


def derive_flags(splits: list[SplitAudit], leakage: LeakageSummary) -> list[DatasetFlag]:
    """Compute flags from counts. Used by the task AND by the validator."""
    flags: list[DatasetFlag] = []
    if any(sum(s.issue_counts.values()) > 0 for s in splits):
        flags.append("label_errors")
    if leakage.near_duplicate_pairs > 0:
        flags.append("cross_split_near_duplicates")
    if leakage.n_groups_spanning_splits:
        flags.append("group_leakage")
    if any(s.n_orphan_labels > 0 for s in splits):
        flags.append("orphan_labels")
    # A split with images but zero boxes of some class cannot train or
    # evaluate that class (this also catches a split with no labels at all).
    if any(s.n_images > 0 and min(s.class_counts.values()) == 0 for s in splits):
        flags.append("class_missing_in_split")
    if _severely_imbalanced(splits):
        flags.append("severe_class_imbalance")
    if any(s.box_sizes.frac_small > SMALL_OBJECT_FRACTION_LIMIT for s in splits):
        flags.append("many_small_objects")
    if leakage.n_images_unmatched_pattern:
        flags.append("unmatched_group_pattern")
    return flags


def derive_verdict(flags: list[DatasetFlag]) -> Verdict:
    """FAIL on any blocking flag, WARN on any other flag, else PASS."""
    if any(f in FAIL_FLAGS for f in flags):
        return "FAIL"
    return "WARN" if flags else "PASS"


class DatasetAuditReport(StrictModel):
    """Accepted DATASET_AUDIT artifact."""

    contract_id: Literal["DatasetAuditReport"] = "DatasetAuditReport"
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    meta: ReportMeta
    class_names: list[str] = Field(min_length=1)
    inference_size: tuple[int, int]
    near_duplicate_max_hamming: NonNegInt
    splits: list[SplitAudit] = Field(min_length=1)
    leakage: LeakageSummary
    issue_examples: list[IssueExample]
    flags: list[DatasetFlag]
    verdict: Verdict

    @model_validator(mode="after")
    def _evidence_derived(self) -> DatasetAuditReport:
        """Flags and verdict must be exactly what the counts imply."""
        for s in self.splits:
            require(list(s.class_counts) == self.class_names, f"{s.split}: class_counts keys != class_names")
        total_issues = sum(sum(s.issue_counts.values()) for s in self.splits)
        require(len(self.issue_examples) <= total_issues, "more issue examples than issues")
        require(self.flags == derive_flags(self.splits, self.leakage), "flags do not match the evidence")
        require(self.verdict == derive_verdict(self.flags), "verdict does not match the flags")
        return self
