"""DATASET_AUDIT report: is this YOLO dataset valid and safe to evaluate on?

What it protects against
------------------------
* Broken labels (class ids out of range, coordinates outside [0, 1]) that
  training frameworks often skip silently, shrinking the dataset without
  telling you.
* **Temporal leakage**: consecutive video frames are near-identical. If frames
  of the same clip land in train and val, validation measures memorization
  and mAP is inflated. Two independent checks:
    - perceptual-hash near duplicates across splits (content-based);
    - shared group ids (clip/camera/day parsed from the file name).
* Objects too small for the substream resolution. A person 20 px tall at
  640x360 is a different problem from one 200 px tall; the report shows the
  pixel-height distribution per class at the inference resolution.

Near-duplicate search with a guarantee (dHash + pigeonhole banding)
-------------------------------------------------------------------
Each image gets a 64-bit difference hash (dHash). Comparing all pairs is
O(n²). Instead the 64 bits are split into 8 bands of 8 bits, and only images
sharing at least one identical band are compared. If two hashes differ in
d <= 7 bits, those d differing bits fall into at most d of the 8 bands, so at
least one band is identical (pigeonhole principle). Candidate generation
therefore has **100 % recall for every threshold <= 7**, which is why the
request caps ``dhash_hamming_threshold`` at 7. This is the deterministic
cousin of MinHash-LSH banding, where the guarantee is only probabilistic.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from vva_contracts.contracts.base import (
    SCHEMA_VERSION,
    ClassName,
    InputFingerprint,
    NonNegativeFloat,
    Sha256,
    StrictModel,
    UnitInterval,
    Verdict,
    require_ordered,
)

# A class is flagged when more than this share of its boxes is smaller than
# min_box_px at the inference resolution.
SMALL_BOX_SHARE_WARN = 0.20
DHASH_BANDS = 8
MAX_ISSUE_EXAMPLES = 200

LabelIssueCode = Literal[
    "malformed_line",
    "class_out_of_range",
    "coord_out_of_range",
    "non_positive_size",
    "box_outside_image",
]

DatasetRiskFlag = Literal[
    "LABEL_ERRORS",
    "ORPHAN_LABELS",
    "CROSS_SPLIT_NEAR_DUPLICATES",
    "GROUP_LEAKAGE",
    "GROUP_LEAKAGE_UNCHECKED",
    "GROUP_REGEX_PARTIAL_MATCH",
    "NEAR_DUPLICATE_SEARCH_TRUNCATED",
    "SMALL_BOXES",
    "CLASS_MISSING_IN_SPLIT",
]

# Flags that make a dataset unusable for a trustworthy evaluation.
FAIL_FLAGS: frozenset[str] = frozenset({"LABEL_ERRORS", "CROSS_SPLIT_NEAR_DUPLICATES", "GROUP_LEAKAGE"})


class LabelIssue(StrictModel):
    file: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    line: int = Field(ge=1)
    code: LabelIssueCode
    detail: Annotated[str, StringConstraints(max_length=280)]


class ClassBoxSize(StrictModel):
    """Pixel height of boxes when the image is resized to the inference size."""

    class_name: ClassName
    n_boxes: int = Field(ge=1)
    height_px_p05: NonNegativeFloat
    height_px_p50: NonNegativeFloat
    height_px_p95: NonNegativeFloat
    share_below_min_px: UnitInterval

    @model_validator(mode="after")
    def _ordered(self) -> ClassBoxSize:
        require_ordered(
            f"{self.class_name} height quantiles", [self.height_px_p05, self.height_px_p50, self.height_px_p95]
        )
        return self


class SplitStats(StrictModel):
    name: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    manifest_sha256: Sha256  # hash over (relative path, file hash) of every image and label
    n_images: int = Field(ge=0)
    n_background_images: int = Field(ge=0)  # no label file or an empty one
    n_boxes: int = Field(ge=0)
    boxes_per_class: dict[ClassName, int]
    box_sizes: list[ClassBoxSize]
    n_exact_duplicate_images: int = Field(ge=0)  # identical dHash to another image in the SAME split

    @model_validator(mode="after")
    def _consistent(self) -> SplitStats:
        if sum(self.boxes_per_class.values()) != self.n_boxes:
            raise ValueError(f"split {self.name}: boxes_per_class does not sum to n_boxes")
        if self.n_background_images > self.n_images:
            raise ValueError(f"split {self.name}: more background images than images")
        for size in self.box_sizes:
            if self.boxes_per_class.get(size.class_name) != size.n_boxes:
                raise ValueError(f"split {self.name}: box_sizes[{size.class_name}] disagrees with boxes_per_class")
        return self


class NearDuplicateCheck(StrictModel):
    method: Literal["dhash64_banded"] = "dhash64_banded"
    n_bands: Literal[8] = 8  # must equal DHASH_BANDS
    hamming_threshold: int = Field(ge=0, le=7)
    reference_split: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    # For every other split: how many of its images have a near duplicate in the reference split.
    images_with_match_in_reference: dict[str, int]
    examples: list[list[str]] = Field(max_length=50)  # [[image_in_split, image_in_reference], ...]
    truncated: bool  # comparison budget exhausted; counts are lower bounds

    @model_validator(mode="after")
    def _guarantee(self) -> NearDuplicateCheck:
        # The pigeonhole guarantee (module docstring) needs threshold < bands.
        if self.hamming_threshold >= self.n_bands:
            raise ValueError("hamming_threshold must be < n_bands for the banding guarantee")
        return self


class GroupLeakageCheck(StrictModel):
    checked: bool
    reason_unchecked: Annotated[str, StringConstraints(max_length=280)] | None
    n_groups: int = Field(ge=0)
    n_images_unmatched: int = Field(ge=0)  # file names the regex did not match
    groups_in_multiple_splits: list[str] = Field(max_length=200)

    @model_validator(mode="after")
    def _explicit_reason(self) -> GroupLeakageCheck:
        # "Not checked" must never look like "checked, zero leaks".
        if self.checked == (self.reason_unchecked is not None):
            raise ValueError("reason_unchecked must be set if and only if checked is false")
        return self


class DatasetAuditReport(StrictModel):
    contract_id: Literal["DatasetAuditReport"] = "DatasetAuditReport"
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    tool_version: str
    inputs: list[InputFingerprint]
    class_names: list[ClassName] = Field(min_length=1)
    inference_size: list[int] = Field(min_length=2, max_length=2)
    min_box_px: int = Field(ge=2)
    splits: list[SplitStats] = Field(min_length=1)
    n_label_issues: int = Field(ge=0)
    label_issues: list[LabelIssue] = Field(max_length=MAX_ISSUE_EXAMPLES)
    n_orphan_labels: int = Field(ge=0)  # label files with no image
    near_duplicates: NearDuplicateCheck
    group_leakage: GroupLeakageCheck
    risk_flags: list[DatasetRiskFlag]
    verdict: Verdict

    @model_validator(mode="after")
    def _derived_fields(self) -> DatasetAuditReport:
        if len(self.label_issues) != min(self.n_label_issues, MAX_ISSUE_EXAMPLES):
            raise ValueError("label_issues must hold the first min(n_label_issues, 200) issues")
        expected_flags = derive_flags(self)
        if self.risk_flags != expected_flags:
            raise ValueError(f"risk_flags {self.risk_flags} do not follow from the evidence: {expected_flags}")
        if self.verdict != derive_verdict(expected_flags):
            raise ValueError("verdict does not follow from risk_flags")
        return self


def derive_flags(report: DatasetAuditReport) -> list[DatasetRiskFlag]:
    """Pure function of the evidence fields. Used by the task AND the validator."""
    flags: list[DatasetRiskFlag] = []
    if report.n_label_issues > 0:
        flags.append("LABEL_ERRORS")
    if report.n_orphan_labels > 0:
        flags.append("ORPHAN_LABELS")
    if any(n > 0 for n in report.near_duplicates.images_with_match_in_reference.values()):
        flags.append("CROSS_SPLIT_NEAR_DUPLICATES")
    if report.near_duplicates.truncated:
        flags.append("NEAR_DUPLICATE_SEARCH_TRUNCATED")
    leak = report.group_leakage
    if not leak.checked:
        flags.append("GROUP_LEAKAGE_UNCHECKED")
    else:
        if leak.groups_in_multiple_splits:
            flags.append("GROUP_LEAKAGE")
        if leak.n_images_unmatched > 0:
            flags.append("GROUP_REGEX_PARTIAL_MATCH")
    if any(s.share_below_min_px > SMALL_BOX_SHARE_WARN for split in report.splits for s in split.box_sizes):
        flags.append("SMALL_BOXES")
    if any(split.boxes_per_class.get(name, 0) == 0 for split in report.splits for name in report.class_names):
        flags.append("CLASS_MISSING_IN_SPLIT")
    return flags


def derive_verdict(flags: list[DatasetRiskFlag]) -> Verdict:
    if any(f in FAIL_FLAGS for f in flags):
        return "FAIL"
    return "WARN" if flags else "PASS"
