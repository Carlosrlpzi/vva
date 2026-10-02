"""DATASET_AUDIT runner.

Flow per split: list images -> match label files -> parse labels (collect
issues, never raise) -> measure box sizes at inference resolution -> hash
images. Then, across splits: near-duplicate search and recording-group
leakage. Flags and verdict come from ``derive_flags``/``derive_verdict``, the
same functions the contract validator calls.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from pathlib import Path

from PIL import Image

from vva_contracts.contracts.dataset_audit import (
    BoxSizeSummary,
    DatasetAuditReport,
    IssueExample,
    IssueKind,
    LeakageSummary,
    NearDuplicatePair,
    SplitAudit,
    derive_flags,
    derive_verdict,
)
from vva_contracts.contracts.requests import DatasetAuditRequest
from vva_contracts.core.phash import cross_near_duplicates, dhash
from vva_contracts.core.stats import percentile
from vva_contracts.readers.yolo import (
    LabelBox,
    LabelIssue,
    SplitDirs,
    list_images,
    load_dataset,
    parse_label_file,
)
from vva_contracts.tasks.common import build_meta

ISSUE_KINDS: tuple[IssueKind, ...] = (
    "malformed_line",
    "class_out_of_range",
    "coord_out_of_range",
    "zero_area",
    "box_exceeds_image",
)
# COCO area thresholds in pixels^2 (small < 32^2 <= medium < 96^2 <= large).
SMALL_AREA, MEDIUM_AREA = 32.0**2, 96.0**2


def letterboxed_size(box: LabelBox, image_size: tuple[int, int], target: tuple[int, int]) -> tuple[float, float]:
    """Box (width, height) in pixels after letterboxing into ``target``.

    Letterboxing scales the image by s = min(W_t / W, H_t / H) so it fits
    without distortion, then pads. The box scales by the same s.
    """
    img_w, img_h = image_size
    scale = min(target[0] / img_w, target[1] / img_h)
    return box.w * img_w * scale, box.h * img_h * scale


def summarise_sizes(sizes: list[tuple[float, float]]) -> BoxSizeSummary:
    """COCO area bins and short-side percentiles of a list of (w, h) in px."""
    if not sizes:
        return BoxSizeSummary(
            n_boxes=0,
            frac_small=0.0,
            frac_medium=0.0,
            frac_large=0.0,
            short_side_px_p05=None,
            short_side_px_p50=None,
            short_side_px_p95=None,
        )
    areas = [w * h for w, h in sizes]
    n = len(areas)
    n_small = sum(a < SMALL_AREA for a in areas)
    n_medium = sum(SMALL_AREA <= a < MEDIUM_AREA for a in areas)
    short = [min(w, h) for w, h in sizes]
    return BoxSizeSummary(
        n_boxes=n,
        frac_small=n_small / n,
        frac_medium=n_medium / n,
        # Computed as the remainder of the integer count, so the three
        # fractions sum to exactly 1 up to float rounding.
        frac_large=(n - n_small - n_medium) / n,
        short_side_px_p05=percentile(short, 5),
        short_side_px_p50=percentile(short, 50),
        short_side_px_p95=percentile(short, 95),
    )


def audit_split(
    name: str, dirs: SplitDirs, class_names: list[str], target: tuple[int, int]
) -> tuple[SplitAudit, list[LabelIssue], dict[str, Path]]:
    """Audit one split; also return its issues and image map for later steps."""
    images = list_images(dirs.images)
    label_files = (
        {p.relative_to(dirs.labels).with_suffix("").as_posix(): p for p in dirs.labels.rglob("*.txt")}
        if dirs.labels.is_dir()
        else {}
    )
    class_counts: Counter[int] = Counter()
    issue_counts: Counter[str] = Counter()
    issues: list[LabelIssue] = []
    sizes: list[tuple[float, float]] = []
    n_background = 0
    for key, image_path in images.items():
        label_path = label_files.get(key)
        if label_path is None:
            n_background += 1  # an image without objects is valid "background"
            continue
        parsed = parse_label_file(label_path, len(class_names), f"{name}/{key}.txt")
        issues.extend(parsed.issues)
        issue_counts.update(i.kind for i in parsed.issues)
        if not parsed.boxes and not parsed.issues:
            n_background += 1
        with Image.open(image_path) as img:
            image_size = img.size  # reads the header only, not the pixels
        for box in parsed.boxes:
            class_counts[box.class_id] += 1
            sizes.append(letterboxed_size(box, image_size, target))
    split = SplitAudit(
        split=name,
        n_images=len(images),
        n_label_files=sum(1 for k in label_files if k in images),
        n_background_images=n_background,
        n_orphan_labels=sum(1 for k in label_files if k not in images),
        n_boxes=sum(class_counts.values()),
        class_counts={cname: class_counts[i] for i, cname in enumerate(class_names)},
        issue_counts={k: issue_counts[k] for k in ISSUE_KINDS},
        box_sizes=summarise_sizes(sizes),
    )
    return split, issues, images


def group_leakage(
    pattern: str | None, images_by_split: dict[str, dict[str, Path]], max_examples: int
) -> tuple[int | None, list[str], int | None]:
    """Recording groups (clip/camera/day) that appear in more than one split."""
    if pattern is None:
        return None, [], None
    regex = re.compile(pattern)
    splits_of_group: dict[str, set[str]] = {}
    unmatched = 0
    for split, images in images_by_split.items():
        for key in images:
            match = regex.search(key)
            if match is None:
                unmatched += 1  # reported: an unmatched image escapes the check
                continue
            splits_of_group.setdefault(match.group("group"), set()).add(split)
    spanning = sorted(g for g, s in splits_of_group.items() if len(s) > 1)
    return len(spanning), spanning[:max_examples], unmatched


def run_dataset_audit(req: DatasetAuditRequest, workspace: Path) -> DatasetAuditReport:
    """Entry point for task DATASET_AUDIT."""
    spec = load_dataset(workspace, req.data_yaml)
    target = (req.inference_width, req.inference_height)
    split_reports: list[SplitAudit] = []
    all_issues: list[LabelIssue] = []
    images_by_split: dict[str, dict[str, Path]] = {}
    for name, dirs in spec.splits.items():
        report, issues, images = audit_split(name, dirs, spec.class_names, target)
        split_reports.append(report)
        all_issues.extend(issues)
        images_by_split[name] = images

    hashes = {split: {key: dhash(p) for key, p in imgs.items()} for split, imgs in images_by_split.items()}
    pairs: list[NearDuplicatePair] = []
    for split_a, split_b in combinations(sorted(hashes), 2):
        for key_a, key_b, dist in cross_near_duplicates(
            hashes[split_a], hashes[split_b], req.near_duplicate_max_hamming
        ):
            pairs.append(
                NearDuplicatePair(split_a=split_a, image_a=key_a, split_b=split_b, image_b=key_b, hamming=dist)
            )

    n_spanning, group_examples, n_unmatched = group_leakage(req.group_pattern, images_by_split, req.max_examples)
    leakage = LeakageSummary(
        near_duplicate_pairs=len(pairs),
        near_duplicate_examples=pairs[: req.max_examples],
        group_pattern=req.group_pattern,
        n_groups_spanning_splits=n_spanning,
        group_examples=group_examples,
        n_images_unmatched_pattern=n_unmatched,
    )
    flags = derive_flags(split_reports, leakage)
    inputs: list[tuple[str, Path]] = [("data_yaml", spec.yaml_path)]
    for name, dirs in spec.splits.items():
        inputs.append((f"images_{name}", dirs.images))
        if dirs.labels.is_dir():
            inputs.append((f"labels_{name}", dirs.labels))
    return DatasetAuditReport(
        meta=build_meta(workspace, inputs),
        class_names=spec.class_names,
        inference_size=target,
        near_duplicate_max_hamming=req.near_duplicate_max_hamming,
        splits=split_reports,
        leakage=leakage,
        issue_examples=[
            IssueExample(file=i.file, line=i.line, kind=i.kind, detail=i.detail)  # type: ignore[arg-type]
            for i in all_issues[: req.max_examples]
        ],
        flags=flags,
        verdict=derive_verdict(flags),
    )
