"""DATASET_AUDIT task. See ``contracts/dataset_audit.py`` for the rationale.

Flow: list each split -> parse labels -> box-size statistics at the
inference resolution -> dHash every image -> near-duplicate search against
the reference split -> optional group leakage via ``group_regex`` -> report.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from vva_contracts.contracts.dataset_audit import (
    DHASH_BANDS,
    MAX_ISSUE_EXAMPLES,
    ClassBoxSize,
    DatasetAuditReport,
    GroupLeakageCheck,
    LabelIssue,
    NearDuplicateCheck,
    SplitStats,
    derive_flags,
    derive_verdict,
)
from vva_contracts.contracts.requests import DatasetAuditRequest
from vva_contracts.errors import InputError
from vva_contracts.io import fingerprint, manifest_digest
from vva_contracts.tasks._common import quantile, resolve_data_yaml, tool_version
from vva_contracts.workspace import display_path, resolve_in_workspace
from vva_contracts.yolo import list_split, load_class_names, parse_label_file

# Upper bound on Hamming-distance comparisons. A static camera can produce
# thousands of frames sharing every band; past this budget the search stops
# and the report says so (truncated=True) instead of running for hours.
MAX_HASH_COMPARISONS = 5_000_000
BAND_BITS = 64 // DHASH_BANDS
BAND_MASK = (1 << BAND_BITS) - 1


def dhash64(path: Path) -> int:
    """64-bit difference hash.

    Downscale to 9x8 grayscale, then for each row compare each pixel with its
    right neighbour: bit = 1 if brightness increases. Encoding *gradients*
    instead of absolute values makes the hash robust to global brightness
    changes and recompression, which is what consecutive video frames differ by.
    """
    with Image.open(path) as img:
        small = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(small, dtype=np.int16)  # shape (8, 9)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()  # 8 rows x 8 comparisons = 64 bits
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _bands(value: int) -> list[int]:
    """Split a 64-bit hash into 8 bands of 8 bits (band index kept implicit by position)."""
    return [(value >> (i * BAND_BITS)) & BAND_MASK for i in range(DHASH_BANDS)]


def run(request: DatasetAuditRequest, workspace: Path) -> DatasetAuditReport:
    root = resolve_in_workspace(workspace, request.dataset_root, kind="dir")
    yaml_path = resolve_data_yaml(workspace, root, request.data_yaml)
    class_names = load_class_names(yaml_path, shown_as=display_path(workspace, yaml_path))
    inference_h = request.inference_size[1]

    inputs = [fingerprint(workspace, "data_yaml", yaml_path)]
    split_stats: list[SplitStats] = []
    issues: list[LabelIssue] = []
    n_issues = 0
    n_orphans = 0
    hashes: dict[str, dict[str, int]] = {}  # split -> image_id -> dhash

    for split in request.splits:
        listing = list_split(workspace, root, split, max_images=request.max_images)
        n_orphans += len(listing.orphan_labels)
        files = [item.image_path for item in listing.items] + [
            item.label_path for item in listing.items if item.label_path is not None
        ]
        digest, _ = manifest_digest(workspace, files)

        heights_by_class: dict[int, list[float]] = defaultdict(list)
        n_background = 0
        for item in listing.items:
            if item.label_path is None:
                n_background += 1
                continue
            parsed = parse_label_file(
                item.label_path, len(class_names), shown_as=display_path(workspace, item.label_path)
            )
            n_issues += len(parsed.issues)
            issues.extend(parsed.issues[: max(0, MAX_ISSUE_EXAMPLES - len(issues))])
            if len(parsed.classes) == 0:
                n_background += 1
            # Normalized height x substream height = pixels the detector will see.
            for cls, h_norm in zip(parsed.classes.tolist(), parsed.boxes_cxcywh[:, 3].tolist(), strict=True):
                heights_by_class[cls].append(h_norm * inference_h)

        boxes_per_class = {name: len(heights_by_class.get(i, [])) for i, name in enumerate(class_names)}
        box_sizes = [
            ClassBoxSize(
                class_name=class_names[cls],
                n_boxes=len(heights),
                height_px_p05=quantile(heights, 0.05),
                height_px_p50=quantile(heights, 0.50),
                height_px_p95=quantile(heights, 0.95),
                share_below_min_px=sum(h < request.min_box_px for h in heights) / len(heights),
            )
            for cls, heights in sorted(heights_by_class.items())
        ]

        split_hashes = {item.image_id: dhash64(item.image_path) for item in listing.items}
        hashes[split] = split_hashes
        # Exact duplicates inside one split: any hash value shared by >1 image.
        counts: dict[int, int] = defaultdict(int)
        for value in split_hashes.values():
            counts[value] += 1
        n_exact_dupes = sum(c for c in counts.values() if c > 1)

        split_stats.append(
            SplitStats(
                name=split,
                manifest_sha256=digest,
                n_images=len(listing.items),
                n_background_images=n_background,
                n_boxes=sum(boxes_per_class.values()),
                boxes_per_class=boxes_per_class,
                box_sizes=box_sizes,
                n_exact_duplicate_images=n_exact_dupes,
            )
        )

    near_dupes = _near_duplicates(hashes, request.splits, request.dhash_hamming_threshold)
    leakage = _group_leakage(hashes, request.group_regex)

    # Build once without flags, derive flags from the evidence with the SAME
    # function the validator uses, then validate the final document.
    draft = DatasetAuditReport.model_construct(
        tool_version=tool_version(),
        inputs=inputs,
        class_names=class_names,
        inference_size=list(request.inference_size),
        min_box_px=request.min_box_px,
        splits=split_stats,
        n_label_issues=n_issues,
        label_issues=issues,
        n_orphan_labels=n_orphans,
        near_duplicates=near_dupes,
        group_leakage=leakage,
        # Placeholders: model_construct skips validation; both are replaced below.
        risk_flags=[],
        verdict="PASS",
    )
    flags = derive_flags(draft)
    return DatasetAuditReport.model_validate({**draft.__dict__, "risk_flags": flags, "verdict": derive_verdict(flags)})


def _near_duplicates(hashes: dict[str, dict[str, int]], splits: list[str], threshold: int) -> NearDuplicateCheck:
    """For each non-reference split, count images with a near duplicate in the reference split.

    The reference is ``train`` when present (leakage means "val looks like
    train"), otherwise the first requested split.
    """
    reference = "train" if "train" in hashes else splits[0]
    ref_items = list(hashes[reference].items())

    # Index: (band position, band value) -> reference image indices.
    index: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (_, value) in enumerate(ref_items):
        for pos, band in enumerate(_bands(value)):
            index[(pos, band)].append(idx)

    comparisons = 0
    truncated = False
    matches: dict[str, int] = {}
    examples: list[list[str]] = []
    for split in splits:
        if split == reference:
            continue
        found = 0
        for image_id, value in hashes[split].items():
            hit = _first_match(value, index, ref_items, threshold)
            comparisons += hit[1]
            if hit[0] is not None:
                found += 1
                if len(examples) < 50:
                    examples.append([f"{split}/{image_id}", f"{reference}/{hit[0]}"])
            if comparisons > MAX_HASH_COMPARISONS:
                truncated = True
                break
        matches[split] = found
        if truncated:
            break
    return NearDuplicateCheck(
        hamming_threshold=threshold,
        reference_split=reference,
        images_with_match_in_reference=matches,
        examples=examples,
        truncated=truncated,
    )


def _first_match(
    value: int, index: dict[tuple[int, int], list[int]], ref_items: list[tuple[str, int]], threshold: int
) -> tuple[str | None, int]:
    """Return (matching reference image_id or None, comparisons spent)."""
    seen: set[int] = set()
    spent = 0
    for pos, band in enumerate(_bands(value)):
        for idx in index.get((pos, band), []):
            if idx in seen:
                continue
            seen.add(idx)
            spent += 1
            # Hamming distance = popcount of XOR.
            if (value ^ ref_items[idx][1]).bit_count() <= threshold:
                return ref_items[idx][0], spent
    return None, spent


def _group_leakage(hashes: dict[str, dict[str, int]], group_regex: str | None) -> GroupLeakageCheck:
    """Groups (clip/camera/day) that appear in more than one split."""
    if group_regex is None:
        return GroupLeakageCheck(
            checked=False,
            reason_unchecked="no group_regex supplied; frames of one clip may be split across train/val",
            n_groups=0,
            n_images_unmatched=0,
            groups_in_multiple_splits=[],
        )
    pattern = re.compile(group_regex)
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    unmatched = 0
    for split, items in hashes.items():
        for image_id in items:
            match = pattern.search(image_id)
            if match is None or not match.group("group"):
                unmatched += 1
                continue
            splits_by_group[match.group("group")].add(split)
    if not splits_by_group:
        raise InputError("group_regex matched no image id; check the pattern against the file names")
    shared = sorted(g for g, s in splits_by_group.items() if len(s) > 1)
    return GroupLeakageCheck(
        checked=True,
        reason_unchecked=None,
        n_groups=len(splits_by_group),
        n_images_unmatched=unmatched,
        groups_in_multiple_splits=shared[:200],
    )
