"""Reader for the YOLO (Ultralytics) dataset layout.

Expected layout (the Ultralytics default)::

    <root>/data.yaml                 # only the ``names`` key is read
    <root>/images/<split>/**/<stem>.<jpg|jpeg|png|bmp|webp>
    <root>/labels/<split>/**/<stem>.txt

Each label line is ``class_id cx cy w h`` with coordinates normalized to
[0, 1] by image width/height. A missing or empty label file means "background
image" (no objects), which is legal and useful against false positives.

Parsing is strict on purpose. Training frameworks tend to *skip* bad lines
with a warning nobody reads; here every bad line becomes a reported issue.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from vva_contracts.contracts.dataset_audit import LabelIssue
from vva_contracts.errors import InputError
from vva_contracts.geometry import FloatArray
from vva_contracts.workspace import display_path, ensure_inside

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
# Centers/edges may exceed [0, 1] by this much because labeling tools round.
COORD_TOLERANCE = 1e-3


@dataclass(frozen=True)
class SplitItem:
    image_id: str  # path under images/<split>/ without suffix, POSIX separators
    image_path: Path
    label_path: Path | None  # None when no label file exists (background)


@dataclass
class SplitListing:
    items: list[SplitItem]
    orphan_labels: list[Path] = field(default_factory=list)  # labels without an image


@dataclass(frozen=True)
class ParsedLabels:
    classes: np.ndarray  # shape (n,), int64 class ids
    boxes_cxcywh: FloatArray  # shape (n, 4), normalized
    issues: list[LabelIssue]


def load_class_names(yaml_path: Path, *, shown_as: str) -> list[str]:
    """Read ``names`` from data.yaml (list form or ``{id: name}`` mapping)."""
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))  # safe_load: no arbitrary objects
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise InputError(f"{shown_as}: not valid YAML") from exc
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, dict):
        # Mapping form must have ids 0..K-1 with no holes, otherwise class ids
        # in label files cannot be mapped unambiguously.
        if sorted(names) != list(range(len(names))):
            raise InputError(f"{shown_as}: names mapping must use ids 0..K-1")
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
        raise InputError(f"{shown_as}: 'names' must be a non-empty list of strings")
    if len(set(names)) != len(names):
        raise InputError(f"{shown_as}: class names must be unique")
    return names


def list_split(workspace: Path, root: Path, split: str, *, max_images: int) -> SplitListing:
    """Pair every image with its label file; collect labels without images."""
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    if not images_dir.is_dir():
        raise InputError(f"missing directory images/{split} under the dataset root")

    items: list[SplitItem] = []
    seen_ids: set[str] = set()
    # os.walk with followlinks=False: a symlinked directory is not traversed;
    # symlinked files are checked individually by ensure_inside below.
    for dirpath, dirnames, filenames in os.walk(images_dir, followlinks=False):
        dirnames.sort()  # deterministic traversal order
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            ensure_inside(workspace, path, shown_as=display_path(workspace, path.parent) + "/" + name)
            image_id = path.relative_to(images_dir).with_suffix("").as_posix()
            if image_id in seen_ids:
                # cam1/x.jpg and cam1/x.png would share label cam1/x.txt.
                raise InputError(f"images/{split}: two images share the id {image_id!r}")
            seen_ids.add(image_id)
            label = labels_dir / (image_id + ".txt")
            if label.is_file():
                ensure_inside(workspace, label)
            items.append(SplitItem(image_id, path, label if label.is_file() else None))
            if len(items) > max_images:
                raise InputError(f"split {split!r} has more than max_images={max_images} images")

    orphans: list[Path] = []
    if labels_dir.is_dir():
        for dirpath, dirnames, filenames in os.walk(labels_dir, followlinks=False):
            dirnames.sort()
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if path.suffix == ".txt" and path.relative_to(labels_dir).with_suffix("").as_posix() not in seen_ids:
                    orphans.append(path)
    return SplitListing(items=items, orphan_labels=orphans)


def parse_label_file(path: Path, n_classes: int, *, shown_as: str) -> ParsedLabels:
    """Parse one label file; invalid lines become issues and are excluded."""
    classes: list[int] = []
    boxes: list[list[float]] = []
    issues: list[LabelIssue] = []
    tol = COORD_TOLERANCE

    def issue(line: int, code: str, detail: str) -> None:
        issues.append(LabelIssue(file=shown_as, line=line, code=code, detail=detail))  # type: ignore[arg-type]

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw.split()
        if not parts:
            continue
        # Exactly 5 fields: segmentation polygons (more fields) are not boxes.
        if len(parts) != 5:
            issue(lineno, "malformed_line", f"expected 5 fields, got {len(parts)}")
            continue
        try:
            cls_float = float(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            issue(lineno, "malformed_line", "non-numeric field")
            continue
        if not cls_float.is_integer() or not 0 <= cls_float < n_classes:
            issue(lineno, "class_out_of_range", f"class {parts[0]} not in [0, {n_classes - 1}]")
            continue
        if w <= 0 or h <= 0:
            issue(lineno, "non_positive_size", f"w={w}, h={h}")
            continue
        if not all(-tol <= v <= 1 + tol for v in (cx, cy, w, h)):
            issue(lineno, "coord_out_of_range", "coordinates must be normalized to [0, 1]")
            continue
        # The center can be valid while the box still sticks out of the image.
        if cx - w / 2 < -tol or cy - h / 2 < -tol or cx + w / 2 > 1 + tol or cy + h / 2 > 1 + tol:
            issue(lineno, "box_outside_image", "box edges exceed the image")
            continue
        classes.append(int(cls_float))
        boxes.append([cx, cy, w, h])

    return ParsedLabels(
        classes=np.asarray(classes, dtype=np.int64),
        boxes_cxcywh=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        issues=issues,
    )


def image_size(path: Path) -> tuple[int, int]:
    """``(width, height)`` read from the image header only (no full decode)."""
    try:
        with Image.open(path) as img:
            return img.size
    except OSError as exc:
        raise InputError(f"cannot read image header: {path.name}") from exc
