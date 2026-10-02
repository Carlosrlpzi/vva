"""Read YOLO (Ultralytics layout) datasets without trusting them.

Layout this module understands::

    data.yaml            path: <root>   names: [person, car, ...]
                         train: images/train   val: images/val
    <root>/images/<split>/<key>.jpg
    <root>/labels/<split>/<key>.txt   one line per box: "class cx cy w h"

Why this module exists
----------------------
Both DATASET_AUDIT (which reports problems) and DETECTION_EVAL (which must
refuse to score against broken ground truth) need the same parser. Parsing
returns *issues* instead of raising, so the audit can list every problem; the
evaluator then decides that any issue is fatal for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import yaml

from vva_contracts.errors import InvalidInput, PolicyBlocked
from vva_contracts.paths import resolve_in_workspace

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
SPLIT_KEYS = ("train", "val", "test")
# Boxes may touch the border; values like 1.0000001 come from float rounding
# in exporters and are not annotation errors.
EDGE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class LabelBox:
    """One valid YOLO box: class id and normalised (cx, cy, w, h)."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float


@dataclass(frozen=True)
class LabelIssue:
    """A problem found in a label file, with its exact location."""

    file: str
    line: int
    kind: str
    detail: str


@dataclass
class ParsedLabels:
    """Valid boxes of one label file plus every issue found in it."""

    boxes: list[LabelBox] = field(default_factory=list)
    issues: list[LabelIssue] = field(default_factory=list)


@dataclass(frozen=True)
class SplitDirs:
    """Image and label directories of one split."""

    images: Path
    labels: Path


@dataclass(frozen=True)
class DatasetSpec:
    """Parsed data.yaml: class names (index = class id) and split folders."""

    yaml_path: Path
    class_names: list[str]
    splits: dict[str, SplitDirs]


def _class_names(raw: object) -> list[str]:
    """Accept ``names`` as a list or as ``{0: "person", 1: "car"}``."""
    if isinstance(raw, list) and all(isinstance(n, str) for n in raw):
        return list(raw)
    if isinstance(raw, dict):
        keys = sorted(raw)
        # Ids must be exactly 0..K-1; a gap would shift every class after it.
        if keys != list(range(len(keys))) or not all(isinstance(v, str) for v in raw.values()):
            raise InvalidInput("data.yaml 'names' dict must map 0..K-1 to strings")
        return [str(raw[k]) for k in keys]
    raise InvalidInput("data.yaml needs 'names' as a list or a {id: name} mapping")


def _labels_dir_for(images_dir: Path) -> Path:
    """Ultralytics rule: replace the LAST 'images' path component by 'labels'."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    raise InvalidInput(f"split directory has no 'images' component: {images_dir}")


def load_dataset(workspace: Path, data_yaml: str) -> DatasetSpec:
    """Parse data.yaml and resolve every split inside the workspace."""
    yaml_path = resolve_in_workspace(workspace, data_yaml)
    with yaml_path.open("r", encoding="utf-8") as handle:
        # safe_load never instantiates arbitrary Python objects.
        doc = yaml.safe_load(handle)
    if not isinstance(doc, dict):
        raise InvalidInput("data.yaml must be a mapping")

    root_value = doc.get("path", ".")
    if not isinstance(root_value, str):
        raise InvalidInput("data.yaml 'path' must be a string")
    if PurePosixPath(root_value).is_absolute():
        # Common in Ultralytics configs, but it would let the dataset point
        # anywhere on disk. Make it relative to the yaml file instead.
        raise PolicyBlocked("data.yaml 'path' must be relative (to the yaml's folder)")
    root_rel = (yaml_path.parent / root_value).relative_to(workspace.resolve()).as_posix()

    splits: dict[str, SplitDirs] = {}
    for key in SPLIT_KEYS:
        value = doc.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise InvalidInput(f"data.yaml '{key}' must be a single directory string")
        images = resolve_in_workspace(workspace, f"{root_rel}/{value}")
        if not images.is_dir():
            raise InvalidInput(f"split '{key}' is not a directory: {value}")
        labels = _labels_dir_for(images)
        splits[key] = SplitDirs(images=images, labels=labels)
    if not splits:
        raise InvalidInput("data.yaml defines no train/val/test split")
    return DatasetSpec(yaml_path=yaml_path, class_names=_class_names(doc.get("names")), splits=splits)


def list_images(images_dir: Path) -> dict[str, Path]:
    """Map image key (relative path without suffix) -> image file, sorted."""
    found: dict[str, Path] = {}
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            key = path.relative_to(images_dir).with_suffix("").as_posix()
            if key in found:
                # img.jpg and img.png would share one label file: ambiguous.
                raise InvalidInput(f"two images share the key {key!r}")
            found[key] = path
    return found


def _parse_line(text: str, n_classes: int) -> tuple[LabelBox | None, tuple[str, str] | None]:  # noqa: PLR0911
    # One early return per rule: each rule can be read (and tested) in isolation.
    """Parse one label line into a box, or return (kind, detail) of the issue."""
    tokens = text.split()
    if len(tokens) != 5:
        # >5 tokens is usually a segmentation polygon in a detection dataset.
        return None, ("malformed_line", f"expected 5 values, got {len(tokens)}")
    try:
        class_id = int(tokens[0])
        cx, cy, w, h = (float(t) for t in tokens[1:])
    except ValueError:
        return None, ("malformed_line", "non-numeric value")
    if not 0 <= class_id < n_classes:
        return None, ("class_out_of_range", f"class {class_id} not in 0..{n_classes - 1}")
    if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
        return None, ("coord_out_of_range", "values must be normalised to [0, 1]")
    if w <= 0.0 or h <= 0.0:
        return None, ("zero_area", "width and height must be > 0")
    if (
        cx - w / 2 < -EDGE_TOLERANCE
        or cx + w / 2 > 1 + EDGE_TOLERANCE
        or (cy - h / 2 < -EDGE_TOLERANCE or cy + h / 2 > 1 + EDGE_TOLERANCE)
    ):
        return None, ("box_exceeds_image", "box extends beyond the image border")
    return LabelBox(class_id, cx, cy, w, h), None


def parse_label_file(path: Path, n_classes: int, display_name: str) -> ParsedLabels:
    """Parse a label file, collecting valid boxes and every issue."""
    result = ParsedLabels()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            box, issue = _parse_line(raw, n_classes)
            if box is not None:
                result.boxes.append(box)
            elif issue is not None:
                result.issues.append(LabelIssue(display_name, line_no, issue[0], issue[1]))
    return result
