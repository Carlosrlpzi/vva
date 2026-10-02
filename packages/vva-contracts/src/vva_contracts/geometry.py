"""Axis-aligned box geometry and polygon tests used by evaluation and rules.

Intersection over Union (IoU)
-----------------------------
For two boxes A and B, ``IoU = area(A and B) / area(A or B)`` with
``area(A or B) = area(A) + area(B) - area(A and B)``. It is the overlap criterion every detection
benchmark uses to decide whether a prediction "hits" a ground-truth object.

Why IoU can be computed in normalized coordinates
-------------------------------------------------
YOLO labels are normalized by image width and height, while predictions are
in pixels. Mapping pixels to normalized units is the linear map
``S = diag(1/W, 1/H)``. Any invertible linear map multiplies every area by the
same factor ``|det S| = 1/(W*H)``, and a diagonal map keeps axis-aligned boxes
axis-aligned. Numerator and denominator of IoU scale identically, so
``IoU(S A, S B) = IoU(A, B)``. Normalizing loses nothing, provided the
prediction was logged in the true image's pixel frame (checked by the task).

Point in polygon (zones)
------------------------
Even-odd ray casting: shoot a horizontal ray from the point to +inf and count
edge crossings; an odd count means inside. It works for any simple polygon,
convex or not. Points exactly on an edge may land on either side; zones
should therefore not be drawn with a boundary exactly through a doorway line.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def iou_matrix(boxes_a: FloatArray, boxes_b: FloatArray) -> FloatArray:
    """Pairwise IoU between two sets of ``xyxy`` boxes.

    Args:
        boxes_a: shape ``(n, 4)``.
        boxes_b: shape ``(m, 4)``.

    Returns:
        Array of shape ``(n, m)`` with values in [0, 1].
    """
    # Broadcast to (n, 1, 4) against (1, m, 4) so every pair is computed in one
    # vectorized pass, without a Python double loop.
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    # Width/height of the intersection rectangle; negative means no overlap,
    # so clip at zero (the same convention as pycocotools' bbIou).
    inter_w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0.0, None)
    inter_h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0.0, None)
    intersection = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - intersection

    # Union is > 0 for any valid box; the guard only avoids a 0/0 warning.
    out: FloatArray = np.zeros(intersection.shape, dtype=np.float64)
    np.divide(intersection, union, out=out, where=union > 0)
    return out


def yolo_to_xyxy(cx: FloatArray, cy: FloatArray, w: FloatArray, h: FloatArray) -> FloatArray:
    """Convert YOLO ``(cx, cy, w, h)`` columns to an ``(n, 4)`` xyxy array."""
    stacked: FloatArray = np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)
    return stacked


def polygon_area(polygon: list[list[float]]) -> float:
    """Unsigned polygon area with the shoelace formula.

    ``A = 1/2 |Σ (x_i * y_{i+1} - x_{i+1} * y_i)|``, indices taken cyclically.
    """
    xs = np.array([p[0] for p in polygon], dtype=np.float64)
    ys = np.array([p[1] for p in polygon], dtype=np.float64)
    # np.roll(..., -1) yields x_{i+1}/y_{i+1} with wrap-around to vertex 0.
    return float(0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(np.roll(xs, -1), ys)))


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    """Even-odd ray casting test (see module docstring)."""
    inside = False
    n = len(polygon)
    j = n - 1  # previous vertex index; starts at the last one to close the ring
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # The edge (j -> i) straddles the horizontal line through y exactly
        # when one endpoint is above and the other is not.
        if (yi > y) != (yj > y):
            # x coordinate where the edge crosses that horizontal line.
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def anchor_point(bbox_xyxy: list[float], width: int, height: int) -> tuple[float, float]:
    """Normalized bottom-center of a box: the point used for zone membership.

    The bottom-center approximates where a person's feet (or a car's wheels)
    touch the ground. Using the box center instead would place a tall person
    standing *outside* a ground-level zone "inside" it whenever their torso
    overlaps the zone in the image.
    """
    x_min, _, x_max, y_max = bbox_xyxy
    return ((x_min + x_max) / 2.0 / width, y_max / height)
