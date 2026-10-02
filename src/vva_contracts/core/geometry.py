"""Box and polygon geometry shared by evaluation and rule replay.

Why this module exists
----------------------
Detection metrics and zone rules both reduce to two primitives:

* **IoU** (intersection over union) decides whether a predicted box "is" a
  ground-truth box: IoU(A, B) = |A n B| / |A u B|, a value in [0, 1].
* **Point-in-polygon** decides whether a detection is inside a camera zone.

Keeping them in one pure module (no I/O, no globals) lets the production
pipeline import exactly the same code that the evaluation contracts use, so a
zone rule cannot behave differently in replay than on the Raspberry Pi.

Coordinate convention: all boxes are ``xyxy`` normalised to [0, 1], i.e.
(x_min, y_min, x_max, y_max) as fractions of image width and height.

Note on normalised IoU: scaling x by W and y by H multiplies *every* area
(intersections and unions alike) by W*H, so the ratio is unchanged. IoU can
therefore be computed in normalised coordinates without knowing image sizes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def xywhn_to_xyxyn(boxes: FloatArray) -> FloatArray:
    """Convert YOLO ``(cx, cy, w, h)`` rows to ``(x1, y1, x2, y2)`` rows."""
    centers = boxes[:, 0:2]
    half_sizes = boxes[:, 2:4] / 2.0
    # Stacking [center - half, center + half] gives the two corners at once.
    return np.hstack([centers - half_sizes, centers + half_sizes])


def iou_matrix(a: FloatArray, b: FloatArray) -> FloatArray:
    """Pairwise IoU between ``a`` (N x 4) and ``b`` (M x 4) xyxy boxes.

    Returns an (N x M) matrix. Vectorised with broadcasting: a Python double
    loop would be O(N*M) interpreter steps, which matters when replaying
    millions of frames on a Raspberry Pi.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    # a[:, None, :] has shape (N,1,4) and b[None, :, :] has shape (1,M,4);
    # broadcasting compares every box in `a` with every box in `b`.
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    # Clip at 0: disjoint boxes would otherwise produce negative "widths".
    wh = np.clip(bottom_right - top_left, 0.0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    # Inclusion-exclusion: |A u B| = |A| + |B| - |A n B|.
    union = area_a[:, None] + area_b[None, :] - intersection
    # Degenerate (zero-area) boxes give union 0; define their IoU as 0 instead
    # of letting a division by zero produce NaN that would poison averages.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(union > 0.0, intersection / union, 0.0)
    return result.astype(np.float64)


def point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Return True if (x, y) lies inside ``polygon`` (ray-casting test).

    A horizontal ray from the point towards +x crosses the polygon boundary an
    odd number of times iff the point is inside (Jordan curve theorem). Each
    edge (x_i, y_i) -> (x_j, y_j) is counted when it straddles the ray's y and
    the crossing lies to the right of the point.
    """
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # (yi > y) != (yj > y) is true only when the edge straddles the ray.
        # Using strict '>' on both ends counts a vertex lying exactly on the
        # ray once, not twice, which avoids the classic double-count bug.
        if (yi > y) != (yj > y):
            # x coordinate where the edge meets the horizontal line at height y.
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def anchor_point(box: Sequence[float]) -> tuple[float, float]:
    """Bottom-centre of an xyxy box.

    Zones are drawn on the ground (driveway, door mat). A person's feet, not
    the centre of their body, is what touches that ground, so the bottom-centre
    is the point tested against the zone polygon.
    """
    x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)
