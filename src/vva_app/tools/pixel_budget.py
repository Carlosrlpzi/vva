"""Geometric pixel budget for one camera target: source height, letterbox scale and model height.

Why this module exists
----------------------
Before any detector code is written, section 0 of the MVP guide asks one question: does the
model receive enough pixels to see a person at the zone boundary at all? This module answers
that geometric question. It is a pure calculator: no camera, no Hailo runtime, no YAML, no
image or stream I/O, and no import of the production pipeline, so it runs during pre-flight
and in unit tests with nothing attached.

Definitions
-----------
``image_width_px`` and ``image_height_px`` are the dimensions of the image that enters the
letterbox, i.e. the frame the model actually receives. They are NOT necessarily the sensor
resolution or the main-stream resolution: when the detector consumes a substream, pass the
substream dimensions. Passing main-stream dimensions here silently inflates every result.

Formulas (guide section 0)
--------------------------
1. Pixel height of a target in the source image (pinhole model)::

       h_source_px = image_height_px * focal_length_mm * target_height_m
                     / (distance_m * sensor_height_mm)

2. Ideal uniform letterbox scale of the whole frame into a square model input::

       letterbox_scale = min(model_input_size_px / image_width_px,
                             model_input_size_px / image_height_px)

3. Pixel height in the model input::

       h_model_px = h_source_px * letterbox_scale

   Padding fills the rest of the square and does not enlarge the target, so the uniform
   scale alone converts source pixels into model pixels. The two axes are never stretched
   independently.

Units
-----
- ``image_width_px`` / ``image_height_px``: Python ``int`` pixels, > 0.
- ``focal_length_mm`` and ``sensor_height_mm``: millimetres, > 0. They share a unit, so
  their ratio is dimensionless.
- ``target_height_m`` and ``distance_m``: metres, > 0. They share a unit, so their ratio is
  dimensionless.
- ``ir_range_m``: metres, optional; when given, > 0.
No unit is parsed from a string: values must already be numbers expressed in these units.

IR range
--------
``ir_range_m`` is evaluated separately from the geometry. It never replaces ``distance_m``
and never changes any pixel height: being inside the declared range only means the point is
not ruled out by IR reach, never that night detection is guaranteed.

Limitations (a valid result is not a guarantee)
-----------------------------------------------
- This is a geometric estimate. It does not model a person's orientation, occlusion, lens
  distortion, illumination, motion blur or detector accuracy.
- ``distance_m`` is the distance from the camera to the target, approximately along the
  optical axis with the target perpendicular to it. A ground-plane distance is shorter than
  the camera distance when the camera is elevated, and camera tilt is not modelled here.
- The IR assessment reuses that same ``distance_m``. IR reach is naturally a radial
  distance, which can differ from the axial one; that difference is not modelled here.
- ``focal_length_mm`` and ``sensor_height_mm`` must describe the same capture mode as
  ``image_height_px``: those pixels are assumed to cover the whole physical
  ``sensor_height_mm`` with no vertical crop and no digital zoom. If the stream is cropped,
  the formula does not apply with these parameters.
- Only heights are computed. If a stream scales the two axes differently (anisotropy), the
  target's width cannot be derived here, so no width is computed.
- The pinhole model ignores barrel distortion. With a wide-angle lens the target size near
  the frame periphery can be overestimated, and this increment does not correct it.
- There is no day/night parameter. With identical geometry and preprocessing the geometric
  estimate is identical; night limits are represented only by the separate IR-range
  assessment, never by an artificially different nighttime height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

MODEL_INPUT_SIZE_PX: Final[int] = 640
"""Side, in pixels, of the square model input the frame is letterboxed into (guide section 1.3)."""


class IrRangeStatus(StrEnum):
    """Verdict of comparing the target distance with the declared IR illuminator range.

    ``NOT_EVALUATED`` means no range was supplied: it is not a claim of coverage. Being
    ``WITHIN_RANGE`` means the declared reach does not rule the point out, never that the
    target is detectable at night.
    """

    NOT_EVALUATED = "not_evaluated"
    WITHIN_RANGE = "within_range"
    BEYOND_RANGE = "beyond_range"


@dataclass(frozen=True)
class PixelBudget:
    """Geometric budget for one target and the optional IR-range verdict.

    Attributes:
        source_height_px: Estimated target height in the source image, before letterbox,
            in pixels.
        letterbox_scale: Uniform scale applied to the whole source frame to fit the model
            input; dimensionless. Padding is not part of it.
        model_height_px: Estimated target height in the model input, in pixels
            (``source_height_px * letterbox_scale``).
        ir_range_status: Whether ``distance_m`` is within the declared IR range, or
            ``NOT_EVALUATED`` when no range was supplied.

    """

    source_height_px: float
    letterbox_scale: float
    model_height_px: float
    ir_range_status: IrRangeStatus = IrRangeStatus.NOT_EVALUATED

    def __post_init__(self) -> None:
        """Reject non-finite or non-positive heights and scale on direct construction."""
        _require_positive_finite(self.source_height_px, "source_height_px")
        _require_positive_finite(self.letterbox_scale, "letterbox_scale")
        _require_positive_finite(self.model_height_px, "model_height_px")


def letterbox_scale(
    image_width_px: int,
    image_height_px: int,
    *,
    model_input_size_px: int = MODEL_INPUT_SIZE_PX,
) -> float:
    """Return the uniform scale that fits the whole frame into a square model input.

    ``min(model_input_size_px / image_width_px, model_input_size_px / image_height_px)``,
    applied to both axes: the frame is scaled uniformly and the short axis is padded, never
    stretched. The value is > 1 when the frame is smaller than the model input, because the
    formula, as fixed by the card, allows upscale.

    Args:
        image_width_px: Source image width in pixels (the image entering the letterbox).
        image_height_px: Source image height in pixels (the image entering the letterbox).
        model_input_size_px: Side of the square model input; defaults to
            :data:`MODEL_INPUT_SIZE_PX`.

    Raises:
        ValueError: If a dimension is not a Python ``int`` > 0 (booleans and floats, even
            integral ones, are rejected) or the model input side is invalid.

    """
    width = _require_positive_int(image_width_px, "image_width_px")
    height = _require_positive_int(image_height_px, "image_height_px")
    side = _require_positive_int(model_input_size_px, "model_input_size_px")
    return min(side / width, side / height)


def estimate_source_height_px(
    image_height_px: int,
    *,
    focal_length_mm: float,
    sensor_height_mm: float,
    target_height_m: float,
    distance_m: float,
) -> float:
    """Return the target's estimated height, in pixels, in the source image (pinhole model).

    ``image_height_px * focal_length_mm * target_height_m / (distance_m * sensor_height_mm)``.
    It depends on the image HEIGHT only, not on the width: a wider frame with the same height
    places the same number of pixels vertically on the target.

    Args:
        image_height_px: Source image height in pixels (the image entering the letterbox).
        focal_length_mm: Lens focal length in millimetres.
        sensor_height_mm: Physical height, in millimetres, of the sensor region that maps to
            ``image_height_px`` in this capture mode.
        target_height_m: Real target height in metres.
        distance_m: Distance from the camera to the target in metres (approximately along
            the optical axis), not a ground-plane distance.

    Raises:
        ValueError: If any input is invalid, or if the result is not finite and positive
            (extreme inputs overflowed or underflowed).

    """
    height = _require_positive_int(image_height_px, "image_height_px")
    focal = _require_positive_finite(focal_length_mm, "focal_length_mm")
    sensor = _require_positive_finite(sensor_height_mm, "sensor_height_mm")
    target = _require_positive_finite(target_height_m, "target_height_m")
    distance = _require_positive_finite(distance_m, "distance_m")
    denominator = distance * sensor
    if denominator == 0.0:
        # Only reachable when two valid positive inputs underflow; fail loud instead of
        # letting a ZeroDivisionError escape with a less useful message.
        raise ValueError("distance_m * sensor_height_mm underflowed to zero")
    result = height * focal * target / denominator
    return _require_positive_result(result, "h_source_px")


def estimate_model_height_px(source_height_px_value: float, letterbox_scale_value: float) -> float:
    """Return the target's estimated height, in pixels, in the model input.

    ``source_height_px_value * letterbox_scale_value``. Letterbox padding occupies the rest
    of the square input; it does not add to the target size, so no padding term appears.

    Args:
        source_height_px_value: Target height in the source image, in pixels.
        letterbox_scale_value: Uniform scale from :func:`letterbox_scale`.

    Raises:
        ValueError: If an input is invalid or the product is not finite and positive.

    """
    source = _require_positive_finite(source_height_px_value, "source_height_px")
    scale = _require_positive_finite(letterbox_scale_value, "letterbox_scale")
    return _require_positive_result(source * scale, "h_model_px")


def ir_range_status(distance_m: float, ir_range_m: float | None = None) -> IrRangeStatus:
    """Compare the target distance with the declared IR illuminator range.

    The comparison is inclusive: a distance equal to the range counts as within reach. A
    missing range yields :attr:`IrRangeStatus.NOT_EVALUATED` rather than an assumed coverage.

    Args:
        distance_m: Distance from the camera to the target in metres.
        ir_range_m: Declared IR range in metres, or ``None`` when unknown.

    Raises:
        ValueError: If ``distance_m`` is invalid, or ``ir_range_m`` is given but invalid.

    """
    distance = _require_positive_finite(distance_m, "distance_m")
    if ir_range_m is None:
        return IrRangeStatus.NOT_EVALUATED
    ir_range = _require_positive_finite(ir_range_m, "ir_range_m")
    if distance <= ir_range:
        return IrRangeStatus.WITHIN_RANGE
    return IrRangeStatus.BEYOND_RANGE


def compute_pixel_budget(
    *,
    image_width_px: int,
    image_height_px: int,
    focal_length_mm: float,
    sensor_height_mm: float,
    target_height_m: float,
    distance_m: float,
    ir_range_m: float | None = None,
    model_input_size_px: int = MODEL_INPUT_SIZE_PX,
) -> PixelBudget:
    """Compose the source height, the letterbox scale, the model height and the IR verdict.

    This is a convenience for callers that want the whole budget at once. It delegates every
    formula to the pure helpers above, so the numbers cannot drift between the two entry
    points. ``ir_range_m`` is optional: geometry-only callers should call the helpers
    directly, and omitting the range leaves ``ir_range_status`` at ``NOT_EVALUATED``.

    Args:
        image_width_px: Source image width in pixels (the image entering the letterbox).
        image_height_px: Source image height in pixels (the image entering the letterbox).
        focal_length_mm: Lens focal length in millimetres.
        sensor_height_mm: Physical sensor-region height in millimetres for this capture mode.
        target_height_m: Real target height in metres.
        distance_m: Distance from the camera to the target in metres (not a ground distance).
        ir_range_m: Optional declared IR range in metres.
        model_input_size_px: Side of the square model input; defaults to
            :data:`MODEL_INPUT_SIZE_PX`.

    Raises:
        ValueError: If any input is invalid or an intermediate result is not finite/positive.

    """
    scale = letterbox_scale(image_width_px, image_height_px, model_input_size_px=model_input_size_px)
    source_height = estimate_source_height_px(
        image_height_px,
        focal_length_mm=focal_length_mm,
        sensor_height_mm=sensor_height_mm,
        target_height_m=target_height_m,
        distance_m=distance_m,
    )
    model_height = estimate_model_height_px(source_height, scale)
    return PixelBudget(
        source_height_px=source_height,
        letterbox_scale=scale,
        model_height_px=model_height,
        ir_range_status=ir_range_status(distance_m, ir_range_m),
    )


def _require_positive_int(value: object, name: str) -> int:
    """Return ``value`` as a Python ``int`` > 0, or raise ``ValueError``.

    Booleans are rejected explicitly (``bool`` is an ``int`` subclass in Python) and floats
    are rejected even when integral: a float dimension signals a unit or type mix-up, and
    this module fails loud instead of guessing.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    try:
        float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is too large to represent as a float: {value}") from exc
    return value


def _require_positive_finite(value: object, name: str) -> float:
    """Return ``value`` as a finite ``float`` > 0, or raise ``ValueError``.

    Strings are rejected outright (no unit parsing), booleans are rejected, and NaN and
    infinities are rejected so they cannot silently poison a ratio.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is too large to represent as a float: {value}") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return number


def _require_positive_result(value: float, name: str) -> float:
    """Return ``value`` if it is finite and > 0, else raise ``ValueError``.

    A non-finite or non-positive result means the inputs were extreme enough to overflow,
    underflow or cancel; a silent wrong pixel height is worse than a loud failure.
    """
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} is not a finite positive number, got {value!r}")
    return value
