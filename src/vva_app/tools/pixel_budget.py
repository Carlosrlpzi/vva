"""Geometric pixel budget for one camera zone: model-input height and the night boundary.

Why this module exists
----------------------
Before any detector code is written, section 0 of the MVP guide asks one question: does the
model receive enough pixels to see a person at the zone boundary at all? It also asks for the
nighttime boundary, where IR illuminator range can be the limiting factor rather than the lens
geometry. This module answers both questions. It is a pure calculator: no camera, no Hailo
runtime, no YAML, no image or stream I/O, and no import of the production pipeline, so it runs
during pre-flight and in unit tests with nothing attached.

Frame geometry
--------------
The inputs describe the frame exactly as it is handed to the letterbox step:
``frame_width_px`` and ``frame_height_px`` are the dimensions of the array passed to
letterbox. They are NOT the camera's main-stream resolution and NOT a display resolution.
When the detector consumes a substream, pass the substream array dimensions. Passing a larger
frame here silently inflates every result.

The focal length in pixels comes from the VERTICAL field of view::

    f_px_v = (frame_height_px / 2) / tan(vfov_deg / 2)

Camera spec sheets usually quote horizontal or diagonal FOV. Convert it to vertical FOV before
calling this module; passing a horizontal or diagonal value here places the wrong number of
pixels on the target. The letterbox scale is::

    s = input_size_px / max(frame_width_px, frame_height_px)

The target height in the model input is::

    h_px = target_height_m * f_px_v * s / distance_m

and the optical floor distance for a minimum acceptable height ``h_min_px`` is::

    optical_floor_m = target_height_m * f_px_v * s / h_min_px

Non-square pixels
-----------------
The camera substream in this project is 720x576 carrying a 16:9 scene, i.e. an effective SAR of
64:45 that the camera does NOT declare in metadata. This module does not add SAR handling: it
models the array it is given. The formula is therefore correct in both feeding modes:

- raw frames (720x576): ``s = 640 / 720``;
- SAR-corrected upstream frames (for example 1024x576): ``s = 640 / 1024``.

The consequence is that SAR correction before letterbox lowers ``h_px`` by a factor of 45/64
(~0.70). Forwarding raw frames instead compresses person boxes horizontally by the same factor;
that is relevant to any future box-aspect-ratio filter and is only noted here, out of scope.

Distances
---------
All ``*_m`` distances are slant (line-of-sight) distances from camera to target:
``distance_m``, ``zone_distance_m``, ``optical_floor_m``, ``ir_range_m`` and ``boundary_m``.
A manufacturer IR range is treated as line-of-sight. This convention is PROVISIONAL — pending
field validation; :func:`slant_distance_m` is the only place the ground-to-slant conversion
lives, so it can be revised in one place after experiments.

Night coverage
--------------
Night coverage is expressed only through :class:`NightBudget`. Its boundary is
``min(optical_floor_m, ir_range_m)`` and its limiting factor records which one won. When the IR
range is unknown, the night fields are null: the optical floor is never used as a fallback,
because that would let a zone claim night coverage without evidence.

Limitations (a valid result is not a guarantee)
-----------------------------------------------
- This is a geometric estimate. It does not model a person's orientation, occlusion, lens
  distortion, illumination, motion blur or detector accuracy.
- ``vfov_deg`` must describe the vertical extent of the array as it is handed to letterbox. A
  vertical crop or a digital zoom changes the effective vertical FOV, so pass the FOV of the
  delivered array, not the lens's nominal value. Resampling only the width (for example SAR
  correction from 720x576 to 1024x576) does not change it.
- Camera tilt is not modelled. The slant distance fixes the line-of-sight length, but a person
  seen from an elevated camera is foreshortened and may sit off the optical axis, so the real
  height in pixels can differ from ``h_px``. Whether to model this is pending field validation.
- The pinhole model ignores barrel distortion. With a wide-angle lens the target size near the
  frame periphery can be overestimated, and this module does not correct it.
- There is no day/night pixel-height parameter. With identical geometry and preprocessing the
  geometric estimate is identical; the night limit is the separate boundary computed above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

MODEL_INPUT_SIZE_PX: Final[int] = 640
"""Side, in pixels, of the square model input the frame is letterboxed into (guide section 1.3)."""


class NightLimit(StrEnum):
    """Which constraint sets the nighttime boundary.

    ``OPTICAL`` means the pixel-height floor is closer than the IR range. ``IR_RANGE`` means
    the IR range is the tighter limit (including a tie). ``NOT_EVALUATED`` means no IR range
    was supplied, so no night coverage claim is made.
    """

    OPTICAL = "optical"
    IR_RANGE = "ir_range"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True)
class NightBudget:
    """Nighttime boundary for one zone, with the diagnostic that explains it.

    Attributes:
        optical_floor_m: Slant distance where the model-input height equals ``h_min_px``,
            in metres.
        ir_range_m: Declared IR illuminator range in metres, or ``None`` when unknown.
        boundary_m: ``min(optical_floor_m, ir_range_m)``, or ``None`` when the IR range is
            unknown.
        limiting_factor: Which constraint set ``boundary_m``; ``NOT_EVALUATED`` when the IR
            range is unknown.
        zone_covered: Whether ``zone_distance_m <= boundary_m``; ``None`` when the IR range is
            unknown. The comparison is inclusive.
        night_h_px: Day model-input height at the zone distance when ``zone_covered`` is true,
            else ``None``. Geometry does not change at night; this mirrors the day value.

    """

    optical_floor_m: float
    ir_range_m: float | None
    boundary_m: float | None
    limiting_factor: NightLimit
    zone_covered: bool | None
    night_h_px: float | None

    def __post_init__(self) -> None:
        """Reject inconsistent or non-finite night fields on direct construction."""
        _require_positive_finite(self.optical_floor_m, "optical_floor_m")
        if not isinstance(self.limiting_factor, NightLimit):
            raise ValueError(f"limiting_factor must be a NightLimit, got {self.limiting_factor!r}")
        if self.ir_range_m is not None:
            _require_positive_finite(self.ir_range_m, "ir_range_m")
        if self.limiting_factor is NightLimit.NOT_EVALUATED:
            _require_unevaluated_night(self)
            return
        _require_evaluated_night(self)


@dataclass(frozen=True)
class PixelBudget:
    """Day model-input height for one zone, with its nested nighttime boundary.

    Attributes:
        zone_distance_m: Slant distance from the camera to the zone boundary, in metres.
        letterbox_scale: Uniform scale applied to the whole frame to fit the model input;
            dimensionless. Padding is not part of it.
        model_height_px: Day model-input height at ``zone_distance_m``, in pixels.
        night: Nighttime boundary and its limiting factor for the same zone.

    """

    zone_distance_m: float
    letterbox_scale: float
    model_height_px: float
    night: NightBudget

    def __post_init__(self) -> None:
        """Reject non-finite or non-positive day fields on direct construction."""
        _require_positive_finite(self.zone_distance_m, "zone_distance_m")
        _require_positive_finite(self.letterbox_scale, "letterbox_scale")
        _require_positive_finite(self.model_height_px, "model_height_px")
        if not isinstance(self.night, NightBudget):
            raise ValueError(f"night must be a NightBudget, got {self.night!r}")


def _require_unevaluated_night(night: NightBudget) -> None:
    """Reject a ``NOT_EVALUATED`` budget that carries any night claim."""
    if (
        night.ir_range_m is not None
        or night.boundary_m is not None
        or night.zone_covered is not None
        or night.night_h_px is not None
    ):
        raise ValueError("NOT_EVALUATED night budget must leave all night fields None")


def _require_evaluated_night(night: NightBudget) -> None:
    """Reject an evaluated budget whose boundary, factor and coverage disagree."""
    if night.ir_range_m is None or night.boundary_m is None:
        raise ValueError("evaluated night budget requires ir_range_m and boundary_m")
    _require_positive_finite(night.boundary_m, "boundary_m")
    expected_boundary = min(night.optical_floor_m, night.ir_range_m)
    if not math.isclose(night.boundary_m, expected_boundary, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("boundary_m must equal min(optical_floor_m, ir_range_m)")
    expected_limit = NightLimit.IR_RANGE if night.ir_range_m <= night.optical_floor_m else NightLimit.OPTICAL
    if night.limiting_factor is not expected_limit:
        raise ValueError("limiting_factor does not match the tighter distance")
    if not isinstance(night.zone_covered, bool):
        raise ValueError(f"zone_covered must be a bool when evaluated, got {night.zone_covered!r}")
    if night.zone_covered:
        if night.night_h_px is None:
            raise ValueError("a covered zone requires night_h_px")
        _require_positive_finite(night.night_h_px, "night_h_px")
    elif night.night_h_px is not None:
        raise ValueError("an uncovered zone must leave night_h_px None")


def slant_distance_m(
    ground_distance_m: float,
    camera_height_m: float,
    target_ref_height_m: float,
) -> float:
    """Return the camera-to-target line-of-sight distance from ground-plane measurements.

    ``hypot(ground_distance_m, camera_height_m - target_ref_height_m)``. This is the only
    place the ground-to-slant conversion lives. ``target_ref_height_m`` is the target's
    vertical reference point; the target's vertical midpoint is a provisional choice.

    PROVISIONAL — pending field validation: the project has not yet measured whether the
    manufacturer IR range and the zone geometry are better modelled as slant or horizontal
    distances. Revise the convention here, in one place, after experiments.

    Args:
        ground_distance_m: Horizontal ground-plane distance to the target, in metres.
        camera_height_m: Camera mounting height, in metres.
        target_ref_height_m: Height of the target reference point, in metres.

    Raises:
        ValueError: If any input is not finite and > 0.

    """
    ground = _require_positive_finite(ground_distance_m, "ground_distance_m")
    camera = _require_positive_finite(camera_height_m, "camera_height_m")
    target = _require_positive_finite(target_ref_height_m, "target_ref_height_m")
    return math.hypot(ground, camera - target)


def letterbox_scale(
    frame_width_px: int,
    frame_height_px: int,
    *,
    input_size_px: int = MODEL_INPUT_SIZE_PX,
) -> float:
    """Return the uniform scale that fits the whole frame into a square model input.

    ``input_size_px / max(frame_width_px, frame_height_px)``, applied to both axes: the frame
    is scaled uniformly and the short axis is padded, never stretched. The value is > 1 when
    the frame is smaller than the model input, because the formula, as fixed by the card,
    allows upscale.

    Args:
        frame_width_px: Width in pixels of the array passed to the letterbox step.
        frame_height_px: Height in pixels of the array passed to the letterbox step.
        input_size_px: Side of the square model input; defaults to
            :data:`MODEL_INPUT_SIZE_PX`.

    Raises:
        ValueError: If a dimension is not a Python ``int`` > 0 (booleans and floats, even
            integral ones, are rejected) or the model input side is invalid.

    """
    width = _require_positive_int(frame_width_px, "frame_width_px")
    height = _require_positive_int(frame_height_px, "frame_height_px")
    side = _require_positive_int(input_size_px, "input_size_px")
    return side / max(width, height)


def model_height_px(
    *,
    frame_width_px: int,
    frame_height_px: int,
    vfov_deg: float,
    target_height_m: float,
    distance_m: float,
    input_size_px: int = MODEL_INPUT_SIZE_PX,
) -> float:
    """Return the target's day height, in pixels, in the model input.

    ``target_height_m * f_px_v * s / distance_m``, where ``f_px_v`` comes from the vertical
    FOV and ``s`` is :func:`letterbox_scale`. Padding fills the rest of the square input and
    does not enlarge the target, so no padding term appears.

    Args:
        frame_width_px: Width in pixels of the array passed to the letterbox step.
        frame_height_px: Height in pixels of the array passed to the letterbox step.
        vfov_deg: Vertical field of view in degrees, strictly between 0 and 180. Convert
            horizontal or diagonal datasheet values to vertical before calling.
        target_height_m: Real target height in metres.
        distance_m: Slant distance from the camera to the target, in metres.
        input_size_px: Side of the square model input; defaults to
            :data:`MODEL_INPUT_SIZE_PX`.

    Raises:
        ValueError: If any input is invalid, or if the result is not finite and positive
            (extreme inputs overflowed or underflowed).

    """
    scale = letterbox_scale(frame_width_px, frame_height_px, input_size_px=input_size_px)
    focal = _vertical_focal_px(frame_height_px, vfov_deg)
    target = _require_positive_finite(target_height_m, "target_height_m")
    distance = _require_positive_finite(distance_m, "distance_m")
    return _require_positive_result(target * focal * scale / distance, "h_px")


def optical_floor_m(
    *,
    frame_width_px: int,
    frame_height_px: int,
    vfov_deg: float,
    target_height_m: float,
    h_min_px: float,
    input_size_px: int = MODEL_INPUT_SIZE_PX,
) -> float:
    """Return the slant distance where the model-input height equals ``h_min_px``.

    ``target_height_m * f_px_v * s / h_min_px``, the inverse of :func:`model_height_px`.

    Args:
        frame_width_px: Width in pixels of the array passed to the letterbox step.
        frame_height_px: Height in pixels of the array passed to the letterbox step.
        vfov_deg: Vertical field of view in degrees, strictly between 0 and 180.
        target_height_m: Real target height in metres.
        h_min_px: Minimum acceptable target height in the model input, in pixels. Required:
            the safe distance depends on the detector's floor and must not have a hidden
            default.
        input_size_px: Side of the square model input; defaults to
            :data:`MODEL_INPUT_SIZE_PX`.

    Raises:
        ValueError: If any input is invalid, or if the result is not finite and positive.

    """
    scale = letterbox_scale(frame_width_px, frame_height_px, input_size_px=input_size_px)
    focal = _vertical_focal_px(frame_height_px, vfov_deg)
    target = _require_positive_finite(target_height_m, "target_height_m")
    minimum = _require_positive_finite(h_min_px, "h_min_px")
    return _require_positive_result(target * focal * scale / minimum, "optical_floor_m")


def night_budget(
    *,
    frame_width_px: int,
    frame_height_px: int,
    vfov_deg: float,
    target_height_m: float,
    h_min_px: float,
    zone_distance_m: float,
    ir_range_m: float | None = None,
    input_size_px: int = MODEL_INPUT_SIZE_PX,
) -> NightBudget:
    """Return the night boundary for one zone and whether the zone is covered.

    The boundary is ``min(optical_floor_m, ir_range_m)``; an IR range equal to the optical
    floor wins the tie (``IR_RANGE``). ``zone_covered`` is inclusive. When ``ir_range_m`` is
    ``None``, all night fields are null: the optical floor is diagnostic only and is never
    used as a fallback that would claim unevidenced night coverage.

    Args:
        frame_width_px: Width in pixels of the array passed to the letterbox step.
        frame_height_px: Height in pixels of the array passed to the letterbox step.
        vfov_deg: Vertical field of view in degrees, strictly between 0 and 180.
        target_height_m: Real target height in metres.
        h_min_px: Minimum acceptable target height in the model input, in pixels.
        zone_distance_m: Slant distance from the camera to the zone boundary, in metres.
        ir_range_m: Declared IR illuminator range in metres, or ``None`` when unknown.
        input_size_px: Side of the square model input; defaults to
            :data:`MODEL_INPUT_SIZE_PX`.

    Raises:
        ValueError: If any input is invalid. ``ir_range_m=None`` is valid ("unknown"); zero or
            a negative value is an error, not "unknown".

    """
    floor = optical_floor_m(
        frame_width_px=frame_width_px,
        frame_height_px=frame_height_px,
        vfov_deg=vfov_deg,
        target_height_m=target_height_m,
        h_min_px=h_min_px,
        input_size_px=input_size_px,
    )
    zone_distance = _require_positive_finite(zone_distance_m, "zone_distance_m")
    if ir_range_m is None:
        return NightBudget(
            optical_floor_m=floor,
            ir_range_m=None,
            boundary_m=None,
            limiting_factor=NightLimit.NOT_EVALUATED,
            zone_covered=None,
            night_h_px=None,
        )
    ir_range = _require_positive_finite(ir_range_m, "ir_range_m")
    boundary = min(floor, ir_range)
    covered = zone_distance <= boundary
    night_h_px = (
        model_height_px(
            frame_width_px=frame_width_px,
            frame_height_px=frame_height_px,
            vfov_deg=vfov_deg,
            target_height_m=target_height_m,
            distance_m=zone_distance,
            input_size_px=input_size_px,
        )
        if covered
        else None
    )
    return NightBudget(
        optical_floor_m=floor,
        ir_range_m=ir_range,
        boundary_m=boundary,
        limiting_factor=NightLimit.IR_RANGE if ir_range <= floor else NightLimit.OPTICAL,
        zone_covered=covered,
        night_h_px=night_h_px,
    )


def compute_pixel_budget(
    *,
    frame_width_px: int,
    frame_height_px: int,
    vfov_deg: float,
    target_height_m: float,
    zone_distance_m: float,
    h_min_px: float,
    ir_range_m: float | None = None,
    input_size_px: int = MODEL_INPUT_SIZE_PX,
) -> PixelBudget:
    """Compose the day model height and the nested night boundary for one zone.

    This is a convenience for callers that want the whole per-zone budget at once. It
    delegates every formula to the pure helpers above, so the numbers cannot drift between
    the two entry points. ``h_min_px`` is required because the optical floor always depends
    on it.

    Args:
        frame_width_px: Width in pixels of the array passed to the letterbox step.
        frame_height_px: Height in pixels of the array passed to the letterbox step.
        vfov_deg: Vertical field of view in degrees, strictly between 0 and 180.
        target_height_m: Real target height in metres.
        zone_distance_m: Slant distance from the camera to the zone boundary, in metres.
        h_min_px: Minimum acceptable target height in the model input, in pixels.
        ir_range_m: Declared IR illuminator range in metres, or ``None`` when unknown.
        input_size_px: Side of the square model input; defaults to
            :data:`MODEL_INPUT_SIZE_PX`.

    Raises:
        ValueError: If any input is invalid or an intermediate result is not finite/positive.

    """
    scale = letterbox_scale(frame_width_px, frame_height_px, input_size_px=input_size_px)
    day_height = model_height_px(
        frame_width_px=frame_width_px,
        frame_height_px=frame_height_px,
        vfov_deg=vfov_deg,
        target_height_m=target_height_m,
        distance_m=zone_distance_m,
        input_size_px=input_size_px,
    )
    night = night_budget(
        frame_width_px=frame_width_px,
        frame_height_px=frame_height_px,
        vfov_deg=vfov_deg,
        target_height_m=target_height_m,
        h_min_px=h_min_px,
        zone_distance_m=zone_distance_m,
        ir_range_m=ir_range_m,
        input_size_px=input_size_px,
    )
    return PixelBudget(
        zone_distance_m=zone_distance_m,
        letterbox_scale=scale,
        model_height_px=day_height,
        night=night,
    )


def _vertical_focal_px(frame_height_px: int, vfov_deg: float) -> float:
    """Return the vertical focal length in pixels: ``(frame_height_px / 2) / tan(vfov/2)``.

    A vanishingly small vertical FOV underflows ``tan`` to zero; fail loud instead of letting a
    ``ZeroDivisionError`` escape with a less useful message.
    """
    height = _require_positive_int(frame_height_px, "frame_height_px")
    vfov = _require_vfov_deg(vfov_deg)
    denominator = math.tan(math.radians(vfov) / 2.0)
    if denominator == 0.0:
        raise ValueError(f"vfov_deg {vfov!r} underflowed tan(vfov/2) to zero")
    return _require_positive_result((height / 2.0) / denominator, "f_px_v")


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


def _require_vfov_deg(value: object) -> float:
    """Return ``value`` as a finite vertical FOV strictly inside ``(0, 180)`` degrees."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"vfov_deg must be a finite number in (0, 180), got {value!r}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"vfov_deg is too large to represent as a float: {value}") from exc
    if not math.isfinite(number) or not 0.0 < number < 180.0:
        raise ValueError(f"vfov_deg must be finite and in (0, 180), got {value!r}")
    return number


def _require_positive_result(value: float, name: str) -> float:
    """Return ``value`` if it is finite and > 0, else raise ``ValueError``.

    A non-finite or non-positive result means the inputs were extreme enough to overflow,
    underflow or cancel; a silent wrong pixel height is worse than a loud failure.
    """
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} is not a finite positive number, got {value!r}")
    return value
