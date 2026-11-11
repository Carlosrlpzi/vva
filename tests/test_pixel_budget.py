"""Unit tests for ``vva_app.tools.pixel_budget``.

No camera, Hailo runtime, network, ffmpeg or YAML is used: the module under test is a pure
calculator. Closed-form values use the hand-computable base fixture in the M0 spec:
720x576 frame, ``vfov_deg = 2 * atan(0.5)`` (so ``f_px_v = 576``), 640 px input
(``s = 8/9``), 1.7 m target and ``h_min_px = 32``.
"""

from __future__ import annotations

import math
import sys

import pytest

from vva_app.tools import pixel_budget
from vva_app.tools.pixel_budget import (
    MODEL_INPUT_SIZE_PX,
    NightBudget,
    NightLimit,
    PixelBudget,
    compute_pixel_budget,
    letterbox_scale,
    model_height_px,
    night_budget,
    optical_floor_m,
    slant_distance_m,
)

FRAME_WIDTH_PX = 720
FRAME_HEIGHT_PX = 576
VFOV_DEG = 2.0 * math.degrees(math.atan(0.5))
INPUT_SIZE_PX = 640
TARGET_HEIGHT_M = 1.7
H_MIN_PX = 32.0
GEOMETRY = {
    "frame_width_px": FRAME_WIDTH_PX,
    "frame_height_px": FRAME_HEIGHT_PX,
    "vfov_deg": VFOV_DEG,
    "target_height_m": TARGET_HEIGHT_M,
    "input_size_px": INPUT_SIZE_PX,
}


def _day_h_px(distance_m: float) -> float:
    return model_height_px(**GEOMETRY, distance_m=distance_m)


def _optical_floor() -> float:
    return optical_floor_m(**GEOMETRY, h_min_px=H_MIN_PX)


def _night(zone_distance_m: float, ir_range_m: float | None) -> NightBudget:
    return night_budget(
        **GEOMETRY,
        h_min_px=H_MIN_PX,
        zone_distance_m=zone_distance_m,
        ir_range_m=ir_range_m,
    )


# --------------------------------------------------------------- base fixture (cases 1-2)
def test_model_height_at_ten_metres() -> None:
    assert _day_h_px(10.0) == pytest.approx(87.04, abs=1e-9)


def test_optical_floor_for_thirty_two_pixels() -> None:
    assert _optical_floor() == pytest.approx(27.2, abs=1e-9)


def test_vertical_fov_produces_the_expected_focal_length() -> None:
    # tan(vfov/2) == 0.5, so the vertical focal length is exactly the frame height.
    assert pixel_budget._vertical_focal_px(FRAME_HEIGHT_PX, VFOV_DEG) == pytest.approx(576.0, abs=1e-9)


# ------------------------------------------------------------------ night boundary (3-8)
def test_ir_beyond_the_optical_floor_leaves_the_optical_limit() -> None:
    budget = _night(10.0, 30.0)
    assert budget.optical_floor_m == pytest.approx(27.2, abs=1e-9)
    assert budget.ir_range_m == pytest.approx(30.0, abs=1e-12)
    assert budget.boundary_m == pytest.approx(27.2, abs=1e-9)
    assert budget.limiting_factor is NightLimit.OPTICAL


def test_ir_below_the_optical_floor_caps_the_boundary() -> None:
    budget = _night(10.0, 20.0)
    assert budget.boundary_m == pytest.approx(20.0, abs=1e-12)
    assert budget.limiting_factor is NightLimit.IR_RANGE


def test_ir_equal_to_the_computed_optical_floor_wins_the_tie() -> None:
    floor = _optical_floor()  # the function's own value, not a literal
    budget = _night(10.0, floor)
    assert budget.ir_range_m is not None
    assert budget.ir_range_m == pytest.approx(floor, rel=0.0, abs=1e-12)
    assert budget.boundary_m == pytest.approx(floor, rel=0.0, abs=1e-12)
    assert budget.limiting_factor is NightLimit.IR_RANGE


def test_zone_at_the_inclusive_boundary_is_covered() -> None:
    budget = _night(20.0, 20.0)
    assert budget.zone_covered is True
    assert budget.night_h_px == pytest.approx(_day_h_px(20.0), rel=0.0, abs=1e-12)
    assert budget.night_h_px == pytest.approx(43.52, abs=1e-9)


def test_zone_beyond_the_boundary_is_not_covered() -> None:
    budget = _night(25.0, 20.0)
    assert budget.zone_covered is False
    assert budget.night_h_px is None


def test_unknown_ir_range_is_not_evaluated_and_does_not_fall_back() -> None:
    budget = _night(10.0, None)
    assert budget.optical_floor_m == pytest.approx(27.2, abs=1e-9)  # diagnostic remains
    assert budget.ir_range_m is None
    assert budget.boundary_m is None
    assert budget.limiting_factor is NightLimit.NOT_EVALUATED
    assert budget.zone_covered is None
    assert budget.night_h_px is None


# ------------------------------------------------------------------ frame geometry (9)
def test_sar_corrected_frame_lowers_the_model_height() -> None:
    corrected = model_height_px(
        frame_width_px=1024,
        frame_height_px=576,
        vfov_deg=VFOV_DEG,
        target_height_m=TARGET_HEIGHT_M,
        distance_m=10.0,
        input_size_px=INPUT_SIZE_PX,
    )
    assert corrected == pytest.approx(61.2, abs=1e-9)
    # 1024x576 has s = 640/1024; raw 720x576 has s = 640/720: the ratio is 45/64.
    assert corrected == pytest.approx(_day_h_px(10.0) * 45.0 / 64.0, rel=1e-12)


def test_letterbox_scale_uses_the_dominant_frame_side() -> None:
    assert letterbox_scale(640, 360) == pytest.approx(1.0, abs=1e-12)  # 16:9, width fits exactly
    assert letterbox_scale(360, 640) == pytest.approx(1.0, abs=1e-12)  # portrait, height fits
    assert letterbox_scale(640, 640) == pytest.approx(1.0, abs=1e-12)  # square, no padding
    assert letterbox_scale(1280, 480) == pytest.approx(0.5, abs=1e-12)  # width is dominant
    assert letterbox_scale(720, 576) == pytest.approx(640 / 720, abs=1e-12)  # 5:4 substream


def test_letterbox_scale_allows_upscale_below_the_model_input() -> None:
    assert letterbox_scale(320, 200) == pytest.approx(2.0, abs=1e-12)  # 640/320 dominates
    assert model_height_px(
        frame_width_px=320,
        frame_height_px=200,
        vfov_deg=VFOV_DEG,
        target_height_m=TARGET_HEIGHT_M,
        distance_m=5.0,
        input_size_px=INPUT_SIZE_PX,
    ) == pytest.approx(136.0, abs=1e-9)


def test_model_input_size_is_a_named_constant_and_overridable() -> None:
    assert MODEL_INPUT_SIZE_PX == 640
    assert letterbox_scale(1280, 720) == pytest.approx(0.5, abs=1e-12)
    assert letterbox_scale(1280, 720, input_size_px=320) == pytest.approx(0.25, abs=1e-12)


def test_model_height_follows_the_dominant_frame_side() -> None:
    tall = model_height_px(
        frame_width_px=640,
        frame_height_px=360,
        vfov_deg=VFOV_DEG,
        target_height_m=TARGET_HEIGHT_M,
        distance_m=5.0,
        input_size_px=INPUT_SIZE_PX,
    )
    wide = model_height_px(
        frame_width_px=1280,
        frame_height_px=360,
        vfov_deg=VFOV_DEG,
        target_height_m=TARGET_HEIGHT_M,
        distance_m=5.0,
        input_size_px=INPUT_SIZE_PX,
    )
    assert tall == pytest.approx(122.4, abs=1e-9)  # s = 1.0
    assert wide == pytest.approx(61.2, abs=1e-9)  # s = 0.5
    assert tall == pytest.approx(2.0 * wide, rel=1e-12)


def test_doubling_distance_halves_model_height() -> None:
    near = _day_h_px(5.0)
    far = _day_h_px(10.0)
    assert near == pytest.approx(2.0 * far, rel=1e-12)


# --------------------------------------------------------------- slant distance (10)
def test_slant_distance_is_hypotenuse_of_ground_and_height_difference() -> None:
    assert slant_distance_m(4.0, 3.1, 0.1) == pytest.approx(5.0, abs=1e-12)


def test_slant_distance_accepts_a_zero_height_difference() -> None:
    assert slant_distance_m(4.0, 2.0, 2.0) == pytest.approx(4.0, abs=1e-12)


def test_slant_distance_accepts_a_target_above_the_camera() -> None:
    assert slant_distance_m(4.0, 1.0, 3.0) == pytest.approx(math.hypot(4.0, -2.0), rel=1e-12)


# ------------------------------------------------------------- input validation (11)
@pytest.mark.parametrize(
    "bad_call",
    [
        pytest.param(
            lambda: night_budget(**GEOMETRY, h_min_px=H_MIN_PX, zone_distance_m=10.0, ir_range_m=0.0),
            id="ir_range_zero",
        ),
        pytest.param(lambda: optical_floor_m(**GEOMETRY, h_min_px=0.0), id="h_min_zero"),
        pytest.param(
            lambda: model_height_px(**{**GEOMETRY, "vfov_deg": 180.0}, distance_m=10.0),
            id="vfov_180",
        ),
        pytest.param(lambda: model_height_px(**GEOMETRY, distance_m=math.nan), id="distance_nan"),
    ],
)
def test_invalid_inputs_raise_value_error(bad_call: object) -> None:
    with pytest.raises(ValueError):
        bad_call()  # type: ignore[operator]


@pytest.mark.parametrize("bad", [0, -1, 360.5, 640.0, True, False])
def test_dimensions_reject_zero_negative_fractional_and_boolean(bad: object) -> None:
    with pytest.raises(ValueError, match="frame_width_px"):
        letterbox_scale(bad, 360)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="frame_height_px"):
        letterbox_scale(640, bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1, 640.0, True, False])
def test_input_size_rejects_zero_negative_fractional_and_boolean(bad: object) -> None:
    with pytest.raises(ValueError, match="input_size_px"):
        letterbox_scale(640, 360, input_size_px=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1, 360.5, 640.0, True])
def test_model_height_rejects_invalid_frame_height(bad: object) -> None:
    with pytest.raises(ValueError, match="frame_height_px"):
        model_height_px(**{**GEOMETRY, "frame_height_px": bad}, distance_m=10.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["target_height_m", "distance_m"])
@pytest.mark.parametrize("bad", [0.0, -1.0, True, False, math.nan, math.inf, -math.inf, "4", "4 m", None])
def test_magnitudes_reject_zero_negative_boolean_non_finite_and_string(field: str, bad: object) -> None:
    kwargs = {**GEOMETRY, "distance_m": 10.0, field: bad}
    with pytest.raises(ValueError, match=field):
        model_height_px(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0.0, -1.0, True, math.nan, math.inf, -math.inf, "4", "4 m", None])
def test_h_min_rejects_invalid_values(bad: object) -> None:
    with pytest.raises(ValueError, match="h_min_px"):
        optical_floor_m(**GEOMETRY, h_min_px=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0.0, 180.0, -1.0, 200.0, True, math.nan, math.inf, "41 deg", None])
def test_vfov_rejects_values_outside_the_open_interval(bad: object) -> None:
    with pytest.raises(ValueError, match="vfov_deg"):
        model_height_px(**{**GEOMETRY, "vfov_deg": bad}, distance_m=10.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["ground_distance_m", "camera_height_m", "target_ref_height_m"])
@pytest.mark.parametrize("bad", [0.0, -1.0, True, math.nan, math.inf, "4 m", None])
def test_slant_distance_rejects_invalid_arguments(field: str, bad: object) -> None:
    kwargs = {"ground_distance_m": 4.0, "camera_height_m": 3.1, "target_ref_height_m": 0.1}
    kwargs[field] = bad
    with pytest.raises(ValueError, match=field):
        slant_distance_m(**kwargs)  # type: ignore[arg-type]


def test_ir_range_rejects_invalid_values_but_accepts_none() -> None:
    for bad in (0.0, -1.0, True, math.nan, math.inf, "15 m"):
        with pytest.raises(ValueError, match="ir_range_m"):
            _night(10.0, bad)  # type: ignore[arg-type]
    assert _night(10.0, None).limiting_factor is NightLimit.NOT_EVALUATED


def test_overflow_to_non_finite_is_rejected() -> None:
    with pytest.raises(ValueError, match="h_px"):
        model_height_px(**GEOMETRY, distance_m=5e-324)


def test_extreme_vertical_fov_overflow_is_rejected() -> None:
    with pytest.raises(ValueError, match="f_px_v"):
        model_height_px(**{**GEOMETRY, "vfov_deg": 1e-308}, distance_m=5.0)


def test_boundary_must_be_the_minimum_of_floor_and_ir_range() -> None:
    with pytest.raises(ValueError, match="boundary_m"):
        NightBudget(
            optical_floor_m=27.2,
            ir_range_m=20.0,
            boundary_m=27.2,
            limiting_factor=NightLimit.IR_RANGE,
            zone_covered=True,
            night_h_px=1.0,
        )


def test_not_evaluated_night_budget_rejects_a_stray_boundary() -> None:
    with pytest.raises(ValueError, match="NOT_EVALUATED"):
        NightBudget(
            optical_floor_m=27.2,
            ir_range_m=None,
            boundary_m=20.0,
            limiting_factor=NightLimit.NOT_EVALUATED,
            zone_covered=None,
            night_h_px=None,
        )


def test_pixel_budget_rejects_non_positive_and_invalid_fields() -> None:
    night = _night(10.0, 30.0)
    for field in ("zone_distance_m", "letterbox_scale", "model_height_px"):
        values = {
            "zone_distance_m": 10.0,
            "letterbox_scale": 1.0,
            "model_height_px": 1.0,
            "night": night,
            field: -1.0,
        }
        with pytest.raises(ValueError):
            PixelBudget(**values)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="night"):
        PixelBudget(zone_distance_m=10.0, letterbox_scale=1.0, model_height_px=1.0, night=object())  # type: ignore[arg-type]


# ------------------------------------------------------------- composition and purity
def test_compute_pixel_budget_nests_the_night_budget() -> None:
    budget = compute_pixel_budget(
        **GEOMETRY,
        zone_distance_m=20.0,
        h_min_px=H_MIN_PX,
        ir_range_m=20.0,
    )
    assert budget.zone_distance_m == 20.0
    assert budget.model_height_px == pytest.approx(43.52, abs=1e-9)
    assert budget.night == _night(20.0, 20.0)


def test_ir_range_does_not_change_the_day_geometry() -> None:
    without = compute_pixel_budget(**GEOMETRY, zone_distance_m=10.0, h_min_px=H_MIN_PX)
    within = compute_pixel_budget(**GEOMETRY, zone_distance_m=10.0, h_min_px=H_MIN_PX, ir_range_m=30.0)
    beyond = compute_pixel_budget(**GEOMETRY, zone_distance_m=10.0, h_min_px=H_MIN_PX, ir_range_m=1.0)
    assert within.model_height_px == without.model_height_px
    assert beyond.model_height_px == without.model_height_px
    assert beyond.night.zone_covered is False


def test_repeated_inputs_produce_identical_results() -> None:
    assert compute_pixel_budget(**GEOMETRY, zone_distance_m=10.0, h_min_px=H_MIN_PX) == compute_pixel_budget(
        **GEOMETRY, zone_distance_m=10.0, h_min_px=H_MIN_PX
    )


def test_module_import_does_not_load_hardware_or_pipeline_runtime() -> None:
    assert pixel_budget.__name__ == "vva_app.tools.pixel_budget"
    assert "hailo_platform" not in sys.modules
