"""Unit tests for ``vva_app.tools.pixel_budget``.

No camera, Hailo runtime, network, ffmpeg or YAML is used: the module under test is a pure
calculator. Closed-form values come from the pixel-budget worked table in guide section 0.
"""

from __future__ import annotations

import math
import sys

import pytest

from vva_app.tools import pixel_budget
from vva_app.tools.pixel_budget import (
    MODEL_INPUT_SIZE_PX,
    IrRangeStatus,
    PixelBudget,
    compute_pixel_budget,
    estimate_model_height_px,
    estimate_source_height_px,
    ir_range_status,
    letterbox_scale,
)

# Guide section 0 reference geometry: 4.0 mm lens, 1/3" sensor (3.0 mm), 1.7 m person, 5 m.
GEOMETRY = {
    "focal_length_mm": 4.0,
    "sensor_height_mm": 3.0,
    "target_height_m": 1.7,
    "distance_m": 5.0,
}
REFERENCE = {"image_width_px": 640, "image_height_px": 360, **GEOMETRY}


# ------------------------------------------------------------------ reference geometry
def test_reference_case_matches_the_guide_worked_table() -> None:
    budget = compute_pixel_budget(**REFERENCE)
    assert budget.source_height_px == pytest.approx(163.2, abs=1e-9)
    assert budget.letterbox_scale == pytest.approx(1.0, abs=1e-12)
    assert budget.model_height_px == pytest.approx(163.2, abs=1e-9)
    assert budget.ir_range_status is IrRangeStatus.NOT_EVALUATED


def test_doubling_distance_halves_source_height() -> None:
    near = estimate_source_height_px(360, **GEOMETRY)
    far = estimate_source_height_px(360, **{**GEOMETRY, "distance_m": 10.0})
    assert near == pytest.approx(2.0 * far, rel=1e-12)


def test_doubling_focal_length_doubles_source_height() -> None:
    base = estimate_source_height_px(360, **GEOMETRY)
    doubled = estimate_source_height_px(360, **{**GEOMETRY, "focal_length_mm": 8.0})
    assert doubled == pytest.approx(2.0 * base, rel=1e-12)


def test_1280x720_halves_the_scale_and_keeps_the_model_height() -> None:
    budget = compute_pixel_budget(image_width_px=1280, image_height_px=720, **GEOMETRY)
    assert budget.source_height_px == pytest.approx(326.4, abs=1e-9)
    assert budget.letterbox_scale == pytest.approx(0.5, abs=1e-12)
    assert budget.model_height_px == pytest.approx(163.2, abs=1e-9)


# ----------------------------------------------------------------- letterbox scale
def test_letterbox_scale_for_horizontal_vertical_and_square_frames() -> None:
    assert letterbox_scale(640, 360) == pytest.approx(1.0, abs=1e-12)  # 16:9, width fits exactly
    assert letterbox_scale(360, 640) == pytest.approx(1.0, abs=1e-12)  # portrait, height fits
    assert letterbox_scale(640, 640) == pytest.approx(1.0, abs=1e-12)  # square, no padding
    assert letterbox_scale(1280, 480) == pytest.approx(0.5, abs=1e-12)  # width is dominant


def test_letterbox_scale_prefers_the_dominant_side_in_a_5_4_frame() -> None:
    # 720x576 is a generic 5:4 format, not a camera model: width dominates, so min() must
    # pick 640/720 over 640/576.
    budget = compute_pixel_budget(image_width_px=720, image_height_px=576, **GEOMETRY)
    assert budget.letterbox_scale == pytest.approx(640 / 720, abs=1e-12)
    assert budget.source_height_px == pytest.approx(261.12, abs=1e-9)
    assert budget.model_height_px == pytest.approx(232.11, abs=1e-2)


def test_letterbox_scale_allows_upscale_below_the_model_input() -> None:
    budget = compute_pixel_budget(image_width_px=320, image_height_px=200, **GEOMETRY)
    assert budget.letterbox_scale == pytest.approx(2.0, abs=1e-12)  # 640/320 dominates
    assert budget.source_height_px == pytest.approx(90.66666666666667, abs=1e-9)
    assert budget.model_height_px > budget.source_height_px


def test_model_input_size_is_a_named_constant_and_overridable() -> None:
    assert MODEL_INPUT_SIZE_PX == 640
    assert letterbox_scale(1280, 720) == pytest.approx(0.5, abs=1e-12)
    assert letterbox_scale(1280, 720, model_input_size_px=320) == pytest.approx(0.25, abs=1e-12)


def test_letterbox_scale_rejects_invalid_model_input_size() -> None:
    with pytest.raises(ValueError, match="model_input_size_px"):
        letterbox_scale(640, 360, model_input_size_px=0)
    with pytest.raises(ValueError, match="model_input_size_px"):
        letterbox_scale(640, 360, model_input_size_px=640.0)


# ------------------------------------------------------------------- input validation
@pytest.mark.parametrize("bad", [0, -1, 360.5, 640.0, True, False])
def test_dimensions_reject_zero_negative_fractional_and_boolean(bad: object) -> None:
    with pytest.raises(ValueError, match="image_width_px"):
        letterbox_scale(bad, 360)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="image_height_px"):
        letterbox_scale(640, bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1, 360.5, 640.0, True])
def test_source_height_rejects_invalid_image_height(bad: object) -> None:
    with pytest.raises(ValueError, match="image_height_px"):
        estimate_source_height_px(bad, **GEOMETRY)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["focal_length_mm", "sensor_height_mm", "target_height_m", "distance_m"])
@pytest.mark.parametrize("bad", [0.0, -1.0, True, False, math.nan, math.inf, -math.inf, "4", "4 mm", None])
def test_magnitudes_reject_zero_negative_boolean_non_finite_and_string(field: str, bad: object) -> None:
    kwargs = {**GEOMETRY, field: bad}
    with pytest.raises(ValueError, match=field):
        estimate_source_height_px(360, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0.0, -1.0, True, math.nan, math.inf, "15 m", None])
def test_model_height_rejects_invalid_inputs(bad: object) -> None:
    with pytest.raises(ValueError):
        estimate_model_height_px(bad, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        estimate_model_height_px(100.0, bad)  # type: ignore[arg-type]


def test_overflow_to_non_finite_is_rejected() -> None:
    with pytest.raises(ValueError, match="h_source_px"):
        estimate_source_height_px(
            360,
            focal_length_mm=1e308,
            sensor_height_mm=1e-308,
            target_height_m=1.7,
            distance_m=5.0,
        )


def test_denominator_underflow_is_rejected() -> None:
    with pytest.raises(ValueError, match="underflow"):
        estimate_source_height_px(
            360,
            focal_length_mm=4.0,
            sensor_height_mm=5e-324,
            target_height_m=1.7,
            distance_m=0.1,
        )


def test_pixel_budget_rejects_non_positive_fields() -> None:
    for field in ("source_height_px", "letterbox_scale", "model_height_px"):
        values = {"source_height_px": 1.0, "letterbox_scale": 1.0, "model_height_px": 1.0, field: -1.0}
        with pytest.raises(ValueError):
            PixelBudget(**values)  # type: ignore[arg-type]


# ------------------------------------------------------------------------- IR range
def test_ir_range_status_covers_less_equal_and_greater() -> None:
    assert ir_range_status(20.0, 30.0) is IrRangeStatus.WITHIN_RANGE
    assert ir_range_status(30.0, 30.0) is IrRangeStatus.WITHIN_RANGE  # equality is inclusive
    assert ir_range_status(30.0001, 30.0) is IrRangeStatus.BEYOND_RANGE


def test_ir_range_status_is_not_evaluated_without_a_range() -> None:
    assert ir_range_status(20.0) is IrRangeStatus.NOT_EVALUATED
    assert ir_range_status(20.0, None) is IrRangeStatus.NOT_EVALUATED


@pytest.mark.parametrize("bad", [0.0, -1.0, True, math.nan, math.inf, "15 m"])
def test_ir_range_rejects_invalid_values(bad: object) -> None:
    with pytest.raises(ValueError, match="ir_range_m"):
        ir_range_status(5.0, bad)  # type: ignore[arg-type]


def test_ir_range_beyond_reach_does_not_change_the_geometry() -> None:
    without = compute_pixel_budget(**REFERENCE)
    within = compute_pixel_budget(**REFERENCE, ir_range_m=30.0)
    beyond = compute_pixel_budget(**REFERENCE, ir_range_m=1.0)  # distance 5 m > 1 m
    assert within.ir_range_status is IrRangeStatus.WITHIN_RANGE
    assert beyond.ir_range_status is IrRangeStatus.BEYOND_RANGE
    assert beyond.source_height_px == without.source_height_px
    assert beyond.model_height_px == without.model_height_px
    assert within.source_height_px == without.source_height_px
    assert within.model_height_px == without.model_height_px


# ---------------------------------------------------------------- invariance / purity
def test_repeated_inputs_produce_identical_results() -> None:
    assert compute_pixel_budget(**REFERENCE) == compute_pixel_budget(**REFERENCE)


def test_source_height_depends_on_image_height_not_width() -> None:
    tall = compute_pixel_budget(image_width_px=640, image_height_px=360, **GEOMETRY)
    wide = compute_pixel_budget(image_width_px=1280, image_height_px=360, **GEOMETRY)
    assert wide.source_height_px == tall.source_height_px
    assert tall.letterbox_scale == pytest.approx(1.0, abs=1e-12)
    assert wide.letterbox_scale == pytest.approx(0.5, abs=1e-12)  # the dominant side changed
    # Only the model height follows the scale: 163.2 px in the source * 0.5.
    assert wide.model_height_px == pytest.approx(81.6, abs=1e-9)


def test_module_import_does_not_load_hardware_or_pipeline_runtime() -> None:
    assert pixel_budget.__name__ == "vva_app.tools.pixel_budget"
    assert "hailo_platform" not in sys.modules
