"""Unit tests for the shared application types in ``vva_app.common.types``.

These tests exercise only the types: construction, the invariants that have
runtime validation, frozen semantics, and the absence of clock defaults or
hardware imports. They run without a camera, Hailo, network or ffmpeg.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from vva_app.common.types import Detection, Event, FramePacket, Track

NAIVE = datetime(2026, 10, 9, 12, 0, 0)
IMAGE = np.zeros((4, 6, 3), dtype=np.uint8)


def _utc(year=2026, month=10, day=9, hour=12, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


# --------------------------------------------------------------------- FramePacket
def test_frame_packet_valid_construction() -> None:
    packet = FramePacket("front", 7, 123_456, _utc(), IMAGE)
    assert packet.camera_id == "front"
    assert packet.frame_idx == 7
    assert packet.capture_mono_ns == 123_456
    assert packet.capture_utc == _utc()
    # The buffer is carried by reference: no defensive copy is made.
    assert packet.image is IMAGE


def test_frame_packet_rejects_naive_utc() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        FramePacket("front", 0, 0, NAIVE, IMAGE)


def test_frame_packet_rejects_non_utc_offset() -> None:
    aware_not_utc = datetime(2026, 10, 9, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    with pytest.raises(ValueError, match="must be UTC"):
        FramePacket("front", 0, 0, aware_not_utc, IMAGE)


def test_frame_packet_rejects_negative_counters() -> None:
    with pytest.raises(ValueError, match="frame_idx"):
        FramePacket("front", -1, 0, _utc(), IMAGE)
    with pytest.raises(ValueError, match="capture_mono_ns"):
        FramePacket("front", 0, -1, _utc(), IMAGE)


def test_temporal_fields_have_no_clock_default() -> None:
    """Producers must supply the stamps; a type never mints 'now' by itself."""
    frame_fields = {field.name: field for field in dataclasses.fields(FramePacket)}
    assert frame_fields["capture_mono_ns"].default is dataclasses.MISSING
    assert frame_fields["capture_utc"].default is dataclasses.MISSING
    event_fields = {field.name: field for field in dataclasses.fields(Event)}
    assert event_fields["capture_mono_ns"].default is dataclasses.MISSING
    assert event_fields["capture_utc"].default is dataclasses.MISSING


def test_frame_packet_is_frozen_but_buffer_is_shared() -> None:
    packet = FramePacket("front", 1, 10, _utc(), IMAGE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        packet.frame_idx = 2  # type: ignore[misc]
    # The array is the producer's buffer, not a copy; consumers must treat it as read-only.
    assert packet.image is IMAGE


def test_frame_packet_does_not_shield_the_image_buffer() -> None:
    """Frozen stops attribute rebinding, not in-place mutation of the shared array."""
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    packet = FramePacket("front", 1, 10, _utc(), image)
    image[0, 0, 0] = 255  # the buffer is shared and mutable; the dataclass cannot prevent it
    assert packet.image is image
    assert packet.image[0, 0, 0] == 255


# ---------------------------------------------------------------------- Detection
def test_detection_valid_coordinates() -> None:
    detection = Detection("person", 0.83, (0.1, 0.2, 0.3, 0.9))
    assert detection.box_xyxyn == (0.1, 0.2, 0.3, 0.9)
    assert detection.confidence == 0.83


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_detection_rejects_confidence_out_of_range(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        Detection("person", confidence, (0.0, 0.0, 1.0, 1.0))


@pytest.mark.parametrize("box", [(0.6, 0.2, 0.3, 0.9), (0.1, 0.9, 0.3, 0.2)])
def test_detection_rejects_unordered_box(box: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError, match="box_xyxyn"):
        Detection("person", 0.5, box)


@pytest.mark.parametrize("box", [(-0.1, 0.0, 0.5, 0.5), (0.0, 0.0, 1.1, 0.5), (0.0, -0.2, 0.5, 0.5)])
def test_detection_rejects_out_of_range_box(box: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError, match="box_xyxyn"):
        Detection("person", 0.5, box)


@pytest.mark.parametrize(
    "box",
    [(math.nan, 0.0, 0.5, 0.5), (0.0, 0.0, math.inf, 0.5), (0.0, 0.0, 0.5, -math.inf)],
)
def test_detection_rejects_non_finite_box(box: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        Detection("person", 0.5, box)


def test_detection_allows_zero_area_box() -> None:
    """The card does not forbid degenerate boxes, so the current policy is kept."""
    detection = Detection("person", 0.5, (0.4, 0.4, 0.4, 0.4))
    assert detection.box_xyxyn == (0.4, 0.4, 0.4, 0.4)


# -------------------------------------------------------------------------- Track
def test_track_valid_construction() -> None:
    track = Track(3, "person", (0.1, 0.1, 0.2, 0.3), 5, True, 123_456, 0.7, 1.2)
    assert track.is_confirmed is True
    assert track.age_seconds == 1.2


def test_track_rejects_invalid_state() -> None:
    with pytest.raises(ValueError, match="track_id"):
        Track(-1, "person", (0, 0, 1, 1), 0, False, 0, 0.5, 0.0)
    with pytest.raises(ValueError, match="hits"):
        Track(1, "person", (0, 0, 1, 1), -1, False, 0, 0.5, 0.0)
    with pytest.raises(ValueError, match="last_seen_mono"):
        Track(1, "person", (0, 0, 1, 1), 0, False, -1, 0.5, 0.0)
    with pytest.raises(ValueError, match="mean_confidence"):
        Track(1, "person", (0, 0, 1, 1), 0, False, 0, 1.5, 0.0)
    with pytest.raises(ValueError, match="age_seconds"):
        Track(1, "person", (0, 0, 1, 1), 0, False, 0, 0.5, -1.0)
    with pytest.raises(ValueError, match="box_xyxyn"):
        Track(1, "person", (0.9, 0, 0.1, 1), 0, False, 0, 0.5, 0.0)


@pytest.mark.parametrize("age_seconds", [-0.1, math.nan, math.inf])
def test_track_rejects_non_finite_or_negative_age(age_seconds: float) -> None:
    with pytest.raises(ValueError, match="age_seconds"):
        Track(1, "person", (0.0, 0.0, 1.0, 1.0), 0, False, 0, 0.5, age_seconds)


@pytest.mark.parametrize("box", [(math.nan, 0.0, 1.0, 1.0), (0.0, 0.0, math.inf, 1.0)])
def test_track_rejects_non_finite_box(box: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        Track(1, "person", box, 0, False, 0, 0.5, 0.0)


def test_track_allows_a_predicted_box_outside_the_frame() -> None:
    # No [0,1] range and no clipping: M5 may predict past the frame edge, and the
    # card does not define how such a box is represented (open M5 decision).
    track = Track(1, "person", (-0.2, 0.0, 1.2, 1.0), 0, False, 0, 0.5, 0.0)
    assert track.box_xyxyn == (-0.2, 0.0, 1.2, 1.0)


def test_track_has_no_deferred_expiration_reason() -> None:
    """P2: the field is deferred to M5; shared types must not carry it yet."""
    assert "expiration_reason" not in {field.name for field in dataclasses.fields(Track)}


# -------------------------------------------------------------------------- Event
def test_event_keeps_the_observation_time_verbatim() -> None:
    stamp = datetime(2026, 10, 9, 12, 0, 0, 500_000, tzinfo=UTC)
    event = Event("ev1", "front", "door", 3, "entry", 999_888_777, stamp)
    # The stamps are the ones passed in, not a value re-read from a clock (D5/P1).
    assert event.capture_utc is stamp
    assert event.capture_mono_ns == 999_888_777
    assert event.kind == "entry"


def test_event_rejects_naive_or_unknown_kind() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Event("ev", "front", "door", 1, "entry", 0, NAIVE)
    with pytest.raises(ValueError, match="kind"):
        Event("ev", "front", "door", 1, "crossing", 0, _utc())


# --------------------------------------------------------------------- Import safety
def test_import_does_not_load_hardware_runtime() -> None:
    assert "vva_app.common.types" in sys.modules
    assert "hailo_platform" not in sys.modules
