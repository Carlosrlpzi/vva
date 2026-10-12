"""Shared application types: :class:`FramePacket`, :class:`Detection`, :class:`Track`, :class:`Event`.

Why this module exists
----------------------
M1-M10 exchange these four objects across package boundaries. Defining them once
(here) means a shape change is one edit and the reviewer reads one contract
instead of a local copy per module.

Representation: plain frozen dataclasses. This is the project's chosen shape for
the shared types (dependency-light, value semantics), and it carries the NumPy
``image`` buffer directly, with no validation layer in between. Freezing stops
attribute rebinding; it does NOT make the buffer immutable.

Conventions used below
----------------------
- Coordinates ``xyxyn``: ``(x1, y1, x2, y2)`` normalised to ``[0, 1]`` in the
  ORIGINAL frame, with the detector's letterbox affine already undone (card
  "Shared types").
- Monotonic time: integer nanoseconds from ``time.monotonic_ns()``. It is not a
  date and its origin is undefined, so only differences inside one session are
  meaningful (decision D2). Never subtract it from, or compare it with, a UTC
  value.
- UTC time: timezone-aware :class:`datetime.datetime` in UTC (decision D2).
- No field is defaulted from a clock: the producer passes every timestamp,
  so a type can never silently mint "now".

This module performs no I/O, reads no YAML, loads no Hailo runtime and touches no
device at import time; it is importable by tests without a camera or NPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Original-frame normalised xyxy box: (x1, y1, x2, y2) in [0, 1].
BoxXyxyn = tuple[float, float, float, float]
# MVP scope: entry events only (guide v11.2, section 1.5).
EventKind = Literal["entry"]

_UTC_ZERO = timedelta(0)


def _require_utc(value: datetime, field: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a timezone-aware UTC datetime.

    D2 requires UTC to be explicit and unambiguous: a naive datetime or a
    non-UTC offset would make persisted history depend on the machine's local
    timezone, so both are rejected instead of silently accepted.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{field} must be timezone-aware UTC, got a naive datetime")
    if offset != _UTC_ZERO:
        raise ValueError(f"{field} must be UTC (offset 0), got offset {offset}")


def _require_ordered_box(box: BoxXyxyn) -> None:
    """Raise ``ValueError`` unless every coordinate is finite and ``x1<=x2, y1<=y2``.

    Finiteness is checked first because NaN/inf would silently poison every IoU
    and point-in-polygon computation downstream. The ``[0, 1]`` range is NOT
    checked here: only :class:`Detection` requires it, while a :class:`Track`
    prediction may legitimately fall outside the frame (see the class docstring).
    """
    if any(not math.isfinite(coordinate) for coordinate in box):
        raise ValueError(f"box_xyxyn must be finite, got {box}")
    x1, y1, x2, y2 = box
    if x1 > x2 or y1 > y2:
        raise ValueError(f"box_xyxyn must be xyxy with x1<=x2 and y1<=y2, got {box}")


@dataclass(frozen=True)
class FramePacket:
    """One decoded frame plus its two capture stamps and its image buffer.

    Attributes:
        camera_id: Stable identifier of the camera that produced the frame.
        frame_idx: Producer-assigned, monotonically increasing frame counter per
            camera/session. With ``camera_id`` it identifies the physical frame.
            The single-slot mailbox may re-serve the same packet, so a consumer
            that must not process one frame twice compares ``capture_mono_ns``
            with the last value it accepted (M2/§1.10); "newness" is a property
            of the delivery, not of this type.
        capture_mono_ns: Monotonic nanoseconds read at ingest, immediately after
            the decode read. Use it for internal intervals (tracker dt, frame
            age, staleness); it is never a date (D2).
        capture_utc: Timezone-aware UTC datetime read at ingest. It is the
            instant the DECODED frame was read on the Pi, not the sensor
            exposure time (the Pi cannot know it) and not the downstream
            processing instant (D2). Both stamps are taken close together, not
            atomically.
        image: BGR, ``uint8``, row-major ``HxWx3`` array owned by the producer.
            The packet is frozen, but the array is not. Lifetime contract:
            consumers must not modify the published buffer, and the producer
            must not overwrite or reuse that buffer while it is still referenced
            by a consumer (the mailbox replaces the slot with a new array rather
            than writing in place).

    """

    camera_id: str
    frame_idx: int
    capture_mono_ns: int
    capture_utc: datetime
    image: NDArray[np.uint8]

    def __post_init__(self) -> None:
        """Enforce the invariants that do not require the producer."""
        if self.frame_idx < 0:
            raise ValueError(f"frame_idx must be >= 0, got {self.frame_idx}")
        if self.capture_mono_ns < 0:
            raise ValueError(f"capture_mono_ns must be >= 0, got {self.capture_mono_ns}")
        _require_utc(self.capture_utc, "capture_utc")


@dataclass(frozen=True)
class Detection:
    """One detector output, in ORIGINAL-frame coordinates.

    Attributes:
        class_name: Detected class label (the MVP watches ``person`` only).
        confidence: Detector confidence in ``[0, 1]``.
        box_xyxyn: ``(x1, y1, x2, y2)`` normalised to ``[0, 1]`` in the original
            frame, letterbox already undone by the detector (card). Every
            coordinate must be finite and inside ``[0, 1]``, with ``x1 <= x2``
            and ``y1 <= y2`` (zero-area boxes are allowed, as the card does not
            forbid them).

    No timing is carried: the enclosing frame's ``capture_mono_ns`` /
    ``capture_utc`` are the observation time of every detection in it (D2), so
    the stamps are not duplicated per detection.

    """

    class_name: str
    confidence: float
    box_xyxyn: BoxXyxyn

    def __post_init__(self) -> None:
        """Reject bad confidence and non-finite, out-of-range or unordered boxes."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        _require_ordered_box(self.box_xyxyn)
        if any(not 0.0 <= coordinate <= 1.0 for coordinate in self.box_xyxyn):
            raise ValueError(f"box_xyxyn must be normalised to [0, 1], got {self.box_xyxyn}")


@dataclass(frozen=True)
class Track:
    """A tracked object as M5 hands it downstream.

    Attributes:
        track_id: Tracker-assigned identity, stable across a track's lifetime.
        class_name: Tracked class label.
        box_xyxyn: Associated/predicted box, original-frame normalised xyxy. A
            prediction may temporarily lead the observation; this type records
            the box only, not whether it came from a match or a Kalman predict
            (that distinction belongs to M5/D4). Only finiteness and order are
            enforced: M5 may predict a box partly outside the frame, and the
            card does not define how such a box is represented, so this type
            imposes no ``[0, 1]`` range and does no clipping (open M5 decision).
        hits: Number of successful associations so far.
        is_confirmed: True once ``hits`` reaches ``min_confirmed_hits`` (owned by
            ``configs/tracker.yaml``). This is the ONLY confirmation signal
            downstream; M6 must read it and never re-count hits.
        last_seen_mono: Monotonic nanoseconds, same domain as
            ``FramePacket.capture_mono_ns``, of the observation time of the last
            SUCCESSFULLY ASSOCIATED frame. It is neither the execution instant of
            the match nor the time of a Kalman prediction.
        mean_confidence: Running mean of the associated detection confidences,
            in ``[0, 1]``; M6 eligibility reads it.
        age_seconds: Monotonic seconds from the first observation that created the
            track up to the logical time of the observation/update this snapshot
            represents (approved option a, 2026-10-09). Computed from marks of the
            same temporal domain; in replay it uses the replayed logical time, not
            real execution duration. It does NOT reset when ``is_confirmed``
            becomes true, and it is NOT time-without-observation nor zone dwell.
            It can grow during a gap while the track stays alive, and that growth
            does not attest continuous presence.

    ``expiration_reason`` is intentionally absent: it is deferred to M5's design
    (decision P2); M5 must define the reasons, their representation and producer,
    and how (whether) M6 consumes them.

    """

    track_id: int
    class_name: str
    box_xyxyn: BoxXyxyn
    hits: int
    is_confirmed: bool
    last_seen_mono: int
    mean_confidence: float
    age_seconds: float

    def __post_init__(self) -> None:
        """Reject negative counters, bad confidence/age and non-finite boxes."""
        if self.track_id < 0:
            raise ValueError(f"track_id must be >= 0, got {self.track_id}")
        if self.hits < 0:
            raise ValueError(f"hits must be >= 0, got {self.hits}")
        if self.last_seen_mono < 0:
            raise ValueError(f"last_seen_mono must be >= 0, got {self.last_seen_mono}")
        if not 0.0 <= self.mean_confidence <= 1.0:
            raise ValueError(f"mean_confidence must be in [0, 1], got {self.mean_confidence}")
        if not math.isfinite(self.age_seconds) or self.age_seconds < 0.0:
            raise ValueError(f"age_seconds must be finite and >= 0, got {self.age_seconds}")
        _require_ordered_box(self.box_xyxyn)


@dataclass(frozen=True)
class Event:
    """A production entry event emitted by M6 (MVP: ``kind == "entry"``).

    Attributes:
        event_id: Unique identifier assigned by M6.
        camera_id: Camera where the event happened.
        zone_id: Zone polygon that was entered.
        track_id: Track that triggered the event.
        kind: ``"entry"`` only in the MVP.
        capture_mono_ns: Observation time (monotonic nanoseconds) of the frame at
            which M6 determined the emission conditions hold, copied from that
            frame's :class:`FramePacket` (decision D5/P1).
        capture_utc: The same observation instant as an aware UTC datetime.

    ``capture_mono_ns``/``capture_utc`` are NOT the real execution instant when
    the event object is built or published, and NOT the first observed entry in
    the zone if that happened earlier. If a future contract adds ``observed_at``
    and ``emitted_at``, those meanings stay separate; this task adds neither an
    ``emitted_at`` nor a first-entry field. Persistence and notification
    throttling use the execution clock, not these stamps (D2/M7/M8).

    """

    event_id: str
    camera_id: str
    zone_id: str
    track_id: int
    kind: EventKind
    capture_mono_ns: int
    capture_utc: datetime

    def __post_init__(self) -> None:
        """Enforce the MVP kind, non-negative time and an aware UTC stamp."""
        if self.kind != "entry":
            raise ValueError(f"MVP kind is 'entry' only, got {self.kind!r}")
        if self.capture_mono_ns < 0:
            raise ValueError(f"capture_mono_ns must be >= 0, got {self.capture_mono_ns}")
        _require_utc(self.capture_utc, "capture_utc")
