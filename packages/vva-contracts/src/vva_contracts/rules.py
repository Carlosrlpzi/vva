"""Reference implementation of the alert rule: zone + confidence + k frames + cooldown.

Why the replay engine lives here and not only in the pipeline
-------------------------------------------------------------
If RULE_REPLAY used its own copy of the state machine, the thresholds it
recommends would be validated against code that is not the code that runs on
the Pi. ``src/events`` in the surveillance project must therefore import
``EventStateMachine`` from this module (or prove equivalence with a test that
feeds both the same frames). One implementation, two callers.

Semantics, per camera, frame by frame (frames must arrive in ts order)
----------------------------------------------------------------------
1. A frame is *positive* when at least one detection has a class in
   ``policy.classes``, ``confidence >= policy.min_confidence``, and its
   bottom-center anchor inside one of the camera's zones (or anywhere in the
   frame when the camera has no zones).
2. ``streak`` counts consecutive positive frames. A negative frame resets it.
   A time gap larger than ``max_frame_gap_s`` also resets it: frames lost to
   a stalled stream are not evidence of continued presence.
3. An alert fires when ``streak >= min_consecutive_frames`` and either no
   alert fired yet or ``ts - last_alert_ts >= cooldown_s``. While presence
   continues, the alert therefore repeats once per cooldown window.

The per-frame logic is plain Python (no vectorization): it processes one
frame at a time, exactly as the live pipeline does, so replay and production
follow the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vva_contracts.contracts.records import FrameRecord, Zone
from vva_contracts.contracts.requests import EventPolicy
from vva_contracts.geometry import anchor_point, point_in_polygon

REFERENCE_IMPLEMENTATION = "vva_contracts.rules.EventStateMachine/1"


@dataclass(frozen=True)
class FiredAlert:
    camera_id: str
    ts: float
    frame_index: int
    streak: int


@dataclass
class _CameraState:
    last_ts: float | None = None
    streak: int = 0
    last_alert_ts: float | None = None


@dataclass
class EventStateMachine:
    """Stateful rule evaluator for one policy across any number of cameras."""

    policy: EventPolicy
    zones: dict[str, list[Zone]]
    max_frame_gap_s: float
    _cameras: dict[str, _CameraState] = field(default_factory=dict)

    def is_positive(self, frame: FrameRecord) -> bool:
        """Rule step 1 for a single frame."""
        camera_zones = self.zones.get(frame.camera_id, [])
        for det in frame.detections:
            if det.class_name not in self.policy.classes:
                continue
            if det.confidence < self.policy.min_confidence:
                continue
            if not camera_zones:
                return True  # no zones configured: the whole frame counts
            x, y = anchor_point(det.bbox_xyxy, frame.width, frame.height)
            if any(point_in_polygon(x, y, zone.polygon) for zone in camera_zones):
                return True
        return False

    def step(self, frame: FrameRecord) -> FiredAlert | None:
        """Feed one frame; return the alert it triggers, if any.

        Raises:
            ValueError: if ``ts`` does not strictly increase for this camera.
        """
        state = self._cameras.setdefault(frame.camera_id, _CameraState())
        if state.last_ts is not None:
            dt = frame.ts - state.last_ts
            if dt <= 0:
                raise ValueError(f"camera {frame.camera_id}: ts must strictly increase (got dt={dt})")
            if dt > self.max_frame_gap_s:
                state.streak = 0  # rule step 2: a gap breaks the chain of evidence
        state.last_ts = frame.ts

        state.streak = state.streak + 1 if self.is_positive(frame) else 0

        if state.streak < self.policy.min_consecutive_frames:
            return None
        if state.last_alert_ts is not None and frame.ts - state.last_alert_ts < self.policy.cooldown_s:
            return None  # rule step 3: still inside the cooldown window
        state.last_alert_ts = frame.ts
        return FiredAlert(frame.camera_id, frame.ts, frame.frame_index, state.streak)
