"""Deterministic alert state machine (one instance per camera).

Why this module exists
----------------------
The project rule is "no LLM decides whether an alert fires". The decision is
this small state machine, and it lives here, in a pure module, so that:

* the production pipeline (``src/events`` of the VVA app) imports it, and
* RULE_REPLAY runs *the same object* over recorded detection logs.

If replay and production used two implementations, tuning thresholds offline
would tell you nothing about the live system.

Rule semantics
--------------
A frame is **positive** if at least one detection has a watched class,
confidence >= ``min_confidence`` and (when a zone is set) its bottom-centre
inside the zone polygon.

``streak`` counts consecutive positive frames. It resets to 0 on a negative
frame, and restarts at 1 if the time gap since the previous frame exceeds
``max_gap_s`` (dropped frames or a stream reconnect break continuity: two
detections 30 s apart are not "consecutive").

An alert fires when ``streak >= min_consecutive_frames`` AND at least
``cooldown_s`` seconds have passed since this camera's previous alert.
Continuous presence therefore re-alerts once per cooldown window.

Why k consecutive frames: if per-frame false positives were independent with
probability p, k in a row would occur with probability ~p^k. They are NOT
independent in practice (a bush mistaken for a person stays a bush), which is
exactly why thresholds must be tuned by replaying real logs, not on paper.
"""

from __future__ import annotations

from dataclasses import dataclass

from vva_contracts.core.geometry import anchor_point, point_in_polygon

# Tolerance for comparing time differences. Timestamps are binary floats, so
# 10.2 - 5.2 evaluates to 4.999999999999999 and a 5 s cooldown would wrongly
# still be "active". 1 microsecond is far below any camera's frame spacing
# (100 ms at 10 fps), so it absorbs rounding without changing real behaviour.
TIME_EPSILON_S = 1e-6


@dataclass(frozen=True)
class Detection:
    """One detector output in normalised xyxy coordinates."""

    class_name: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Frame:
    """All detections of one processed frame (an empty tuple is meaningful)."""

    camera_id: str
    frame_idx: int
    ts: float  # seconds since epoch, monotonic per camera
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class EventRule:
    """Tunable policy. Every field is a decision documented in the project."""

    classes: frozenset[str]
    min_confidence: float
    min_consecutive_frames: int
    cooldown_s: float
    max_gap_s: float
    zone: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class Alert:
    """An alert decision plus the evidence that triggered it."""

    camera_id: str
    frame_idx: int
    ts: float
    streak: int
    max_confidence: float


@dataclass
class EventStateMachine:
    """Per-camera state. Feed frames in timestamp order through ``step``."""

    rule: EventRule
    streak: int = 0
    last_ts: float | None = None
    last_alert_ts: float | None = None

    def _qualifying_confidence(self, frame: Frame) -> float | None:
        """Highest confidence among detections that satisfy the rule, or None."""
        best: float | None = None
        for det in frame.detections:
            if det.class_name not in self.rule.classes or det.confidence < self.rule.min_confidence:
                continue
            if self.rule.zone is not None and not point_in_polygon(*anchor_point(det.box), self.rule.zone):
                continue
            best = det.confidence if best is None else max(best, det.confidence)
        return best

    def step(self, frame: Frame) -> Alert | None:
        """Advance one frame; return an Alert when the rule fires."""
        if self.last_ts is not None and frame.ts < self.last_ts:
            # Fail fast: silently re-ordering would hide a clock or log bug.
            raise ValueError(f"timestamps went backwards on camera {frame.camera_id!r}")
        gap_broken = self.last_ts is not None and (frame.ts - self.last_ts) > self.rule.max_gap_s + TIME_EPSILON_S
        self.last_ts = frame.ts

        confidence = self._qualifying_confidence(frame)
        if confidence is None:
            self.streak = 0
            return None
        # A large gap means these are not consecutive observations: restart.
        self.streak = 1 if gap_broken else self.streak + 1

        if self.streak < self.rule.min_consecutive_frames:
            return None
        if self.last_alert_ts is not None and frame.ts - self.last_alert_ts < self.rule.cooldown_s - TIME_EPSILON_S:
            return None  # still inside the cooldown window -> deduplicated
        self.last_alert_ts = frame.ts
        return Alert(frame.camera_id, frame.frame_idx, frame.ts, self.streak, confidence)
