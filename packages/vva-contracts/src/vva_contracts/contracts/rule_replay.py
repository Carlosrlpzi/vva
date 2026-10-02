"""RULE_REPLAY report: what would each candidate alert policy have done?

Why replay instead of reasoning on paper
----------------------------------------
If false-positive frames were independent with probability p, requiring k
consecutive positive frames would cut false triggers to roughly p^k (with
p = 0.05 and k = 3, ~1.3e-4 per window). Real false positives are *not*
independent: a bush the detector mistakes for a person keeps being mistaken
for many frames. The correlation makes the real reduction much smaller than
p^k, and only the logged frames can tell by how much. Replay measures it.

Denominators
------------
``covered_hours`` only counts time between consecutive frames that are at most
``max_frame_gap_s`` apart. If the pipeline was down for an hour, that hour is
not "an hour without alerts"; it is an hour without evidence.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from vva_contracts.contracts.base import (
    SCHEMA_VERSION,
    CameraId,
    InputFingerprint,
    NonNegativeFloat,
    StrictModel,
    UnitInterval,
    require_close,
    require_ordered,
    safe_ratio,
)
from vva_contracts.contracts.requests import EventPolicy

MAX_ALERTS_LISTED = 500


class Alert(StrictModel):
    camera_id: CameraId
    ts: float = Field(ge=0.0)
    frame_index: int = Field(ge=0)
    streak: int = Field(ge=1)  # consecutive positive frames at the moment of firing


class CameraAlerts(StrictModel):
    camera_id: CameraId
    n_alerts: int = Field(ge=0)


class EventMatching(StrictModel):
    """Comparison with human-annotated events (only when they were supplied).

    An event counts as detected when some alert on the same camera fires in
    ``[start_ts, end_ts + match_tolerance_s]``; latency is that first alert's
    ``ts - start_ts`` and is therefore >= 0 by construction.
    """

    n_events: int = Field(ge=0)
    n_detected: int = Field(ge=0)
    event_recall: UnitInterval | None
    n_alerts_inside_events: int = Field(ge=0)
    alert_precision: UnitInterval | None
    latency_s_p50: NonNegativeFloat | None
    latency_s_p90: NonNegativeFloat | None
    latency_s_max: NonNegativeFloat | None

    @model_validator(mode="after")
    def _recompute(self) -> EventMatching:
        if self.n_detected > self.n_events:
            raise ValueError("n_detected cannot exceed n_events")
        require_close("event_recall", self.event_recall, safe_ratio(self.n_detected, self.n_events))
        latencies = [self.latency_s_p50, self.latency_s_p90, self.latency_s_max]
        if (self.n_detected == 0) != all(v is None for v in latencies):
            raise ValueError("latency fields must be null exactly when nothing was detected")
        if self.n_detected > 0:
            require_ordered("latency quantiles", [v for v in latencies if v is not None])
        return self


class PolicyResult(StrictModel):
    policy: EventPolicy
    n_alerts: int = Field(ge=0)
    alerts_per_hour: NonNegativeFloat
    per_camera: list[CameraAlerts]
    alerts: list[Alert] = Field(max_length=MAX_ALERTS_LISTED)
    alerts_truncated: bool
    events: EventMatching | None

    @model_validator(mode="after")
    def _consistent(self) -> PolicyResult:
        if sum(c.n_alerts for c in self.per_camera) != self.n_alerts:
            raise ValueError(f"{self.policy.name}: per_camera alerts do not sum to n_alerts")
        if len(self.alerts) != min(self.n_alerts, MAX_ALERTS_LISTED):
            raise ValueError(f"{self.policy.name}: alerts must list the first min(n_alerts, 500)")
        if self.alerts_truncated != (self.n_alerts > MAX_ALERTS_LISTED):
            raise ValueError(f"{self.policy.name}: alerts_truncated is inconsistent")
        if self.events is not None:
            require_close(
                f"{self.policy.name}.alert_precision",
                self.events.alert_precision,
                safe_ratio(self.events.n_alerts_inside_events, self.n_alerts),
            )
            if self.events.n_alerts_inside_events > self.n_alerts:
                raise ValueError(f"{self.policy.name}: more alerts inside events than alerts")
        return self


class RuleReplayReport(StrictModel):
    contract_id: Literal["RuleReplayReport"] = "RuleReplayReport"
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    tool_version: str
    inputs: list[InputFingerprint]
    n_frames: int = Field(ge=2)
    cameras: list[CameraId] = Field(min_length=1)
    covered_hours: float = Field(gt=0.0)
    n_frame_gaps: int = Field(ge=0)
    max_frame_gap_s: float = Field(gt=0.0)
    match_tolerance_s: NonNegativeFloat
    reference_implementation: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    ground_truth_available: bool
    results: list[PolicyResult] = Field(min_length=1)

    @model_validator(mode="after")
    def _recompute(self) -> RuleReplayReport:
        for result in self.results:
            require_close(
                f"{result.policy.name}.alerts_per_hour", result.alerts_per_hour, result.n_alerts / self.covered_hours
            )
            if (result.events is not None) != self.ground_truth_available:
                raise ValueError(f"{result.policy.name}: events must be present iff ground truth was supplied")
            if sorted(c.camera_id for c in result.per_camera) != sorted(self.cameras):
                raise ValueError(f"{result.policy.name}: per_camera must list every camera exactly once")
        return self
