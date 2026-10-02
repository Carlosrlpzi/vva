"""RULE_REPLAY report contract.

Why this contract exists
------------------------
The open decision "min_confidence 0.60, 3 frames, 45 s cooldown?" is a
trade-off between two measurable quantities:

* **alert load**: alerts per observed hour (how often your phone buzzes), and
* **event recall**: the fraction of real events (labelled intervals) that got
  at least one alert, and how late it arrived.

Replaying several candidate policies over the *same* recorded frames makes the
trade-off visible. The validator re-checks what the state machine promised:
alerts on one camera are at least ``cooldown_s`` apart, each alert had a
streak of at least ``min_consecutive_frames``, and the rates follow from the
counts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from vva_contracts.contracts.base import (
    SCHEMA_VERSION,
    Identifier,
    NonNegFloat,
    NonNegInt,
    Ratio,
    ReportMeta,
    StrictModel,
    close,
    require,
)
from vva_contracts.contracts.requests import PolicySpec
from vva_contracts.core.state_machine import TIME_EPSILON_S
from vva_contracts.core.stats import safe_ratio

SECONDS_PER_HOUR = 3600.0


class AlertRecord(StrictModel):
    """One alert the policy would have fired, with its trigger evidence."""

    camera_id: Identifier
    frame_idx: NonNegInt
    ts: NonNegFloat
    streak: int = Field(ge=1)
    max_confidence: Ratio
    # Line number (1-based) of the ground-truth event it falls in, if any.
    matched_event_line: int | None


class CameraStats(StrictModel):
    """Alert load on one camera."""

    camera_id: Identifier
    n_frames: NonNegInt
    observed_hours: NonNegFloat
    n_alerts: NonNegInt
    alerts_per_hour: NonNegFloat

    @model_validator(mode="after")
    def _rate(self) -> CameraStats:
        """alerts_per_hour = n_alerts / observed_hours."""
        require(close(self.alerts_per_hour, safe_ratio(self.n_alerts, self.observed_hours)), "alerts_per_hour mismatch")
        return self


class EventOutcome(StrictModel):
    """How the policy performed against labelled events."""

    n_events: NonNegInt
    n_detected: NonNegInt
    recall: Ratio
    n_alerts_in_events: NonNegInt
    alert_precision: Ratio
    latency_mean_s: NonNegFloat | None
    latency_p50_s: NonNegFloat | None
    latency_max_s: NonNegFloat | None

    @model_validator(mode="after")
    def _recompute(self) -> EventOutcome:
        """Recall and latency stats follow from the counts."""
        require(self.n_detected <= self.n_events, "n_detected > n_events")
        require(close(self.recall, safe_ratio(self.n_detected, self.n_events)), "recall mismatch")
        latencies = (self.latency_mean_s, self.latency_p50_s, self.latency_max_s)
        require(all(v is None for v in latencies) == (self.n_detected == 0), "latency null iff nothing detected")
        if self.latency_max_s is not None and self.latency_p50_s is not None and self.latency_mean_s is not None:
            require(self.latency_p50_s <= self.latency_max_s, "latency p50 > max")
            require(self.latency_mean_s <= self.latency_max_s + 1e-9, "latency mean > max")
        return self


class PolicyResult(StrictModel):
    """Replay outcome of one policy over every camera in the log."""

    policy: PolicySpec
    n_frames: NonNegInt
    observed_hours: NonNegFloat
    n_alerts: NonNegInt
    alerts_per_hour: NonNegFloat
    per_camera: list[CameraStats] = Field(min_length=1)
    ground_truth: EventOutcome | None
    alerts: list[AlertRecord]
    alerts_truncated: bool

    @model_validator(mode="after")
    def _recompute(self) -> PolicyResult:
        """Totals, rates and state-machine guarantees are re-checked."""
        require(self.n_alerts == sum(c.n_alerts for c in self.per_camera), "n_alerts != sum per camera")
        require(self.n_frames == sum(c.n_frames for c in self.per_camera), "n_frames != sum per camera")
        require(close(self.observed_hours, sum(c.observed_hours for c in self.per_camera)), "observed_hours mismatch")
        require(close(self.alerts_per_hour, safe_ratio(self.n_alerts, self.observed_hours)), "alerts_per_hour mismatch")
        if self.alerts_truncated:
            require(len(self.alerts) < self.n_alerts, "truncated list must be shorter than n_alerts")
        else:
            require(len(self.alerts) == self.n_alerts, "alert list length != n_alerts")
        self._check_state_machine_guarantees()
        if self.ground_truth is not None:
            gt = self.ground_truth
            require(gt.n_alerts_in_events <= self.n_alerts, "more in-event alerts than alerts")
            require(close(gt.alert_precision, safe_ratio(gt.n_alerts_in_events, self.n_alerts)), "precision mismatch")
        return self

    def _check_state_machine_guarantees(self) -> None:
        """Cooldown spacing, minimum streak and minimum confidence per alert.

        The listed alerts are a time-ordered prefix, so consecutive listed
        alerts of a camera are also consecutive in reality and the spacing
        check is valid even when the list is truncated.
        """
        last_ts: dict[str, float] = {}
        previous_ts = -1.0
        for alert in self.alerts:
            require(alert.ts >= previous_ts, "alerts must be listed in time order")
            previous_ts = alert.ts
            require(alert.streak >= self.policy.min_consecutive_frames, "alert fired below min_consecutive_frames")
            require(alert.max_confidence >= self.policy.min_confidence, "alert fired below min_confidence")
            if alert.camera_id in last_ts:
                gap = alert.ts - last_ts[alert.camera_id]
                # Same tolerance as the state machine, or valid replays would be rejected.
                ok = gap >= self.policy.cooldown_s - TIME_EPSILON_S
                require(ok, f"alerts on {alert.camera_id} closer than cooldown_s")
            last_ts[alert.camera_id] = alert.ts


class RuleReplayReport(StrictModel):
    """Accepted RULE_REPLAY artifact."""

    contract_id: Literal["RuleReplayReport"] = "RuleReplayReport"
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    meta: ReportMeta
    alert_grace_s: NonNegFloat
    camera_ids: list[Identifier] = Field(min_length=1)
    policies: list[PolicyResult] = Field(min_length=1)

    @model_validator(mode="after")
    def _same_footage(self) -> RuleReplayReport:
        """Every policy was replayed over the same frames and cameras."""
        for result in self.policies:
            require([c.camera_id for c in result.per_camera] == self.camera_ids, "camera set differs between policies")
            require(result.n_frames == self.policies[0].n_frames, "policies replayed different frame counts")
        return self
