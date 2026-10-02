"""RULE_REPLAY runner.

Flow: frames JSONL -> per-camera frame sequences -> for each policy, one
``EventStateMachine`` per camera (the production class) -> alerts -> alert
load and, if labelled events are given, recall / precision / latency.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import fmean

from vva_contracts.contracts.logs import EventRow, FrameRow
from vva_contracts.contracts.requests import PolicySpec, RuleReplayRequest
from vva_contracts.contracts.rule_replay import (
    SECONDS_PER_HOUR,
    AlertRecord,
    CameraStats,
    EventOutcome,
    PolicyResult,
    RuleReplayReport,
)
from vva_contracts.core.state_machine import Alert, Detection, EventRule, EventStateMachine, Frame
from vva_contracts.core.stats import percentile, safe_ratio
from vva_contracts.errors import InvalidInput
from vva_contracts.paths import resolve_in_workspace
from vva_contracts.readers.jsonl import read_all
from vva_contracts.tasks.common import build_meta

# (line_number, event) pairs, so results can point back to the labels file.
LabelledEvents = list[tuple[int, EventRow]]


def load_frames(path: Path) -> dict[str, list[Frame]]:
    """Group frame rows by camera and check per-camera time order."""
    by_camera: dict[str, list[Frame]] = defaultdict(list)
    last_ts: dict[str, float] = {}
    for line_no, row in read_all(path, FrameRow):
        if row.camera_id in last_ts and row.ts < last_ts[row.camera_id]:
            # The state machine would raise too; failing here gives a line number.
            raise InvalidInput(f"{path.name}:{line_no}: timestamp goes backwards for {row.camera_id!r}")
        last_ts[row.camera_id] = row.ts
        detections = tuple(Detection(d.class_name, d.confidence, d.box) for d in row.detections)
        by_camera[row.camera_id].append(Frame(row.camera_id, row.frame_idx, row.ts, detections))
    return dict(sorted(by_camera.items()))


def replay_camera(policy: PolicySpec, camera_id: str, frames: list[Frame]) -> list[Alert]:
    """Run the production state machine over one camera's frames."""
    zone = policy.zones.get(camera_id)
    rule = EventRule(
        classes=frozenset(policy.classes),
        min_confidence=policy.min_confidence,
        min_consecutive_frames=policy.min_consecutive_frames,
        cooldown_s=policy.cooldown_s,
        max_gap_s=policy.max_gap_s,
        zone=tuple(zone) if zone is not None else None,
    )
    machine = EventStateMachine(rule)
    return [alert for frame in frames if (alert := machine.step(frame)) is not None]


def match_events(alerts: list[Alert], events: LabelledEvents, grace_s: float) -> tuple[EventOutcome, dict[int, int]]:
    """Score alerts against labelled intervals [start, end + grace].

    Returns the outcome and a map ``id(alert index) -> event line`` for the
    alerts that fall inside some event window.
    """
    alert_to_event: dict[int, int] = {}
    latencies: list[float] = []
    for line_no, event in events:
        window_start, window_end = event.start_ts, event.end_ts + grace_s
        first_hit: float | None = None
        for idx, alert in enumerate(alerts):
            if alert.camera_id != event.camera_id or not window_start <= alert.ts <= window_end:
                continue
            # setdefault: an alert inside overlapping events keeps the first.
            alert_to_event.setdefault(idx, line_no)
            first_hit = alert.ts if first_hit is None else min(first_hit, alert.ts)
        if first_hit is not None:
            latencies.append(first_hit - event.start_ts)  # >= 0 by construction
    outcome = EventOutcome(
        n_events=len(events),
        n_detected=len(latencies),
        recall=safe_ratio(len(latencies), len(events)),
        n_alerts_in_events=len(alert_to_event),
        alert_precision=safe_ratio(len(alert_to_event), len(alerts)),
        latency_mean_s=fmean(latencies) if latencies else None,
        latency_p50_s=percentile(latencies, 50) if latencies else None,
        latency_max_s=max(latencies) if latencies else None,
    )
    return outcome, alert_to_event


def replay_policy(
    policy: PolicySpec, frames: dict[str, list[Frame]], events: LabelledEvents | None, req: RuleReplayRequest
) -> PolicyResult:
    """Replay one policy over every camera and assemble its result."""
    per_camera: list[CameraStats] = []
    alerts: list[Alert] = []
    for camera_id, camera_frames in frames.items():
        camera_alerts = replay_camera(policy, camera_id, camera_frames)
        hours = (camera_frames[-1].ts - camera_frames[0].ts) / SECONDS_PER_HOUR
        per_camera.append(
            CameraStats(
                camera_id=camera_id,
                n_frames=len(camera_frames),
                observed_hours=hours,
                n_alerts=len(camera_alerts),
                alerts_per_hour=safe_ratio(len(camera_alerts), hours),
            )
        )
        alerts.extend(camera_alerts)
    # Time order across cameras (camera id breaks ties deterministically).
    alerts.sort(key=lambda a: (a.ts, a.camera_id))
    outcome, alert_to_event = match_events(alerts, events, req.alert_grace_s) if events is not None else (None, {})
    total_hours = sum(c.observed_hours for c in per_camera)
    listed = [
        AlertRecord(
            camera_id=a.camera_id,
            frame_idx=a.frame_idx,
            ts=a.ts,
            streak=a.streak,
            max_confidence=a.max_confidence,
            matched_event_line=alert_to_event.get(i),
        )
        for i, a in enumerate(alerts[: req.max_alerts_listed])
    ]
    return PolicyResult(
        policy=policy,
        n_frames=sum(c.n_frames for c in per_camera),
        observed_hours=total_hours,
        n_alerts=len(alerts),
        alerts_per_hour=safe_ratio(len(alerts), total_hours),
        per_camera=per_camera,
        ground_truth=outcome,
        alerts=listed,
        alerts_truncated=len(alerts) > req.max_alerts_listed,
    )


def run_rule_replay(req: RuleReplayRequest, workspace: Path) -> RuleReplayReport:
    """Entry point for task RULE_REPLAY."""
    frames_path = resolve_in_workspace(workspace, req.frames_log)
    frames = load_frames(frames_path)
    if sum((f[-1].ts - f[0].ts) for f in frames.values()) <= 0:
        raise InvalidInput("frames log spans zero seconds; alert rates would be undefined")
    inputs: list[tuple[str, Path]] = [("frames_log", frames_path)]
    events: LabelledEvents | None = None
    if req.events is not None:
        events_path = resolve_in_workspace(workspace, req.events)
        events = read_all(events_path, EventRow)
        unknown = {e.camera_id for _, e in events} - set(frames)
        if unknown:
            raise InvalidInput(f"events reference cameras absent from the frames log: {sorted(unknown)}")
        inputs.append(("events", events_path))
    for policy in req.policies:
        unknown_zones = set(policy.zones) - set(frames)
        if unknown_zones:
            # A zone for a camera that never appears is almost always a typo.
            raise InvalidInput(f"policy {policy.name!r} has zones for unknown cameras {sorted(unknown_zones)}")
    return RuleReplayReport(
        meta=build_meta(workspace, inputs),
        alert_grace_s=req.alert_grace_s,
        camera_ids=list(frames),
        policies=[replay_policy(p, frames, events, req) for p in req.policies],
    )
