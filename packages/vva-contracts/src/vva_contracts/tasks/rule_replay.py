"""RULE_REPLAY task: stream the frame log once through every candidate policy.

Memory stays flat for a 24 h log (millions of lines): frames are not kept,
only one ``EventStateMachine`` per policy and the alerts they fire.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

from vva_contracts.contracts.base import safe_ratio
from vva_contracts.contracts.records import FrameRecord, GroundTruthEvent, Zone, ZonesFile
from vva_contracts.contracts.requests import RuleReplayRequest
from vva_contracts.contracts.rule_replay import (
    MAX_ALERTS_LISTED,
    Alert,
    CameraAlerts,
    EventMatching,
    PolicyResult,
    RuleReplayReport,
)
from vva_contracts.errors import InputError
from vva_contracts.io import fingerprint, iter_jsonl, read_json
from vva_contracts.rules import REFERENCE_IMPLEMENTATION, EventStateMachine, FiredAlert
from vva_contracts.tasks._common import quantile, tool_version
from vva_contracts.workspace import resolve_in_workspace


def run(request: RuleReplayRequest, workspace: Path) -> RuleReplayReport:
    frames_path = resolve_in_workspace(workspace, request.frames, kind="file")
    inputs = [fingerprint(workspace, "frames", frames_path)]

    zones: dict[str, list[Zone]] = {}
    if request.zones is not None:
        zones_path = resolve_in_workspace(workspace, request.zones, kind="file")
        zones = dict(read_json(zones_path, ZonesFile, shown_as=request.zones).cameras)
        inputs.append(fingerprint(workspace, "zones", zones_path))

    machines = [EventStateMachine(p, zones, request.max_frame_gap_s) for p in request.policies]
    fired: list[list[FiredAlert]] = [[] for _ in machines]

    last_ts: dict[str, float] = {}
    covered_s = 0.0
    n_gaps = 0
    n_frames = 0
    for lineno, frame in iter_jsonl(frames_path, FrameRecord, shown_as=request.frames):
        previous = last_ts.get(frame.camera_id)
        if previous is not None:
            dt = frame.ts - previous
            if dt <= 0:
                raise InputError(
                    f"{request.frames}:{lineno}: ts not increasing for camera {frame.camera_id!r}; "
                    "the log must be in capture order per camera"
                )
            # Only intervals that are short enough count as observed time
            # (see "Denominators" in contracts/rule_replay.py).
            if dt <= request.max_frame_gap_s:
                covered_s += dt
            else:
                n_gaps += 1
        last_ts[frame.camera_id] = frame.ts
        n_frames += 1
        for machine, sink in zip(machines, fired, strict=True):
            alert = machine.step(frame)
            if alert is not None:
                sink.append(alert)

    if n_frames < 2 or covered_s <= 0:
        raise InputError("the frame log covers no time (need >= 2 frames per camera within max_frame_gap_s)")
    cameras = sorted(last_ts)
    unknown_zone_cams = sorted(set(zones) - set(cameras))
    if unknown_zone_cams:
        raise InputError(f"zones defined for cameras absent from the log: {unknown_zone_cams}")

    events: list[GroundTruthEvent] | None = None
    if request.ground_truth_events is not None:
        gt_path = resolve_in_workspace(workspace, request.ground_truth_events, kind="file")
        events = [e for _, e in iter_jsonl(gt_path, GroundTruthEvent, shown_as=request.ground_truth_events)]
        inputs.append(fingerprint(workspace, "ground_truth_events", gt_path))
        missing = sorted({e.camera_id for e in events} - set(cameras))
        if missing:
            raise InputError(f"ground truth refers to cameras absent from the log: {missing}")

    results = [
        _policy_result(machine, alerts, cameras, covered_s, events, request.match_tolerance_s)
        for machine, alerts in zip(machines, fired, strict=True)
    ]
    return RuleReplayReport(
        tool_version=tool_version(),
        inputs=inputs,
        n_frames=n_frames,
        cameras=cameras,
        covered_hours=covered_s / 3600.0,
        n_frame_gaps=n_gaps,
        max_frame_gap_s=request.max_frame_gap_s,
        match_tolerance_s=request.match_tolerance_s,
        reference_implementation=REFERENCE_IMPLEMENTATION,
        ground_truth_available=events is not None,
        results=results,
    )


def _policy_result(
    machine: EventStateMachine,
    alerts: list[FiredAlert],
    cameras: list[str],
    covered_s: float,
    events: list[GroundTruthEvent] | None,
    tolerance_s: float,
) -> PolicyResult:
    per_camera: dict[str, int] = defaultdict(int)
    for a in alerts:
        per_camera[a.camera_id] += 1
    return PolicyResult(
        policy=machine.policy,
        n_alerts=len(alerts),
        alerts_per_hour=len(alerts) / (covered_s / 3600.0),
        per_camera=[CameraAlerts(camera_id=c, n_alerts=per_camera[c]) for c in cameras],
        alerts=[
            Alert(camera_id=a.camera_id, ts=a.ts, frame_index=a.frame_index, streak=a.streak)
            for a in alerts[:MAX_ALERTS_LISTED]
        ],
        alerts_truncated=len(alerts) > MAX_ALERTS_LISTED,
        events=None if events is None else _match_events(alerts, events, tolerance_s),
    )


def _match_events(alerts: list[FiredAlert], events: list[GroundTruthEvent], tolerance_s: float) -> EventMatching:
    """Match alerts to annotated events per camera (window [start, end + tolerance])."""
    times_by_camera: dict[str, list[float]] = defaultdict(list)
    for a in alerts:
        times_by_camera[a.camera_id].append(a.ts)  # already ts-ordered per camera

    latencies: list[float] = []
    for event in events:
        times = times_by_camera.get(event.camera_id, [])
        # Binary search: first alert at or after the event start.
        i = bisect_left(times, event.start_ts)
        if i < len(times) and times[i] <= event.end_ts + tolerance_s:
            latencies.append(times[i] - event.start_ts)

    inside = 0
    for a in alerts:
        if any(e.camera_id == a.camera_id and e.start_ts <= a.ts <= e.end_ts + tolerance_s for e in events):
            inside += 1

    detected = len(latencies)
    return EventMatching(
        n_events=len(events),
        n_detected=detected,
        event_recall=safe_ratio(detected, len(events)),
        n_alerts_inside_events=inside,
        alert_precision=safe_ratio(inside, len(alerts)),
        latency_s_p50=quantile(latencies, 0.5) if latencies else None,
        latency_s_p90=quantile(latencies, 0.9) if latencies else None,
        latency_s_max=max(latencies) if latencies else None,
    )
