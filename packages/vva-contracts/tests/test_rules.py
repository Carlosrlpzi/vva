"""Event state machine semantics and RULE_REPLAY end to end."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import CliRunner

from vva_contracts.contracts.records import FrameRecord, Zone
from vva_contracts.contracts.requests import EventPolicy
from vva_contracts.rules import EventStateMachine

POLICY = EventPolicy(name="p", classes=["person"], min_confidence=0.6, min_consecutive_frames=3, cooldown_s=45.0)


def frame(ts: float, conf: float | None, box: list[float] | None = None, cam: str = "front") -> FrameRecord:
    dets = (
        [] if conf is None else [{"class_name": "person", "confidence": conf, "bbox_xyxy": box or [100, 50, 140, 200]}]
    )
    return FrameRecord.model_validate(
        {"camera_id": cam, "frame_index": int(ts * 10), "ts": ts, "width": 640, "height": 360, "detections": dets}
    )


def test_needs_k_consecutive_frames() -> None:
    sm = EventStateMachine(POLICY, {}, max_frame_gap_s=1.0)
    outcomes = [sm.step(frame(t / 10, c)) for t, c in enumerate([0.9, 0.9, None, 0.9, 0.9, 0.9], start=1)]
    # The negative third frame resets the streak; the alert fires on the 6th frame.
    assert [o is not None for o in outcomes] == [False, False, False, False, False, True]


def test_low_confidence_does_not_count() -> None:
    sm = EventStateMachine(POLICY, {}, max_frame_gap_s=1.0)
    assert all(sm.step(frame(t / 10, 0.59)) is None for t in range(1, 10))


def test_cooldown_and_repeat() -> None:
    sm = EventStateMachine(POLICY, {}, max_frame_gap_s=1.0)
    fired = [f.ts for t in range(1, 1001) if (f := sm.step(frame(t / 10, 0.9))) is not None]
    # Continuous presence for 100 s: first alert at frame 3, then one per 45 s window.
    assert fired == pytest.approx([0.3, 45.3, 90.3])


def test_gap_resets_streak() -> None:
    sm = EventStateMachine(POLICY, {}, max_frame_gap_s=1.0)
    sm.step(frame(0.1, 0.9))
    sm.step(frame(0.2, 0.9))
    # 5 s of missing frames: the next positive frame starts a new streak at 1.
    assert sm.step(frame(5.2, 0.9)) is None


def test_zone_uses_feet_not_box_center() -> None:
    """Zone covers the bottom 30 % of the image (the porch floor)."""
    porch = Zone(name="porch", polygon=[[0.0, 0.7], [1.0, 0.7], [1.0, 1.0], [0.0, 1.0]])
    sm = EventStateMachine(POLICY, {"front": [porch]}, max_frame_gap_s=1.0)
    # Feet at y=200/360=0.56 -> outside, even though the box is large.
    assert not sm.is_positive(frame(0.1, 0.9, [100, 20, 140, 200]))
    # Feet at y=300/360=0.83 -> inside.
    assert sm.is_positive(frame(0.2, 0.9, [100, 120, 140, 300]))


def test_degenerate_zone_rejected() -> None:
    with pytest.raises(ValueError, match="area"):
        Zone(name="line", polygon=[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_replay_end_to_end(workspace: Path, run_cli: CliRunner) -> None:
    rows = []
    for i in range(600):  # 60 s at 10 fps; a person is present from 20 s to 30 s
        ts = 1_000.0 + i / 10
        present = 20.0 <= i / 10 < 30.0
        dets = [{"class_name": "person", "confidence": 0.8, "bbox_xyxy": [100, 50, 140, 200]}] if present else []
        rows.append({"camera_id": "front", "frame_index": i, "ts": ts, "width": 640, "height": 360, "detections": dets})
    _write_jsonl(workspace / "frames.jsonl", rows)
    _write_jsonl(
        workspace / "events.jsonl", [{"camera_id": "front", "start_ts": 1020.0, "end_ts": 1030.0, "label": "person"}]
    )
    policies = [
        {"name": "strict", "classes": ["person"], "min_confidence": 0.9, "min_consecutive_frames": 3, "cooldown_s": 45},
        {
            "name": "default",
            "classes": ["person"],
            "min_confidence": 0.6,
            "min_consecutive_frames": 3,
            "cooldown_s": 45,
        },
    ]
    code, report = run_cli(
        {"task": "RULE_REPLAY", "frames": "frames.jsonl", "policies": policies, "ground_truth_events": "events.jsonl"},
        workspace,
    )
    assert code == 0, report
    strict, default = report["results"]
    assert strict["n_alerts"] == 0 and strict["events"]["event_recall"] == 0.0
    assert default["n_alerts"] == 1 and default["events"]["event_recall"] == 1.0
    # Third positive frame is 0.2 s after the first one.
    assert default["events"]["latency_s_p50"] == pytest.approx(0.2)
    assert report["covered_hours"] == pytest.approx(59.9 / 3600)


def test_replay_rejects_unordered_log(workspace: Path, run_cli: CliRunner) -> None:
    rows = [
        {"camera_id": "front", "frame_index": i, "ts": ts, "width": 640, "height": 360, "detections": []}
        for i, ts in enumerate([1.0, 2.0, 1.5])
    ]
    _write_jsonl(workspace / "frames.jsonl", rows)
    policy = {"name": "p", "classes": ["person"], "min_confidence": 0.6, "min_consecutive_frames": 3, "cooldown_s": 45}
    code, report = run_cli({"task": "RULE_REPLAY", "frames": "frames.jsonl", "policies": [policy]}, workspace)
    assert code == 3
    assert "frames.jsonl:3" in report["error"]["message"]
