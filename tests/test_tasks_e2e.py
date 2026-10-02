"""End-to-end: every task through the real CLI entry point, on synthetic data."""

from __future__ import annotations

import pytest
from builders import (
    events_log,
    frames_log,
    gt_predictions,
    make_dataset,
    make_video,
    timing_log,
    write_jsonl,
)

from vva_contracts.contracts.requests import PipelineBenchRequest


# ---------------------------------------------------------------- DATASET_AUDIT
def test_dataset_audit_clean_dataset(tmp_path, run_cli):
    make_dataset(tmp_path / "ds")
    code, rep = run_cli(
        tmp_path, {"task": "DATASET_AUDIT", "data_yaml": "ds/data.yaml", "group_pattern": r"^(?P<group>cam\d+)_"}
    )
    assert code == 0, rep
    assert rep["verdict"] in {"PASS", "WARN"}
    assert "cross_split_near_duplicates" not in rep["flags"]
    train = rep["splits"][0]
    assert train["class_counts"] == {"person": 4, "car": 2} and train["n_boxes"] == 6
    assert rep["leakage"]["n_groups_spanning_splits"] == 0


def test_dataset_audit_detects_leakage_and_bad_labels(tmp_path, run_cli):
    make_dataset(tmp_path / "ds", leak=True, bad_label=True)
    code, rep = run_cli(
        tmp_path, {"task": "DATASET_AUDIT", "data_yaml": "ds/data.yaml", "group_pattern": r"^(?P<group>cam\d+)_"}
    )
    assert code == 0, rep  # a FAIL verdict is still an accepted, valid report
    assert rep["verdict"] == "FAIL"
    assert {"label_errors", "cross_split_near_duplicates", "group_leakage"} <= set(rep["flags"])
    pair = rep["leakage"]["near_duplicate_examples"][0]
    assert (pair["image_a"], pair["image_b"]) == ("cam1_20261001_0001", "cam1_20261001_0004")
    assert rep["issue_examples"][0]["kind"] == "class_out_of_range"


def test_dataset_audit_letterbox_sizes(tmp_path, run_cli):
    make_dataset(tmp_path / "ds")
    # 320x240 images into 640x640: scale 2 -> a 0.05 x 0.10 box is 32 x 48 px.
    code, rep = run_cli(tmp_path, {"task": "DATASET_AUDIT", "data_yaml": "ds/data.yaml"})
    assert code == 0
    assert rep["splits"][0]["box_sizes"]["short_side_px_p05"] >= 32.0 - 1e-9


# ---------------------------------------------------------------- DETECTION_EVAL
def test_detection_eval_perfect_predictions(tmp_path, run_cli):
    make_dataset(tmp_path / "ds")
    write_jsonl(tmp_path / "pred.jsonl", gt_predictions(tmp_path / "ds"))
    code, rep = run_cli(tmp_path, {"task": "DETECTION_EVAL", "data_yaml": "ds/data.yaml", "predictions": "pred.jsonl"})
    assert code == 0, rep
    assert rep["metrics"]["map50"] == pytest.approx(1.0)
    assert rep["metrics"]["map50_95"] == pytest.approx(1.0)
    assert all(c["precision"] == 1.0 and c["recall"] == 1.0 for c in rep["metrics"]["classes"])


def test_detection_eval_refuses_broken_ground_truth(tmp_path, run_cli):
    make_dataset(tmp_path / "ds", bad_label=True)
    write_jsonl(tmp_path / "pred.jsonl", gt_predictions(tmp_path / "ds"))
    code, rep = run_cli(tmp_path, {"task": "DETECTION_EVAL", "data_yaml": "ds/data.yaml", "predictions": "pred.jsonl"})
    assert code == 3 and "DATASET_AUDIT" in rep["error"]["message"]


def test_detection_eval_rejects_unknown_image(tmp_path, run_cli):
    make_dataset(tmp_path / "ds")
    write_jsonl(
        tmp_path / "pred.jsonl", [{"image": "nope", "class_id": 0, "confidence": 0.9, "box": [0.1, 0.1, 0.2, 0.2]}]
    )
    code, rep = run_cli(tmp_path, {"task": "DETECTION_EVAL", "data_yaml": "ds/data.yaml", "predictions": "pred.jsonl"})
    assert code == 3 and "pred.jsonl:1" in rep["error"]["message"]


# ---------------------------------------------------------------- QUANT_PARITY
def test_quant_parity_pass_and_fail(tmp_path, run_cli):
    make_dataset(tmp_path / "ds")
    write_jsonl(tmp_path / "fp32.jsonl", gt_predictions(tmp_path / "ds", jitter=0.0))
    write_jsonl(tmp_path / "int8_ok.jsonl", gt_predictions(tmp_path / "ds", jitter=0.002, seed=3))
    write_jsonl(tmp_path / "int8_bad.jsonl", gt_predictions(tmp_path / "ds", jitter=0.05, seed=4))
    base = {"task": "QUANT_PARITY", "data_yaml": "ds/data.yaml", "reference_predictions": "fp32.jsonl"}
    code, ok = run_cli(tmp_path, {**base, "candidate_predictions": "int8_ok.jsonl"})
    assert code == 0 and ok["verdict"] == "PASS", ok
    code, bad = run_cli(tmp_path, {**base, "candidate_predictions": "int8_bad.jsonl"})
    assert code == 0 and bad["verdict"] == "FAIL" and "map50_95_drop" in bad["failed_checks"]
    assert bad["delta_map50_95"] < 0


# ---------------------------------------------------------------- RULE_REPLAY
def test_rule_replay_tradeoff(tmp_path, run_cli):
    write_jsonl(tmp_path / "frames.jsonl", frames_log())
    write_jsonl(tmp_path / "events.jsonl", events_log())
    policies = [
        {
            "name": "loose",
            "classes": ["person"],
            "min_confidence": 0.5,
            "min_consecutive_frames": 1,
            "cooldown_s": 45,
            "max_gap_s": 1.0,
        },
        {
            "name": "project_default",
            "classes": ["person"],
            "min_confidence": 0.6,
            "min_consecutive_frames": 3,
            "cooldown_s": 45,
            "max_gap_s": 1.0,
        },
    ]
    code, rep = run_cli(
        tmp_path, {"task": "RULE_REPLAY", "frames_log": "frames.jsonl", "events": "events.jsonl", "policies": policies}
    )
    assert code == 0, rep
    loose, default = rep["policies"]
    # Regression test of the CURRENT frame-level RULE_REPLAY logic (decided 2026-10-02).
    # Known limitation, kept on purpose: the loose policy fires on an isolated false
    # positive just before the real event; that alert opens the 45 s per-camera
    # cooldown and suppresses the alert for the real event (t=100..130 s), so its
    # recall is 0.0. The stricter default policy ignores the blip and catches it.
    # Production cooldowns (30 s per track/zone, 5 s re-arm in M6; 10 s notification
    # throttle in M8) are validated separately by the M3 replay harness, not here.
    assert loose["ground_truth"]["recall"] == 0.0
    assert default["ground_truth"]["recall"] == 1.0
    # The loose policy also alerts on the bush and on isolated blips.
    assert loose["n_alerts"] > default["n_alerts"]
    assert default["ground_truth"]["alert_precision"] == 1.0
    # Latency of the default policy: 3 frames at 10 fps -> first alert at +0.2 s.
    assert default["ground_truth"]["latency_max_s"] == pytest.approx(0.2)


def test_rule_replay_zone_for_unknown_camera_is_rejected(tmp_path, run_cli):
    write_jsonl(tmp_path / "frames.jsonl", frames_log(seconds=10))
    policy = {
        "name": "p",
        "classes": ["person"],
        "min_confidence": 0.6,
        "min_consecutive_frames": 3,
        "cooldown_s": 45,
        "max_gap_s": 1.0,
        "zones": {"frnot": [[0, 0], [1, 0], [1, 1]]},
    }
    code, rep = run_cli(tmp_path, {"task": "RULE_REPLAY", "frames_log": "frames.jsonl", "policies": [policy]})
    assert code == 3 and "frnot" in rep["error"]["message"]


# ---------------------------------------------------------------- STREAM_PROBE
def test_stream_probe_on_file(tmp_path, run_cli):
    make_video(tmp_path / "clip.mp4", fps=10, seconds=6)
    code, rep = run_cli(tmp_path, {"task": "STREAM_PROBE", "camera_id": "front", "path": "clip.mp4", "duration_s": 5})
    assert code == 0, rep
    assert (rep["width"], rep["height"], rep["codec"]) == (320, 240, "h264")
    assert rep["declared_fps"] == pytest.approx(10.0)
    assert rep["measured_fps"] == pytest.approx(10.0, rel=0.01)
    assert rep["keyframe_interval_frames"] == pytest.approx(20.0)
    assert rep["flags"] == []


def test_stream_probe_env_value_must_be_rtsp(tmp_path, run_cli, monkeypatch):
    monkeypatch.setenv("VVA_FRONT_SUB_URL", "/etc/passwd")
    code, _rep = run_cli(tmp_path, {"task": "STREAM_PROBE", "camera_id": "front", "url_env": "VVA_FRONT_SUB_URL"})
    assert code == 4


def test_stream_probe_never_leaks_credentials(tmp_path, run_cli, monkeypatch):
    # Unreachable host on a reserved TEST-NET address; ffprobe fails or times out.
    monkeypatch.setenv("VVA_FRONT_SUB_URL", "rtsp://admin:S3cr3t@192.0.2.1:554/stream2")
    code, rep = run_cli(
        tmp_path, {"task": "STREAM_PROBE", "camera_id": "front", "url_env": "VVA_FRONT_SUB_URL", "timeout_s": 5}
    )
    assert code in (3, 6)
    assert "S3cr3t" not in str(rep) and "admin" not in str(rep)


# ---------------------------------------------------------------- PIPELINE_BENCH
def test_pipeline_bench_pass_and_drop_failure(tmp_path, run_cli):
    write_jsonl(tmp_path / "ok.jsonl", timing_log())
    write_jsonl(tmp_path / "drops.jsonl", timing_log(drop_every=20))  # 5 % dropped
    write_jsonl(tmp_path / "sys.jsonl", [{"ts": 1_000_000.0 + 5 * i, "cpu_temp_c": 70.0 + i % 15} for i in range(40)])
    code, ok = run_cli(tmp_path, {"task": "PIPELINE_BENCH", "timings_log": "ok.jsonl"})
    assert code == 0 and ok["verdict"] == "PASS", ok
    assert 60 < ok["overall"]["end_to_end"]["p50_ms"] < 110
    code, bad = run_cli(tmp_path, {"task": "PIPELINE_BENCH", "timings_log": "drops.jsonl", "system_log": "sys.jsonl"})
    assert code == 0 and bad["verdict"] == "FAIL"
    assert {"drop_rate_high", "thermal_throttle_risk"} <= set(bad["flags"])


def test_pipeline_bench_default_budget_is_guide_alarm(tmp_path, run_cli):
    """The default budget is the guide's end-to-end p95 alarm (400 ms), applied to p95."""
    assert PipelineBenchRequest(task="PIPELINE_BENCH", timings_log="t.jsonl").latency_budget_ms == 400.0
    write_jsonl(tmp_path / "ok.jsonl", timing_log())
    code, ok = run_cli(tmp_path, {"task": "PIPELINE_BENCH", "timings_log": "ok.jsonl"})
    assert code == 0 and ok["latency_budget_ms"] == 400.0
    # A budget just below the measured p95 must flag it; one at p95 must not (strict ">").
    p95 = ok["overall"]["end_to_end"]["p95_ms"]
    base = {"task": "PIPELINE_BENCH", "timings_log": "ok.jsonl"}
    code, tight = run_cli(tmp_path, {**base, "latency_budget_ms": p95 * 0.999})
    assert code == 0 and "latency_budget_exceeded" in tight["flags"] and tight["verdict"] == "FAIL"
    code, exact = run_cli(tmp_path, {**base, "latency_budget_ms": p95})
    assert code == 0 and "latency_budget_exceeded" not in exact["flags"]


def test_timing_rows_must_be_causal(tmp_path, run_cli):
    write_jsonl(
        tmp_path / "t.jsonl",
        [{"camera_id": "c", "frame_idx": 0, "t_capture": 5.0, "t_infer_start": 4.0, "t_infer_end": 6.0, "t_done": 7.0}],
    )
    code, rep = run_cli(tmp_path, {"task": "PIPELINE_BENCH", "timings_log": "t.jsonl", "warmup_s": 0})
    assert code == 3 and "t.jsonl:1" in rep["error"]["message"]
