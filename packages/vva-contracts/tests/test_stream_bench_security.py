"""STREAM_PROBE, PIPELINE_BENCH, and the security boundary of the CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import CliRunner

from vva_contracts.cli import main
from vva_contracts.tasks.stream_probe import compute_timing, redact_url

# ---------------------------------------------------------------- STREAM_PROBE


def test_drop_estimate_from_gaps() -> None:
    # 10 fps with one interval of 0.4 s: three frames missing.
    pts = [i / 10 for i in range(10)] + [1.3 + i / 10 for i in range(10)]
    timing = compute_timing(pts, expected_fps=10.0, declared_fps=None)
    assert timing.n_gaps == 1
    assert timing.estimated_dropped_frames == 3
    assert timing.measured_fps == pytest.approx(19 / 2.2)


def test_redaction_removes_credentials_and_tokens() -> None:
    url = "rtsp://admin:S3cret@192.168.1.20:554/media/video2?token=abc"
    assert redact_url(url) == "rtsp://192.168.1.20:554/media/video2"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_probe_recorded_clip(workspace: Path, run_cli: CliRunner) -> None:
    clip = workspace / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10", "-t", "4",
         "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )  # fmt: skip
    code, report = run_cli({"task": "STREAM_PROBE", "path": "clip.mp4", "duration_s": 3, "expected_fps": 10}, workspace)
    assert code == 0, report
    assert (report["width"], report["height"], report["codec"]) == (640, 360, "h264")
    assert report["timing"]["measured_fps"] == pytest.approx(10.0, rel=0.01)
    assert report["verdict"] == "PASS"


def test_live_probe_requires_operator_gate(
    workspace: Path, run_cli: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VVA_ALLOW_STREAM_PROBE", raising=False)
    monkeypatch.setenv("VVA_CAM_FRONT_URL", "rtsp://u:p@10.0.0.2/stream")
    code, report = run_cli({"task": "STREAM_PROBE", "url_env": "VVA_CAM_FRONT_URL"}, workspace)
    assert code == 4
    assert "p@" not in json.dumps(report)


def test_url_env_cannot_point_at_other_secrets(workspace: Path, run_cli: CliRunner) -> None:
    code, _ = run_cli({"task": "STREAM_PROBE", "url_env": "DEEPSEEK_API_KEY"}, workspace)
    assert code == 3


# -------------------------------------------------------------- PIPELINE_BENCH


def _bench_rows(throttled: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(100):
        t = 1_000.0 + i / 10
        rows.append({"kind": "timing", "camera_id": "front", "frame_index": i, "ts_capture": t,
                     "ts_infer_start": t + 0.01, "ts_infer_end": t + 0.03, "ts_done": t + 0.05})  # fmt: skip
    for i in range(5):
        rows.append({"kind": "system", "ts": 1_000.0 + 2 * i, "cpu_temp_c": 60.0 + i, "cpu_percent": 40.0,
                     "throttled_flags": throttled})  # fmt: skip
    return rows


def test_bench_pass_and_latency_breakdown(workspace: Path, run_cli: CliRunner) -> None:
    (workspace / "bench.jsonl").write_text("\n".join(json.dumps(r) for r in _bench_rows()) + "\n")
    code, report = run_cli({"task": "PIPELINE_BENCH", "bench_log": "bench.jsonl", "target_fps": 10}, workspace)
    assert code == 0, report
    cam = report["cameras"][0]
    assert cam["throughput_fps"] == pytest.approx(10.0)
    assert cam["inference"]["p50_ms"] == pytest.approx(20.0)
    assert report["verdict"] == "PASS"


def test_bench_throttling_fails(workspace: Path, run_cli: CliRunner) -> None:
    (workspace / "bench.jsonl").write_text("\n".join(json.dumps(r) for r in _bench_rows(throttled=0x50005)) + "\n")
    code, report = run_cli({"task": "PIPELINE_BENCH", "bench_log": "bench.jsonl"}, workspace)
    assert code == 0
    assert "THROTTLED" in report["flags"] and report["verdict"] == "FAIL"


# ------------------------------------------------------------ CLI / security


@pytest.mark.parametrize(
    ("request_body", "fragment"),
    [
        ({"task": "PIPELINE_BENCH", "bench_log": "b.jsonl", "colour": 1}, "Extra inputs"),
        ({"task": "PIPELINE_BENCH", "bench_log": "b.jsonl", "target_fps": "10"}, "valid number"),
        ({"task": "PIPELINE_BENCH", "bench_log": "b.jsonl", "frames": "x"}, "Extra inputs"),
        ({"task": "TRAIN_EVERYTHING"}, "does not match any of the expected tags"),
    ],
)
def test_strict_requests(workspace: Path, run_cli: CliRunner, request_body: dict[str, Any], fragment: str) -> None:
    code, report = run_cli(request_body, workspace)
    assert code == 3
    assert fragment in report["error"]["message"]


@pytest.mark.parametrize("path", ["/etc/passwd", "../outside.jsonl", "C:\\Windows\\win.ini"])
def test_paths_cannot_escape(workspace: Path, run_cli: CliRunner, path: str) -> None:
    code, report = run_cli({"task": "PIPELINE_BENCH", "bench_log": path}, workspace)
    assert code == 3
    assert "workspace" in report["error"]["message"]


def test_symlink_escape_rejected(workspace: Path, run_cli: CliRunner, tmp_path: Path) -> None:
    secret = tmp_path / "secret.jsonl"
    secret.write_text("{}\n")
    (workspace / "link.jsonl").symlink_to(secret)
    code, report = run_cli({"task": "PIPELINE_BENCH", "bench_log": "link.jsonl"}, workspace)
    assert code == 3 and "escapes" in report["error"]["message"]


def test_schema_export(tmp_path: Path) -> None:
    assert main(["schemas", "--output", str(tmp_path)]) == 0
    names = sorted(p.name for p in tmp_path.glob("*.schema.json"))
    assert "DetectionEvalReport.schema.json" in names and "FrameRecord.schema.json" in names
    assert len(names) == 12
