"""STREAM_PROBE and PIPELINE_BENCH runners.

STREAM_PROBE calls ``ffprobe`` twice: once for stream metadata (codec, size,
declared frame rate) and once to read packet timestamps for ``duration_s``
seconds, from which the *measured* frame rate, keyframe spacing, jitter and
gaps are computed. The RTSP URL (with credentials) is read from an allowlisted
environment variable and never written to the report or to error messages.

PIPELINE_BENCH reads the timing log the Pi pipeline writes for every frame
and summarises latency, throughput, drops and temperature.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from statistics import fmean, median, pstdev

from vva_contracts.contracts.logs import SystemRow, TimingRow
from vva_contracts.contracts.requests import PipelineBenchRequest, StreamProbeRequest
from vva_contracts.contracts.stream import (
    LatencySummary,
    PipelineBenchReport,
    StageBench,
    StreamProbeReport,
    ThermalSummary,
    derive_bench_flags,
    derive_bench_verdict,
    derive_stream_flags,
)
from vva_contracts.core.stats import parse_rational, percentile, safe_ratio
from vva_contracts.errors import InvalidInput, PolicyBlocked, SubprocessTimeout, redact
from vva_contracts.paths import resolve_in_workspace
from vva_contracts.readers.jsonl import read_all
from vva_contracts.tasks.common import build_meta

# An interval longer than this many expected intervals counts as a gap
# (at 10 fps: any hole longer than 200 ms, i.e. at least one lost frame).
GAP_FACTOR = 2.0


def _ffprobe(args: list[str], source: str, timeout_s: float) -> dict[str, object]:
    """Run ffprobe with an argv list (no shell) and parse its JSON output."""
    binary = shutil.which("ffprobe")
    if binary is None:
        raise InvalidInput("ffprobe not found on PATH (install ffmpeg)")
    try:
        # argv list + shell=False: the URL is passed as one argument and can
        # never be interpreted by a shell, whatever characters it contains.
        proc = subprocess.run(
            [binary, "-v", "error", *args, "-of", "json", source],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SubprocessTimeout(f"ffprobe did not finish within {timeout_s} s") from exc
    if proc.returncode != 0:
        # stderr usually repeats the URL; redact() strips user:password@.
        raise InvalidInput(f"ffprobe failed: {redact(proc.stderr.strip())[:500]}")
    parsed = json.loads(proc.stdout)
    if not isinstance(parsed, dict):
        raise InvalidInput("ffprobe returned unexpected JSON")
    return parsed


def _source(req: StreamProbeRequest, workspace: Path) -> tuple[str, str, list[str], Path | None]:
    """Return (source string for ffprobe, report reference, input options, file)."""
    if req.url_env is not None:
        url = os.environ.get(req.url_env)
        if not url:
            raise InvalidInput(f"environment variable {req.url_env} is not set")
        if not url.startswith(("rtsp://", "rtsps://")):
            # The env var is allowlisted by name; its *value* must still be a
            # camera stream, not e.g. a local file path smuggled in.
            raise PolicyBlocked(f"{req.url_env} must contain an rtsp:// or rtsps:// URL")
        return url, req.url_env, ["-rtsp_transport", req.rtsp_transport], None
    if req.path is None:  # unreachable: the request validator demands one source
        raise InvalidInput("no stream source given")
    path = resolve_in_workspace(workspace, req.path)
    return str(path), req.path, [], path


def run_stream_probe(req: StreamProbeRequest, workspace: Path) -> StreamProbeReport:
    """Entry point for task STREAM_PROBE."""
    source, source_ref, input_opts, file_path = _source(req, workspace)
    meta_doc = _ffprobe(
        [
            *input_opts,
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,width,height,pix_fmt,avg_frame_rate,r_frame_rate",
        ],
        source,
        req.timeout_s,
    )
    streams = meta_doc.get("streams")
    if not isinstance(streams, list) or not streams:
        raise InvalidInput("no video stream found")
    stream = streams[0]
    # avg_frame_rate is what the encoder actually targets; r_frame_rate is a
    # timebase guess and is only used when the average is unknown ("0/0").
    declared = parse_rational(str(stream.get("avg_frame_rate", "0/0"))) or parse_rational(
        str(stream.get("r_frame_rate", "0/0"))
    )

    packets_doc = _ffprobe(
        [
            *input_opts,
            "-select_streams",
            "v:0",
            "-read_intervals",
            f"%+{req.duration_s}",
            "-show_entries",
            "packet=pts_time,dts_time,flags",
        ],
        source,
        req.timeout_s,
    )
    raw_packets = packets_doc.get("packets")
    if not isinstance(raw_packets, list):
        raise InvalidInput("ffprobe returned no packets")
    times: list[float] = []
    key_indices: list[int] = []
    for index, packet in enumerate(raw_packets):
        stamp = packet.get("pts_time", packet.get("dts_time"))
        if stamp in (None, "N/A"):
            continue  # a packet without timestamps cannot be placed in time
        times.append(float(stamp))
        if "K" in str(packet.get("flags", "")):
            key_indices.append(index)
    if len(times) < 2:
        raise InvalidInput("fewer than 2 timestamped packets; stream too short or not decodable")

    times.sort()  # presentation order; IP cameras rarely use B-frames anyway
    deltas = [b - a for a, b in pairwise(times)]
    duration = times[-1] - times[0]
    if duration <= 0:
        raise InvalidInput("packet timestamps span zero seconds")
    measured_fps = (len(times) - 1) / duration  # n packets span n-1 intervals
    fps_ratio = measured_fps / declared if declared else None
    # Without a declared rate, the median interval is the robust "expected"
    # spacing: unlike the mean, a few long holes cannot drag it upwards.
    expected_interval = 1.0 / declared if declared else median(deltas)
    n_gaps = sum(d > GAP_FACTOR * expected_interval for d in deltas)
    key_spacing = [b - a for a, b in pairwise(key_indices)]

    return StreamProbeReport(
        meta=build_meta(workspace, [("stream_file", file_path)] if file_path is not None else []),
        camera_id=req.camera_id,
        source_kind="file" if file_path is not None else "rtsp",
        source_ref=source_ref,
        codec=str(stream.get("codec_name", "unknown")),
        profile=_optional_str(stream.get("profile")),
        width=int(stream["width"]),
        height=int(stream["height"]),
        pix_fmt=_optional_str(stream.get("pix_fmt")),
        declared_fps=declared,
        requested_duration_s=req.duration_s,
        n_packets=len(times),
        n_keyframes=len(key_indices),
        measured_duration_s=duration,
        measured_fps=measured_fps,
        fps_ratio=fps_ratio,
        keyframe_interval_frames=fmean(key_spacing) if key_spacing else None,
        interval_jitter_ms=pstdev(deltas) * 1000.0,  # population std of intervals
        n_gaps=n_gaps,
        min_fps_ratio=req.min_fps_ratio,
        flags=derive_stream_flags(fps_ratio, req.min_fps_ratio, len(key_indices), n_gaps),
    )


def _optional_str(value: object) -> str | None:
    """Ffprobe omits unknown fields; keep None instead of the string 'None'."""
    return None if value is None else str(value)


def _latency(values_s: list[float]) -> LatencySummary | None:
    """Summarise a latency sample given in seconds, reported in milliseconds."""
    if not values_s:
        return None
    ms = [v * 1000.0 for v in values_s]
    return LatencySummary(
        n=len(ms),
        mean_ms=fmean(ms),
        p50_ms=percentile(ms, 50),
        p95_ms=percentile(ms, 95),
        p99_ms=percentile(ms, 99),
        max_ms=max(ms),
    )


def _stage_bench(camera_id: str, rows: list[TimingRow]) -> StageBench:
    """Throughput, drops and stage latencies for a list of timing rows."""
    processed = [r for r in rows if not r.dropped]
    done = sorted(r.t_done for r in processed if r.t_done is not None)
    # Throughput over the span of completions: n completions span n-1 gaps.
    throughput = safe_ratio(len(done) - 1, done[-1] - done[0]) if len(done) >= 2 else 0.0
    end_to_end = [r.t_done - r.t_capture for r in processed if r.t_done is not None]
    inference = [
        r.t_infer_end - r.t_infer_start for r in processed if r.t_infer_end is not None and r.t_infer_start is not None
    ]
    queue_wait = [r.t_infer_start - r.t_capture for r in processed if r.t_infer_start is not None]
    n_dropped = len(rows) - len(processed)
    return StageBench(
        camera_id=camera_id,
        n_records=len(rows),
        n_dropped=n_dropped,
        n_processed=len(processed),
        drop_rate=safe_ratio(n_dropped, len(rows)),
        throughput_fps=throughput,
        end_to_end=_latency(end_to_end),
        inference=_latency(inference),
        queue_wait=_latency(queue_wait),
    )


def run_pipeline_bench(req: PipelineBenchRequest, workspace: Path) -> PipelineBenchReport:
    """Entry point for task PIPELINE_BENCH."""
    timings_path = resolve_in_workspace(workspace, req.timings_log)
    rows = [row for _, row in read_all(timings_path, TimingRow)]
    # Warm-up is excluded: the first frames include model loading and cache
    # misses, which would inflate p99 without describing steady state.
    start = min(r.t_capture for r in rows) + req.warmup_s
    steady = [r for r in rows if r.t_capture >= start]
    if not steady:
        raise InvalidInput(f"no timing rows after the {req.warmup_s} s warm-up window")
    by_camera: dict[str, list[TimingRow]] = defaultdict(list)
    for row in steady:
        by_camera[row.camera_id].append(row)
    cameras = [_stage_bench(cam, cam_rows) for cam, cam_rows in sorted(by_camera.items())]
    overall = _stage_bench("ALL", steady)

    inputs: list[tuple[str, Path]] = [("timings_log", timings_path)]
    thermal: ThermalSummary | None = None
    if req.system_log is not None:
        system_path = resolve_in_workspace(workspace, req.system_log)
        samples = [s for _, s in read_all(system_path, SystemRow) if s.ts >= start]
        if samples:
            temps = [s.cpu_temp_c for s in samples]
            thermal = ThermalSummary(
                n_samples=len(temps),
                mean_temp_c=fmean(temps),
                max_temp_c=max(temps),
                frac_at_or_above_throttle=sum(t >= req.throttle_temp_c for t in temps) / len(temps),
            )
        inputs.append(("system_log", system_path))

    flags = derive_bench_flags(overall, thermal, req.latency_budget_ms, req.max_drop_rate)
    return PipelineBenchReport(
        meta=build_meta(workspace, inputs),
        warmup_s=req.warmup_s,
        latency_budget_ms=req.latency_budget_ms,
        max_drop_rate=req.max_drop_rate,
        throttle_temp_c=req.throttle_temp_c,
        cameras=cameras,
        overall=overall,
        thermal=thermal,
        flags=flags,
        verdict=derive_bench_verdict(flags),
    )
