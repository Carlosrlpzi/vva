"""STREAM_PROBE task: measure a camera stream (or a recorded clip) with ffprobe.

Security model
--------------
* The RTSP URL comes from an environment variable whose *name* is in the
  request (``VVA_CAM_<NAME>_URL``). The operator sets it in ``.env`` on the
  machine; the model never sees the URL or the password inside it.
* Probing a live device is network access, so it requires the operator gate
  ``VVA_ALLOW_STREAM_PROBE=1``. Probing a workspace file needs no gate.
* ffprobe runs without a shell (argument list), so nothing in the URL can be
  interpreted as shell syntax.
* ffprobe's stderr may echo the URL. It is never forwarded verbatim; error
  messages are rebuilt from the redacted URL only.

Why packets instead of decoded frames
-------------------------------------
``-show_packets`` reads timestamps from the container without decoding the
video, so the probe costs almost no CPU on the Pi and does not disturb the
pipeline it is measuring. For IP cameras without B-frames, one video packet
is one frame. Packets are sorted by PTS before computing intervals anyway.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from vva_contracts.contracts.requests import StreamProbeRequest
from vva_contracts.contracts.stream import (
    FrameTiming,
    StreamProbeReport,
    StreamSource,
    derive_stream_flags,
    derive_stream_verdict,
)
from vva_contracts.errors import InputError, PolicyError, TaskTimeoutError
from vva_contracts.tasks._common import tool_version
from vva_contracts.workspace import display_path, resolve_in_workspace

GAP_FACTOR = 1.5  # an interval longer than 1.5 nominal intervals is a gap
TIMEOUT_MARGIN_S = 20.0  # connection set-up + teardown on top of the read window
RTSP_SOCKET_TIMEOUT_US = 5_000_000  # 5 s: fail fast on an unreachable camera


def redact_url(url: str) -> str:
    """Remove ``user:password@`` from a URL; keep scheme, host, port, path."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    # Query strings on some cameras carry tokens (?token=...): drop them too.
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _source(request: StreamProbeRequest, workspace: Path) -> tuple[str, StreamSource, list[str]]:
    """Return (ffprobe input argument, redacted description, extra input options)."""
    if request.url_env is not None:
        if os.environ.get("VVA_ALLOW_STREAM_PROBE") != "1":
            raise PolicyError("probing a live camera requires the operator to set VVA_ALLOW_STREAM_PROBE=1")
        url = os.environ.get(request.url_env)
        if not url:
            raise InputError(f"environment variable {request.url_env} is not set")
        if urlsplit(url).scheme not in {"rtsp", "rtsps"}:
            raise InputError(f"{request.url_env} must hold an rtsp:// or rtsps:// URL")
        options = ["-rtsp_transport", request.rtsp_transport, "-timeout", str(RTSP_SOCKET_TIMEOUT_US)]
        return url, StreamSource(kind="rtsp", redacted=redact_url(url)), options
    assert request.path is not None  # guaranteed by the request validator
    path = resolve_in_workspace(workspace, request.path, kind="file")
    return str(path), StreamSource(kind="file", redacted=display_path(workspace, path)), []


def _run_ffprobe(arguments: list[str], timeout_s: float, redacted: str, secret: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            arguments, capture_output=True, text=True, timeout=timeout_s, check=False, shell=False
        )
    except FileNotFoundError as exc:
        raise InputError("ffprobe not found on PATH (install ffmpeg)") from exc
    except subprocess.TimeoutExpired as exc:
        raise TaskTimeoutError(f"ffprobe did not finish within {timeout_s:.0f} s for {redacted}") from exc
    if completed.returncode != 0:
        # Keep only the last stderr line and scrub the secret URL out of it.
        tail = (completed.stderr.strip().splitlines() or ["no error output"])[-1]
        tail = tail.replace(secret, redacted)
        tail = re.sub(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s@/]+@", "<redacted>@", tail)
        raise InputError(f"ffprobe failed for {redacted}: {tail[:200]}")
    try:
        parsed: dict[str, Any] = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise InputError(f"ffprobe returned non-JSON output for {redacted}") from exc
    return parsed


def _rate(value: str | None) -> float | None:
    """Parse ffprobe rates like ``"30000/1001"``; ``"0/0"`` means unknown."""
    if not value or value in {"0/0", "0"}:
        return None
    rate = Fraction(value)
    return float(rate) if rate > 0 else None


def compute_timing(pts_s: list[float], expected_fps: float | None, declared_fps: float | None) -> FrameTiming:
    """Interval statistics and drop estimate from packet timestamps (seconds)."""
    pts = np.sort(np.asarray(pts_s, dtype=np.float64))
    n = len(pts)
    span = float(pts[-1] - pts[0]) if n >= 2 else 0.0
    if n < 2 or span <= 0:
        return FrameTiming(
            n_frames=n, span_s=span, measured_fps=None, nominal_interval_ms=None, nominal_interval_origin=None,
            interval_ms_p50=None, interval_ms_p95=None, interval_ms_max=None,
            n_gaps=0, estimated_dropped_frames=0, drop_share=None if n == 0 else 0.0,
        )  # fmt: skip

    intervals = np.diff(pts)
    # Nominal interval preference: what the operator expects > what the stream
    # declares > what it actually delivers (median is robust to the gaps).
    if expected_fps is not None:
        nominal, origin = 1.0 / expected_fps, "expected_fps"
    elif declared_fps is not None:
        nominal, origin = 1.0 / declared_fps, "declared_fps"
    else:
        nominal, origin = float(np.median(intervals)), "median_interval"
    if nominal <= 0:
        raise InputError("stream has duplicate timestamps; cannot estimate a nominal interval")

    gaps = intervals[intervals > GAP_FACTOR * nominal]
    # A gap of length k*nominal hides k-1 frames (see contracts/stream.py).
    dropped = int(np.sum(np.round(gaps / nominal) - 1))
    return FrameTiming(
        n_frames=n,
        span_s=span,
        measured_fps=(n - 1) / span,
        nominal_interval_ms=nominal * 1000.0,
        nominal_interval_origin=origin,  # type: ignore[arg-type]
        interval_ms_p50=float(np.quantile(intervals, 0.50) * 1000.0),
        interval_ms_p95=float(np.quantile(intervals, 0.95) * 1000.0),
        interval_ms_max=float(intervals.max() * 1000.0),
        n_gaps=len(gaps),
        estimated_dropped_frames=dropped,
        drop_share=dropped / (n + dropped),
    )


def run(request: StreamProbeRequest, workspace: Path) -> StreamProbeReport:
    target, source, input_options = _source(request, workspace)
    arguments = [
        "ffprobe",
        "-v", "error",
        *input_options,
        "-read_intervals", f"%+{request.duration_s}",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,profile,width,height,avg_frame_rate,r_frame_rate:packet=pts_time",
        "-show_packets",
        "-of", "json",
        target,
    ]  # fmt: skip
    data = _run_ffprobe(arguments, request.duration_s + TIMEOUT_MARGIN_S, source.redacted, target)

    streams = data.get("streams") or []
    if not streams:
        raise InputError(f"no video stream found in {source.redacted}")
    stream = streams[0]
    pts = [float(p["pts_time"]) for p in data.get("packets", []) if p.get("pts_time") not in (None, "N/A")]
    declared = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))
    timing = compute_timing(pts, request.expected_fps, declared)
    flags = derive_stream_flags(timing, request.expected_fps)

    return StreamProbeReport(
        tool_version=tool_version(),
        source=source,
        codec=str(stream.get("codec_name", "unknown")),
        profile=stream.get("profile"),
        width=int(stream["width"]),
        height=int(stream["height"]),
        declared_fps=declared,
        duration_requested_s=request.duration_s,
        expected_fps=request.expected_fps,
        timing=timing,
        flags=flags,
        verdict=derive_stream_verdict(flags),
    )
