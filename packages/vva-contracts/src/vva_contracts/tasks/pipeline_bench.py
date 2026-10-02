"""PIPELINE_BENCH task: throughput, stage latencies and host health from a bench log.

The bench log is written by the pipeline itself (``TimingRecord`` per frame,
``SystemSample`` every few seconds). This task only summarizes it; it never
measures anything on its own, so it can run on the laptop against a log
copied from the Pi.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from vva_contracts.contracts.records import BenchLine, SystemSample, TimingRecord
from vva_contracts.contracts.requests import PipelineBenchRequest
from vva_contracts.contracts.stream import (
    CameraBench,
    LatencyQuantiles,
    PipelineBenchReport,
    SystemSummary,
    derive_bench_flags,
    derive_bench_verdict,
)
from vva_contracts.errors import InputError
from vva_contracts.io import fingerprint, iter_jsonl
from vva_contracts.tasks._common import quantile, tool_version
from vva_contracts.workspace import resolve_in_workspace


def _latency(seconds: list[float]) -> LatencyQuantiles:
    ms = np.asarray(seconds, dtype=np.float64) * 1000.0
    return LatencyQuantiles(
        p50_ms=quantile(ms, 0.50), p90_ms=quantile(ms, 0.90), p99_ms=quantile(ms, 0.99), max_ms=float(ms.max())
    )


def run(request: PipelineBenchRequest, workspace: Path) -> PipelineBenchReport:
    path = resolve_in_workspace(workspace, request.bench_log, kind="file")
    timings: dict[str, list[TimingRecord]] = defaultdict(list)
    samples: list[SystemSample] = []
    for _, line in iter_jsonl(path, BenchLine, shown_as=request.bench_log):
        record = line.root
        if isinstance(record, TimingRecord):
            timings[record.camera_id].append(record)
        else:
            samples.append(record)

    cameras: list[CameraBench] = []
    for camera_id in sorted(timings):
        # Order by capture time: that is the order frames entered the pipeline.
        records = sorted(timings[camera_id], key=lambda r: r.ts_capture)
        if len(records) < 2:
            raise InputError(f"camera {camera_id!r} has fewer than 2 timing records")
        span = records[-1].ts_capture - records[0].ts_capture
        if span <= 0:
            raise InputError(f"camera {camera_id!r}: all capture timestamps are equal")
        indices = [r.frame_index for r in records]
        # Frame indices that go backwards along capture time reveal a
        # mislabeled log or a reordering bug in the producer.
        out_of_order = sum(1 for a, b in pairwise(indices) if b <= a)
        cameras.append(
            CameraBench(
                camera_id=camera_id,
                n_frames=len(records),
                span_s=span,
                throughput_fps=(len(records) - 1) / span,
                n_out_of_order=out_of_order,
                end_to_end=_latency([r.ts_done - r.ts_capture for r in records]),
                inference=_latency([r.ts_infer_end - r.ts_infer_start for r in records]),
                queueing=_latency([r.ts_infer_start - r.ts_capture for r in records]),
            )
        )
    if not cameras:
        raise InputError("bench log contains no timing records")

    flags_seen = [s.throttled_flags for s in samples if s.throttled_flags is not None]
    system = SystemSummary(
        n_samples=len(samples),
        cpu_temp_c_max=max(s.cpu_temp_c for s in samples) if samples else None,
        cpu_percent_p95=quantile([s.cpu_percent for s in samples], 0.95) if samples else None,
        throttled_any=any(f != 0 for f in flags_seen) if flags_seen else None,
    )

    draft = PipelineBenchReport.model_construct(
        tool_version=tool_version(),
        inputs=[fingerprint(workspace, "bench_log", path)],
        target_fps=request.target_fps,
        temp_warn_c=request.temp_warn_c,
        latency_budget_ms=request.latency_budget_ms,
        cameras=cameras,
        system=system,
        # Placeholders: model_construct skips validation; both are replaced below.
        flags=[],
        verdict="PASS",
    )
    flags = derive_bench_flags(draft)
    return PipelineBenchReport.model_validate(
        {**draft.__dict__, "flags": flags, "verdict": derive_bench_verdict(flags)}
    )
