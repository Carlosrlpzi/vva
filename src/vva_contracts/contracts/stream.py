"""STREAM_PROBE and PIPELINE_BENCH report contracts.

Why these contracts exist
-------------------------
The architecture relies on two assumptions that are easy to state and easy to
get wrong: (1) each camera's *substream* really delivers the configured
resolution and frame rate, and (2) the Pi + Hailo pipeline keeps up with all
cameras without dropping frames, blowing the latency budget or overheating.
These reports measure both, with flags derived only from the numbers.

Latency percentiles use the method recorded in ``percentile_method`` (see
core/stats.py), because "p95" means different things in different tools.
"""

from __future__ import annotations

from typing import Annotated, Literal

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
from vva_contracts.core.stats import PERCENTILE_METHOD, safe_ratio

StreamFlag = Literal["fps_below_declared", "declared_fps_unknown", "no_keyframes", "timestamp_gaps"]
BenchFlag = Literal["no_processed_frames", "latency_budget_exceeded", "drop_rate_high", "thermal_throttle_risk"]
BENCH_FAIL_FLAGS: frozenset[str] = frozenset({"no_processed_frames", "latency_budget_exceeded", "drop_rate_high"})


class StreamProbeReport(StrictModel):
    """Accepted STREAM_PROBE artifact. Never contains the stream URL."""

    contract_id: Literal["StreamProbeReport"] = "StreamProbeReport"
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    meta: ReportMeta
    camera_id: Identifier
    source_kind: Literal["rtsp", "file"]
    # Env var NAME (rtsp) or workspace path (file); the URL itself is a secret.
    source_ref: str
    codec: str
    profile: str | None
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    pix_fmt: str | None
    declared_fps: Annotated[float, Field(gt=0.0)] | None
    requested_duration_s: NonNegFloat
    n_packets: int = Field(ge=2)
    n_keyframes: NonNegInt
    measured_duration_s: float = Field(gt=0.0)
    measured_fps: float = Field(gt=0.0)
    fps_ratio: NonNegFloat | None
    keyframe_interval_frames: NonNegFloat | None
    interval_jitter_ms: NonNegFloat
    n_gaps: NonNegInt
    min_fps_ratio: Ratio
    flags: list[StreamFlag]

    @model_validator(mode="after")
    def _recompute(self) -> StreamProbeReport:
        """Rates follow from counts; flags follow from rates."""
        require(self.n_keyframes <= self.n_packets, "more keyframes than packets")
        require(self.n_gaps <= self.n_packets - 1, "more gaps than packet intervals")
        # n packets span n-1 intervals; this is the fence-post correct rate.
        expected_fps = (self.n_packets - 1) / self.measured_duration_s
        require(close(self.measured_fps, expected_fps), "measured_fps != (n_packets - 1) / measured_duration_s")
        if self.declared_fps is None:
            require(self.fps_ratio is None, "fps_ratio must be null when declared_fps is unknown")
        else:
            require(
                self.fps_ratio is not None and close(self.fps_ratio, self.measured_fps / self.declared_fps),
                "fps_ratio != measured_fps / declared_fps",
            )
        expected = derive_stream_flags(self.fps_ratio, self.min_fps_ratio, self.n_keyframes, self.n_gaps)
        require(self.flags == expected, "flags do not match the measurements")
        return self


def derive_stream_flags(
    fps_ratio: float | None, min_fps_ratio: float, n_keyframes: int, n_gaps: int
) -> list[StreamFlag]:
    """Flags from measured numbers only. Called by the runner AND the validator.

    ``fps_ratio`` is None exactly when the declared frame rate is unknown.
    """
    flags: list[StreamFlag] = []
    if fps_ratio is None:
        flags.append("declared_fps_unknown")
    elif fps_ratio < min_fps_ratio:
        flags.append("fps_below_declared")
    if n_keyframes == 0:
        flags.append("no_keyframes")
    if n_gaps > 0:
        flags.append("timestamp_gaps")
    return flags


class LatencySummary(StrictModel):
    """Distribution of one latency (milliseconds)."""

    n: int = Field(ge=1)
    mean_ms: NonNegFloat
    p50_ms: NonNegFloat
    p95_ms: NonNegFloat
    p99_ms: NonNegFloat
    max_ms: NonNegFloat

    @model_validator(mode="after")
    def _ordered(self) -> LatencySummary:
        """Percentiles of one sample are monotone and bounded by the max."""
        require(self.p50_ms <= self.p95_ms <= self.p99_ms <= self.max_ms, "percentiles not ordered")
        require(self.mean_ms <= self.max_ms + 1e-9, "mean exceeds max")
        return self


class StageBench(StrictModel):
    """Throughput and latency for one camera (or 'ALL' for the pooled view)."""

    camera_id: Identifier
    n_records: NonNegInt
    n_dropped: NonNegInt
    n_processed: NonNegInt
    drop_rate: Ratio
    throughput_fps: NonNegFloat
    end_to_end: LatencySummary | None
    inference: LatencySummary | None
    queue_wait: LatencySummary | None

    @model_validator(mode="after")
    def _recompute(self) -> StageBench:
        """Counts partition; drop rate follows; summaries exist iff processed > 0."""
        require(self.n_processed == self.n_records - self.n_dropped, "n_processed != n_records - n_dropped")
        require(close(self.drop_rate, safe_ratio(self.n_dropped, self.n_records)), "drop_rate mismatch")
        summaries = (self.end_to_end, self.inference, self.queue_wait)
        require(all(s is None for s in summaries) == (self.n_processed == 0), "latency null iff nothing processed")
        for s in summaries:
            if s is not None:
                require(s.n == self.n_processed, "latency sample size != n_processed")
        return self


class ThermalSummary(StrictModel):
    """CPU temperature over the benchmark window."""

    n_samples: int = Field(ge=1)
    mean_temp_c: float
    max_temp_c: float
    frac_at_or_above_throttle: Ratio


class PipelineBenchReport(StrictModel):
    """Accepted PIPELINE_BENCH artifact."""

    contract_id: Literal["PipelineBenchReport"] = "PipelineBenchReport"
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    meta: ReportMeta
    percentile_method: Literal["linear"] = PERCENTILE_METHOD
    warmup_s: NonNegFloat
    latency_budget_ms: float = Field(gt=0.0)
    max_drop_rate: Ratio
    throttle_temp_c: float
    cameras: list[StageBench] = Field(min_length=1)
    overall: StageBench
    thermal: ThermalSummary | None
    flags: list[BenchFlag]
    verdict: Literal["PASS", "WARN", "FAIL"]

    @model_validator(mode="after")
    def _recompute(self) -> PipelineBenchReport:
        """The pooled view adds up; flags and verdict follow from it."""
        require(self.overall.camera_id == "ALL", "overall.camera_id must be 'ALL'")
        require(self.overall.n_records == sum(c.n_records for c in self.cameras), "overall n_records mismatch")
        require(self.overall.n_dropped == sum(c.n_dropped for c in self.cameras), "overall n_dropped mismatch")
        expected = derive_bench_flags(self.overall, self.thermal, self.latency_budget_ms, self.max_drop_rate)
        require(self.flags == expected, "flags do not match the measurements")
        require(self.verdict == derive_bench_verdict(expected), "verdict does not match the flags")
        return self


def derive_bench_flags(
    overall: StageBench, thermal: ThermalSummary | None, latency_budget_ms: float, max_drop_rate: float
) -> list[BenchFlag]:
    """Flags from the pooled numbers; the budget applies to end-to-end p95.

    p95 rather than the mean: an alert pipeline is judged by its slow frames
    (the person walking past *while* the queue is backed up), not the average.
    """
    flags: list[BenchFlag] = []
    if overall.n_processed == 0:
        flags.append("no_processed_frames")
    elif overall.end_to_end is not None and overall.end_to_end.p95_ms > latency_budget_ms:
        flags.append("latency_budget_exceeded")
    if overall.drop_rate > max_drop_rate:
        flags.append("drop_rate_high")
    if thermal is not None and thermal.frac_at_or_above_throttle > 0.0:
        flags.append("thermal_throttle_risk")
    return flags


def derive_bench_verdict(flags: list[BenchFlag]) -> Literal["PASS", "WARN", "FAIL"]:
    """FAIL on blocking flags, WARN on thermal risk only, else PASS."""
    if any(f in BENCH_FAIL_FLAGS for f in flags):
        return "FAIL"
    return "WARN" if flags else "PASS"
