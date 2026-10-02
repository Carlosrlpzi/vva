"""STREAM_PROBE and PIPELINE_BENCH reports: is the hardware keeping up?

Frame-rate and drop estimation (STREAM_PROBE)
---------------------------------------------
With n packet timestamps spanning ``span`` seconds there are n - 1 intervals,
so ``measured_fps = (n - 1) / span`` (using n / span overestimates by one
frame per window). Given the nominal interval Δ = 1 / fps, an interval
δ > 1.5 Δ is a *gap*, and it hides ``round(δ / Δ) - 1`` missing frames: an
interval of 3Δ means two frames never arrived. Every gap hides at least one
frame (δ/Δ > 1.5 rounds to >= 2), hence ``estimated_dropped_frames >= n_gaps``.

Throughput and latency (PIPELINE_BENCH)
---------------------------------------
Throughput uses the same (n - 1) / span rule on capture timestamps. Latency
quantiles use numpy's default linear interpolation (Hyndman-Fan type 7).
Throughput and latency are different questions: a pipeline can sustain
10 fps while every frame is 2 s late (deep queue), which is useless for an
alert at the door.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from vva_contracts.contracts.base import (
    SCHEMA_VERSION,
    CameraId,
    InputFingerprint,
    NonNegativeFloat,
    StrictModel,
    UnitInterval,
    Verdict,
    require_close,
    require_ordered,
)

FPS_SHORTFALL_WARN = 0.90  # measured below 90 % of expected -> WARN
DROP_SHARE_WARN = 0.02  # more than 2 % of frames missing -> WARN

StreamFlag = Literal["TOO_FEW_FRAMES", "FPS_BELOW_EXPECTED", "FRAME_DROPS"]
BenchFlag = Literal["FPS_BELOW_TARGET", "LATENCY_OVER_BUDGET", "HIGH_TEMPERATURE", "THROTTLED", "OUT_OF_ORDER_FRAMES"]


class StreamSource(StrictModel):
    kind: Literal["rtsp", "file"]
    # Credentials are stripped before this string is built (see tasks/stream_probe.py).
    redacted: Annotated[str, StringConstraints(min_length=1, max_length=4096)]


class FrameTiming(StrictModel):
    n_frames: int = Field(ge=0)
    span_s: NonNegativeFloat
    measured_fps: NonNegativeFloat | None
    nominal_interval_ms: float | None = Field(gt=0.0)
    nominal_interval_origin: Literal["expected_fps", "declared_fps", "median_interval"] | None
    interval_ms_p50: NonNegativeFloat | None
    interval_ms_p95: NonNegativeFloat | None
    interval_ms_max: NonNegativeFloat | None
    n_gaps: int = Field(ge=0)
    estimated_dropped_frames: int = Field(ge=0)
    drop_share: UnitInterval | None  # dropped / (received + dropped)

    @model_validator(mode="after")
    def _recompute(self) -> FrameTiming:
        if self.n_frames >= 2 and self.span_s > 0:
            require_close("measured_fps", self.measured_fps, (self.n_frames - 1) / self.span_s)
            quantiles = [self.interval_ms_p50, self.interval_ms_p95, self.interval_ms_max]
            require_ordered("interval quantiles", [q for q in quantiles if q is not None])
        elif self.measured_fps is not None:
            raise ValueError("measured_fps must be null with fewer than 2 frames or zero span")
        if self.estimated_dropped_frames < self.n_gaps:
            raise ValueError("each gap hides at least one frame: dropped >= n_gaps")
        total = self.n_frames + self.estimated_dropped_frames
        expected_share = None if total == 0 else self.estimated_dropped_frames / total
        require_close("drop_share", self.drop_share, expected_share)
        return self


class StreamProbeReport(StrictModel):
    contract_id: Literal["StreamProbeReport"] = "StreamProbeReport"
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    tool_version: str
    source: StreamSource
    codec: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    profile: Annotated[str, StringConstraints(max_length=64)] | None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    declared_fps: float | None = Field(gt=0.0)
    duration_requested_s: float = Field(gt=0.0)
    expected_fps: float | None = Field(gt=0.0)
    timing: FrameTiming
    flags: list[StreamFlag]
    verdict: Verdict

    @model_validator(mode="after")
    def _derived(self) -> StreamProbeReport:
        expected = derive_stream_flags(self.timing, self.expected_fps)
        if self.flags != expected:
            raise ValueError(f"flags {self.flags} do not follow from timing: {expected}")
        if self.verdict != derive_stream_verdict(expected):
            raise ValueError("verdict does not follow from flags")
        return self


def derive_stream_flags(timing: FrameTiming, expected_fps: float | None) -> list[StreamFlag]:
    flags: list[StreamFlag] = []
    if timing.measured_fps is None:
        return ["TOO_FEW_FRAMES"]
    if expected_fps is not None and timing.measured_fps < FPS_SHORTFALL_WARN * expected_fps:
        flags.append("FPS_BELOW_EXPECTED")
    if timing.drop_share is not None and timing.drop_share > DROP_SHARE_WARN:
        flags.append("FRAME_DROPS")
    return flags


def derive_stream_verdict(flags: list[StreamFlag]) -> Verdict:
    if "TOO_FEW_FRAMES" in flags:
        return "FAIL"
    return "WARN" if flags else "PASS"


class LatencyQuantiles(StrictModel):
    p50_ms: NonNegativeFloat
    p90_ms: NonNegativeFloat
    p99_ms: NonNegativeFloat
    max_ms: NonNegativeFloat

    @model_validator(mode="after")
    def _ordered(self) -> LatencyQuantiles:
        require_ordered("latency quantiles", [self.p50_ms, self.p90_ms, self.p99_ms, self.max_ms])
        return self


class CameraBench(StrictModel):
    camera_id: CameraId
    n_frames: int = Field(ge=2)
    span_s: float = Field(gt=0.0)
    throughput_fps: float = Field(gt=0.0)
    n_out_of_order: int = Field(ge=0)  # frame_index not increasing along capture time
    end_to_end: LatencyQuantiles  # capture -> done
    inference: LatencyQuantiles  # infer_start -> infer_end (Hailo)
    queueing: LatencyQuantiles  # capture -> infer_start

    @model_validator(mode="after")
    def _recompute(self) -> CameraBench:
        require_close(f"{self.camera_id}.throughput_fps", self.throughput_fps, (self.n_frames - 1) / self.span_s)
        return self


class SystemSummary(StrictModel):
    n_samples: int = Field(ge=0)
    cpu_temp_c_max: float | None
    cpu_percent_p95: float | None = Field(ge=0.0, le=100.0)
    throttled_any: bool | None  # None when no sample carried throttled_flags

    @model_validator(mode="after")
    def _nulls(self) -> SystemSummary:
        if (self.n_samples == 0) != (self.cpu_temp_c_max is None):
            raise ValueError("system metrics must be null exactly when there are no samples")
        return self


class PipelineBenchReport(StrictModel):
    contract_id: Literal["PipelineBenchReport"] = "PipelineBenchReport"
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    tool_version: str
    inputs: list[InputFingerprint]
    target_fps: float | None = Field(gt=0.0)
    temp_warn_c: float
    latency_budget_ms: float = Field(gt=0.0)
    cameras: list[CameraBench] = Field(min_length=1)
    system: SystemSummary
    flags: list[BenchFlag]
    verdict: Verdict

    @model_validator(mode="after")
    def _derived(self) -> PipelineBenchReport:
        expected = derive_bench_flags(self)
        if self.flags != expected:
            raise ValueError(f"flags {self.flags} do not follow from the evidence: {expected}")
        if self.verdict != derive_bench_verdict(expected):
            raise ValueError("verdict does not follow from flags")
        return self


def derive_bench_flags(report: PipelineBenchReport) -> list[BenchFlag]:
    flags: list[BenchFlag] = []
    if report.target_fps is not None and any(
        c.throughput_fps < FPS_SHORTFALL_WARN * report.target_fps for c in report.cameras
    ):
        flags.append("FPS_BELOW_TARGET")
    # p90 rather than max: one slow frame is noise, a slow decile is a queue.
    if any(c.end_to_end.p90_ms > report.latency_budget_ms for c in report.cameras):
        flags.append("LATENCY_OVER_BUDGET")
    if report.system.cpu_temp_c_max is not None and report.system.cpu_temp_c_max >= report.temp_warn_c:
        flags.append("HIGH_TEMPERATURE")
    if report.system.throttled_any:
        flags.append("THROTTLED")
    if any(c.n_out_of_order > 0 for c in report.cameras):
        flags.append("OUT_OF_ORDER_FRAMES")
    return flags


def derive_bench_verdict(flags: list[BenchFlag]) -> Verdict:
    # Throttling or a latency budget miss means alerts arrive late: FAIL.
    if "THROTTLED" in flags or "LATENCY_OVER_BUDGET" in flags:
        return "FAIL"
    return "WARN" if flags else "PASS"
