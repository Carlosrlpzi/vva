"""Request contracts: what the agent is allowed to ask for.

Why this module exists
----------------------
The agent selects a task explicitly; there is no intent classifier. The
``task`` field is a *discriminator*: Pydantic picks the request model from it
and then applies that model's rules, so a field that belongs to another task
(e.g. ``policies`` on a DETECTION_EVAL request) is rejected as an extra field.

Every numeric default below is a documented starting point, not a magic
number; the README explains where each comes from.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from vva_contracts.contracts.base import Identifier, Ratio, RelPath, StrictModel, require
from vva_contracts.core.phash import MAX_GUARANTEED_DISTANCE
from vva_contracts.errors import ContractViolation

BoxFormat = Literal["xyxyn", "xywhn"]
Point = tuple[Ratio, Ratio]

# Only environment variables matching this pattern can be named by a request.
# Without the allowlist a model could ask the tool to "probe" DEEPSEEK_API_KEY
# and the secret would surface in an error message.
URL_ENV_PATTERN = r"^VVA_[A-Z0-9_]+_URL$"


class DatasetAuditRequest(StrictModel):
    """Audit a YOLO dataset described by an Ultralytics-style data.yaml."""

    task: Literal["DATASET_AUDIT"]
    data_yaml: RelPath
    # Letterbox target used to express box sizes in *inference* pixels.
    inference_width: int = Field(default=640, ge=32, le=4096)
    inference_height: int = Field(default=640, ge=32, le=4096)
    # <= 7 keeps the banded search exact (see core/phash.py).
    near_duplicate_max_hamming: int = Field(default=4, ge=0, le=MAX_GUARANTEED_DISTANCE)
    # Regex applied to image file stems; its named group "group" identifies
    # the recording (clip/camera/day) a frame belongs to.
    group_pattern: str | None = None
    max_examples: int = Field(default=20, ge=1, le=100)

    @field_validator("group_pattern")
    @classmethod
    def _pattern_has_group(cls, value: str | None) -> str | None:
        """Compile the regex now so a bad pattern fails before any I/O."""
        if value is not None:
            try:
                compiled = re.compile(value)
            except re.error as exc:
                raise ContractViolation(f"group_pattern is not a valid regex: {exc}") from exc
            require("group" in compiled.groupindex, "group_pattern needs a named group (?P<group>...)")
        return value


class _DetectionInputs(StrictModel):
    """Fields shared by DETECTION_EVAL and QUANT_PARITY."""

    data_yaml: RelPath
    split: Identifier = "val"
    box_format: BoxFormat = "xyxyn"
    # Project policy starting point (min_confidence: 0.60).
    operating_confidence: Ratio = 0.60
    # COCO default: at most 100 detections per image and class are scored.
    max_dets_per_image: int = Field(default=100, ge=1, le=1000)


class DetectionEvalRequest(_DetectionInputs):
    """Score one predictions file against the ground truth of one split."""

    task: Literal["DETECTION_EVAL"]
    predictions: RelPath


class QuantParityRequest(_DetectionInputs):
    """Compare a reference model (e.g. FP32 ONNX) with a candidate (INT8 HEF)."""

    task: Literal["QUANT_PARITY"]
    reference_predictions: RelPath
    candidate_predictions: RelPath
    max_map50_drop: Ratio = 0.02
    max_map50_95_drop: Ratio = 0.03
    agreement_iou: Ratio = 0.5


class PolicySpec(StrictModel):
    """One candidate alert policy to replay (mirrors core.state_machine.EventRule)."""

    name: Identifier
    classes: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(min_length=1)
    min_confidence: Ratio
    min_consecutive_frames: int = Field(ge=1, le=100)
    cooldown_s: float = Field(ge=0.0, le=86_400.0)
    max_gap_s: float = Field(gt=0.0, le=3_600.0)
    # Per-camera zone polygons in normalised coordinates; a camera without an
    # entry uses the whole frame.
    zones: dict[Identifier, list[Point]] = Field(default_factory=dict)

    @field_validator("zones")
    @classmethod
    def _polygons_are_polygons(
        cls, value: dict[str, list[tuple[float, float]]]
    ) -> dict[str, list[tuple[float, float]]]:
        """A polygon needs at least three vertices to enclose an area."""
        for camera, polygon in value.items():
            require(len(polygon) >= 3, f"zone for camera {camera!r} needs >= 3 points")
        return value


class RuleReplayRequest(StrictModel):
    """Replay recorded frames through candidate policies."""

    task: Literal["RULE_REPLAY"]
    frames_log: RelPath
    events: RelPath | None = None
    policies: list[PolicySpec] = Field(min_length=1, max_length=50)
    # An alert counts as on-time if it arrives before event end + grace.
    alert_grace_s: float = Field(default=10.0, ge=0.0, le=600.0)
    max_alerts_listed: int = Field(default=500, ge=0, le=5000)

    @model_validator(mode="after")
    def _unique_policy_names(self) -> RuleReplayRequest:
        """Policy names key the results, so duplicates would be ambiguous."""
        names = [p.name for p in self.policies]
        require(len(names) == len(set(names)), "policy names must be unique")
        return self


class StreamProbeRequest(StrictModel):
    """Measure a camera stream (RTSP via env var) or a recorded file."""

    task: Literal["STREAM_PROBE"]
    camera_id: Identifier
    url_env: Annotated[str, Field(pattern=URL_ENV_PATTERN)] | None = None
    path: RelPath | None = None
    duration_s: float = Field(default=10.0, ge=2.0, le=120.0)
    timeout_s: float = Field(default=30.0, ge=5.0, le=300.0)
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    min_fps_ratio: Ratio = 0.9

    @model_validator(mode="after")
    def _exactly_one_source(self) -> StreamProbeRequest:
        """Exactly one of url_env / path: ambiguity must fail, not pick one."""
        require((self.url_env is None) != (self.path is None), "set exactly one of url_env or path")
        return self


class PipelineBenchRequest(StrictModel):
    """Summarise per-frame timing (and optional thermal) logs from the Pi."""

    task: Literal["PIPELINE_BENCH"]
    timings_log: RelPath
    system_log: RelPath | None = None
    warmup_s: float = Field(default=30.0, ge=0.0, le=3_600.0)
    latency_budget_ms: float = Field(default=500.0, gt=0.0)
    max_drop_rate: Ratio = 0.01
    throttle_temp_c: float = Field(default=80.0, ge=40.0, le=110.0)


VVARequest = Annotated[
    DatasetAuditRequest
    | DetectionEvalRequest
    | QuantParityRequest
    | RuleReplayRequest
    | StreamProbeRequest
    | PipelineBenchRequest,
    Field(discriminator="task"),
]
REQUEST_ADAPTER: TypeAdapter[VVARequest] = TypeAdapter(VVARequest)
