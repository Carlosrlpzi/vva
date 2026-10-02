"""Requests accepted by ``vva-contract run`` (and the ``vva_contract`` tool).

Why a discriminated union instead of one flat model
---------------------------------------------------
The original ``MLRequest`` is one model with every optional field, plus custom
checks that forbid, say, ``target`` on a code review. A discriminated union
makes that structural: each task has its own model with ``extra="forbid"``, so
a field that does not belong to the selected task is rejected automatically,
and the JSON schema an agent reads lists exactly the valid fields per task.

Every path is workspace-relative and re-checked by ``workspace.py``.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, RootModel, StringConstraints, field_validator, model_validator

from vva_contracts.contracts.base import ClassName, StrictModel, UnitInterval

RelPath = Annotated[str, StringConstraints(min_length=1, max_length=4096)]
SplitName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,32}$")]


class DatasetAuditRequest(StrictModel):
    """Audit a YOLO-format dataset before training or evaluating on it."""

    task: Literal["DATASET_AUDIT"]
    dataset_root: RelPath
    data_yaml: RelPath | None = None  # default: <dataset_root>/data.yaml
    splits: list[SplitName] = Field(default_factory=lambda: ["train", "val"], min_length=1, max_length=4)
    # Resolution of the camera SUBSTREAM the detector will see, [width, height].
    # Required on purpose: the "is this object big enough to detect" question
    # has no meaningful default.
    inference_size: list[int] = Field(min_length=2, max_length=2)
    min_box_px: int = Field(default=16, ge=2, le=512)
    group_regex: Annotated[str, StringConstraints(min_length=1, max_length=200)] | None = None
    dhash_hamming_threshold: int = Field(default=4, ge=0, le=7)
    max_images: int = Field(default=50_000, ge=1, le=200_000)

    @field_validator("inference_size")
    @classmethod
    def _positive_size(cls, value: list[int]) -> list[int]:
        if not all(16 <= v <= 8192 for v in value):
            raise ValueError("inference_size values must be in [16, 8192] pixels")
        return value

    @field_validator("group_regex")
    @classmethod
    def _regex_has_group(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            compiled = re.compile(value)
        except re.error as exc:
            raise ValueError(f"group_regex does not compile: {exc}") from exc
        if "group" not in compiled.groupindex:
            raise ValueError("group_regex must define a named group: (?P<group>...)")
        return value

    @field_validator("splits")
    @classmethod
    def _unique_splits(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("splits must be unique")
        return value


class DetectionEvalRequest(StrictModel):
    """Evaluate one predictions file against a labeled split."""

    task: Literal["DETECTION_EVAL"]
    dataset_root: RelPath
    data_yaml: RelPath | None = None
    split: SplitName = "val"
    predictions: RelPath  # JSONL of PredictionRecord
    operating_confidence: UnitInterval = 0.60
    classes: list[ClassName] | None = Field(default=None, min_length=1, max_length=100)


class QuantParityRequest(StrictModel):
    """Compare a reference model (e.g. ONNX FP32) with a candidate (e.g. HEF INT8)."""

    task: Literal["QUANT_PARITY"]
    dataset_root: RelPath
    data_yaml: RelPath | None = None
    split: SplitName = "val"
    reference_predictions: RelPath
    candidate_predictions: RelPath
    operating_confidence: UnitInterval = 0.60
    max_map50_drop: UnitInterval = 0.02
    classes: list[ClassName] | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def _different_files(self) -> QuantParityRequest:
        if self.reference_predictions == self.candidate_predictions:
            raise ValueError("reference_predictions and candidate_predictions must differ")
        return self


class EventPolicy(StrictModel):
    """One candidate alert rule to replay. Units are in the field names."""

    name: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]
    classes: list[ClassName] = Field(min_length=1, max_length=20)
    min_confidence: UnitInterval
    min_consecutive_frames: int = Field(ge=1, le=100)
    cooldown_s: float = Field(ge=0.0, le=86_400.0)


class RuleReplayRequest(StrictModel):
    """Replay logged frames through candidate alert policies."""

    task: Literal["RULE_REPLAY"]
    frames: RelPath  # JSONL of FrameRecord, ts-ordered within each camera
    policies: list[EventPolicy] = Field(min_length=1, max_length=50)
    zones: RelPath | None = None  # JSON ZonesFile; cameras without zones use the full frame
    ground_truth_events: RelPath | None = None  # JSONL of GroundTruthEvent
    match_tolerance_s: float = Field(default=2.0, ge=0.0, le=600.0)
    max_frame_gap_s: float = Field(default=1.0, gt=0.0, le=3600.0)

    @field_validator("policies")
    @classmethod
    def _unique_names(cls, value: list[EventPolicy]) -> list[EventPolicy]:
        names = [p.name for p in value]
        if len(set(names)) != len(names):
            raise ValueError("policy names must be unique")
        return value


class StreamProbeRequest(StrictModel):
    """Measure a video source: codec, resolution, real frame rate, drops.

    Exactly one source:
    * ``url_env``: NAME of an environment variable holding the RTSP URL. The
      URL (which contains the camera password) never travels through the
      model; only the variable name does. The name pattern prevents using the
      tool to read unrelated secrets such as ``DEEPSEEK_API_KEY``.
    * ``path``: a recorded clip inside the workspace (also used by tests).
    """

    task: Literal["STREAM_PROBE"]
    url_env: Annotated[str, StringConstraints(pattern=r"^VVA_CAM_[A-Z0-9_]{1,48}_URL$")] | None = None
    path: RelPath | None = None
    duration_s: float = Field(default=10.0, ge=2.0, le=120.0)
    expected_fps: float | None = Field(default=None, gt=0.0, le=120.0)
    rtsp_transport: Literal["tcp", "udp"] = "tcp"

    @model_validator(mode="after")
    def _exactly_one_source(self) -> StreamProbeRequest:
        if (self.url_env is None) == (self.path is None):
            raise ValueError("provide exactly one of url_env or path")
        return self


class PipelineBenchRequest(StrictModel):
    """Summarize a bench log (timing records + system samples)."""

    task: Literal["PIPELINE_BENCH"]
    bench_log: RelPath  # JSONL of BenchLine
    target_fps: float | None = Field(default=None, gt=0.0, le=120.0)
    temp_warn_c: float = Field(default=80.0, ge=40.0, le=110.0)
    latency_budget_ms: float = Field(default=500.0, gt=0.0, le=60_000.0)


TaskRequest = Annotated[
    DatasetAuditRequest
    | DetectionEvalRequest
    | QuantParityRequest
    | RuleReplayRequest
    | StreamProbeRequest
    | PipelineBenchRequest,
    Field(discriminator="task"),
]


class VVARequest(RootModel[TaskRequest]):
    """Top-level request: the ``task`` field selects the model."""
