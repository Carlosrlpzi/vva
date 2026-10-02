"""Line formats that the surveillance pipeline must write, and the tasks read.

Why this module exists
----------------------
The audit tasks never look at video. They read *logs* produced by the
pipeline running on the Pi. If those logs are loosely defined, every metric
computed from them is ambiguous. These models are the single definition of
each log line, and the pipeline itself should import them to validate what it
writes (``FrameRecord.model_validate(...)`` before ``json.dumps``).

Coordinate conventions (one convention per field, stated once)
--------------------------------------------------------------
* ``bbox_xyxy`` is in **pixels of the frame described by width/height of the
  same record**: ``[x_min, y_min, x_max, y_max]``, origin at top-left.
  If the detector ran on a letterboxed 640x640 tensor, the producer must map
  boxes back to the substream frame before logging. The evaluation task
  checks width/height against the real image size to catch that bug.
* Zone polygons are in **normalized coordinates** in [0, 1]. The same zone
  then applies unchanged to the substream (inference) and the main stream
  (evidence clips), whatever their resolutions.
* ``ts`` values are UNIX epoch seconds (float). Wall-clock time is used so
  that logs from different cameras can be merged and compared.

The "absence of evidence must be explicit" rule
-----------------------------------------------
A ``FrameRecord`` is written for **every processed frame, including frames
with zero detections**. The event rule "N consecutive positive frames" can
only be replayed if negative frames are in the log; a missing line would
otherwise be indistinguishable from a dropped frame.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Literal

from pydantic import Field, RootModel, StringConstraints, model_validator

from vva_contracts.contracts.base import (
    SCHEMA_VERSION,
    CameraId,
    ClassName,
    StrictModel,
    UnitInterval,
)

# Boxes may exceed the frame by up to one pixel because detectors round
# coordinates after rescaling. Anything larger is a coordinate-frame bug.
BOX_EDGE_TOLERANCE_PX = 1.0
MAX_FRAME_SIDE_PX = 16384
MAX_DETECTIONS_PER_FRAME = 1000


class Detection(StrictModel):
    """One detector output after NMS."""

    class_name: ClassName
    confidence: UnitInterval
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def _positive_extent(self) -> Detection:
        x_min, y_min, x_max, y_max = self.bbox_xyxy
        # A zero-area box has IoU 0 with everything; it is always a bug upstream.
        if not (x_max > x_min and y_max > y_min):
            raise ValueError(f"bbox_xyxy must satisfy x_max > x_min and y_max > y_min, got {self.bbox_xyxy}")
        return self


class _ImageDetections(StrictModel):
    """Fields shared by live frames and offline image predictions."""

    width: int = Field(gt=0, le=MAX_FRAME_SIDE_PX)
    height: int = Field(gt=0, le=MAX_FRAME_SIDE_PX)
    detections: list[Detection] = Field(max_length=MAX_DETECTIONS_PER_FRAME)

    @model_validator(mode="after")
    def _boxes_inside_frame(self) -> _ImageDetections:
        tol = BOX_EDGE_TOLERANCE_PX
        for det in self.detections:
            x_min, y_min, x_max, y_max = det.bbox_xyxy
            if x_min < -tol or y_min < -tol or x_max > self.width + tol or y_max > self.height + tol:
                raise ValueError(
                    f"box {det.bbox_xyxy} lies outside the {self.width}x{self.height} frame; "
                    "were coordinates logged in the model-input (letterbox) space?"
                )
        return self


class FrameRecord(_ImageDetections):
    """One processed frame from a live camera (input of RULE_REPLAY)."""

    camera_id: CameraId
    frame_index: int = Field(ge=0)
    ts: float = Field(ge=0.0)


class PredictionRecord(_ImageDetections):
    """Detector output for one dataset image (input of DETECTION_EVAL).

    ``image_id`` is the image path relative to ``images/<split>/`` without the
    file extension, using ``/`` as separator, e.g. ``cam1/2026-10-01_000123``.
    """

    image_id: Annotated[str, StringConstraints(min_length=1, max_length=1024)]


class GroundTruthEvent(StrictModel):
    """A human-annotated real event: someone/something was really there."""

    camera_id: CameraId
    start_ts: float = Field(ge=0.0)
    end_ts: float = Field(ge=0.0)
    label: ClassName

    @model_validator(mode="after")
    def _ordered(self) -> GroundTruthEvent:
        if self.end_ts <= self.start_ts:
            raise ValueError("end_ts must be strictly greater than start_ts")
        return self


class TimingRecord(StrictModel):
    """Per-frame stage timestamps (input of PIPELINE_BENCH).

    The four timestamps split end-to-end latency into stages:
    capture -> inference start (queueing/decoding), inference start -> end
    (Hailo), inference end -> done (tracker, rules, persistence).
    Use ``time.time()`` for all four, on the same host.
    """

    kind: Literal["timing"]
    camera_id: CameraId
    frame_index: int = Field(ge=0)
    ts_capture: float = Field(ge=0.0)
    ts_infer_start: float = Field(ge=0.0)
    ts_infer_end: float = Field(ge=0.0)
    ts_done: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _stages_in_order(self) -> TimingRecord:
        stages = [self.ts_capture, self.ts_infer_start, self.ts_infer_end, self.ts_done]
        if any(later < earlier for earlier, later in pairwise(stages)):
            raise ValueError(f"stage timestamps must be non-decreasing, got {stages}")
        return self


class SystemSample(StrictModel):
    """Periodic host health sample (input of PIPELINE_BENCH).

    ``throttled_flags`` is the integer from ``vcgencmd get_throttled`` on the
    Pi (0 means no under-voltage/throttling ever observed); ``None`` when not
    sampled.
    """

    kind: Literal["system"]
    ts: float = Field(ge=0.0)
    cpu_temp_c: float = Field(ge=-40.0, le=150.0)
    cpu_percent: float = Field(ge=0.0, le=100.0)
    throttled_flags: int | None = Field(default=None, ge=0)


class BenchLine(RootModel[Annotated[TimingRecord | SystemSample, Field(discriminator="kind")]]):
    """One line of a bench log: either a timing record or a system sample."""


class Zone(StrictModel):
    """A named polygon in normalized coordinates."""

    name: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    polygon: list[list[float]] = Field(min_length=3, max_length=64)

    @model_validator(mode="after")
    def _valid_polygon(self) -> Zone:
        for vertex in self.polygon:
            if len(vertex) != 2 or not all(0.0 <= v <= 1.0 for v in vertex):
                raise ValueError(f"zone {self.name!r}: vertices must be [x, y] pairs in [0, 1], got {vertex}")
        # Imported lazily to keep this module free of numpy at import time.
        from vva_contracts.geometry import polygon_area

        # A degenerate polygon (collinear points) contains no point, so a zone
        # rule using it would silently never fire. Reject it up front.
        if polygon_area(self.polygon) < 1e-6:
            raise ValueError(f"zone {self.name!r}: polygon area is ~0 (collinear or repeated vertices)")
        return self


class ZonesFile(StrictModel):
    """Content of ``configs/zones.json``: zones per camera."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    cameras: dict[CameraId, list[Zone]]
