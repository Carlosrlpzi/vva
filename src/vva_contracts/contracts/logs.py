"""Row contracts for the JSONL files exchanged with the VVA application.

Why this module exists
----------------------
The Raspberry Pi pipeline *produces* evidence (detections, frame logs, timing
records) and these tools *measure* it. The boundary between the two is a set
of line formats. Defining each line as a strict model means:

* the VVA app can import these models and validate before writing, and
* a malformed line is reported with its line number instead of being skipped.

All boxes are ``xyxyn`` unless a request says otherwise: normalised
(x_min, y_min, x_max, y_max) in [0, 1].
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from vva_contracts.contracts.base import Identifier, NonNegFloat, Ratio, StrictModel, require

Box = tuple[float, float, float, float]
ClassName = Annotated[str, Field(min_length=1, max_length=64)]


class PredictionRow(StrictModel):
    """One detector output for DETECTION_EVAL / QUANT_PARITY.

    ``image`` is the image key: its path relative to the split's image
    directory, without extension (e.g. ``"cam1_20261001_000123"``).
    """

    image: Annotated[str, Field(min_length=1, max_length=1024)]
    class_id: Annotated[int, Field(ge=0)]
    confidence: Ratio
    box: Box


class DetectionRow(StrictModel):
    """One detection inside a frame log line (always xyxyn)."""

    class_name: ClassName
    confidence: Ratio
    box: Box

    @model_validator(mode="after")
    def _box_is_ordered(self) -> DetectionRow:
        """x1 <= x2 and y1 <= y2, otherwise anchors and zones are meaningless."""
        x1, y1, x2, y2 = self.box
        require(x1 <= x2 and y1 <= y2, f"box must be xyxy with x1<=x2, y1<=y2: {self.box}")
        return self


class FrameRow(StrictModel):
    """One processed frame. Frames WITHOUT detections must be logged too.

    An empty ``detections`` list is what breaks a streak in the state machine;
    a log that only records positive frames would make every streak look
    unbroken and the replay would overstate alerts.
    """

    camera_id: Identifier
    frame_idx: Annotated[int, Field(ge=0)]
    ts: NonNegFloat
    detections: list[DetectionRow]


class EventRow(StrictModel):
    """A human-labelled ground-truth event interval for RULE_REPLAY."""

    camera_id: Identifier
    start_ts: NonNegFloat
    end_ts: NonNegFloat
    label: ClassName

    @model_validator(mode="after")
    def _interval_is_ordered(self) -> EventRow:
        """An event cannot end before it starts."""
        require(self.end_ts >= self.start_ts, "end_ts must be >= start_ts")
        return self


class TimingRow(StrictModel):
    """Per-frame timing record written by the pipeline (seconds, one clock).

    Stages: capture -> inference start -> inference end -> done (rules,
    persistence). A dropped frame only has ``t_capture``.
    """

    camera_id: Identifier
    frame_idx: Annotated[int, Field(ge=0)]
    t_capture: NonNegFloat
    t_infer_start: NonNegFloat | None = None
    t_infer_end: NonNegFloat | None = None
    t_done: NonNegFloat | None = None
    dropped: bool = False

    @model_validator(mode="after")
    def _stages_are_ordered(self) -> TimingRow:
        """Processed frames need all stamps, in causal order."""
        if self.dropped:
            return self
        stamps = (self.t_infer_start, self.t_infer_end, self.t_done)
        require(all(s is not None for s in stamps), "processed frames need every timestamp")
        start, end, done = (s for s in stamps if s is not None)
        require(self.t_capture <= start <= end <= done, "timestamps must satisfy capture<=start<=end<=done")
        return self


class SystemRow(StrictModel):
    """Periodic system sample (e.g. every 5 s) for thermal analysis."""

    ts: NonNegFloat
    cpu_temp_c: Annotated[float, Field(ge=-40.0, le=150.0)]
    cpu_percent: Annotated[float, Field(ge=0.0, le=100.0)] | None = None
