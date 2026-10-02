"""Task registry: the single, explicit map from task name to runner and contract.

Why this module exists
----------------------
Dispatch is a dictionary lookup, not a model decision. Reading this file tells
you every task the tool can execute and which contract its output must pass.
Adding a task means adding one line here, plus its request, contract, runner
and tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vva_contracts.contracts.base import StrictModel
from vva_contracts.contracts.dataset_audit import DatasetAuditReport
from vva_contracts.contracts.detection import DetectionEvalReport, QuantParityReport
from vva_contracts.contracts.rule_replay import RuleReplayReport
from vva_contracts.contracts.stream import PipelineBenchReport, StreamProbeReport
from vva_contracts.tasks.dataset_audit import run_dataset_audit
from vva_contracts.tasks.detection import run_detection_eval, run_quant_parity
from vva_contracts.tasks.rule_replay import run_rule_replay
from vva_contracts.tasks.stream import run_pipeline_bench, run_stream_probe


@dataclass(frozen=True)
class TaskSpec:
    """A runner and the report contract its output must satisfy."""

    runner: Callable[[Any, Path], StrictModel]
    report: type[StrictModel]


TASKS: dict[str, TaskSpec] = {
    "DATASET_AUDIT": TaskSpec(run_dataset_audit, DatasetAuditReport),
    "DETECTION_EVAL": TaskSpec(run_detection_eval, DetectionEvalReport),
    "QUANT_PARITY": TaskSpec(run_quant_parity, QuantParityReport),
    "RULE_REPLAY": TaskSpec(run_rule_replay, RuleReplayReport),
    "STREAM_PROBE": TaskSpec(run_stream_probe, StreamProbeReport),
    "PIPELINE_BENCH": TaskSpec(run_pipeline_bench, PipelineBenchReport),
}
