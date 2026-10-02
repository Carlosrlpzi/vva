"""Deterministic dispatch: the request's ``task`` field selects exactly one function.

There is no intent classifier and no fallback. The model chooses the task
explicitly; an unknown task is rejected by the request contract before
reaching this table.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from vva_contracts.tasks import dataset_audit, detection_eval, pipeline_bench, rule_replay, stream_probe

TaskHandler = Callable[[Any, Path], BaseModel]

HANDLERS: dict[str, TaskHandler] = {
    "DATASET_AUDIT": dataset_audit.run,
    "DETECTION_EVAL": detection_eval.run,
    "QUANT_PARITY": detection_eval.run_parity,
    "RULE_REPLAY": rule_replay.run,
    "STREAM_PROBE": stream_probe.run,
    "PIPELINE_BENCH": pipeline_bench.run,
}
