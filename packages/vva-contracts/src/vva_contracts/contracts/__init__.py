"""Registry of every report contract, keyed by its ``contract_id``.

Used by ``vva-contract schemas`` (JSON-schema export) and
``vva-contract validate`` (re-checking a report file offline).
"""

from __future__ import annotations

from pydantic import BaseModel

from vva_contracts.contracts.dataset_audit import DatasetAuditReport
from vva_contracts.contracts.detection_eval import DetectionEvalReport, QuantParityReport
from vva_contracts.contracts.records import BenchLine, FrameRecord, GroundTruthEvent, PredictionRecord, ZonesFile
from vva_contracts.contracts.requests import VVARequest
from vva_contracts.contracts.rule_replay import RuleReplayReport
from vva_contracts.contracts.stream import PipelineBenchReport, StreamProbeReport

REPORTS: dict[str, type[BaseModel]] = {
    "DatasetAuditReport": DatasetAuditReport,
    "DetectionEvalReport": DetectionEvalReport,
    "QuantParityReport": QuantParityReport,
    "RuleReplayReport": RuleReplayReport,
    "StreamProbeReport": StreamProbeReport,
    "PipelineBenchReport": PipelineBenchReport,
}

# Input-side formats are exported too, so producers on the Pi can validate
# their logs against the same schemas the auditor uses.
INPUT_FORMATS: dict[str, type[BaseModel]] = {
    "VVARequest": VVARequest,
    "FrameRecord": FrameRecord,
    "PredictionRecord": PredictionRecord,
    "GroundTruthEvent": GroundTruthEvent,
    "BenchLine": BenchLine,
    "ZonesFile": ZonesFile,
}
