---
name: dev-emit-evidence-logs
description: Rules for making pipeline modules write the JSONL evidence logs (FrameRow, TimingRow, SystemRow, PredictionRow) that the vva_contract measurement tasks consume. Use when implementing ingest, detector, tracker, events, observability or the replay harness.
compatibility: opencode
metadata:
  role: vva-builder
  phase: build
---
# Emit evidence logs

## 1. Why
The Pi pipeline PRODUCES evidence; `vva_contracts` MEASURES it. The boundary is a set of
JSONL line formats defined in `src/vva_contracts/contracts/logs.py`. Import those models
and validate each row before writing it. Never define a parallel format.

## 2. Which module writes what
| Row | Writer | Consumer task | Key rule |
|---|---|---|---|
| `FrameRow` | event stage, one row per processed frame and camera | RULE_REPLAY | Frames with zero detections MUST be written (`"detections": []`) |
| `TimingRow` | observability, one row per captured frame | PIPELINE_BENCH | Dropped frame: only `t_capture` and `dropped: true`; processed: `t_capture <= t_infer_start <= t_infer_end <= t_done` |
| `SystemRow` | scheduler, periodic (example every 5 s) | PIPELINE_BENCH | `cpu_temp_c` in [-40, 150] |
| `PredictionRow` | offline detector run on a labelled split | DETECTION_EVAL, QUANT_PARITY | `image` is the stem without extension; boxes `xyxyn` unless the request says otherwise |

## 3. Rules
- Boxes in logs are normalised `xyxyn` with x1 <= x2 and y1 <= y2, after undoing the
  letterbox (inverse affine), never in letterboxed coordinates.
- One clock per file: all `t_*` fields of a TimingRow come from the same monotonic clock.
- Timestamps per camera never go backwards; the replay rejects it.
- Logs are written by a dedicated writer (never on the per-frame hot path), with rotation.
- Real logs stay on the Pi or the local disk. They never go into prompts or fixtures;
  tests use `tests/builders.py` synthetic data.

## 4. Verification
Add a test that writes rows through your writer and reads them back with the row model
(`model_validate_json`) for each line.
