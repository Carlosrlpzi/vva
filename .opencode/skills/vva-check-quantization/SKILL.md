---
name: vva-check-quantization
description: Compare a reference model (for example FP32 ONNX) with a quantised candidate (INT8 HEF on the Hailo-10H) on the same labelled split and operating confidence using the QUANT_PARITY task of vva_contract.
compatibility: opencode
metadata:
  task: QUANT_PARITY
  report: QuantParityReport
---
# QUANT_PARITY

## 1. When to use / When NOT to use
- Use: "how much accuracy did compiling to INT8 HEF cost?" (milestone 1 exit criterion).
- Do NOT use to score a single model: use `vva-eval-detection`. Unaudited dataset: run `vva-audit-dataset` first.

## 2. Preconditions (check with `read`, `glob`, `grep`)
- Two PredictionRow JSONL files exist: reference and candidate, produced on the SAME images of the SAME split.
- Both use the same `box_format`.
If one check fails, ask ONE concise question.

## 3. Request
Required: `data_yaml`, `reference_predictions`, `candidate_predictions`.
Optional (defaults copied from code): `split` "val", `box_format` "xyxyn", `operating_confidence` 0.6, `max_dets_per_image` 100, `max_map50_drop` 0.02, `max_map50_95_drop` 0.03, `agreement_iou` 0.5.
Illustrative tool arguments:
```json
{"task": "QUANT_PARITY", "params": {"data_yaml": "data/datasets/front/data.yaml", "reference_predictions": "data/predictions/val_fp32.jsonl", "candidate_predictions": "data/predictions/val_int8.jsonl", "max_map50_drop": 0.02, "max_map50_95_drop": 0.03}}
```
Schema: `schemas/QuantParityReport.schema.json`.

## 4. Call
Call `vva_contract` once with that single object. The "task" key is a top-level argument, never inside `params`.

## 5. Result handling
| Outcome | Action |
|---|---|
| exit 0 (report JSON) | Return the tool JSON exactly as received. |
| exit 2 `contract_violation` | Return the error JSON unchanged; ask for the ONE field to fix. |
| exit 3 `invalid_input` | Return the error JSON unchanged; ask for the ONE missing or broken input. |
| exit 4 `policy_blocked` | Return the error JSON unchanged; ask for a workspace-relative path or an allowed env var name. Never try to bypass the policy. |
| exit 6 `subprocess_timeout` | Return the error JSON unchanged; ask whether to retry with a larger `timeout_s` or check the source. |
| crash (exit 1, no typed error) | Report it as an internal error. Do not retry more than once. Never present it as a report. |

## 6. Interpretation boundary
You may quote: `verdict`, `failed_checks`, `delta_map50`, `delta_map50_95` (candidate minus reference), `agreement.n_matched`, `agreement.mean_iou`.
The verdict follows `failed_checks` only: PASS when it is empty, FAIL otherwise (`map50_drop`, `map50_95_drop`, `no_ground_truth`).
Never recompute, round, average or "improve" any number.

## 7. Domain pitfalls (why the rules exist)
- INT8 quantisation changes weights and activations; the drop must be measured on your footage, not assumed from a benchmark.
- A gain never fails the check: only reference minus candidate above the tolerance does.
- Low agreement with a small mAP drop means the boxes moved: inspect it before trusting the tracker on the candidate.

## 8. Output rule
The final answer is the raw JSON document only: no markdown fences, no prose.
(The fenced example in section 3 is illustrative only.)
