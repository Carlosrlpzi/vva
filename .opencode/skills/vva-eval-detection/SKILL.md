---
name: vva-eval-detection
description: Score one detector predictions file against the ground truth of one labelled split (COCO AP50, AP50-95, precision/recall at the operating confidence) with the DETECTION_EVAL task of vva_contract.
compatibility: opencode
metadata:
  task: DETECTION_EVAL
  report: DetectionEvalReport
---
# DETECTION_EVAL

## 1. When to use / When NOT to use
- Use: "how good is the detector on our footage?" for one model and one split.
- Do NOT use to compare FP32 against INT8/HEF: use `vva-check-quantization`. Do NOT use on a dataset that was never audited: run `vva-audit-dataset` first.

## 2. Preconditions (check with `read`, `glob`, `grep`)
- data.yaml exists and defines the requested split (default `val`).
- The predictions file exists and is PredictionRow JSONL: one object per line
  `{"image": "<stem without extension>", "class_id": <int>, "confidence": <0..1>, "box": [a, b, c, d]}`.
- The predictions were produced on the SAME split, and `box_format` matches the file.
If one check fails, ask ONE concise question.

## 3. Request
Required: `data_yaml`, `predictions`.
Optional (defaults copied from code): `split` "val", `box_format` "xyxyn" (or "xywhn"), `operating_confidence` 0.6, `max_dets_per_image` 100 (1..1000).
Illustrative tool arguments:
```json
{"task": "DETECTION_EVAL", "params": {"data_yaml": "data/datasets/front/data.yaml", "predictions": "data/predictions/val_int8.jsonl", "split": "val", "box_format": "xyxyn", "operating_confidence": 0.6}}
```
Schema: `schemas/DetectionEvalReport.schema.json`, row schema: `schemas/PredictionRow.schema.json`.

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
Any label error in the ground truth aborts with exit 3 and a message that asks to run DATASET_AUDIT: tell the user to run `vva-audit-dataset`.

## 6. Interpretation boundary
You may quote: `metrics.map50`, `metrics.map50_95`, and per class `ap50`, `ap50_95`, `precision`, `recall`, `f1`, `tp`, `fp`, `fn`, `n_gt`.
Never say "the detector works" without citing `map50`/`map50_95` AND precision/recall at `operating_confidence`.
A class with `n_gt` 0 has null AP on purpose: never treat it as zero or average it in.
Never recompute, round, average or "improve" any number.

## 7. Domain pitfalls (why the rules exist)
- AP summarises the whole confidence range; the alert rule runs at ONE confidence, so precision/recall at the operating point is what the user will experience.
- AP50-95 averages 10 IoU thresholds (0.50..0.95): it punishes loose boxes that AP50 accepts.
- Day and night_ir footage behave differently: evaluate each condition as its own split instead of pooling.
- Scoring on frames from recordings used in training inflates every metric (see `vva-audit-dataset`).

## 8. Output rule
The final answer is the raw JSON document only: no markdown fences, no prose.
(The fenced example in section 3 is illustrative only.)
