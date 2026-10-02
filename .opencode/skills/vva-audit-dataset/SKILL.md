---
name: vva-audit-dataset
description: Audit a YOLO dataset (labels, class balance, box sizes at inference resolution, train/val leakage by near-duplicates and by recording group) with the DATASET_AUDIT task of vva_contract. Use before any detection evaluation on a dataset not yet audited.
compatibility: opencode
metadata:
  task: DATASET_AUDIT
  report: DatasetAuditReport
---
# DATASET_AUDIT

## 1. When to use / When NOT to use
- Use: the user wants to know whether a labelled dataset is trustworthy, or wants to evaluate a detector on a dataset that has no audit yet.
- Do NOT use for scoring a detector: use `vva-eval-detection`. Not for FP32 vs INT8: use `vva-check-quantization`.

## 2. Preconditions (check with `read`, `glob`, `grep` before calling)
- The data.yaml path exists and is workspace-relative (no absolute path, no `..`).
- data.yaml has `names` (list or {id: name}) and at least one of `train`, `val`, `test`.
- If file stems encode the recording (example `cam1_20261001_0001`), propose a `group_pattern`.
If one check fails, ask ONE concise question.

## 3. Request
Required: `data_yaml`.
Optional (defaults copied from code): `inference_width` 640 (32..4096), `inference_height` 640 (32..4096), `near_duplicate_max_hamming` 4 (0..7), `group_pattern` null, `max_examples` 20 (1..100).
`group_pattern` is a regex over image stems and MUST contain the named group `(?P<group>...)`.
Illustrative tool arguments:
```json
{"task": "DATASET_AUDIT", "params": {"data_yaml": "data/datasets/front/data.yaml", "inference_width": 640, "inference_height": 640, "group_pattern": "^(?P<group>cam[0-9]+_[0-9]{8})_[0-9]+$"}}
```
Schema: `schemas/DatasetAuditReport.schema.json`, request schema: `schemas/VVARequest.schema.json`.

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
You may quote: `verdict`, `flags`, per-split counts (`n_images`, `n_boxes`, `class_counts`, `issue_counts`), `leakage.near_duplicate_pairs`, `leakage.n_groups_spanning_splits`, `box_sizes.frac_small`.
A FAIL verdict is a valid report, not a tool error: relay it. FAIL comes from `label_errors`, `cross_split_near_duplicates` or `group_leakage`; any other flag gives WARN.
Never recompute, round, average or "improve" any number.

## 7. Domain pitfalls (why the rules exist)
- Split by recording (clip/camera/day), never by frame: consecutive frames are near-duplicates, so a frame-level split leaks training data into validation and inflates mAP.
- Box sizes are measured after letterboxing to the inference size: a person that is large in the main stream can be "small" (< 32x32 px) at 640x640.
- `near_duplicate_max_hamming` above 7 is rejected because the banded search is only exact up to that distance.
- `unmatched_group_pattern` means some stems did not match the regex: leakage by group is then only partially measured.

## 8. Output rule
The final answer is the raw JSON document only: no markdown fences, no prose.
(The fenced example in section 3 is illustrative only.)
