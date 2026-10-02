---
name: vva-bench-pipeline
description: Summarise per-frame timing logs and optional thermal logs from the Raspberry Pi pipeline (throughput, drop rate, end-to-end p95 latency against a budget, throttle risk) with the PIPELINE_BENCH task of vva_contract.
compatibility: opencode
metadata:
  task: PIPELINE_BENCH
  report: PipelineBenchReport
---
# PIPELINE_BENCH

## 1. When to use / When NOT to use
- Use: "does the Pi keep up with the cameras without dropping frames or overheating?" (milestones 1, 7, 10, 11).
- Do NOT use to probe a camera stream: use `vva-probe-stream`.

## 2. Preconditions (check with `read`, `glob`, `grep`)
- `timings_log` exists and is TimingRow JSONL: `{"camera_id", "frame_idx", "t_capture", "t_infer_start", "t_infer_end", "t_done", "dropped"}`; a dropped frame only has `t_capture` and `"dropped": true`.
- Optional `system_log` is SystemRow JSONL: `{"ts", "cpu_temp_c", "cpu_percent"}`.
- These logs are produced on the Pi, outside OpenCode. If one is missing, tell the user which one (TimingRow or SystemRow) is missing; never create it.
If one check fails, ask ONE concise question.

## 3. Request
Required: `timings_log`.
Optional (defaults copied from code): `system_log` null, `warmup_s` 30.0 (0..3600), `latency_budget_ms` 400.0 (> 0; equals the guide's end-to-end p95 alarm), `max_drop_rate` 0.01, `throttle_temp_c` 80.0 (40..110).
Illustrative tool arguments:
```json
{"task": "PIPELINE_BENCH", "params": {"timings_log": "data/logs/timings.jsonl", "system_log": "data/logs/system.jsonl", "warmup_s": 30.0, "latency_budget_ms": 400.0, "max_drop_rate": 0.01}}
```
Schema: `schemas/PipelineBenchReport.schema.json`, rows: `schemas/TimingRow.schema.json`, `schemas/SystemRow.schema.json`.

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
You may quote: `verdict`, `flags`, `overall.drop_rate`, `overall.throughput_fps`, `overall.end_to_end.p95_ms`, `overall.end_to_end.p50_ms`, per-camera values, `thermal.max_temp_c`.
FAIL comes from `no_processed_frames`, `latency_budget_exceeded` or `drop_rate_high`; `thermal_throttle_risk` alone gives WARN.
Never recompute, round, average or "improve" any number.

## 7. Domain pitfalls (why the rules exist)
- Frames inside the first `warmup_s` seconds are excluded: model load and cache warm-up are not steady state.
- The budget applies to end-to-end p95, not the mean: an alert system is judged by its slow frames.
- Percentiles use the `linear` method recorded in `percentile_method`; other tools may give different p95 on small samples.
- Performance measured on a laptop says nothing about the Pi + Hailo: only Pi-side logs count.

## 8. Output rule
The final answer is the raw JSON document only: no markdown fences, no prose.
(The fenced example in section 3 is illustrative only.)
