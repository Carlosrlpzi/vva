---
name: vva-replay-rules
description: Replay recorded frame logs through one or more candidate alert policies (min confidence, consecutive frames, cooldown, zones) and measure alert load and event recall with the RULE_REPLAY task of vva_contract.
compatibility: opencode
metadata:
  task: RULE_REPLAY
  report: RuleReplayReport
---
# RULE_REPLAY

## 1. When to use / When NOT to use
- Use: comparing FRAME-LEVEL detection policies (confidence, consecutive frames, per-camera cooldown) on recorded footage.
- Do NOT use to validate production events: those are track-based (milestone 6) and are validated by the milestone-3 replay harness, not by this task.
- Do NOT use to score boxes: use `vva-eval-detection`. Do NOT use for latency: use `vva-bench-pipeline`.

## 2. Preconditions (check with `read`, `glob`, `grep`)
- `frames_log` exists and is FrameRow JSONL: `{"camera_id", "frame_idx", "ts", "detections": [{"class_name", "confidence", "box"}]}`, box always xyxyn.
- The log MUST include zero-detection frames (`"detections": []`): they break streaks.
- Optional `events` is EventRow JSONL: `{"camera_id", "start_ts", "end_ts", "label"}`.
- Zone keys must be camera ids present in the frames log.
If one check fails, ask ONE concise question.

## 3. Request
Required: `frames_log`, `policies` (1..50, unique `name`).
Each policy requires: `name`, `classes` (>= 1), `min_confidence` (0..1), `min_consecutive_frames` (1..100), `cooldown_s` (0..86400), `max_gap_s` (> 0, <= 3600); optional `zones` {camera_id: [[x, y], ...]} with >= 3 normalised points.
Optional: `events` null, `alert_grace_s` 10.0 (0..600), `max_alerts_listed` 500 (0..5000).
Illustrative tool arguments:
```json
{"task": "RULE_REPLAY", "params": {"frames_log": "data/logs/frames.jsonl", "events": "data/logs/events.jsonl", "policies": [{"name": "project_default", "classes": ["person"], "min_confidence": 0.6, "min_consecutive_frames": 3, "cooldown_s": 45.0, "max_gap_s": 1.0}, {"name": "front_zone", "classes": ["person"], "min_confidence": 0.6, "min_consecutive_frames": 3, "cooldown_s": 45.0, "max_gap_s": 1.0, "zones": {"front": [[0.1, 0.5], [0.9, 0.5], [0.9, 1.0], [0.1, 1.0]]}}]}}
```
Schema: `schemas/RuleReplayReport.schema.json`, rows: `schemas/FrameRow.schema.json`, `schemas/EventRow.schema.json`.

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
You may quote per policy: `n_alerts`, `alerts_per_hour`, `per_camera`, and with events `ground_truth.recall`, `ground_truth.alert_precision`, `ground_truth.latency_p50_s`, `ground_truth.latency_max_s`.
Compare policies only within the same report (same log). Never recompute, round, average or "improve" any number.

## 7. Domain pitfalls (why the rules exist)
- Per-frame false positives are correlated (a bush stays a bush), so p^k understates the real alert rate: never choose thresholds "on paper"; compare policies on the same log.
- Known limitation of this task: a false-positive alert opens the per-camera cooldown (for example 45 s) and can suppress a real event right after it. Say so whenever a loose policy shows low recall; production uses different, track-based cooldowns.
- A log without empty frames makes every streak look unbroken and overstates alerts.
- `max_gap_s` restarts the streak after dropped frames or reconnects: two detections 30 s apart are not consecutive.

## 8. Output rule
The final answer is the raw JSON document only: no markdown fences, no prose.
(The fenced example in section 3 is illustrative only.)
