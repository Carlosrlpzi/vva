---
name: vva-probe-stream
description: Measure a camera stream (codec, resolution, declared vs measured fps, keyframes, timestamp gaps) through ffprobe with the STREAM_PROBE task of vva_contract, using an env var name for RTSP or a recorded file path.
compatibility: opencode
metadata:
  task: STREAM_PROBE
  report: StreamProbeReport
---
# STREAM_PROBE

## 1. When to use / When NOT to use
- Use: "does this camera substream really deliver the configured fps and resolution?" (milestone 2), or to check a recorded clip.
- Do NOT use for end-to-end pipeline latency: use `vva-bench-pipeline`.

## 2. Preconditions (check with `read`, `glob`, `grep`)
- Ask for the env var NAME only, never a URL, IP, user or password. If the user pastes a URL, do not repeat it; ask for the variable name instead.
- The name must match `^VVA_[A-Z0-9_]+_URL$` and, to reach the tool process, `^VVA_CAM_[A-Z0-9_]+_URL$` (example `VVA_CAM_FRONT_SUB_URL`).
- For a recorded file, the path is workspace-relative and exists.
- Exactly one of `url_env` or `path`.
If one check fails, ask ONE concise question.

## 3. Request
Required: `camera_id` and exactly one of `url_env` / `path`.
Optional (defaults copied from code): `duration_s` 10.0 (2..120), `timeout_s` 30.0 (5..300), `rtsp_transport` "tcp" (or "udp"), `min_fps_ratio` 0.9.
Illustrative tool arguments:
```json
{"task": "STREAM_PROBE", "params": {"camera_id": "front_sub", "url_env": "VVA_CAM_FRONT_SUB_URL", "duration_s": 10.0, "rtsp_transport": "tcp"}}
```
Schema: `schemas/StreamProbeReport.schema.json`.

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
You may quote: `codec`, `width`, `height`, `declared_fps`, `measured_fps`, `fps_ratio`, `n_keyframes`, `keyframe_interval_frames`, `n_gaps`, `flags`.
Relay `declared_fps_unknown` and `timestamp_gaps` as-is. `source_ref` holds the env var name, never the URL.
Never recompute, round, average or "improve" any number.

## 7. Domain pitfalls (why the rules exist)
- Probe the substream (inference) and the main stream (evidence) separately: they have different resolution, fps and GOP.
- The pre-roll buffer can only cut at keyframes: `keyframe_interval_frames` divided by fps should be about 1 s.
- RTSP URLs carry user:password; that is why only the variable name ever appears in chat.
- A short probe can miss intermittent gaps; repeat over time rather than extending one probe indefinitely.

## 8. Output rule
The final answer is the raw JSON document only: no markdown fences, no prose.
(The fenced example in section 3 is illustrative only.)
