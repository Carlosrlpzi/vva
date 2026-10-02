# VVA module contracts (MVP v11.1)

One card per module. A card is the contract the builder implements and the reviewer
checks. Section numbers refer to `docs/mvp-guide.md` (v11.2). Application code lives in
`src/vva_app/`; every guide path `src/<x>/...` maps to `src/vva_app/<x>/...`. "Tier" says who does the work:
Flash = vva-builder alone; Pro = escalate per `dev-escalate-pro` for the marked parts.
"Verified by" names the measurement that proves the module on real data.

Shared types (create once, in `src/vva_app/common/types.py`, before M1):

| Type | Fields (minimum) | Notes |
|---|---|---|
| `FramePacket` | `camera_id`, `frame_idx`, `ts_mono`, `ts_utc`, `image` (ndarray) | Both clocks read at capture time |
| `Detection` | `class_name`, `confidence`, `box_xyxyn` | Original-frame normalised coords (letterbox undone) |
| `Track` | `track_id`, `class_name`, `box_xyxyn`, `hits`, `is_confirmed`, `last_seen_mono` | `is_confirmed` is the only confirmation signal downstream |
| `Event` | `event_id`, `camera_id`, `zone_id`, `track_id`, `kind`, `ts_mono`, `ts_utc` | MVP `kind` = `entry` only |

## C1 Config validation (§1.18) - cross-cutting, start at M1
- Path: `src/vva_app/config/validate.py`. Each milestone adds the schema of the YAML it introduces.
- Invariants: Pydantic models (extra="forbid") per YAML; cross-file checks (`detector.confidence_threshold == tracker.low_confidence_threshold`, zone files exist, no unresolved `<...>` placeholders); `same_camera_same_event_seconds` exists only in `notify.yaml` (rejected in `events.yaml`); runs once at startup, config changes need a restart.
- Required tests: each cross-file rule rejects a broken synthetic config.
- Tier: Flash.

Log rows written to JSONL are NOT redefined: import them from
`vva_contracts.contracts.logs` (see skill `dev-emit-evidence-logs`).

---

## M0 Pixel budget (§0) - milestone 0
- Path: `src/vva_app/tools/pixel_budget.py` + values recorded in `configs/cameras/<id>.yaml`.
- Consumes: image height P_h, focal length f, sensor height S_h, target height H, distance D, IR range.
- Produces: `h_px = P_h * f * H / (D * S_h)` per camera and zone boundary, day and night (night limited by IR range); letterbox scale to 640x640.
- Invariants: same length units for H and D; letterbox scale = min(640/W, 640/H_img).
- Required tests: the guide's worked table (4.0 mm, 5 m, 1/3", 640x360 -> ~163 px) reproduced within rounding.
- Verified by: measured pixel height of a person at the zone boundary (manual), DATASET_AUDIT `box_sizes`.
- Tier: Flash.

## M1 Detector on Hailo-10H (§1.3) - milestone 1
- Path: `src/vva_app/inference/detector.py`, config `configs/models.yaml` (owner of `confidence_threshold`, `nms_iou_threshold`, `input_size`, `hef_path`).
- Consumes: BGR frame. Produces: list[`Detection`].
- Invariants: letterbox with pad 114, never stretch; inverse affine applied to boxes; `confidence_threshold == tracker.low_confidence_threshold` (0.15); failure policy fail-loud (startup identify + HEF load; mid-run failure marks `/health` degraded, never silent).
- Required tests: letterbox forward/inverse round trip on synthetic boxes; fake NPU raising mid-run -> health degraded.
- Verified by: QUANT_PARITY (FP32 ONNX vs INT8 HEF, same split), DETECTION_EVAL per condition (day / night_ir), PIPELINE_BENCH (inference latency).
- Tier: Flash; Pro for performance (T4).

## M2 RTSP/ONVIF ingest (§1.1, §1.10) - milestone 2
- Path: `src/vva_app/ingest/rtsp_reader.py`, `src/vva_app/ingest/onvif_client.py`.
- Consumes: env var names only: `VVA_CAM_<ID>_<SUB|MAIN>_URL`, `VVA_CAM_<ID>_ONVIF_HOST`, `VVA_CAM_<ID>_ONVIF_USER`, `VVA_CAM_<ID>_ONVIF_PASSWORD`. Produces: `FramePacket` through a single-slot mailbox.
- Invariants: newest frame overwrites the slot (no queue growth); exponential-backoff reconnect; stale-data signal after `frame_stale_after_seconds`; monotonic + UTC stamps at read time; credentials never logged (redact `user:pass@`).
- Required tests: mailbox drops old frames under a slow consumer (fake source); backoff sequence; redaction of URLs in log messages.
- Verified by: STREAM_PROBE on substream and main stream separately (fps ratio, keyframe interval ~1 s, gaps); PIPELINE_BENCH drop rate.
- Tier: Flash.

## M3 Zones, labelling and replay harness (§1.5 zones, §1.11) - milestone 3
- Path: `configs/zones/<camera_id>.json`, `src/vva_app/eval/replay_runner.py`, config `configs/eval.yaml`.
- Consumes: recordings, ground-truth CSV with `condition` (`day` / `night_ir`). Produces: precision/recall/F1 per condition with Wilson intervals, per-stage traces.
- Role (decided 2026-10-02): this harness is the authority for PRODUCTION event validation. It runs the real M6 state machine (and the M8 throttle decision, without sending) over recordings with a fake clock. RULE_REPLAY only compares frame-level detection policies.
- Invariants: fake clock; matching rule explicit (`one_to_one_greedy_nearest`, window 5 s); held-out days never used for tuning; fail fast on unresolved `<...>` placeholders; `night_ir_insect_activity` must emit zero events.
- Required tests: Wilson interval against closed-form values; matching rule on a synthetic timeline with known TP/FP/FN; the harness imports `vva_app.events.state_machine` (no re-implementation).
- Verified by: RULE_REPLAY (alert load vs recall per policy), DATASET_AUDIT before any evaluation.
- Tier: Flash for I/O; Pro for the matching rule and Wilson implementation (T3).

## M4 Motion gate (§1.2) - milestone 4
- Path: `src/vva_app/ingest/motion_gate.py`, config `configs/motion_gate.yaml`.
- Consumes: `FramePacket`, zone polygons. Produces: pass/drop decision + motion ratio.
- Invariants: MOG2 foreground test is `== 255` (127 = shadow), never `> 0`; `learningRate = dt / background_adaptation_seconds`; ratio computed only inside zone+margin mask; forced inference after `maximum_idle_inference_interval_seconds`.
- Required tests: synthetic mask with shadow pixels counts only 255; learning rate scales with dt.
- Verified by: replay harness (thresholds tuned by re-running, not by eye); PIPELINE_BENCH.
- Tier: Flash.

## M5 Tracker Kalman + IoU (§1.4, §4.1) - milestone 5
- Path: `src/vva_app/inference/tracker.py`, config `configs/tracker.yaml` (sole owner of `min_confirmed_hits`).
- Consumes: list[`Detection`] + dt. Produces: list[`Track`].
- Invariants: F(dt) rebuilt every predict; Q(dt) continuous white-noise block (1/3 dt^3, 1/2 dt^2, dt) times `sigma_accel_sq` (units px^2/s^4); expiry in seconds (`max_missed_seconds`); two-tier association (high 0.45, low 0.15).
- Required tests: Q(dt) against closed-form values; IoU against hand-computed boxes; identity kept through a short synthetic gap.
- Verified by: replay harness per-stage traces (confirmed-track stage).
- Tier: Pro (T3) for F/Q/association; Flash for wiring and config.

## M6 Event state machine, entry only (§1.5, v11.2) - milestone 6
- Path: `src/vva_app/events/state_machine.py`, config `configs/events.yaml` (owner of `cooldowns.same_track_same_zone_seconds: 30`, `cooldowns.rearm_after_track_lost_seconds: 5`).
- Consumes: confirmed `Track`s + zone polygons + monotonic time. Produces: `Event` (`kind = entry`).
- Invariants:
  - Reads `track.is_confirmed`, never re-counts hits; anchor = bottom-centre of the box; only `entry_event_enabled` is true.
  - Eligibility (guide §1.5): confirmed AND `mean_confidence >= minimum_mean_confidence` AND `age_seconds >= minimum_track_age_seconds` AND anchor enters the zone AND not in the (track_id, zone_id) cooldown.
  - (track_id, zone_id) cooldown: a second entry of the SAME track into the SAME zone is suppressed while `now - last_entry < 30 s`; allowed at `>= 30 s` (compare with a 1 us epsilon, like `TIME_EPSILON_S`).
  - Re-arm: the cooldown state of a track is kept until 5 s after the track is LOST and then released (bounded memory); a track the tracker recovers within that window keeps its cooldown. (Interpretation to confirm, see manual.)
  - Distinct people: different confirmed `track_id`s entering the same zone produce distinct entry events, however close in time. There is NO per-camera/event-type limit here.
  - Every emitted event is persisted (M7) before any notification decision (M8).
- Required tests (boundaries):
  - Same track re-enters at +29.999 s -> suppressed; at +30.000 s -> new event.
  - Two distinct confirmed tracks enter the same zone 0.1 s apart -> two events.
  - Unconfirmed track inside the zone -> no event; it becomes confirmed -> one event.
  - Track lost, recovered at +4.9 s -> still in cooldown; state released at +5.0 s after loss.
  - Point-in-polygon on a known shape (vertex, edge and outside cases).
- Verified by: M3 replay harness on the held-out day (real M6 code). NOT by RULE_REPLAY.
- Tier: Flash; Pro if the cooldown design changes.

## M7 Persistence, API, metrics (§1.6, §1.7, §1.10) - milestone 7
- Path: `src/vva_app/events/store.py`, `src/vva_app/api/`, `src/vva_app/observability/metrics.py`, config `configs/observability.yaml`.
- Invariants: SQLite writes on a dedicated writer thread (never on the frame path); both timestamps persisted; `/health` has three states; `/metrics` reports per-stage throughput AND end-to-end latency separately; Uvicorn bound to a specific interface; bearer token on every route except `/health`.
- Required tests: writer thread drains a queue without blocking the producer; auth rejects missing token.
- Verified by: PIPELINE_BENCH from TimingRow logs (default budget 400 ms = end-to-end p95 alarm).
- Tier: Flash.

## M8 Alerts and notification (§1.14, v11.2) - milestone 8
- Path: `src/vva_app/notify/`, config `configs/notify.yaml` (owner of `throttle.same_camera_same_event_seconds: 10`, moved from `events.yaml`).
- Consumes: committed event IDs from the dedicated post-commit queue. Produces: delivered notifications and throttle records.
- Invariants:
  - Dedicated post-commit queue fed by the writer (never a second consumer of the writer's input queue); webhook first, Telegram second; retries with backoff never block ingest; dead-man's switch URL from env or 0600 file.
  - Throttle per (camera_id, event_type): a notification is suppressed while `now - last_sent < 10 s`; sent at `>= 10 s`. The window starts at the last SENT notification, not at the suppressed one.
  - Throttling never deletes or alters events: every suppression is recorded (row with event_id, camera_id, event_type, ts, reason `throttled`) and counted in `notifications_throttled_total` on `/metrics`.
  - Different cameras or event types never throttle each other.
- Required tests (boundaries):
  - Two events, same camera/type, +9.999 s -> second suppressed and recorded; +10.000 s -> sent.
  - Three events at 0, 6, 11 s -> sent, suppressed, sent (window measured from the last sent).
  - Same time, different cameras -> both sent.
  - Two distinct people (M6 emits two events) within 10 s -> two event rows, one notification, one suppression record.
  - Failing webhook retries without blocking; synthetic event delivered end to end through a fake transport.
- Verified by: M3 harness (throttle decision replayed), synthetic end-to-end event on the Pi (manual).
- Tier: Flash.

## M9 Pre-roll evidence buffer (§1.12) - milestone 9
- Path: `src/vva_app/ingest/ring_buffer.py`, config `configs/pre_roll.yaml`.
- Invariants: PyAV encoded packets from the main stream; cut on the preceding keyframe; record achieved pre-roll; own RTSP connection with its own reconnect and counters.
- Required tests: synthetic packet sequence -> cut starts at the nearest preceding keyframe.
- Verified by: STREAM_PROBE on the main stream (keyframe interval).
- Tier: Flash; Pro for timing edge cases (T2/T3).

## M10 Local enrichment (§1.9) - milestone 10
- Path: `src/vva_app/enrichment/`, config `configs/enrichment.yaml`.
- Invariants: asynchronous, bounded queue, drop oldest; consumes persisted event metadata only (text); localhost only; no external network calls; never influences whether an alert fires.
- Required tests: full queue drops oldest; enrichment failure does not affect the event row.
- Verified by: PIPELINE_BENCH with the LLM resident (detector latency re-measured).
- Tier: Flash; Pro for the boundary design (T1).

## M11 Hardening, soak, demo (§1.13) - milestone 11
- Path: systemd units, `configs/hardening.yaml` (C1 validator complete by now).
- Invariants: startup checks (Hailo identify, HEF load, RTC plausible, storage writable) fail fast; `Restart=on-failure`, watchdog, `OnFailure=` unit; retention purge independent of disk-full eviction.
- Required tests: startup checks fail fast with fakes; retention purge and disk-full eviction tested separately.
- Verified by: 24 h soak with PIPELINE_BENCH (drops, latency, thermal) on Pi-side logs.
- Tier: Flash.
