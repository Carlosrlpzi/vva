# Smart Camera System — MVP Development Guide

This guide picks up exactly where `01-setup-guide.md` leaves off. Hardware is assembled, the OS is flashed, the Hailo‑10H is verified at three levels (`lspci`, `hailortcli scan`, `hailortcli fw-control identify`), cameras answer `ffprobe`, and the Python venv imports `cv2`, `onvif`, and `hailo_platform` cleanly. This document is about the **application code** for the single‑camera MVP: what each element does, why it exists, how it works internally, and where to read further if you want to go deeper than the summary.

It follows the patched architecture already agreed for this project — **YOLOv8n** (not YOLO26, which the Hailo‑10H does not currently support), a **motion gate** ahead of the NPU, a **time‑parameterized Kalman/IoU tracker** instead of naive consecutive‑frame confirmation, and **on‑chip GenAI enrichment** through Hailo‑Ollama/HailoRT instead of any cloud API.

```text
RTSP substream → threaded single-slot capture → MOG2 motion gate (zone-masked) →
Hailo-10H YOLOv8n inference (letterboxed) → time-parameterized Kalman/IoU tracking →
track-based zone/event state machine → async SQLite write + pre-roll clip flush → API →
optional local Hailo GenAI enrichment
```

The MVP target is deliberately narrow: **one camera, one class (`person`), one zone rule**, running end to end. Once that slice is stable, the same modules generalize to multiple cameras.

---

## Revision Notes

### v2 — First Architectural Review (Sept 8 2026)

A design review of the v1 guide surfaced several places where the pipeline as specified would have quietly broken itself or produced misleading results. This revision folded in four categories of fixes:

1. **Mathematical/architectural corrections** — the detector's confidence threshold was silently discarding boxes the tracker's low-confidence tier was designed to use; `min_confirmed_hits` was duplicated across two config files; the Kalman filter assumed a fixed frame interval that the motion gate's irregular scheduling breaks; track expiry was specified in frames instead of seconds; the frame-differencing motion gate is a poor fit for a 24/7 outdoor scene; and the detector's square input was stretching a 16:9 source instead of letterboxing it.
2. **System bottlenecks** — `cv2.VideoCapture`'s internal buffering (the "OpenCV smear") can silently serve increasingly stale frames under load with no visible error; no explicit latency budget or `/metrics` endpoint was defined; and event timestamps needed to be pinned to frame capture time, not processing time.
3. **Missing components** — there was no offline replay/evaluation harness to calibrate thresholds against ground truth, no pre-roll evidence buffer, no INT8-vs-FP32 accuracy check on the compiled HEF, no defined disk-full/log-rotation/SD-wear behavior, and no note on the AGPL-3.0 license implications of the Ultralytics YOLOv8 toolchain.
4. **Timeline reprioritization** — the highest-risk item (HEF compatibility) moved to week 1 as an offline spike, RTSP ingestion moved to week 2, the offline replay harness was built before any threshold tuning, and the event state machine ships entry-only.

### v3 — Second Review (Sept 8 2026)

A follow-up review of v2 found that several of the v2 fixes introduced new defects, and that one correction was applied in one place but not consistently elsewhere. This revision adds:

1. **Bugs in the v2 code samples** — the MOG2 snippet counts shadow pixels (value 127) as motion, cancelling out `detectShadows: true` entirely; `mog2_history` is still a *frame count* even though the identical Δt problem was fixed for the Kalman filter; `FreshFrameReader` silently drops the reconnect logic §1.1 requires and busy-loops on read failure with no staleness signal; and `time.monotonic()` cannot serve as an event's recorded timestamp.
2. **Conceptual corrections** — §1.10 conflated *throughput* (must fit in one frame period) with *latency* (capture → event), and its budget summed to exactly 100% of the available time with zero headroom; the SQLite write does not belong on the per-frame critical path; the replay harness never defined its event-matching rule, without which precision and recall are not computable; and threshold tuning against a small hand-labeled set needs a held-out split.
3. **Missing components** — a **pre-flight pixel budget** (§0) that determines whether YOLOv8n can see a person at your zone distance *at all*, which could invalidate the model or lens choice before any code is written; `storage_mode: encoded_packets` (§1.12) is not achievable with OpenCV and requires **PyAV** plus keyframe-aligned cutting and a camera-side GOP change; NPU failure handling; and an RTC battery note.
4. **Phase 2 mathematical refinements (§4)** — the physically correct continuous white-noise form of \(Q(\Delta t)\) (cubic in position, not merely \(\propto \Delta t^2\)), and Inverse Perspective Mapping via a homography matrix to replace depth-dependent pixel distances with metric ground-plane distances.
5. **Timeline corrections** — milestone 4 depended on an artifact (zone polygons) not produced until milestone 6; continuous footage recording is the *calendar* long pole and must start passively at the end of milestone 2; and milestone 8 bundled the two riskiest remaining items into one week.

### v4 — Independent Senior-Engineer Review (Sept 8 2026)

An external review of v3 found that the prior revisions were tactically sound but left several domain-level and operational gaps unaddressed — the kind that don't surface as code bugs but as "the pipeline works but the product doesn't." This revision adds:

1. **Detector domain-gap risk** — the INT8-vs-FP32 cross-check in §1.3 only proves quantization didn't hurt accuracy relative to the *same* pretrained weights; it says nothing about whether COCO2017-calibrated weights generalize to this project's fixed oblique doorstep camera, distance, and — critically — nighttime IR footage, which is architecturally the least-tested and highest-value case for a home security system.
2. **A structural, not incidental, decode bottleneck** — the Raspberry Pi 5 has **no hardware H.264 decoder at all**; §1.10 undersold this as "a stage to watch on a dashboard" when it is actually a hard CPU ceiling with a concrete mitigation (H.265/HEVC substreams, if the cameras support it).
3. **A statistically wrong confidence interval in the harness's own worked example** — the Wald/normal-approximation interval used in §1.11 is documented to under-cover at small n and near-1.0 proportions, which is exactly the regime this project's label sets will sit in for months; Wilson or Jeffreys intervals are the correct tool.
4. **No runtime crash supervision** — §1.13's startup checks cover failure to *start*; nothing in v1–v3 covers a mid-run crash, hang, or OOM once the pipeline is already running.
5. **Product-level gaps that no amount of pipeline correctness fixes** — no notification/alerting channel, no way to view a clip without manually pulling files, no per-stage (only end-to-end) evaluation, no stated API auth/network boundary, no whole-system memory budget, no camera-tamper/obstruction detection, no unit-test/CI story for the pure-function pieces (IoU, \(Q(\Delta t)\), point-in-polygon, homography), and no retention policy independent of disk pressure.

New sections **1.14–1.17** cover these (notification channel, minimal operator UI, camera tamper/obstruction detection, and unit-test/CI for pure functions); §1.3, §1.7, §1.10, §1.11, and §1.13 get targeted corrections; and the milestone timeline and readiness checklist are updated accordingly.

### v5 — Independent Review of the v4 Patch (Sept 8 2026)

A further review checked v4's corrections against external evidence and the project's own history, and found the substance sound but several specifics needed sharpening. This revision:

1. **Reframes the §1.10 decode bottleneck from a hard ceiling to a milestone-1 measurement item** — the Pi 5's software H.264 decode is widely reported to outperform the Pi 4's old hardware decoder rather than being strictly worse, so the §1.10 `decode: 25 ms` budget is now explicitly provisional pending a direct measurement, and the H.265 recommendation is corrected to note the `-hwaccel drm` flag specifically (not just `v4l2m2m`), since it measurably lowers CPU cost on Pi 5.
2. **Replaces the §1.11 CASRAI citation** with the canonical Brown, Cai & DasGupta (2001) proportion-CI paper, which is the actual source underlying the Wilson-vs-Wald recommendation.
3. **Strengthens the §1.3 low-light/night-IR citations** with two concrete published benchmarks (a YOLOv8 nighttime surveillance evaluation and a Springer systematic review across YOLOv8–v11 on ExDark) in place of two weaker sources.
4. **Traces the night-IR gap to three concrete downstream consequences** that v4 named but didn't connect to specific fixes: the BGR-vs-grayscale MOG2 choice (§1.2) is a day-mode-only setting once cameras switch to monochrome IR video; IR-attracted insects are a false-positive source distinct from daytime shadows/branches that neither MOG2 nor the ROI margin filters, requiring a downstream tracker/classifier fix and an explicit `night_ir` negative-test case; and IR illuminator range (typically 20–30 m) caps the nighttime zone boundary independently of the §0 pixel-height geometry.
5. **Fixes the §1.16 tamper heuristic** so a routine day/night IR cutover — which every camera in this project performs twice daily — no longer trips `tamper_suspected`, via an IR-mode cross-check and/or a persistence-after-step requirement.
6. **Gives notification its own milestone slot.** Milestone 7 (v4) bundled persistence/API/`/metrics` with notification-consumer wiring into one week; notification now has its own milestone 8, pushing pre-roll, local enrichment, and the hardening/soak/demo milestones back one week each (final milestone is now 11, not 10). §1.14 is also tied explicitly to the project's previously-decided webhook-first, Telegram-second alert outbox pattern.

### v6 — Five Targeted Corrections (Sept 8 2026)

A focused pass fixing five specific issues identified in v5, without a broader re-review. This revision:

1. **Reframes the §1.3 night-IR benchmark as a ceiling, not degradation evidence.** The cited nighttime numbers (mAP@50 0.908/0.819/0.886) are above YOLOv8n's own COCO mAP (~0.52), so they describe what a domain-tuned model achieves at night, not what the project's COCO-pretrained weights will do — the ExDark systematic review remains the correct source for the degradation claim, and milestone-3 fine-tuning on the `night_ir` labels is the stated phase-2 action.
2. **Renames `kalman_sigma_accel` to `kalman_sigma_accel_sq`** (§1.4/§4.1), since the field held \(\sigma_a^2\) (as `filterpy`'s `spectral_density` parameter requires) despite its name implying \(\sigma\). States its units explicitly as px\(^2\)/s\(^4\) (pixel space, pre-homography), replaces the implausible placeholder `1.0` with a starting value of `30.0` for the §1.11 harness to tune, and notes that too small a value reproduces the overconfident-covariance failure §4.1 describes.
3. **Adds config provenance to the §1.11 replay harness** — `replay_runner.py` now emits a hash of the merged config alongside `n_labeled_events`, with a matching field added to `eval.yaml`, so a reported precision/recall number can be traced back to the exact configuration (across the pipeline's eleven YAML files) that produced it.
4. **Moves pure-function unit tests out of the phase-2 backlog** (§1.17/§3). IoU and \(Q(\Delta t)\) are now written at milestone 5 and point-in-polygon at milestone 6 — alongside the code that first needs them — rather than deferred to a single later cleanup week.
5. **Relocates the H.265/HEVC decode change** (§3 milestone 2 / §1.10) out of milestone 2's exit criteria and the single-camera readiness checklist, since the cited saving (13% → 9% of one core, ~1% of total CPU) isn't worth the added §1.12 remux-path complexity at one camera. It now lives in the multi-camera generalization paragraph at the end of §3, where four streams make the saving worthwhile; the `-hwaccel drm` detail moves with it.

### v7 — Two Targeted Corrections (Sept 9 2026)

A further focused pass fixing two specific issues, without a broader re-review. This revision:

1. **Implements the `day` / `night_ir` split and per-frame detector recall in `config/eval.yaml`** (§1.11). The split had been called for in five places (this section's v4 note, §1.3's v4 correction, §1.2's v5 addition, milestone 3's exit criteria, and the readiness checklist) but was never added to the config block that implements it, so no per-frame detector recall was ever computed to tell a genuine domain gap apart from a mistuned threshold. Adds `conditions: [day, night_ir]` and `condition_source` to the `split:` block, a separate `detector_metrics: [per_frame_recall, per_frame_precision]` list, and a `negative_test_cases:` entry for the §1.2 IR-insect case; adds a `condition` column to the ground-truth CSV schema, since there was otherwise nowhere to record it.
2. **Fixes the `kalman_sigma_accel_sq` arithmetic and value** (§1.4). The v6 correction's own claim was internally inconsistent — "tens of px/s²" squared is hundreds to thousands, not "tens" of px²/s⁴ — making the v6 value of `30.0` roughly two orders of magnitude too small. Rederiving from §0's pixel-budget table (54 px at 15 m, 163 px at 5 m, both for the 4.0 mm lens) gives a defensible range of 10³–10⁴ px²/s⁴ depending on distance-to-camera; sets `kalman_sigma_accel_sq: 2000.0` as the new starting point for the §1.11 harness to tune.

### v8 — Three Targeted Corrections (Sept 9 2026)

A further focused pass fixing three specific issues, without a broader re-review. This revision:

1. **Adds `kalman_sigma_accel_sq` to what Inverse Perspective Mapping fixes** (§4.2). §1.4's rederived arithmetic shows \(\sigma_a^2\) swinging roughly 9× between 5 m and 15 m in the same frame — the same depth dependence §4.2 already calls out for `max_centroid_distance_px`. In ground-plane metric coordinates, \(\sigma_a^2\) becomes a physical acceleration variance (roughly 2–3 m²/s⁴ for a walking person) that is constant across the zone and transfers between cameras, same as `max_association_distance_m`.
2. **Corrects the phase-2 fine-tuning action** (§1.3). The v6 correction said fine-tuning proceeds "on those labels," referring to milestone 3's event-level `night_ir` split — but detector fine-tuning needs per-frame bounding boxes, which those labels don't contain and can't be derived from event timestamps. Reworded to state that the milestone-3 labels only reveal whether a night gap exists (via per-frame detector recall, §1.11); closing it is a separate, scoped phase-2 project requiring its own bounding-box annotation pass, a retrain, and a fresh HEF compile with a domain-specific calibration set, and the retrain inherits the section's existing AGPL-3.0 concern since it runs through the Ultralytics training tooling.
3. **Fixes the `eval.yaml` / milestone 3 date contradiction** (§1.11/§3). `split.tuning_days` and `heldout_days` were hardcoded to Sep 28–Oct 1, 2026, which falls inside milestone 3's own week — after the ground-truth CSV is labeled from week-2 (Sep 21–27) footage, not before. Replaced the literal dates with placeholders and a comment noting they move with the schedule, rather than asserting a different fixed date that could drift out of sync again.

### v9 — Three Targeted Corrections (Sept 9 2026)

A focused pass applying three of the highest-priority items from a senior-engineer review of v8, without a broader re-review. This revision:

1. **Extends the §1.14 notification channel to system-health signals, not just emitted events.** `min_severity: entry_event` meant `/health` degraded/down transitions, `npu_failures_total`, and the §1.16 `tamper_suspected` signal never reached a human — the same "nobody is told" blind spot §1.14 was built to close, just for infrastructure failure instead of a missed event. Adds `health_degraded` / `health_down` / `tamper_suspected` severity tiers and a second consumer path watching `/health` transitions alongside the existing event-commit path.
2. **Adds a startup configuration-validation layer** (§1.18, new). Four of the prior eight revisions were caused by the same root problem — drift across the project's independent YAML files (the v1 `min_confirmed_hits`/`required_track_hits` duplication, v6/v7's `kalman_sigma_accel` naming and value errors, v8's `eval.yaml`/milestone date contradiction). §1.18 adds a fail-fast check, run before the ingestion thread starts, asserting `detector.confidence_threshold == tracker.low_confidence_threshold` and that every configured camera's zone-polygon file exists — the same class of invariant that was previously only enforced by careful reading.
3. **Adds a fail-fast placeholder check to the §1.11 replay harness.** v8 replaced `eval.yaml`'s literal, out-of-sync dates with `<tuning-day-1>`-style placeholders, but nothing stopped `replay_runner.py` from silently running against an unfilled placeholder. Adds `reject_unresolved_placeholders: true` to `eval.yaml` and a corresponding startup check in `replay_runner.py` that refuses to run — rather than producing a confusing failure or a silent no-op — while any `split.tuning_days`/`heldout_days` entry still matches the `<...>` placeholder pattern.

### v10 — Three Targeted Corrections (Sept 9 2026)

A further focused pass applying three more items from the same senior-engineer review of v8, without a broader re-review. This revision:

1. **Adds a rollback and shadow-validation path for the phase-2 detector retrain** (§1.3). The v6 correction already documents how to produce a fine-tuned HEF for the night-IR domain gap; it did not say what happens if that HEF turns out worse than the one already deployed. Adds a requirement to validate the retrained model against the same held-out split (§1.11) before promoting it, and to keep the prior HEF and its `models.yaml` values addressable so a regression can be rolled back.
2. **Adds an ongoing, rolling labeling cadence beyond milestone 3** (§1.11/§3). `minimum_labeled_events_for_hard_gate: 100` is unlikely to be reached from a single week of milestone-3 labeling, which would leave the CI gates advisory indefinitely. Adds a note in both sections that labeling continues on a regular cadence from the already-running 24/7 recording, rather than treating milestone 3 as a one-time labeling pass.
3. **Adds a bearer-token lifecycle and transport-security note, and removes the hardcoded camera credentials from the §1.1 example** (§1.1/§1.7). The ONVIF code sample previously hardcoded a username and password; replaced with environment-variable lookups. The v4 addition in §1.7 said to add a bearer-token check but not how to generate, store, rotate, or transport it; adds concrete guidance for each, including putting the API behind TLS.

### v11 — Out-of-Process Liveness & Three Gating Fixes (Sept 10 2026)

A focused pass closing the one blind spot a v10 audit rated most serious, plus three items that were documented but never gated. This revision:

1. **Adds out-of-process liveness monitoring** (§1.13/§1.14/§3). §1.14's v9 `/health` consumer runs inside the pipeline process, so it cannot report a crash, an OOM kill, systemd exhausting `start_limit_burst: 5`, power loss, or network loss. Adds two mechanisms covering disjoint failure sets: an `OnFailure=camera-alert@%n.service` oneshot unit (§1.13) that webhooks from outside the failed process on crash, hang, or restart-limit exhaustion; and a dead-man's switch (§1.14, `dead_mans_switch:` in `notify.yaml`) — an APScheduler job pings an external endpoint, which alerts when pings stop — the only path that survives power loss, network loss, or a wedged kernel. Both are verified in milestone 11 (kill the process; power the Pi down) and gated in the readiness checklist.
2. **Gates two existing fixes that nothing enforced** (§3 only). The `night_ir_insect_activity` negative test already in `eval.yaml` must now pass (zero emitted events) at milestone 3 and in the checklist; §1.1's env-var camera credentials must be verified at milestone 2 and in the checklist, with no credentials anywhere in the repository.
3. **Scopes two claims honestly** (§1.16/§1.18). §1.16 now states it detects obstruction/re-aiming only while the Pi is powered and running, deferring the power/network vector to the §1.14 dead-man's switch. The YAML file count is corrected from eight to eleven wherever it appears (§1.18, §1.11's v6 addition, and the v6 revision note) — eleven is the number of `config/*.yaml` blocks actually in this document — and §1.18 now states validation is startup-only with no hot-reload path planned for the MVP.
4. **Creates `docs/post-mvp-backlog.md`** for deferred items (further §1.18 invariants, config hot-reload, a traceability CI check), with a one-line pointer at the top of §3.

### v11.1 — Four Blocking Fixes from Model Council Review (Sept 10 2026)

A three-model independent review, cross-examination, and adjudication pass (Astra, Claude Fable 5.1, Gemini 3.1 Pro) found the v11 liveness fixes above were each undermined by one small, concrete defect in the literal text. All four are fixed here with 1–3 line edits each; no new sections or components were added. Everything else the council considered was explicitly deferred or dropped as non-blocking for the single-camera MVP.

1. **`camera-alert@.service`'s `EnvironmentFile=` line carried a trailing comment** (§1.13) — systemd does not support inline comments on unit directives, so the comment was parsed as part of the path and the unit could not activate, silently defeating the v11 crash-alert webhook on an ordinary crash-and-restart. Fixed by moving the comment above the directive.
2. **The notifier was specified as a second consumer of the §1.6 writer thread's own input queue** (§1.14, and repeated in the milestone 8 table at §3) — two consumers racing on one queue split events between them with no error raised. Fixed by having the writer publish committed event IDs to a separate, dedicated notification queue.
3. **`FreshFrameReader.get_latest()` had no new-frame signal** (§1.10) — a consumer faster than the camera's frame rate could reprocess one physical frame multiple times, letting a single-frame artifact satisfy `min_confirmed_hits` against itself and reintroducing the flicker false-positive the tracker was built to fix. Fixed by adding an `is_new` flag to `get_latest()`'s return value and gating MOG2/inference/tracker-hit updates on it.
4. **The `sd_notify` heartbeat only proved the main loop was alive, not the SQLite writer or notifier threads** (§1.13) — if either worker thread died, the heartbeat, dead-man's switch, and `/health` would all keep reporting healthy while events silently dropped. Fixed by gating the heartbeat on `all(t.is_alive() for t in (reader, writer, notifier))`.

### v11.2 — Cooldown scopes split between events and notifications; single repository (Oct 2 2026)

Owner decisions, recorded so the agents that write the code follow them literally:

1. **Event cooldowns stay in the state machine (§1.5, milestone 6):** `same_track_same_zone_seconds: 30` and `rearm_after_track_lost_seconds: 5`. **Distinct people produce distinct entry events:** two different confirmed `track_id`s entering the same zone are two events, however close in time.
2. **The 10 s per `(camera_id, event_type)` limit moves to the notification channel (§1.14, milestone 8)** as `notify.throttle.same_camera_same_event_seconds: 10`. It throttles *notifications*, never events: every event is still persisted, and every suppressed notification is recorded (row + `notifications_throttled_total` counter), so the event history stays complete.
3. **Validation split.** The `vva_contracts` RULE_REPLAY task evaluates *frame-level* detection policies only (confidence, consecutive frames, per-camera cooldown). Its known limitation — a false-positive alert opens the per-camera cooldown and can suppress a real event right after it — is pinned by a regression test, not "fixed". Production event behaviour (M6 + M8) is validated by the milestone-3 replay harness (§1.11) running the real M6 code. A track-based replay task may be added to `vva_contracts` later.
4. **Single repository.** The application lives in `src/vva_app/` next to `src/vva_contracts/`; every guide path `src/<x>/...` maps to `src/vva_app/<x>/...` (for example `src/events/state_machine.py` → `src/vva_app/events/state_machine.py`).
5. **PIPELINE_BENCH default budget = 400 ms**, matching `end_to_end_latency_ms.alarm_p95` (§1.10).

---

## 0. Pre-Flight Check: The Pixel Budget

**Do this before writing any code.** It takes one frame and a tape measure, and its answer can invalidate the model choice, the substream resolution, or the lens selection — all of which are expensive to discover at milestone 5.

**The problem.** Every threshold in this document assumes the detector can actually see a person at the distance that matters. YOLOv8n's finest detection stride is 8 pixels, and detection reliability degrades sharply once an object's height in the model's input frame falls below roughly 30–40 px. Nothing in v1 or v2 checked whether that condition holds for your cameras.

**The calculation.** From the pinhole camera model, the pixel height of an object of real height \(H\) at distance \(D\) is:

\[
h_{px} = \frac{P_{h} \cdot f \cdot H}{D \cdot S_{h}}
\]

where \(P_h\) is the image height in pixels, \(f\) the lens focal length (mm), \(S_h\) the sensor height (mm, from the camera datasheet), and \(H, D\) in the same units.

Worked example, assuming a 1/3" sensor (\(S_h \approx 3.0\) mm), a 1.7 m person, and a 640×360 substream:

| Lens | Distance | Person height (px) | Verdict |
|---|---|---|---|
| 4.0 mm | 5 m | ~163 px | Comfortable |
| 4.0 mm | 15 m | ~54 px | Workable |
| 4.0 mm | 25 m | ~33 px | Marginal — expect misses |
| 2.8 mm | 15 m | ~38 px | Marginal |
| 2.8 mm | 25 m | ~23 px | Below the floor — will fail |

**Substitute your own datasheet values.** The table is illustrative; the sensor height in particular varies meaningfully between 1/3", 1/2.8", and 1/2.7" parts.

**The non-obvious consequence: a higher-resolution substream does not automatically help.** Because the detector letterboxes into a fixed 640×640 input (§1.3), a 640×360 substream is scaled by 1.0 (width already fits, only vertical padding is added) while a 1280×720 substream is scaled by 0.5. A person 108 px tall in 720p becomes 54 px after letterboxing — identical to the 640×360 case. Raising substream resolution buys you nothing unless you also change *how* the frame reaches the model.

If the pixel budget comes out marginal, you have three real options, in increasing order of effort:

- **Shorter field of view.** A 4.0 mm lens instead of 2.8 mm puts roughly 43% more pixels on target at the same distance. This is also the quantitative basis for settling the still-open 2.8 mm vs 4.0 mm question for camera #3 in the architecture doc — measure the zone distance and compute, rather than estimating "backyard width."
- **Crop-and-infer.** Instead of letterboxing the whole frame, crop the zone ROI (plus margin) at native substream resolution and feed *that* to the 640×640 input. A person occupying 33 px of a full frame may occupy 90+ px of a tight crop. Costs you awareness outside the crop, which is acceptable when your zone rule only cares about one region anyway.
- **Escalate the model.** YOLOv8s has meaningfully better small-object recall than YOLOv8n, and the Hailo‑10H has ample headroom for it (§1.10). This is the same decision point the INT8-vs-FP32 check in §1.3 feeds into.

**v5 addition — the nighttime budget is capped by IR range, not just the lens equation above.** The pinhole-model table above assumes enough ambient light for the sensor to resolve detail at the calculated distance; after dark, most consumer/prosumer PoE cameras switch to IR illumination with a specified effective range — commonly around 20–30 m for this camera class — beyond which the image is simply too dark to expose, regardless of how many pixels the geometry places on target. Run the same pixel-height table using your camera's actual IR illuminator range instead of (or alongside) its daytime sightline, and treat whichever is smaller — the optical pixel-height floor or the IR illumination floor — as the real nighttime zone boundary. Record both distances in `configs/`: a zone drawn from daytime visibility alone can silently claim nighttime coverage the camera cannot deliver.

**Exit criterion:** a measured pixel height for a person standing at each camera's zone boundary, recorded in `configs/` alongside the zone polygon, and a written decision on lens/resolution/model before milestone 1 begins — covering both the daytime and IR-range-limited nighttime cases.

---

## 1. Component Reference

Each section below maps to a module in `src/`. For every component you get: **Purpose** (the problem it exists to solve), **How it works** (the mechanism, with the math where it matters), and **Deeper reading**.

### 1.1 RTSP/ONVIF Camera Ingestion — `src/ingest/rtsp_reader.py`, `onvif_client.py`

**Purpose.** Turn a physical IP camera into a steady stream of decoded frames your Python code can operate on, without hard‑coding vendor‑specific URL formats that break the moment a camera firmware updates.

**How it works.** ONVIF is a standardized device‑discovery and media‑profile protocol that most commercial IP cameras (Uniarch, Tiandy, Hikvision‑OEM, etc.) implement. Instead of guessing a stream path like `/Streaming/Channels/102`, the code calls the camera's `GetProfiles()` and `GetStreamUri()` SOAP methods to retrieve the actual RTSP URL for the low‑resolution **substream** — the smaller stream is what you want for continuous AI inference, keeping the full‑resolution mainstream free for evidence clips. `cv2.VideoCapture(uri)` then pulls decoded frames using FFmpeg/GStreamer under the hood. A thin reconnect loop (catch failed `.read()`, back off, retry `VideoCapture`) is required because RTSP over a LAN will occasionally drop a TCP session; without it, one Wi‑Fi hiccup or camera reboot silently kills the whole pipeline.

```python
import os
from onvif import ONVIFCamera
cam = ONVIFCamera(
    os.environ['CAMERA_HOST'],
    80,
    os.environ['CAMERA_USER'],
    os.environ['CAMERA_PASSWORD'],
)
media = cam.create_media_service()
profiles = media.GetProfiles()
uri = media.GetStreamUri({
    'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
    'ProfileToken': profiles[1].token  # substream profile
}).Uri
```

**v10 correction — don't hardcode camera credentials in source.** The example above previously wrote the ONVIF username and password directly in the code. Read them from environment variables (or an equivalent secrets mechanism) instead, since this file is the kind of thing that ends up committed, pasted into a writeup, or shared for debugging — and the failure mode is a leaked camera credential, not just a leaked API key.

**Correction (v2) — don't let `VideoCapture` silently go stale.** `cv2.VideoCapture` buffers frames internally (via FFmpeg/GStreamer), and if your processing loop ever falls behind the camera's real frame rate, it does not drop frames — it serves increasingly old ones while `.read()` keeps returning successfully. Nothing errors, so the system *looks* healthy while end-to-end alert latency silently grows unbounded. Fix this with a dedicated ingestion thread that continuously reads frames into a single‑slot "mailbox" (each new frame overwrites the previous one), so the processing loop always consumes the freshest available frame and stale ones are discarded rather than queued. Where the backend honors it, `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` helps too, but it's not reliably supported across all FFmpeg builds, so treat the threaded mailbox pattern as the dependable fix. Attach a **capture timestamp** to each frame the instant it's read (not when it's later processed) — every downstream event should carry this capture time, since a lagging pipeline would otherwise misrepresent when something actually happened, which destroys forensic value. See §1.10 for the full latency‑budget and timestamp discipline this feeds into.

**Camera-side configuration (v3).** Two settings in the camera's own web UI matter to the software and are easy to forget because they live outside the repository:

- **I-frame / GOP interval → ~1 second.** The pre-roll buffer (§1.12) can only cut a clip at a keyframe. If the camera emits an I-frame every 4 seconds, your "6-second pre-roll" is really "somewhere between 4 and 8 seconds, undecodable at the start."
- **Substream resolution and frame rate** should be recorded in `configs/` alongside the §0 pixel budget, because both the letterbox scale factor and the latency budget derive from them.

**Deeper reading:**
- [python-onvif-zeep on GitHub](https://github.com/FalkTannhaeuser/python-onvif-zeep) — the ONVIF client library used in this project
- [ONVIF Core Specification](https://www.onvif.org/specs/core/ONVIF-Core-Specification.pdf) — the underlying standard, if you want to see exactly what `GetStreamUri` returns and why
- [OpenCV `VideoCapture` documentation](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) — the class doing the actual RTSP decode
- [PyImageSearch — Faster video file FPS with cv2.VideoCapture and OpenCV](https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/) — the threaded-reader pattern that fixes the buffering smear
- [Stack Overflow — OpenCV VideoCapture lag due to the capture buffer](https://stackoverflow.com/questions/30032063/opencv-videocapture-lag-due-to-the-capture-buffer) — a concise explanation of exactly this failure mode

---

### 1.2 Pre‑Inference Motion Gate — `src/ingest/motion_gate.py`

**Purpose.** The Hailo‑10H's 40 TOPS budget is finite, and a static security camera spends the overwhelming majority of its time looking at an unchanged scene. Running YOLOv8n on every decoded frame wastes NPU throughput on empty hallways and also invites false positives from compression artifacts on an otherwise idle frame. The motion gate is a cheap CPU‑only pre‑filter: only frames with actual pixel‑level change reach the accelerator.

**Correction (v2) — plain frame differencing does not survive a real outdoor camera.** The original design compared each frame only to the *previous* frame (\(|G_t - G_{t-1}|\)). That has no persistent notion of "background," so it fails in three ways a 24/7 camera will hit constantly: a person walking straight toward the lens changes very few pixels frame‑to‑frame and can go undetected; swaying trees, moving shadows, and IR‑illuminated insects at night change pixels every frame and trigger constant false motion; and any gradual lighting shift (clouds, sunset) is indistinguishable from real motion. The fix is a proper **background model** rather than a two‑frame comparison.

**How it works.** Replace the two‑frame difference with OpenCV's `BackgroundSubtractorMOG2` — a per‑pixel Gaussian Mixture Model (GMM) that is continuously updated with a learning rate, so it can represent *multiple* background states per pixel (e.g., a branch swaying between two positions counts as background on both, not as motion) and adapts gradually to lighting drift instead of reacting to it as foreground. MOG2's `apply()` call returns a foreground mask directly — you no longer compute an explicit `absdiff`/threshold step by hand.

The second correction is **where** you measure motion: computing the motion ratio over the *entire* frame wastes sensitivity on background regions (a driveway edge, a neighbor's tree) that will never contain an actual zone event. Instead, derive a static region‑of‑interest mask \(Z(x,y)\) from the configured zone polygons plus a safety margin (so objects are caught approaching a zone, not just already inside it), and restrict the motion ratio to that region:

\[
r_t = \frac{\sum_{x,y} F_t(x,y)\, Z(x,y)}{\sum_{x,y} Z(x,y)}
\]

Submit the frame to YOLOv8n when \(r_t \geq \tau_{\text{motion}}\). A morphological opening/dilation pass on \(F_t\) before summing still helps merge scattered noise pixels into coherent blobs. Two rules still matter in practice: keep running inference at the tracker's cadence while a track is already active (so a person pausing mid‑frame doesn't silently disappear from the event pipeline), and force one "heartbeat" inference every few seconds regardless of motion, so a frozen camera feed looks different from an idle‑but‑healthy one.

**v3 correction — MOG2's mask is three-valued, and `> 0` counts shadows as motion.** With `detectShadows=True`, `apply()` does **not** return a binary mask. It returns **0 for background, 255 for foreground, and 127 for detected shadow**. The v2 snippet tested `fg_mask > 0`, which classifies every shadow pixel as motion — meaning you pay the CPU cost of shadow detection and then discard its entire benefit. Since moving shadows across a driveway in late afternoon are one of the exact false-positive sources this section exists to suppress, this one comparison operator undoes the whole MOG2 upgrade. **Test `== 255`.**

A related detail: MOG2's shadow detector uses chromatic information, so feeding it a **BGR frame rather than a grayscale one** gives materially better shadow suppression. It costs roughly 3× the memory bandwidth in that stage, so benchmark both against your latency budget (§1.10) before committing.

**v5 addition — the BGR-vs-grayscale choice is a day-mode setting, and IR-attracted insects need their own answer, not a background-model fix.** Most PoE home cameras switch to monochrome, IR-illuminated video after dark (§1.3): every channel reads R=G=B, so MOG2's chromatic shadow discrimination has nothing to work with, and `input_color_space: bgr` buys you nothing at night — treat it as a `day` setting and fall back to grayscale (or simply accept that the BGR/grayscale choice is moot) once IR mode is active, rather than assuming one input format covers both. Separately, insects drawn to the IR illuminator at close range to the lens are a false-positive source distinct from the swaying-tree/moving-shadow cases the v2 correction above named MOG2 as the fix for: a GMM background model only learns motion that recurs in a small number of stable states, but an insect's erratic, non-repeating flight path in front of the lens never settles into "background," and because it happens directly in the camera's field of view it typically falls inside the zone ROI margin too, so `roi_source: zone_polygons_plus_margin` does not filter it out either. The fix lives downstream of the gate: let these frames through to the detector as designed, and rely on the `classes: [person]` filter plus the tracker's `min_confirmed_hits` (§1.4) to reject the resulting churn rather than trying to suppress it at the motion-gate stage. Add an explicit `night_ir` negative-test case to the §1.11 replay harness so a regression that lets insect-triggered frames through as sustained NPU submissions shows up as a measured false-positive/throughput number, not as a mysteriously busy Pi at 2 a.m.

**v3 correction — `mog2_history` is a frame count, and the Δt fix was never applied here.** §1.4 correctly replaces `max_missed_frames` with a wall-clock expiry because the motion gate makes frame arrival irregular. But `mog2_history: 500` has exactly the same defect and was left alone: it is a *frame* count, and MOG2 derives its automatic learning rate from it. Under the single-slot mailbox (§1.10), frames are deliberately dropped whenever processing falls behind, so 500 frames is an unknown and load-dependent amount of wall-clock time. The consequence is that the background model adapts faster or slower depending on how busy the Pi is, which silently changes gate sensitivity.

Fix: pass an **explicit `learningRate`** to `apply()`, computed from the measured elapsed time since the last call, rather than letting OpenCV derive it from a frame count:

\[
\alpha = \min\left(1,\ \frac{\Delta t}{T_{\text{adapt}}}\right)
\]

where \(T_{\text{adapt}}\) is the intended background adaptation time constant in seconds. This is the same principle as the time-parameterized Kalman filter in §1.4, applied consistently.

```python
now = time.monotonic()
dt = now - last_gate_ts
learning_rate = min(1.0, dt / config.background_adaptation_seconds)

fg_mask = bg_subtractor.apply(bgr_frame, learningRate=learning_rate)
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

# 255 = foreground, 127 = shadow, 0 = background.
# Test == 255, NOT > 0, or shadows are counted as motion.
motion_ratio = (fg_mask[zone_mask > 0] == 255).mean()

should_run_npu = (
    motion_ratio >= config.minimum_motion_ratio
    or has_active_tracks
    or now - last_inference_time >= config.maximum_idle_inference_interval_seconds
)
last_gate_ts = now
```

```yaml
# config/motion_gate.yaml (v3)
motion_gate:
  algorithm: mog2                        # v2: was previous_frame differencing
  input_color_space: bgr                 # v3: BGR improves MOG2 shadow discrimination vs. grayscale
  background_adaptation_seconds: 50.0    # v3: replaces mog2_history (a frame count) — learningRate = dt / this
  mog2_var_threshold: 16
  detect_shadows: true
  shadow_pixel_value: 127                # v3: documented so the == 255 test is never "simplified" back to > 0
  roi_source: zone_polygons_plus_margin  # restrict r_t to this mask, not the full frame
  roi_margin_px: 40
  minimum_motion_ratio: 0.01             # recalibrate against the ROI-only baseline via the §1.11 harness
  maximum_idle_inference_interval_seconds: 5.0
```

**Note on the `has_active_tracks` clause.** While any track is alive, the gate is bypassed and the NPU runs on every frame. That is intentional and correct, but it means your worst-case NPU load is 100% of the frame rate, not the gated average. Budget §1.10 for the ungated case; the gate is a power and thermal optimization, not a capacity assumption.

**Deeper reading:**
- [OpenCV `BackgroundSubtractorMOG2` reference](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html) — including the `apply(image, fgmask, learningRate)` signature and the 0/127/255 mask semantics
- [OpenCV background subtraction tutorial](https://docs.opencv.org/4.x/de/df4/tutorial_js_bg_subtraction.html) — conceptual overview of MOG2 vs. simpler methods
- [PyImageSearch — Basic motion detection and tracking with Python and OpenCV](https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/) — background on the simpler frame‑differencing approach and why it's a reasonable starting point but not the final design

---

### 1.3 YOLOv8n Detector on the Hailo‑10H — `src/inference/detector.py`

**Purpose.** This is the actual "what is in this frame" stage — the object detector that turns a motion‑gated frame into a list of bounding boxes with class labels and confidence scores. YOLOv8n (Nano) is the pinned MVP model because it has a mature, pre‑compiled path to Hailo hardware; YOLO26 does not.

**How it works.** Hailo devices don't run ONNX or PyTorch directly — they execute a **HEF** (Hailo Executable Format) binary produced by the Hailo Dataflow Compiler, which quantizes and compiles a trained model for a specific chip architecture (an HEF built for Hailo‑8/8L is not interchangeable with a Hailo‑10H). The Hailo Model Zoo publishes YOLOv8 configurations and either ships or documents how to obtain HEFs for supported target devices, and the Model Zoo's public benchmark data is the credible source for expected FPS/accuracy at this model size. At inference time, the Python code loads the compiled `.hef`, feeds it a preprocessed 640×640 frame through HailoRT, and receives raw detector outputs which then go through non‑max suppression using the configured `nms_iou_threshold` to remove duplicate/overlapping boxes for the same object.

**Correction (v2) — the detector must not pre‑filter what the tracker needs.** The v1 config set `confidence_threshold: 0.45` on the detector while the tracker's two‑tier association (§1.4) expects a `low_confidence_threshold: 0.15` tier to exist. If the detector discards every box below 0.45 before the tracker ever sees it, the low‑confidence tier is fed nothing — the two‑tier association becomes impossible, not just degraded. **The detector must emit at the tracker's low threshold (0.15)** and let the tracker (and, ultimately, the event state machine's `minimum_mean_confidence`) do the tiering. Any stricter cut belongs downstream, where there's identity and temporal context to reason with, not at the point where boxes are irreversibly thrown away.

**Correction (v2) — letterbox, don't stretch.** Your camera substream is very likely 16:9, but the model input is a square 640×640. Naively resizing (stretching) a 16:9 frame into a square distorts the aspect ratio the model was trained on (COCO images), which measurably bleeds mAP — particularly on humans, whose bounding boxes are aspect‑sensitive. Use **letterbox** preprocessing instead: scale the frame to fit within 640×640 while preserving aspect ratio, then pad the remaining space (typically with neutral gray, 114/114/114) rather than stretching. Store the scale factor and padding offsets used for this frame, and apply the **inverse** of that same affine transform to every returned box before it reaches the tracker — otherwise your bounding boxes will be correct in "model input space" but wrong in the original frame's coordinate space, which silently corrupts zone/line geometry downstream.

Note the interaction with §0: at a 640×360 substream the letterbox scale factor is 1.0 (only vertical padding is added), so the substream resolution *is* the effective detection resolution. Raising the substream to 720p halves the scale factor and yields the same on-target pixel height — which is why crop-and-infer, not a bigger substream, is the lever if the pixel budget comes out short.

```yaml
# config/models.yaml (v3)
detector:
  name: yolov8n
  hef_path: models/yolov8n_h10h.hef
  input_size: [640, 640]
  preprocessing: letterbox        # v2: was naive stretch-to-square resize
  pad_value: [114, 114, 114]
  confidence_threshold: 0.15      # v2: was 0.45 — must match tracker.low_confidence_threshold
  nms_iou_threshold: 0.50
  classes: [person]
  on_device_nms: check            # v3: prefer an NMS-on-chip HEF variant if available; CPU-side NMS costs latency
  failure_policy: fail_loud       # v3: see "NPU failure handling" below
```

Treat `confidence_threshold` purely as the detector's *emission floor*, not a calibration knob — the tracker and event state machine own the actual confirmation logic, so this value should stay low and recall‑biased.

**Missing step (v2) — verify INT8 accuracy before trusting YOLOv8n.** The Dataflow Compiler quantizes the model to INT8 for the Hailo‑10H, and quantization degrades accuracy disproportionately on already‑small models and small/distant targets — exactly the case (a person far from a doorway camera) this project cares about most. Before committing to YOLOv8n, run the same set of recorded frames through both the compiled HEF and the original FP32 Ultralytics model, and compare detections (missed small boxes, confidence drift, localization error). If the INT8 gap is unacceptable on your actual camera footage, that's the signal to escalate to YOLOv8s rather than discovering it later as unexplained false negatives in production. Pair this check with the §0 pixel budget: small-object INT8 degradation and marginal pixel height compound each other, and the two together decide the model.

**v4 correction — the INT8-vs-FP32 check validates quantization, not the deployment domain.** Passing the check above only proves the compiled HEF is faithful to the *original* Ultralytics weights — it says nothing about whether those weights are good at this project's specific scene. Hailo's published model-zoo artifacts, and Ultralytics' own Hailo export tooling by default, are optimized/calibrated against COCO2017 ([Hailo Model Zoo — DATA.rst](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/DATA.rst), [Ultralytics Hailo export guide](https://docs.ultralytics.com/integrations/hailo)), which recommends at least 1,024 *representative* calibration images for a custom domain. COCO's person images are daytime, well-lit, ground-level, unoccluded, and nothing like a fixed oblique doorstep camera at 15–25 m. This is a distinct failure mode from quantization error and will not show up in the check above — it will only surface once the §1.11 replay harness starts reporting recall lower than the pixel-budget math predicted, at which point it is easy to misdiagnose as a threshold problem rather than a domain problem. Once the labeled dataset exists (milestone 3), compute raw per-frame detector recall as its own number, separate from event-level precision/recall, so a systematic domain gap is visible directly instead of buried inside the tracker/state-machine metrics.

**v4 correction — nighttime IR is architecturally the least-tested, highest-value case.** Most PoE home cameras switch to monochrome, IR-illuminated video at night — a domain further still from COCO's daytime RGB. A systematic review comparing YOLOv8 through YOLOv11 on the ExDark low-light benchmark dataset confirms this costs real accuracy — the degradation holds across the model family, not just one version ([systematic review of low-light detection, YOLOv8–v11 on ExDark](https://link.springer.com/article/10.1007/s42452-025-08051-5)). The §0 pixel-budget math implicitly assumes daytime color video; a doorstep security camera's highest-value alerts (someone approaching after dark) are precisely the ones this project has not yet measured. Add an explicit `day` / `night_ir` split to `eval.yaml` (§1.11) rather than treating dusk/dawn footage as sufficient coverage, and be prepared for `confidence_threshold` to need a day/night-conditional value rather than one global constant.

**v6 correction — the cited nighttime benchmark shows a ceiling, not a COCO-weights baseline.** A YOLOv8-family model evaluated specifically on nighttime small-object surveillance footage reports Precision 0.908 / Recall 0.819 / mAP@50 0.886 against its low-light/night test set ([YOLOv8 nighttime small-object surveillance benchmark](https://www.iieta.org/journals/ijsse/paper/10.18280/ijsse.140611)) — but that model was *trained* for night, and its mAP@50 is above what YOLOv8n reaches on COCO (~0.52), so it cannot be read as evidence that this project's COCO-calibrated weights will degrade at night; it shows what a domain-tuned model achieves once it has been trained on the right domain. The ExDark systematic review above is the citation that actually supports the degradation claim for out-of-domain (COCO-only) weights. Read the nighttime benchmark instead as the target this project can reach, not the baseline it starts from. The milestone-3 `night_ir` label split is event-level, so it can only reveal *whether* a night domain gap exists via the per-frame detector recall it now feeds (§1.11) — it cannot itself be used to fine-tune the detector, since fine-tuning needs per-frame bounding boxes, not event timestamps, and producing those is a materially larger annotation effort than labeling event windows. Closing a confirmed gap is therefore a scoped phase-2 project, not a direct follow-on to milestone 3: a separate bounding-box annotation pass, a retrain, and a fresh HEF compile through the Hailo Dataflow Compiler using a domain-specific calibration set (the same ~1,024 representative-image minimum §1.3 already cites for COCO-to-scene drift). That retrain runs through the Ultralytics training tooling, so it inherits the same AGPL-3.0 concern this section's licensing note already flags — worth deciding on before committing to fine-tuning, not just before publishing the repository.

**v10 addition — the phase-2 retrain needs a rollback and shadow-validation path before promotion, not just a recipe for producing a new HEF.** The v6 correction above describes producing a new HEF; it does not say what happens if that HEF turns out worse than the one already deployed. Validate the retrained model against the same held-out split (§1.11) used for the original YOLOv8n before promoting it — report precision/recall/mAP on that split, not just training-time metrics, and treat a worse held-out result as a reason to keep the current HEF. Keep the previous HEF and its exact `models.yaml` values addressable so a regression can be rolled back by reverting a config value and a file path, rather than by re-running the phase-2 project from scratch.

**v5 addition — the night-IR gap has three concrete downstream consequences, not just a warning to test it.** First, IR-illuminated video is monochrome (R=G=B in every pixel), which moots the BGR-vs-grayscale MOG2 shadow-discrimination choice at night and reopens IR-attracted insects as a distinct false-positive source the motion gate alone can't filter — see §1.2 for both. Second, IR illuminator range on this camera class is typically 20–30 m regardless of lens focal length, which caps the effective nighttime zone boundary independently of the §0 pixel-height geometry — see §0 for the night-specific variant of that calculation.

**v3 addition — NPU failure handling.** Nothing in v1 or v2 says what the pipeline does when HailoRT throws, the device resets, or a firmware/driver mismatch appears after an `apt upgrade`. For a surveillance system the worst possible failure mode is one that believes it is watching while it isn't, so the policy should be explicit:

- **At startup:** verify the device and load the HEF before the ingestion thread starts. If either fails, refuse to start rather than running a pipeline that decodes frames it will never analyze. This matches the fail‑fast philosophy already used elsewhere in this project.
- **Mid-run:** catch HailoRT exceptions at the inference call site, log at ERROR, increment a `npu_failures_total` counter on `/metrics` (§1.10), and mark `/health` **degraded** — never silently continue.
- **Recovery:** attempt bounded re-initialization (a few retries with backoff). If re-init fails, keep the ingestion and pre-roll buffer alive (so evidence is still being recorded) but report the pipeline as down. Recording without analysis is a degraded state worth preserving; pretending to analyze is not.

**Licensing note.** Ultralytics YOLOv8 is distributed under **AGPL‑3.0**. That's unrestricted for private, non‑distributed use exactly like this project, but if the code is ever published as an open‑source portfolio piece with derivative code built on the Ultralytics training/export tooling, AGPL's copyleft terms extend outward unless an Ultralytics Enterprise license is purchased. Decide this early — it doesn't block the MVP, but it does constrain how the repository can be shared later.

**Deeper reading:**
- [Hailo Model Zoo on GitHub](https://github.com/hailo-ai/hailo_model_zoo) — HEF generation, benchmark numbers, and supported model configs, including [`yolov8n.yaml`](https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/networks/yolov8n.yaml)
- [Hailo RPi5 Examples — object detection pipeline](https://github.com/hailo-ai/hailo-rpi5-examples/blob/main/doc/basic-pipelines.md) — a working reference GStreamer/Python pipeline for exactly this hardware combination
- [Hailo Application Code Examples](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime/python) — plain Python (non‑GStreamer) HailoRT inference examples, closer to what a FastAPI‑embedded pipeline needs
- [Raspberry Pi AI accelerator documentation](https://www.raspberrypi.com/documentation/computers/ai.html) — the official `hailo-h10-all` package and model pipeline docs
- [Ultralytics — Hailo export integration](https://docs.ultralytics.com/integrations/hailo) — the official export path and where to cross-check INT8 HEF behavior against the FP32 model
- [Ultralytics `data.augment` API reference (LetterBox)](https://docs.ultralytics.com/reference/data/augment) — the reference letterbox implementation to model your own preprocessing on
- [Ultralytics License page](https://www.ultralytics.com/license) — the AGPL-3.0 vs. Enterprise licensing terms
- [Hailo Model Zoo — DATA.rst](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/DATA.rst) — confirms COCO2017 as the calibration/evaluation set behind the shipped model-zoo artifacts (v4)
- [YOLOv8 nighttime small-object surveillance benchmark](https://www.iieta.org/journals/ijsse/paper/10.18280/ijsse.140611) — concrete published precision/recall/mAP numbers for low-light/night detection (v5)
- [Systematic review of low-light object detection — YOLOv8–v11 on ExDark](https://link.springer.com/article/10.1007/s42452-025-08051-5) — confirms the low-light accuracy gap holds across the YOLO model family (v5)

---

### 1.4 Multi‑Object Tracker (Kalman + IoU) — `src/inference/tracker.py`

**Purpose.** A raw detection box has no memory — it can flicker frame to frame from occlusion, motion blur, or a skipped motion‑gated frame, and the old "3 consecutive frames above 0.60 confidence" rule broke exactly there. A tracker's job is to give each detected object a persistent identity (`track_id`) across frames, so the event logic downstream can reason about *stable objects*, not noisy boxes.

**How it works.** This is a **tracking‑by‑detection** design, the same family as SORT and ByteTrack: the detector proposes boxes every frame; the tracker's only job is to decide which box belongs to which existing track (or whether it's a new one). Two mechanisms do the work:

1. **Kalman filter prediction** — each track keeps a state vector estimating position, size, and velocity, and predicts where the object should be *before* seeing the next frame's detections:
   \[
   x_t = [c_x, c_y, w, h, v_x, v_y, v_w, v_h]^T
   \]
   This lets a track survive a frame or two with no matching detection (e.g., a brief occlusion or a motion‑gate skip) because the filter keeps extrapolating motion instead of the track simply vanishing.

2. **IoU‑based association** — a predicted track box \(b_i\) is matched to an actual detection \(d_j\) using Intersection‑over‑Union:
   \[
   \operatorname{IoU}(b_i, d_j) = \frac{\operatorname{area}(b_i \cap d_j)}{\operatorname{area}(b_i \cup d_j)}
   \]
   A match is accepted only when \(\operatorname{IoU}(b_i, d_j) \geq \tau_{\text{IoU}}\) (and optionally a centroid‑distance check for extra robustness). Unmatched detections seed new tentative tracks; unmatched predictions increase a track's "missed time," and a track is dropped once it exceeds the expiry window.

ByteTrack's specific contribution over a plain SORT‑style tracker is associating **every** detection box, including low‑confidence ones, rather than throwing them away before matching — low‑score boxes are still useful for keeping an already‑confirmed track alive through partial occlusion, even though they'd be too weak to *start* a new track on their own ([FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack)). This only works end‑to‑end now that the detector (§1.3) emits at 0.15 instead of discarding that tier before it reaches the tracker.

**Correction (v2) — the Kalman filter must be time‑parameterized, not frame‑parameterized.** A standard SORT‑style Kalman filter assumes a constant \(\Delta t = 1\) between updates — fine when frames arrive on a fixed clock, but the motion gate (§1.2) deliberately makes frame arrival irregular: a frame might follow the last one by 100 ms, or by 4 seconds if nothing moved and only the heartbeat fired. Feeding a fixed‑\(\Delta t\) filter irregular real time causes it to badly over‑ or under‑extrapolate an object's position, breaking the very IoU association it's supposed to support. The state transition matrix \(F\) and process noise covariance \(Q\) must both scale with the **actual wall‑clock elapsed time** since the track's last update:

\[
F(\Delta t) = \begin{bmatrix} 1 & 0 & 0 & 0 & \Delta t & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & \Delta t & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & \Delta t & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & \Delta t \\ 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}
\]

In practice this means recomputing \(F\) (and re‑scaling \(Q\)) from the measured \(\Delta t\) on every predict step, using the track's own last‑update timestamp rather than assuming a fixed frame interval — most Kalman libraries (including `filterpy`) support this by rebuilding `F`/`Q` per call instead of once at initialization.

**v3 refinement — \(Q(\Delta t) \propto \Delta t^2\) is an approximation, and the exact form matters for long gaps.** v2 stated the scaling as \(Q(\Delta t) \propto \Delta t^2\), which is convenient but not physically correct. Under a continuous white noise acceleration model, the uncertainty in *velocity* grows linearly with elapsed time, but the uncertainty in *position* grows with the **cube** of elapsed time — because position error accumulates the integral of an already-growing velocity error. For each independent position/velocity pair the correct block is:

\[
Q = \begin{bmatrix} \tfrac{1}{3}\Delta t^{3} & \tfrac{1}{2}\Delta t^{2} \\[2pt] \tfrac{1}{2}\Delta t^{2} & \Delta t \end{bmatrix} \sigma_a^{2}
\]

where \(\sigma_a^2\) is the variance of the object's acceleration — physically, how abruptly you expect a person to change speed or direction. The off-diagonal \(\tfrac{1}{2}\Delta t^2\) terms encode the correlation between position and velocity error, which the naive \(\Delta t^2\) scaling throws away entirely.

Why this matters here specifically: the motion gate can produce \(\Delta t\) values of several seconds. At \(\Delta t = 3\) s the cubic term is an order of magnitude larger than the quadratic approximation, so the naive form leaves the filter **overconfident** about where a person is after a long gap. An overconfident prediction produces a too-tight gate, the IoU association fails, and the track is lost and re-created with a new `track_id` — which is precisely the identity break the tracker exists to prevent. Build \(Q\) from the block above, and treat \(\sigma_a^2\) as a tunable calibrated through the §1.11 harness rather than a constant guessed once. Full derivation and the discrete-vs-continuous choice: §4.1.

**Correction (v2) — track expiry must be measured in seconds, not frames.** `max_missed_frames: 12` is meaningless once frames arrive at a variable rate: 12 *skipped* frames could span half a second of real inactivity or, under aggressive motion gating, tens of seconds — in the latter case the track expires far later than intended and stale identities linger. Replace it with a wall‑clock expiry, `max_missed_seconds`, evaluated against the same monotonic clock used for \(\Delta t\).

**Correction (v2) — single source of truth for confirmation.** v1 defined `min_confirmed_hits: 3` here *and* `required_track_hits: 3` in `events.yaml` — two names for the same concept, guaranteed to drift the moment one config changes and the other doesn't. **The tracker owns this variable exclusively.** It exposes a `track.is_confirmed` boolean (true once `min_confirmed_hits` successful matches have accumulated), and the event state machine (§1.5) simply reads that flag rather than re‑counting hits itself.

```yaml
# config/tracker.yaml (v3)
tracker:
  algorithm: kalman_iou
  min_iou_match: 0.25
  max_centroid_distance_px: 120    # see §4.2 — depth-dependent; replace with metric distance in phase 2
  min_confirmed_hits: 3            # single source of truth — exposed downstream as track.is_confirmed
  max_missed_seconds: 1.5          # v2: was max_missed_frames: 12 — now wall-clock, not frame-count
  max_track_age_seconds: 3.0
  kalman_time_parameterized: true  # F(Δt) rebuilt from measured elapsed time each predict step
  kalman_q_model: continuous_white_noise   # v3: full (1/3 Δt³, 1/2 Δt², Δt) block, not the Δt² approximation
  kalman_sigma_accel_sq: 2000.0     # v7: was 30.0 (v6) — arithmetic error corrected, see correction below; units px²/s⁴, tune via the §1.11 harness
  low_confidence_threshold: 0.15   # matches detector.confidence_threshold exactly — no gap in the tiering
  high_confidence_threshold: 0.45
  class_aware_matching: true
  tracked_classes: [person]
```

A track only becomes `confirmed` (`track.is_confirmed = true`, eligible to trigger events) after `min_confirmed_hits` successful matches — this is what replaces the old naive frame‑counting rule with something that actually models identity over time, and it is now the *only* place this threshold is defined.

**v6 correction — `kalman_sigma_accel` was named for σ but valued and used as σ².** The field name implies a standard deviation, but §4.1's block uses \(\sigma_a^2\) directly and `filterpy`'s `Q_continuous_white_noise(..., spectral_density=...)` expects that same squared quantity, not its square root — the v3 name was simply wrong for what the config held. Renamed to **`kalman_sigma_accel_sq`**, with units stated explicitly as **px²/s⁴**: the tracker operates in raw pixel space, before the §4.2 homography projection, so this is acceleration variance in pixels-per-second-squared, squared again — not metric units.

**v7 correction — the v6 value of `30.0` was derived from arithmetic that contradicted its own claim, and is roughly two orders of magnitude too small.** v6 asserted that a pixel acceleration "on the order of tens of px/s²" implies a variance "in the tens of px²/s⁴" — but squaring a quantity in the tens yields hundreds to thousands, not tens; that step was simply wrong. Rederiving from §0's own pixel-budget table gives a defensible starting range instead of a guessed one. §0's 4.0 mm lens at 15 m puts a 1.7 m person at ~54 px, a scale of ~54 / 1.7 ≈ 32 px/m; a walking person changing pace or direction accelerates at roughly 1–2 m/s², so at that distance the apparent pixel acceleration is on the order of ~32 × 1.5 ≈ 48 px/s², and \(\sigma_a^2 \approx 48^2 \approx 2{,}300\) px²/s⁴. At the same lens's 5 m distance, §0 gives ~163 px, a scale of ~163 / 1.7 ≈ 96 px/m, so the same 1–2 m/s² physical acceleration is ~144 px/s² and \(\sigma_a^2 \approx 144^2 \approx 21{,}000\) px²/s⁴. Scale — and therefore the defensible variance — depends on distance-to-camera, so treat **10³–10⁴ px²/s⁴** as the starting range this project's zone geometry actually supports, not a single constant. Set **`kalman_sigma_accel_sq: 2000.0`** (§1.4 config) as the starting point for the §1.11 harness to tune — a mid-range value for a person tracked at moderate zone distance — rather than the arithmetically-unsupportable `30.0`. A value this small reproduces exactly the overconfident-covariance failure §4.1 describes: it makes the filter *more* certain than it should be about position after a long \(\Delta t\), tightening the IoU association gate and triggering the same identity-break/duplicate-alert symptom that motivated the \(\Delta t^3\) correction in the first place — which is exactly the failure mode a two-orders-of-magnitude-too-small value would have walked straight back into.

**Deeper reading:**
- [ByteTrack (ECCV 2022) on GitHub](https://github.com/FoundationVision/ByteTrack) — the "associate every detection box" idea and its reported MOT17 accuracy numbers
- [SORT — Simple Online and Realtime Tracking](https://github.com/abewley/sort) — the simpler Kalman+IoU baseline this MVP tracker most closely resembles
- [PyImageSearch — Intersection over Union (IoU) for object detection](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/) — IoU explained with worked examples
- [kalmanfilter.net](https://www.kalmanfilter.net/) — an approachable, math‑first walkthrough of how the Kalman filter's predict/update cycle works
- [Cross Validated — Kalman smoothing with irregular time steps](https://stats.stackexchange.com/questions/49300/how-does-one-apply-kalman-smoothing-with-irregular-time-steps) — the math for scaling \(F\) and \(Q\) by actual elapsed time rather than a fixed step
- [filterpy issue — handling variable dt](https://github.com/rlabbe/filterpy/issues/196) — a concrete discussion of rebuilding `F`/`Q` per update in a widely used Python Kalman library
- [`filterpy.common.Q_continuous_white_noise`](https://filterpy.readthedocs.io/en/latest/common/discretization.html) — a ready-made implementation of the \(\tfrac{1}{3}\Delta t^3\) block described above

---

### 1.5 Event State Machine & Zones — `src/events/state_machine.py`

**Purpose.** Decide, from a confirmed track's trajectory, whether something actually worth alerting on happened — a person entering a doorstep zone, crossing a boundary line, or dwelling too long — while avoiding duplicate alerts for the same ongoing situation.

**How it works.** Each track moves through an explicit state machine:

```text
NEW → TENTATIVE → CONFIRMED → EVENT_EMITTED (cooldown) → LOST → REMOVED
```

Zone membership is evaluated geometrically per frame (point‑in‑polygon test for the track's centroid against a configured zone polygon), and a line‑crossing event is detected by checking whether the track's centroid trajectory crosses a defined segment between consecutive frames — the same underlying idea used in commercial VMS "virtual tripwire" features. An event is only emitted when **all** of these hold at once:

\[
\text{event eligible} = \text{confirmed track} \land \text{valid class} \land \text{zone or crossing condition} \land \text{not in cooldown}
\]

The cooldown term matters as much as the detection logic: without it, a person standing near a zone boundary for 10 seconds could otherwise generate a new "event" on every frame. Two cooldown scopes are useful in practice — per `(track_id, zone_id)` so the same person doesn't re‑trigger while lingering, and per `(camera_id, event_type)` so a burst of different people doesn't flood the log within a short window.

**v11.2 correction — the `(camera_id, event_type)` scope moved to notifications (§1.14).** Suppressing a *different* person's entry would delete real history. The state machine now keeps only the per-`(track_id, zone_id)` cooldown and the re-arm timer, emits one entry event per distinct confirmed track, and the 10 s burst limit is applied by the notifier as a throttle that records every suppressed notification.

**Correction (v2) — read `is_confirmed`, don't re‑count hits.** v1 re‑specified `required_track_hits: 3` here, duplicating `min_confirmed_hits` from `tracker.yaml` (§1.4). Delete that field entirely. The state machine's confirmation check becomes a direct read of the tracker's boolean:

```python
event_eligible = (
    track.is_confirmed          # owned and computed solely by the tracker (§1.4)
    and track.mean_confidence >= config.minimum_mean_confidence
    and track.age_seconds >= config.minimum_track_age_seconds
    and zone_or_crossing_condition
    and not in_cooldown
)
```

**Scope-down correction (v2) — ship entry-only for the MVP.** Shipping entry, crossing, and dwell detection simultaneously means tuning three independent false-positive rates at once with no way to isolate which rule is misbehaving. The MVP should enable **only** the entry‑zone rule, drive its false‑positive rate down to an acceptable level using the offline replay harness (§1.11), and only then turn on crossing and dwell as a phase‑two addition once entry is trusted.

**v3 correction — zone polygons must be authored before the motion gate is calibrated.** The motion gate now derives its ROI mask from `zone_polygons_plus_margin` (§1.2), which makes the zone polygon an *input* to gate calibration rather than an artifact of this section. Zone polygons are pure configuration (a list of image-space points per camera), not code, so there is no reason to wait: **author them during the ground-truth labeling pass (milestone 3)**, while you are already reviewing frames from that camera and can see exactly where the doorstep boundary falls. Store them in `configs/zones/<camera_id>.json` alongside the §0 pixel-budget measurement for the same camera.

```yaml
# config/events.yaml (v3)
events:
  confirmation:
    # required_track_hits removed (v2) — read track.is_confirmed from the tracker instead
    minimum_track_age_seconds: 0.5
    minimum_mean_confidence: 0.35
  zone_rules:
    zone_definitions: configs/zones/     # v3: authored at milestone 3, consumed by §1.2 and §1.5
    require_confirmed_track: true
    entry_event_enabled: true            # MVP scope: ship this rule first
    crossing_event_enabled: false        # phase 2 — enable only after entry's FP rate is validated
    dwell_event_enabled: false           # phase 2 — enable only after entry's FP rate is validated
    dwell_seconds: 2.0
  cooldowns:
    same_track_same_zone_seconds: 30
    # same_camera_same_event_seconds moved to notify.yaml as a notification throttle (v11.2)
    rearm_after_track_lost_seconds: 5
```

**Deeper reading:**
- [Hikvision — Line Crossing Detection](https://enpinfo.hikvision.com/hkwsen/unzip/20230410194813_20373_doc/GUID-246BF07A-3F33-48FC-99D9-DE1AFC3E9144.html) — how a commercial system defines and evaluates virtual line crossing, useful as a spec reference
- [yas-sim/object-tracking-line-crossing-area-intrusion](https://github.com/yas-sim/object-tracking-line-crossing-area-intrusion) — an open reference implementation of zone/line logic on top of tracked objects
- [Debounce design pattern (community.openhab.org)](https://community.openhab.org/t/design-pattern-debounce/101566) — the general software pattern behind the cooldown/deduplication logic

---

### 1.6 Persistence Layer — `src/events/store.py`

**Purpose.** Give every emitted event a durable, queryable record — what happened, on which camera, at what time, linked to which track — without requiring a database server on a single‑board computer.

**How it works.** SQLite is a serverless, file‑based relational database built into Python's standard library via the `sqlite3` module — no separate process to run, no network port, and the entire event history lives in one file that's trivial to back up or copy off the Pi. An `events` table (camera_id, track_id, event_type, zone_id, first_seen, last_seen, confidence, metadata JSON) is written once per emitted event, immediately after the state machine confirms eligibility — before any enrichment step runs, so the API always has the raw event available even if enrichment is slow or fails. Enable WAL (write‑ahead log) mode if the FastAPI process and any background enrichment worker will read/write concurrently; it lets readers and a single writer proceed without blocking each other, which matters once the API is serving `/events` queries while new events are still being written.

**v3 correction — the write does not belong on the per-frame critical path.** v2's latency budget allocated 15 ms per frame to `sqlite_write`, but the overwhelming majority of frames emit no event at all, so that allocation is both wasteful in the budget and dangerous in the loop: a WAL checkpoint or an fsync stall would block frame processing directly. Push emitted events onto an in-memory queue consumed by a **dedicated writer thread**, and keep the detection loop's only obligation an O(1) enqueue.

Two properties to preserve: the queue must be bounded with an explicit overflow policy (block briefly, then log and drop with a `events_dropped_total` counter — never grow without limit), and the writer must still complete the row **before** enrichment is dispatched, which it does naturally since enrichment consumes from the persisted record.

**Timestamp columns (v3).** Store both clocks per event, for the reasons in §1.10: `captured_at_utc` (wall-clock, the forensic answer to "when did this happen") and `captured_at_monotonic` (for interval arithmetic and for correlating against `/metrics` without NTP steps corrupting the math).

**Deeper reading:**
- [Python `sqlite3` module documentation](https://docs.python.org/3/library/sqlite3.html) — the standard library interface used directly in this project
- [SQLite WAL mode documentation](https://sqlite.org/wal.html) — why and how to enable write‑ahead logging for concurrent read/write access

---

### 1.7 API Layer — `src/api/`

**Purpose.** Expose camera status and event history to anything outside the Pi — a dashboard, a phone app, a curl command during debugging — over plain HTTP, without building a custom protocol.

**How it works.** FastAPI is a Python web framework that generates request validation and interactive API documentation directly from type‑annotated function signatures and Pydantic models, and runs behind the Uvicorn ASGI server. A minimal MVP surface is small on purpose: `GET /health` (camera/pipeline liveness), `GET /events` (recent event history, filterable by camera/time/zone), `GET /events/{id}` (single event detail, including enrichment text once available), and `GET /metrics` (§1.10). Because FastAPI is async‑native, the event endpoints can query SQLite without blocking frame processing running in a separate thread/process — this separation (ingestion/inference in one loop, API serving in another) is what keeps a slow HTTP client from ever stalling the detection pipeline.

**`/health` should be tri-state, not boolean (v3).** Given the NPU failure policy (§1.3) and the stale-frame detection (§1.10), `healthy` / `degraded` / `down` carries far more operational information than up/down. "Ingesting and buffering evidence but not analyzing" is a real and important state.

**v4 addition — the API needs a stated network/auth boundary.** §1.9 is explicit that Hailo-Ollama binds to `127.0.0.1` only; this section never makes the equivalent statement for the API that actually serves event history, enrichment text, and links to clips of real visitors. Even for a private LAN deployment, bind Uvicorn to a specific interface rather than `0.0.0.0`, and add a minimal bearer-token check on the non-`/health` routes — a camera event log is exactly the kind of data that shouldn't be reachable by "anything else on the LAN" by default.

**v10 addition — bearer-token lifecycle and transport security.** The v4 addition above says to add a bearer-token check but not how to generate, store, or rotate the token, or whether the connection is encrypted. Generate the token once with `secrets.token_urlsafe(32)` and store it outside the repository — an environment variable or a `0600`-permissioned file, never a committed config value — since this token gates access to event history, enrichment text, and links to clips of real visitors. Treat rotation as a manual, documented procedure for the MVP (regenerate and redistribute the token if compromise is suspected) rather than building automatic rotation this early. Because the API is plain HTTP by default, put it behind a minimal TLS-terminating reverse proxy (e.g., Caddy with automatic certificates, or nginx with a self-signed certificate for LAN-only use) rather than serving bearer tokens in plaintext, even on a trusted network.

**Deeper reading:**
- [FastAPI official tutorial](https://fastapi.tiangolo.com/tutorial/) — the canonical getting‑started guide, including automatic docs generation
- [FastAPI project homepage](https://fastapi.tiangolo.com/) — framework overview and design rationale

---

### 1.8 Scheduler / Heartbeat — APScheduler

**Purpose.** Run periodic, non‑request‑driven jobs — the motion gate's idle heartbeat inference, stale‑track cleanup, cooldown expiry sweeps, or a nightly SQLite integrity check — without hand‑rolling timer threads.

**How it works.** APScheduler runs jobs on interval, cron‑like, or one‑off schedules inside the same Python process, backed by a configurable job store (in‑memory is sufficient for the MVP). It's the natural place to implement `maximum_idle_inference_interval_seconds` from the motion gate and any "sweep and expire" logic the event state machine needs on a fixed cadence rather than being triggered by a new frame.

**Deeper reading:**
- [APScheduler documentation](https://apscheduler.readthedocs.io/) — user guide covering triggers, job stores, and executors

---

### 1.9 Local GenAI Enrichment — `src/enrichment/` (Hailo‑Ollama / HailoRT VLM)

**Purpose.** Turn a bare event record ("person entered zone doorstep at 14:20, track 42") into a short human‑readable summary, without sending any event data, frames, or metadata to an external cloud API — enrichment happens **after** the event is already persisted and served, so it can never slow down or block the live detection path.

**How it works.** The Hailo Model Zoo GenAI package ships **Hailo‑Ollama**, a REST server (Ollama‑compatible API surface, implemented in C++ on top of HailoRT) that runs small local LLMs directly on the Hailo‑10H's dedicated memory, independent of the Pi's own CPU/RAM. You `curl http://127.0.0.1:8000/hailo/v1/list` to see which models are locally available, then POST structured event metadata as a prompt and get back a short natural‑language description. For the MVP, keep this to **text‑only** summarization of structured metadata (camera, zone, dwell time, track id) — that's a well‑supported, low‑risk use of Hailo‑Ollama. Frame‑ or crop‑aware analysis (actually looking at the image) is a materially different capability path — it runs through a HailoRT VLM application rather than the Ollama‑style text endpoint — and should stay disabled in `config/enrichment.yaml` until it's separately tested, since correctness and latency characteristics differ from the text path.

**v3 note — the enrichment model shares the NPU with the detector.** Loading a 1B-class LLM onto the Hailo‑10H consumes device memory that YOLOv8n also needs. Measure detector latency with and without the enrichment model resident before assuming the 40 TOPS headroom absorbs both, and give the enrichment queue an explicit drop policy (`queue_max_size` with oldest-first eviction) so a slow or wedged LLM can never apply backpressure to the detection path.

```yaml
# config/enrichment.yaml
enrichment:
  enabled: true
  execution_mode: asynchronous
  queue_max_size: 100
  queue_overflow_policy: drop_oldest   # v3: enrichment must never backpressure detection
  llm:
    provider: hailo_ollama
    base_url: http://127.0.0.1:8000
    model: <validated-local-model>
    timeout_seconds: 20
    max_tokens: 160
  vlm:
    enabled: false
    provider: hailort_vlm
  privacy:
    external_network_calls: false
    store_prompts: false
    store_raw_frames_in_prompt_logs: false
```

**Deeper reading:**
- [Hailo Model Zoo GenAI on GitHub](https://github.com/hailo-ai/hailo_model_zoo_genai) — Hailo‑Ollama server, model pull/list/chat endpoints, and hardware/OS prerequisites
- [Raspberry Pi AI HAT+ 2 — Hailo‑10H local LLM walkthrough](https://raspberry.tips/en/raspberrypi-tutorials/raspberry-pi-ai-hat-2-hailo-10h-40-tops-local-llms) — real throughput numbers (roughly 30–50 tokens/sec for a 1B‑class model) and a worked vision+LLM example on this exact hardware
- [Hailo GenAI Model Explorer — VLM models](https://hailo.ai/products/hailo-software/model-explorer/generative-ai/type/vlm/) — the separate vision‑language model catalog, for when image‑aware enrichment is revisited

---

### 1.10 Ingestion Backpressure, Latency Budget & Timestamp Discipline — `src/ingest/rtsp_reader.py`, `src/observability/metrics.py`

**Purpose.** Everything upstream (§1.1–§1.5) assumes the pipeline is keeping pace with the camera in real time. This section makes that assumption explicit, measurable, and enforced — instead of discovering months later that alerts have quietly become minutes late.

**How it works.** Run frame capture in a **dedicated thread** that continuously reads from `cv2.VideoCapture` into a single‑slot "mailbox" — each new frame simply overwrites whatever was there, and the processing loop always reads the freshest available frame rather than draining a queue of stale ones.

**v3 correction — the v2 reference implementation was unsafe in three ways.** The snippet shipped in v2 dropped the reconnect logic that §1.1 explicitly requires, spun a CPU core at 100% when `read()` returned `False`, and gave the consumer no way to tell a frozen camera from a healthy idle one. Since §1.10 is the code someone will actually copy, it needs to be correct. All three fixes below:

```python
import cv2, threading, time
from datetime import datetime, timezone


class FreshFrameReader:
    """Single-slot mailbox: always yields the newest frame, never a queued stale one."""

    def __init__(self, uri, stale_after_s=2.0, max_backoff_s=30.0):
        self._uri = uri
        self._stale_after_s = stale_after_s
        self._max_backoff_s = max_backoff_s
        self._lock = threading.Lock()
        self._latest = None            # (frame, monotonic_ts, utc_ts)
        self._stop = threading.Event()
        self._reconnects = 0
        self._last_served_mono_ts = None   # v11: lets get_latest() report whether this is a new physical frame
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self._uri)
        # Honored by some FFmpeg builds, ignored by others — the mailbox is the real fix.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _read_loop(self):
        cap = self._open()
        backoff = 0.5
        while not self._stop.is_set():
            ok, frame = cap.read()
            if ok:
                backoff = 0.5                       # v3: reset backoff on success
                with self._lock:
                    # Both clocks, captured at read time (see "timestamp semantics" below).
                    self._latest = (frame,
                                    time.monotonic(),
                                    datetime.now(timezone.utc))
                continue

            # v3: reconnect with exponential backoff instead of busy-looping forever.
            cap.release()
            time.sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff_s)
            cap = self._open()
            self._reconnects += 1

    def get_latest(self):
        """Returns (frame, monotonic_ts, utc_ts, is_stale, is_new) or None before the first frame.

        v11 fix: the mailbox has no queue depth, so a consumer running faster than the
        camera's frame rate previously had no way to tell it had just re-read the same
        physical frame it already processed. `is_new` is False whenever `mono_ts` is
        unchanged since the last call — i.e. no new frame has arrived since this consumer
        last looked.
        """
        with self._lock:
            if self._latest is None:
                return None
            frame, mono_ts, utc_ts = self._latest
        # v3: a frozen camera returns successfully forever unless staleness is checked.
        is_stale = (time.monotonic() - mono_ts) > self._stale_after_s
        is_new = mono_ts != self._last_served_mono_ts   # v11: new-frame signal
        self._last_served_mono_ts = mono_ts
        return frame, mono_ts, utc_ts, is_stale, is_new
```

A stale read is a **health signal, not an error**: the consumer should stop submitting stale frames to the NPU, mark `/health` degraded (§1.7), and increment `frames_stale_total`. A camera that is powered but frozen is a common and otherwise invisible failure.

**v11 correction — a fast consumer must not reprocess one physical frame as if it were several.** Nothing paces this loop to the camera's frame rate, and once `has_active_tracks` is true the motion gate (§1.2) is bypassed on every iteration (§1.2's own note). A gated no-NPU iteration and, especially, a full NPU inference iteration can each finish well inside one 100 ms frame period, so the consumer can call `get_latest()` and run MOG2/inference/tracker-update against the *same* frame several times before the reader thread produces a new one. Check `is_new` before doing any of that work: skip the MOG2 update, the NPU submission, and the tracker's hit-count update whenever `is_new` is `False` (staleness and reconnect handling still run every iteration regardless). Correspondingly, `min_confirmed_hits` (§1.4) must count one hit per distinct `mono_ts`, never per loop iteration — otherwise a single-frame detector artifact (a compression glitch, an IR-attracted insect, a headlight flare) can satisfy all `min_confirmed_hits` matches against itself at Δt≈0, reintroducing the same single-frame flicker false positive the tracker's hit-confirmation design was meant to eliminate (§1.4). Note this failure mode does not appear in the §1.11 replay harness, which feeds each recorded frame exactly once — so harness-measured false-positive rates will not reflect this risk; it must be reasoned about directly, not tuned away.

**v3 correction — throughput and latency are different budgets, and v2 conflated them.** v2 defined a single 100 ms `latency_budget_ms` whose components summed to exactly 100. Two separate problems:

1. **They measure different things.** 100 ms at 10 FPS is the **inter-frame period** — a *throughput* constraint, meaning each frame's processing must finish before the next arrives or the pipeline falls behind. End-to-end **latency** (capture → event row written) is a distinct quantity that can legitimately exceed one frame period when stages are pipelined. Track both, with separate targets and separate alarms. Confusing them means a pipeline that is keeping up perfectly can look like it is failing its budget, and vice versa.
2. **A budget summing to 100% of the available time is already over budget.** There is no headroom for GC pauses, the enrichment worker, the API process, the pre-roll buffer's I/O, or a second camera later. Target roughly **60–70% utilization** of the frame period, so aim the per-stage sum at 60–70 ms, not 100.

Also removed from the per-frame budget: `sqlite_write`, which moves to the dedicated writer thread (§1.6) since almost no frames emit events.

Instrument each stage's duration and expose it on a `/metrics` endpoint (Prometheus text format works well with FastAPI via `prometheus-fastapi-instrumentator`). In practice, on this hardware the Hailo‑10H's 40 TOPS handles YOLOv8n with room to spare — the more likely bottlenecks are H.264 decode load on the Pi's CPU and per‑call HailoRT memory‑copy overhead, so build your dashboards around those two stages first.

**v5 correction — reframed as a milestone-1 measurement item, not a hard ceiling.** The Raspberry Pi 5 removed the hardware H.264 decode block that the Pi 4 had; its only hardware video decoder is H.265/HEVC ([Raspberry Pi Forums — no H.264 hardware decode on Pi 5](https://forums.raspberrypi.com/viewtopic.php?t=364180), [decode architecture comparison across Pi generations](https://salivity.github.io/ffmpeg/article/hardware-accelerated-video-decoding-on-raspberry-pi-with-ffmpeg)). That much is real, but the v4 framing overstated its consequence: Raspberry Pi engineers report the Pi 5's quad Cortex-A76 cores software-decode H.264 fast enough to outperform the Pi 4's old hardware block outright, including at resolutions that block couldn't handle at all ([Raspberry Pi Forums — Pi 5 software decode vs. Pi 4 hardware decode](https://forums.raspberrypi.com/viewtopic.php?t=391283), [Raspberry Pi Forums — NEON-optimised software H.264 decode](https://forums.raspberrypi.com/viewtopic.php?t=357870)), and a 640×360@10fps substream sits at only roughly 1/25th to 1/40th the pixel rate of the 1080p30/4K workloads those reports describe. That is a reason to measure, not a reason to assume in either direction: **add a decode-cost measurement to milestone 1's exit criteria** (§3) — time `cv2.VideoCapture` reads against the recorded `.mp4` at the target substream resolution/frame rate and record the actual per-frame decode duration and core utilization, rather than carrying the placeholder `decode: 25` ms below as an assumed worst case. The cost still scales linearly with every camera added, which is what actually matters for the multi-camera generalization mentioned at the end of §3 — a cost that is negligible at one camera can still be the first budget that runs out at four, so measure it once here and re-check it as cameras are added rather than re-deriving the assumption later. **v6 correction — the H.265/HEVC codec change itself is deferred to the multi-camera generalization step, not recommended here.** The measured saving from decoding H.265 through the Pi 5's hardware block via `-hwaccel drm` is modest at one camera (13% → 9% of one core — roughly 1% of total CPU) and the change also touches the §1.12 PyAV remux path, so it is not worth taking on during single-camera milestone 2; see the multi-camera generalization paragraph at the end of §3 for the recommendation and the `-hwaccel drm` detail, where four camera streams make the same per-core saving worth the added complexity.

**Correction (v2) — timestamp semantics.** Every event record must carry the frame's **capture timestamp** (set the instant the ingestion thread reads the frame) as its authoritative time, never the timestamp of whichever downstream stage happens to process it. Under load, processing time can legitimately lag capture time by hundreds of milliseconds or more; if events are stamped with processing time, the recorded sequence of "what happened when" becomes wrong exactly when you need it most.

**v3 correction — `time.monotonic()` cannot be the event timestamp.** v2's reader stamped frames with `time.monotonic()` alone. Monotonic time is an unanchored counter from an arbitrary origin: it is exactly right for computing \(\Delta t\) (immune to NTP steps, which matters a great deal now that the Kalman filter and the MOG2 learning rate both depend on \(\Delta t\)), but it is **not a date** and cannot answer "when did this happen." Capture **both** clocks at read time, as the code above does, and persist both (§1.6).

**Related hardware note:** the Raspberry Pi 5 has an RTC, but it requires a **coin-cell battery** to retain time across power loss. Without one, every event recorded between cold boot and NTP sync carries a wrong wall-clock timestamp. For a system whose entire value is forensic, fit the battery.

```yaml
# config/observability.yaml (v3)
observability:
  metrics_endpoint: /metrics

  # Throughput: per-frame processing must fit inside the frame period, with headroom.
  frame_period_ms: 100              # 10 FPS substream
  throughput_target_utilization: 0.65   # v3: aim the per-stage sum at ~65 ms, not 100 ms
  stage_budget_ms:
    decode: 25                     # v5: provisional — replace with milestone-1's measured value (§1.10)
    motion_gate: 10
    npu_inference: 25
    tracker_update: 5
    # sqlite_write removed (v3) — moved to the dedicated writer thread (§1.6)

  # Latency: a separate end-to-end measurement with its own alarm.
  end_to_end_latency_ms:
    target_p50: 150
    alarm_p95: 400

  staleness:
    frame_stale_after_seconds: 2.0
  counters:
    - frames_captured_total
    - frames_dropped_total
    - frames_stale_total
    - rtsp_reconnects_total
    - npu_submissions_total
    - npu_failures_total
    - events_emitted_total
    - events_dropped_total
  alert_on_budget_overrun: true
  overrun_log_level: warning
```

**Deeper reading:**
- [PyImageSearch — Increasing webcam FPS with a threaded video stream](https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/) — the threaded‑capture pattern this design is built on
- [Stack Overflow — OpenCV VideoCapture lag due to the capture buffer](https://stackoverflow.com/questions/30032063/opencv-videocapture-lag-due-to-the-capture-buffer) — why naive `VideoCapture` usage silently accumulates latency
- [prometheus-fastapi-instrumentator (GitHub)](https://github.com/trallnag/prometheus-fastapi-instrumentator) — drop‑in Prometheus metrics for a FastAPI app, used for the `/metrics` endpoint
- [Raspberry Pi 5 RTC documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html) — the coin-cell battery connector and RTC behavior across power loss
- [Raspberry Pi Forums — Pi 5 has no hardware H.264 decoder](https://forums.raspberrypi.com/viewtopic.php?t=364180) — confirms the codec/decode gap driving the H.265 substream recommendation (v4)
- [Hardware-accelerated video decoding on Raspberry Pi with FFmpeg](https://salivity.github.io/ffmpeg/article/hardware-accelerated-video-decoding-on-raspberry-pi-with-ffmpeg) — decode capability comparison across Pi 4/5 generations (v4)
- [Raspberry Pi Forums — Pi 5 software decode outperforms Pi 4 hardware decode](https://forums.raspberrypi.com/viewtopic.php?t=391283) — evidence the software-decode cost is not automatically a hard ceiling (v5)
- [Raspberry Pi Forums — NEON-optimised software H.264 decode on Pi 5](https://forums.raspberrypi.com/viewtopic.php?t=357870) — same point, with implementation detail on why it's fast (v5)
- [Frigate GitHub discussion — Pi 5 `hwaccel drm` vs. `v4l2m2m`](https://github.com/blakeblackshear/frigate/discussions/18431) — the correct decode flag for Pi 5, with measured real-world CPU deltas (v5)

---

### 1.11 Offline Replay & Evaluation Harness — `src/eval/replay_runner.py`

**Purpose.** Every threshold in this guide — `minimum_motion_ratio`, `confidence_threshold`, `min_iou_match`, `min_confirmed_hits`, `minimum_mean_confidence`, `kalman_sigma_accel_sq` — has so far been described as "calibrated per camera scene," without specifying calibrated *against what*. Without ground truth, tuning is just guessing with extra steps. This is the single biggest gap the v2 review identified.

**How it works.** Record raw substream (and, ideally, mainstream) video to disk for each camera over a representative period — including edge cases like dusk/dawn lighting transitions, wind‑blown trees, and delivery‑vehicle visits. Hand‑label a CSV of ground‑truth events (`camera_id, event_type, start_ts, end_ts, condition, notes`) against that recorded footage. Then build a **replay runner** that feeds the recorded video through the *exact same* production pipeline code (motion gate → detector → tracker → state machine) using a **fake clock** driven by the recorded frame timestamps rather than `time.monotonic()`, so the time‑parameterized Kalman filter (§1.4), the MOG2 learning rate (§1.2), and the cooldown logic (§1.5) all behave identically to a live run.

```python
class FakeClock:
    def __init__(self, start_ts):
        self._t = start_ts
    def now(self):
        return self._t
    def advance_to(self, frame_ts):
        self._t = frame_ts   # driven by recorded frame timestamps, not wall-clock time
```

**v3 addition — define the matching rule, or precision and recall are not computable.** v2 specified CI gates (`fail_ci_below_precision: 0.85`) without ever defining what counts as a *match* between an emitted event and a labeled ground-truth event. That definition is the single most consequential parameter in the harness, and every reasonable choice yields a materially different number:

- **Temporal window.** Does an emitted event match a label if it falls inside `[start_ts - δ, end_ts + δ]`? What is δ? A generous δ inflates recall; a tight one punishes the pipeline for a legitimate half-second of confirmation delay.
- **Cardinality.** If the system fires three times during one real visit, is that one true positive, or one TP plus two false positives? (Recommended: **one-to-one greedy matching by nearest timestamp** — each label may be claimed by at most one emitted event, and every unclaimed emission is a false positive. This is the choice that actually penalizes duplicate-alert bugs, which is the behavior you care about.)
- **Unmatched labels** are false negatives by definition; make sure the harness reports them individually with timestamps, not just as a count, so you can go watch the footage of what was missed.

Write these into `eval.yaml` as explicit, versioned parameters. A precision number computed under an undocumented matching rule is not reproducible and does not belong in a portfolio writeup.

**v3 addition — hold out a validation split, and be honest about sample size.** You will realistically hand-label somewhere between a few dozen and a couple hundred events. Tuning five-plus thresholds against that set is a high-dimensional search over a small sample, and it *will* overfit: the thresholds you land on will look excellent on the labeled footage and generalize worse than the metrics suggest.

The fix is the standard one and it costs nothing here: **split the recorded footage by day.** Tune on the development days, and report final numbers on a held-out day that was never used for tuning. Only the held-out number goes in the writeup or the README.

On the CI gates specifically: with 23 labeled events, an observed precision of 0.87 carries a 95% confidence interval of roughly ±0.14. A hard gate at 0.85 against that sample is measuring noise, not quality. Record `n_labeled_events` in the harness output, report the interval alongside the point estimate, and treat the gates as advisory until the label set is large enough to support them.

**v10 addition — one calendar week of milestone-3 labeling won't reach the hard-gate threshold on its own.** `minimum_labeled_events_for_hard_gate: 100` (below) is unlikely to be met from a single week of footage at typical doorstep-camera event rates, which means the CI gates could stay in their advisory state indefinitely rather than the milestone-3 pass being a one-time step toward them. Keep labeling as an ongoing, rolling task past milestone 3 — continue adding labeled events from the already-running 24/7 recording (§3, milestone 2) at a regular cadence — rather than treating the single milestone-3 pass as the harness's permanent ground-truth set.

**v4 correction — that ±0.14 is a Wald interval, and Wald is the wrong tool at this sample size.** The interval above uses the textbook normal-approximation (Wald) formula, which is documented to under-cover — and can even produce bounds outside [0, 1] — precisely in the small-\(n\), near-boundary regime this harness will be operating in for months ([comparative analysis of Wald/Wilson/Jeffreys coverage](https://arxiv.org/html/2508.10223v1), [Brown, Cai & DasGupta (2001) — Interval Estimation for a Binomial Proportion](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full)). Use a **Wilson score interval** instead — a five-line change (`statsmodels.stats.proportion.proportion_confint(count, nobs, method="wilson")`), and honest exactly when the label set is smallest, which is when this project needs its numbers trusted most:

\[
\tilde p = \frac{\hat p + \dfrac{z^2}{2n} \pm z\sqrt{\dfrac{\hat p(1-\hat p)}{n} + \dfrac{z^2}{4n^2}}}{1 + \dfrac{z^2}{n}}
\]

**v4 addition — the harness should also report per-stage traces, not only the final event list.** As specified, a failed match tells you *that* the system missed or duplicated an alert, but not *where* in the pipeline the fault lies — the detector may never have boxed the person, the tracker may have boxed but never confirmed (or confirmed then lost identity), or the state machine may have confirmed but the zone geometry or cooldown ate the event. Have `replay_runner.py` emit an intermediate trace per labeled window — frames with a raw detection, frames with a confirmed track, and the track's full lifetime — alongside the final precision/recall numbers. Without this, root-causing a missed event still requires re-instrumenting the pipeline by hand every time; with it, the harness answers "which stage failed" directly.

**v6 addition — record which config produced each reported number.** By this point eleven separate YAML files feed the pipeline, and nothing currently ties a reported precision/recall number back to the exact configuration that produced it — re-running the harness after any tuning change silently invalidates the previous number's provenance. Have `replay_runner.py` compute a hash (e.g., SHA-256 of the concatenated, canonically-serialized contents of every config file it loads) and emit it alongside `n_labeled_events` in its output, and add a matching `config_hash` field to `eval.yaml` for the harness to populate at run time. This turns "precision was 0.87" into "precision was 0.87 under config `a3f9e1...`," which is what makes a later regression or a portfolio writeup's numbers actually reproducible.

**v7 addition — the `day` / `night_ir` split was specified in five places (this section's v4 revision note, §1.3's v4 correction, §1.2's v5 addition, milestone 3's exit criteria, and the readiness checklist) but never actually implemented in `eval.yaml`.** The `metrics:` list below was entirely event-level, so nothing computed the per-frame detector recall that §1.3 needs to tell a genuine domain gap apart from a mistuned `confidence_threshold` — an event-level number conflates detector performance with the tracker's and state machine's behavior on top of it. Added a `conditions: [day, night_ir]` field and a `condition_source` field to the `split:` block so every metric is reported per condition rather than pooled across both; added a separate `detector_metrics: [per_frame_recall, per_frame_precision]` list alongside `metrics:`, since these are computed directly against raw per-frame detections rather than matched events; and added a `negative_test_cases:` entry for the §1.2 IR-insect case, which expects zero emitted events despite sustained detector-stage activity. The ground-truth CSV schema above now includes a `condition` column, since there was otherwise nowhere to record which of `day` / `night_ir` a labeled event belongs to — without it, the harness has no way to compute anything per-condition regardless of what `eval.yaml` asks for.

```yaml
# config/eval.yaml (v8)
eval:
  recordings_dir: data/recordings/
  ground_truth_csv: data/ground_truth_events.csv

  # v3: the matching rule — without this, precision/recall are undefined
  matching:
    strategy: one_to_one_greedy_nearest
    temporal_window_seconds: 5.0      # emitted event must fall within [start - w, end + w]
    unmatched_emission: false_positive
    unmatched_label: false_negative
    report_unmatched_individually: true

  # v3: honest evaluation protocol
  split:
    tuning_days: [<tuning-day-1>, <tuning-day-2>, <tuning-day-3>]   # v8: placeholders, not literal dates — these must fall inside the week-2 (§3 milestone 2) recording window this harness actually labels from, and shift with the schedule rather than staying fixed
    heldout_days: [<heldout-day-1>]   # v8: same as above; never used for tuning, the only number that gets reported
    conditions: [day, night_ir]       # v7: report every metric per condition, not pooled
    condition_source: ground_truth_csv.condition   # v7: read from the CSV's condition column
    reject_unresolved_placeholders: true   # v9: replay_runner.py fails fast if tuning_days/heldout_days still match <...>
  report_sample_size: true
  report_confidence_intervals: true
  confidence_interval_method: wilson       # v4: was an implicit Wald/normal approximation
  emit_per_stage_traces: true              # v4: detection / confirmed-track / event-level, not just final matches
  config_hash: null                        # v6: populated at run time — SHA-256 over every loaded config file, reported alongside n_labeled_events

  metrics: [precision, recall, f1, mean_latency_to_event_ms]
  detector_metrics: [per_frame_recall, per_frame_precision]   # v7: per-frame, detector-stage only — isolates a domain gap from a threshold problem
  fail_ci_below_precision: 0.85       # advisory until n_labeled_events is large enough to support it
  fail_ci_below_recall: 0.80
  minimum_labeled_events_for_hard_gate: 100

  negative_test_cases:                # v7: cases that must emit zero events regardless of detector-stage activity
    - name: night_ir_insect_activity   # §1.2 v5 addition — IR-attracted insects at close range to the lens
      condition: night_ir
      expect: no_events
```

**v8 correction — the `split:` dates above previously named calendar days that hadn't happened yet.** `tuning_days` was `2026-09-28` through `2026-09-30` and `heldout_days` was `2026-10-01`, but milestone 3's exit criteria (§3) label the ground-truth CSV from **week-2 footage** (the Sep 21 – Sep 27 recording window) — so the literal split dates fell entirely inside milestone 3's own week, after labeling begins, not inside the footage that labeling actually draws from. Replaced with placeholders (`<tuning-day-1>` etc.) rather than a different set of hardcoded dates, since the real values depend on wherever the schedule lands when milestone 3 executes; both fields are illustrative and are meant to move with the schedule, not to be copied verbatim.

**v9 addition — fail fast if the placeholders above are never filled in.** Nothing previously stopped `replay_runner.py` from running against the literal `<tuning-day-1>`-style placeholders themselves, which would silently produce a meaningless or crashing run rather than a clear error. Adds `reject_unresolved_placeholders: true` above, and a corresponding startup check in `replay_runner.py` — consistent with the fail-fast philosophy already applied in §1.3 and §1.13 — that refuses to start a harness run while any `split.tuning_days`/`heldout_days` entry still matches the `<...>` placeholder pattern.

**Deeper reading:**
- [Evaluating object detection models: methods and metrics (GeeksforGeeks)](https://www.geeksforgeeks.org/computer-vision/evaluating-object-detection-models-methods-and-metrics/) — precision/recall/F1 definitions applied to detection pipelines
- [Object detection metrics explained (Label Your Data)](https://labelyourdata.com/articles/object-detection-metrics) — a practical walkthrough of the same metrics with worked examples
- [Comparative analysis of Wald, Wilson, and other proportion CIs](https://arxiv.org/html/2508.10223v1) — coverage behavior at small n and boundary proportions (v4)
- [Brown, Cai & DasGupta (2001) — Interval Estimation for a Binomial Proportion](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full) — the canonical paper establishing Wald's small-n/near-boundary coverage failure and recommending Wilson (Statistical Science 16(2):101–133) (v5)

---

### 1.12 Pre‑Roll Evidence Buffer — `src/ingest/ring_buffer.py`

**Purpose.** Because a track only becomes `is_confirmed` after `min_confirmed_hits` matches and `minimum_track_age_seconds` (§1.4–§1.5), the event itself always fires **after** the interesting behavior already started — by definition, confirmation requires having already observed it happening for a moment. Without a buffer, the saved clip for an event starts mid‑action instead of showing the approach.

**How it works.** Maintain a continuous, fixed‑duration **ring buffer** of the mainstream (the full‑resolution feed, not the substream used for motion gating/detection), storing **encoded packets** rather than decoded frames to keep CPU and memory overhead low. On event trigger, flush a window spanning a configurable pre‑roll (e.g., 5–10 seconds before the track's first detection) plus a post‑roll (e.g., 5–10 seconds after the event fires or the track is lost) into a saved clip linked to the event record.

**v3 correction — `storage_mode: encoded_packets` is not achievable with OpenCV.** v2 correctly specified buffering encoded packets, but `cv2.VideoCapture` **decodes**; it exposes no access to the underlying H.264 packets. Nothing in the v2 dependency list can do what that config field describes. Two workable options:

- **PyAV** (Python bindings to libav) — lets you demux the RTSP stream and hold raw `AVPacket`s in a `collections.deque`, then remux them into an MP4 on trigger with no re-encode. This is the right answer and should be added to the stack.
- **ffmpeg subprocess** writing rolling segments (`-f segment -segment_time 2`) to disk, keeping the last N segments and concatenating on trigger. Simpler to implement and debug, at the cost of continuous disk writes and coarser cut granularity.

**v3 correction — clips can only be cut at keyframes.** An H.264 stream is decodable only from an IDR (keyframe) onward; starting a clip mid-GOP produces garbage until the next one. Two consequences:

1. `buffer_seconds` must comfortably exceed the camera's keyframe interval, or the buffer may contain no cut point at all.
2. The camera's **I-frame / GOP interval must be lowered to roughly 1 second** in its web UI (§1.1). At a default 4-second GOP, a "6-second pre-roll" is really "4 to 8 seconds, with an undecodable head." This is a camera-side setting with no representation in the repository, which is exactly why it gets discovered late.

The flush logic must therefore search backward from the requested pre-roll point to the **nearest preceding keyframe**, and record the actual achieved pre-roll duration on the event record rather than assuming the configured value.

**v3 note — the pre-roll buffer is a second continuous RTSP connection.** It runs 24/7 against the mainstream, independent of the substream used for inference. Budget its CPU (demux only, no decode, so modest) and its bandwidth, and give it its own reconnect logic and `/metrics` counters. It should also be the *last* thing to be shut down on failure: recording evidence without analysis (§1.3) is a useful degraded state.

```yaml
# config/pre_roll.yaml (v3)
pre_roll:
  enabled: true
  source_stream: mainstream
  backend: pyav                    # v3: OpenCV cannot expose encoded packets — PyAV or ffmpeg segments
  storage_mode: encoded_packets    # remux on trigger, no re-encode
  buffer_seconds: 15               # must exceed the camera GOP interval by a wide margin
  pre_roll_seconds: 6
  post_roll_seconds: 8
  cut_on_keyframe: true            # v3: search back to the nearest preceding IDR
  record_achieved_pre_roll: true   # v3: log what you actually got, not what you asked for
  expected_camera_gop_seconds: 1.0 # v3: set camera-side (§1.1); assert at startup if detectable
  clip_output_dir: /mnt/ssd/clips/
```

**Deeper reading:**
- [PyAV documentation](https://pyav.org/docs/stable/) — demuxing to raw packets and remuxing without re-encoding, the mechanism this section depends on
- [picamera circular streams (deepwiki)](https://deepwiki.com/waveform80/picamera/4.2-circular-streams) — a Raspberry Pi‑native reference implementation of exactly this circular pre‑roll pattern
- [VisioForge — Pre-event recording guide](https://www.visioforge.com/help/docs/dotnet/mediablocks/Guides/pre-event-recording/) — general design considerations for pre/post-roll buffering around a trigger
- [Battleroid/seccam (GitHub)](https://github.com/Battleroid/seccam) — an open security-camera reference implementing motion-triggered pre-roll clip capture

---

### 1.13 Operational Hardening & Licensing

**Purpose.** A 24‑hour soak test (§3) is only meaningful if failure modes that take hours or days to manifest — disk exhaustion, log bloat, flash storage wear — are defined and handled ahead of time, and if the project's legal footing is settled before any code is shared publicly.

**How it works.**

- **Disk‑full behavior.** Decide explicitly what happens when the clips/database volume fills up: the recommended MVP behavior is to stop accepting new pre‑roll clips (oldest‑first eviction) while still writing event *rows* (small, cheap) so the alert history itself never breaks even if video storage is exhausted.
- **Log rotation.** Use Python's built‑in `RotatingFileHandler` (or `TimedRotatingFileHandler`) so debug/motion‑gate/inference logs cap out at a fixed total size instead of growing unbounded over a multi‑day soak test.
- **SD card write wear.** Continuous SQLite WAL writes (§1.6) plus frequent log writes are exactly the access pattern that shortens a microSD card's life. Before the soak test, move the SQLite database file **and** the clips directory onto a **USB‑attached SSD** rather than the boot microSD — SSDs have far higher write endurance and the failure mode (slow degradation, SMART‑reportable) is much safer than a silently corrupting SD card. With the pre-roll buffer (§1.12) now writing clips continuously on trigger, this is no longer optional.
- **RTC battery (v3).** Fit the coin cell (§1.10). Event timestamps from a Pi that booted without one are wrong until NTP sync.
- **NPU health (v3).** The failure policy in §1.3 needs a corresponding startup check in the service unit: verify `hailortcli fw-control identify` succeeds before the application starts, so a driver/firmware mismatch after an `apt upgrade` surfaces as a clean startup failure rather than a runtime exception at 3 a.m.
- **Licensing.** As noted in §1.3, Ultralytics YOLOv8 ships under AGPL‑3.0. Not a blocker for a private MVP, but resolve it *before* deciding to open‑source the repository as a portfolio piece — either keep the repo private, isolate the Ultralytics‑derived training/export code from any published application code, or budget for an Ultralytics Enterprise license if wide redistribution is a goal.
- **Process supervision (v4).** Everything above covers failure to *start*; nothing in v1–v3 covers a crash, hang, or OOM once the pipeline is already running — a segfault in the `cv2`/HailoRT native bindings, the kernel OOM-killer once the resident enrichment LLM (§1.9) competes with the rest of the process for RAM, or a background thread that wedges without killing the main process. Run the application under a `systemd` unit with `Restart=on-failure`, a `RestartSec` backoff, and `StartLimitIntervalSec`/`StartLimitBurst` so a persistent fault reboots cleanly instead of crash-looping forever ([systemd restart/reliability patterns](https://forums.raspberrypi.com/viewtopic.php?t=376126), [practical Raspberry Pi service-reliability guide](https://www.dzombak.com/blog/2023/12/keep-your-software-up-and-running-on-the-raspberry-pi/)). Add `WatchdogSec=` with a periodic `sd_notify` heartbeat from the main loop so systemd can detect a **hung** process too, not only a crashed one — this is the cheapest reliability improvement available and it directly serves the soak test's own goal.
- **v11 correction — the heartbeat must check worker threads, not just its own pulse.** As specified above, the heartbeat only proves the thread that calls `sd_notify` is still looping; it says nothing about the dedicated SQLite writer thread (§1.6) or the notification consumer thread (§1.14). If either dies from an uncaught exception (a disk I/O error, a malformed row, an HTTP client exception), the main loop keeps sending `WATCHDOG=1`, the dead-man's switch (§1.14) keeps getting pinged, and `/health` stays healthy — because none of those three signals ever inspect worker-thread state, only process liveness and NPU/staleness (§1.7). Events then queue up, hit `queue_max_size` (§1.14), and are silently dropped via `events_dropped_total`, a counter nothing alerts on. Gate the heartbeat on all three threads being alive, so a dead worker stops the heartbeat, trips the watchdog, and forces exactly the crash-recovery + out-of-process-alert path already built above:

  ```python
  # v11: heartbeat only fires if every required worker thread is still alive
  if all(t.is_alive() for t in (reader_thread, writer_thread, notifier_thread)):
      sd_notify("WATCHDOG=1")
  # else: skip the ping — WatchdogSec times out, systemd restarts the unit,
  # and (once EnvironmentFile above is fixed) OnFailure= fires the webhook.
  ```
- **Out-of-process failure notification (v11).** `Restart=`/`WatchdogSec=` above *recover* from a crash or hang; nothing *reports* it, because the only notification consumer (§1.14, v9) lives inside the process that just died — and once `StartLimitBurst` is exhausted the unit sits in `failed` indefinitely with nobody told. Add `OnFailure=camera-alert@%n.service` to the pipeline unit: systemd activates that oneshot unit each time the pipeline enters the `failed` state (a crash, a watchdog-detected hang, and the final restart-limit exhaustion all pass through `failed` under the default `RestartMode=normal`), and the oneshot delivers a webhook from outside the failed process. This covers "the process is not running"; it does **not** cover power loss, network loss, or a wedged kernel, since systemd itself is gone in those cases — see the dead-man's switch in §1.14, which covers exactly that disjoint set.

  ```ini
  # /etc/systemd/system/camera-pipeline.service — [Unit] section addition (v11)
  [Unit]
  OnFailure=camera-alert@%n.service

  # /etc/systemd/system/camera-alert@.service (v11) — runs outside the failed process
  [Unit]
  Description=Out-of-process failure alert for %i

  [Service]
  Type=oneshot
  # NOTIFY_WEBHOOK_URL — same webhook as config/notify.yaml, never committed.
  # v11 fix: systemd does NOT support trailing/inline comments on a unit directive —
  # anything after `EnvironmentFile=<path>` on the same line becomes part of the path
  # value, so the line below must carry no comment of its own.
  EnvironmentFile=/etc/camera/notify.env
  ExecStart=/usr/bin/curl -fsS -d "%i entered failed state on %H" "${NOTIFY_WEBHOOK_URL}"
  ```

- **Whole-system memory budget (v4).** §1.9 already flags NPU *device* memory contention between the detector and the resident enrichment LLM; the same discipline applies to host RAM. The concurrent set on one Pi 5 is: the RTSP mailbox thread, the PyAV pre-roll demux thread, the HailoRT runtime, Uvicorn/FastAPI, APScheduler, SQLite (WAL), and logging — with no RAM ceiling defined anywhere. Do a rough budget during the milestone 1 offline spike (with the enrichment model resident, since that's the worst case) and expose host memory on `/metrics` alongside the existing NPU/device counters, so a slow memory creep during the soak test is visible before it becomes an OOM kill.
- **Retention vs. capacity (v4).** `disk_full_policy` above is a *capacity* policy — it only fires under storage pressure. There is no independent maximum-age purge, and no stated position on incidentally recording people who are not the household (delivery workers, mail carriers, neighbors passing the zone boundary). Even for a private residential MVP, define an explicit `max_retention_days` for clips independent of disk usage, and decide up front how long footage of non-household people is kept.

```yaml
# config/hardening.yaml (v4)
hardening:
  disk_full_policy: evict_oldest_clips_keep_event_rows
  max_retention_days: 30                        # v4: age-based purge, independent of disk pressure
  log_rotation:
    handler: RotatingFileHandler
    max_bytes: 10485760      # 10 MB per file
    backup_count: 5
  storage:
    sqlite_db_path: /mnt/ssd/camera_events.db   # moved off the boot microSD before soak testing
    clips_dir: /mnt/ssd/clips/
    minimum_free_gb: 5                          # v3: eviction trigger, not "wait for ENOSPC"
  startup_checks:                               # v3: fail-fast, before the ingestion thread starts
    - hailo_device_identify
    - hef_load
    - rtc_time_plausible
    - storage_writable
  process_supervision:                          # v4: covers mid-run crash/hang, not just startup
    manager: systemd
    restart_policy: on-failure
    restart_sec: 5
    start_limit_interval_sec: 300
    start_limit_burst: 5
    watchdog_sec: 30
    watchdog_heartbeat: sd_notify
    on_failure_unit: camera-alert@%n.service   # v11: out-of-process webhook when the unit enters `failed`, incl. restart-limit exhaustion
  observability:
    host_memory_metric: process_rss_bytes       # v4: exposed on /metrics alongside NPU counters
  licensing:
    yolo_toolchain_license: AGPL-3.0
    safe_for_private_use: true
    requires_review_before_open_source: true
```

**Deeper reading:**
- [Python logging cookbook](https://docs.python.org/3/howto/logging-cookbook.html) — practical recipes including rotating file handlers
- [Python `logging.handlers` reference](https://docs.python.org/3/library/logging.handlers.html) — `RotatingFileHandler`/`TimedRotatingFileHandler` API details
- [SQLite on a Raspberry Pi (Atomic Object)](https://spin.atomicobject.com/sqlite-raspberry-pi/) — practical notes on SQLite write patterns and SD card considerations on this exact class of hardware
- [SD card lifespan calculator (raspberry.tips)](https://raspberry.tips/en/sd-card-lifespan-calculator-how-long-will-your-storage-last) — a concrete way to estimate write‑wear budget before moving storage to SSD
- [Ultralytics License page](https://www.ultralytics.com/license) — AGPL-3.0 vs. Enterprise terms
- [systemd restart/watchdog patterns for Raspberry Pi services](https://forums.raspberrypi.com/viewtopic.php?t=376126) — `Restart=`, `WatchdogSec=`, and the crash-vs-hang distinction (v4)
- [Keeping software running on the Raspberry Pi (dzombak.com)](https://www.dzombak.com/blog/2023/12/keep-your-software-up-and-running-on-the-raspberry-pi/) — a practical systemd reliability walkthrough (v4)

---

### 1.14 Alerting & Notification Channel — `src/notify/` *(v4 addition)*

**Purpose.** Nothing in §1.1–§1.13 ever reaches a human. An emitted, persisted, and even enriched event is still just a row in SQLite until someone polls `/events` — and a security system nobody is notified about only functions as security if someone remembers to check it. This is a product gap, not a pipeline bug, but it is the single missing piece that turns this project from "a working pipeline" into "an actual alert."

**How it works.** **v11 correction — the notifier must not drain the writer's input queue.** Do not attach the notifier as a second `.get()` consumer of the §1.6 writer-thread queue: standard queue semantics hand each enqueued item to exactly one consumer, so two consumers racing on one `queue.Queue` silently split events between them — roughly half persisted-but-never-alerted, the other half alerted-but-never-persisted, with no exception raised. Instead, after the SQLite writer thread successfully commits an event row, have it publish the committed event's ID onto a separate, bounded notification queue that only the notifier reads (or use the polling `/events` watcher already offered as an alternative, which sidesteps the queue-sharing question entirely). That notifier then fires a push notification — a webhook to [ntfy](https://ntfy.sh/) or [Pushover](https://pushover.net/), or a Home Assistant/MQTT publish if the household already runs one. Keep this off the frame-processing critical path the same way SQLite writes and enrichment already are: a slow or unreachable notification endpoint must never block ingestion.

**v5 addition — this is the project's previously-decided alert outbox, not a new design.** The webhook-first, additive-second-channel shape here was already settled for this project before this MVP guide existed: deliver through a generic webhook (ntfy/Pushover/Home Assistant all satisfy this) as the always-on baseline channel, and treat a dedicated Telegram bot integration as a second channel layered on top once the webhook path is proven, not a replacement for it. Keeping the webhook as the baseline avoids coupling the MVP's only notification path to one third-party API's uptime and rate limits. Now that this section has its own milestone slot (§3), scope it explicitly to that decision rather than re-deriving it from scratch.

**v9 addition — the notification channel now also watches system-health signals, not only emitted events.** `min_severity: entry_event` meant `/health` degraded/down transitions (§1.3, §1.7, §1.10) and the `tamper_suspected` signal (§1.16) never reached a human — the exact blind spot this section exists to close, just for infrastructure failure instead of a missed event. Adds a second consumer path that subscribes to `/health` state transitions and the tamper signal directly, independent of the writer-thread event queue described above, and routes both through the same delivery channels (webhook first, Telegram second) at their own severity tier.

**v11 addition — a dead-man's switch, because the v9 consumer shares fate with what it watches.** The `/health` consumer above runs *inside* the pipeline process, so it can report "degraded but running" and nothing else: a crash, an OOM kill, systemd giving up after `start_limit_burst: 5`, a power cut, or a lost network connection all silence it at exactly the moment it matters. Two mechanisms close this, and they cover **disjoint** failure sets: the `OnFailure=` unit in §1.13 reports a process that has died, hung, or exhausted its restart budget, but needs systemd and the network alive to do so; the dead-man's switch below needs neither — an APScheduler job (§1.8, already in the stack) issues an HTTP ping to an external endpoint every `interval_seconds`, and the *external* service raises the alert when pings stop for longer than `grace_period_seconds`. It is the only mechanism in this document that catches power loss, network loss, or a wedged kernel, precisely because the alert is decided somewhere the Pi cannot take down with it. Pinging is a job, not a health check — it must not depend on the frame loop or the writer thread, or it inherits their failure modes.

```yaml
# config/notify.yaml (v11)
notify:
  enabled: true
  provider: webhook             # v5: webhook is the baseline outbox channel — ntfy/pushover/mqtt are all webhook-shaped
  base_url: https://ntfy.sh/<topic>
  secondary_provider: telegram  # v5: additive second channel once the webhook path is proven, per the outbox decision above
  throttle:                     # v11.2: moved from events.yaml; throttles notifications, never events
    same_camera_same_event_seconds: 10
    record_suppressed: true     # every suppressed notification is persisted and counted
  include_enrichment_summary: true   # attach the §1.9 text summary once it's ready, don't block on it
  queue_max_size: 50
  queue_overflow_policy: drop_oldest
  min_severity: entry_event
  health_severity_tiers:               # v9: watches /health transitions and the §1.16 tamper signal, not just emitted events
    - health_degraded
    - health_down
    - tamper_suspected
  watch_health_endpoint: true          # v9: second consumer path, independent of the §1.6 writer-thread queue above
  dead_mans_switch:                    # v11: external service alerts when pings stop — the only path that survives power/network/kernel loss
    enabled: true
    ping_url: https://<dead-mans-switch-service>/ping/<check-id>   # secret-bearing URL: env var or 0600 file, never committed (same rule as §1.7)
    interval_seconds: 60
    grace_period_seconds: 180          # external alert fires after this long without a ping; ≥ 2× interval to tolerate one lost ping
```

**Deeper reading:**
- [ntfy.sh documentation](https://docs.ntfy.sh/) — self-hostable push notifications over a simple HTTP API
- [Pushover API](https://pushover.net/api) — a hosted push-notification service commonly used for home-automation alerts

---

### 1.15 Minimal Operator UI — `src/ui/` *(v4 addition)*

**Purpose.** The API (§1.7) is JSON-only. Watching what actually happened still means manually pulling clip files off the Pi — which makes the MVP demoable but not genuinely usable day to day.

**How it works.** A single static HTML page, served by the same FastAPI app, listing recent events (thumbnail or first frame, timestamp, zone, enrichment summary) with an embedded `<video>` tag pointing at the corresponding pre-roll clip (§1.12). This does not need a frontend framework, build step, or authentication scheme beyond what §1.7 already specifies — it is a thin read-only view over `/events` and the clips directory, not a new architectural layer.

**Deeper reading:**
- [FastAPI — serving static files](https://fastapi.tiangolo.com/tutorial/static-files/) — the minimal mechanism needed to serve this page from the same process

---

### 1.16 Camera Tamper & Obstruction Detection *(v4 addition)*

**Purpose.** The motion gate (§1.2) cannot distinguish "nothing moved" from "someone covered, blacked-out, or re-aimed the lens" — a well-known and specifically security-relevant attack against exactly this class of system, and one the current design has no way to notice.

**How it works.** Add a lightweight tamper heuristic alongside the MOG2 gate: a sustained run of frames with near-zero variance or entropy, especially combined with a sudden step change in mean luminance (lens covered) or a persistent, large full-frame motion event with no plausible cause (camera physically re-aimed), should raise a distinct `tamper_suspected` signal on `/health` — separate from the existing NPU/staleness degradation states (§1.3, §1.7, §1.10) — rather than silently reading as "a very quiet scene."

**v5 correction — a day/night IR cutover is indistinguishable from the tamper signal above unless exempted explicitly.** Every camera in this project switches between color and IR-illuminated modes at dusk and dawn, and that switch is itself a sudden, large step change in mean luminance — exactly what `luminance_step_threshold` is designed to catch. Left as specified, this heuristic raises `tamper_suspected` twice a day, every day, which is the textbook way an operator learns to ignore a security signal. Two fixes, either sufficient and stronger combined: (1) **cross-check against the camera's own reported day/night mode** — most ONVIF cameras expose an IR-cut-filter/imaging-mode setting or event, so a luminance step that coincides with a reported mode change is a routine cutover, not tamper; (2) if that signal isn't reliably exposed by your specific camera, use a **persistence check** instead of a bare step threshold — a real cover/blackout event stays low-variance *after* the luminance step (the lens is covered, so structure never returns), while an IR cutover's step is immediately followed by the image regaining normal variance and detail as the illuminator settles. Require `low_variance_frame_count` consecutive low-variance frames *following* the step, not just the step itself, before raising `tamper_suspected`.

**v11 scope note.** This heuristic detects lens obstruction and re-aiming only while the Pi remains powered and running — it runs in-process on decoded frames, so cutting power or network to the Pi disables the detector along with the camera. That vector is covered instead by the dead-man's switch in §1.14 (v11), not by anything in this section.

```yaml
# config/tamper.yaml (v5)
tamper:
  enabled: true
  luminance_step_threshold: 40        # sudden mean-brightness jump between consecutive samples
  ir_cutover_grace_period_seconds: 10  # v5: suppress the step check around a reported/expected day-night mode change
  require_persistence_after_step: true # v5: step alone is not sufficient — variance must stay low afterward, not just spike momentarily
  low_variance_frame_count: 30        # consecutive near-zero-variance frames before flagging
  low_variance_threshold: 5.0
  action: mark_health_tamper_suspected
```

**Deeper reading:**
- [OpenCV — image statistics and histogram basics](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html) — the building blocks (mean/variance) this heuristic is built from

---

### 1.17 Testing & CI for Pure-Function Components *(v4 addition)*

**Purpose.** The §1.11 replay harness validates the system end to end against real footage, which is exactly right for calibration — but it is slow, requires labeled data, and is the wrong tool for catching a regression in a small, pure function. IoU, the \(Q(\Delta t)\) block (§1.4/§4.1), point-in-polygon zone membership, and the homography round-trip (§4.2) are all independently and cheaply unit-testable, and a bug in any of them would otherwise surface weeks later as an unexplained precision drop in the harness rather than as a failing test today.

**v6 correction — these tests are written alongside the code that needs them, not deferred to a later cleanup pass.** This section describes what the pure-function test suite covers; it is no longer a standalone phase-2 backlog item (see §3). IoU and the \(Q(\Delta t)\) block are written at milestone 5, when the tracker is built; point-in-polygon is written at milestone 6, when zone logic is built. Only the homography round-trip test remains genuinely phase-2, since §4.2 itself is a phase-2 refinement not yet on the MVP critical path.

**How it works.** A small `pytest` suite, run in CI on every commit, covering: IoU against hand-computed cases (identical boxes, no overlap, partial overlap); the \(Q(\Delta t)\) block against the closed-form values in §4.1 for a few chosen \(\Delta t\); point-in-polygon against a known zone shape and a handful of inside/outside/edge points; and the homography round-trip (project four calibration points through \(H\) and back, assert sub-pixel/sub-centimeter error). None of this replaces the replay harness — it catches a different, cheaper class of bug before it ever reaches real footage.

**Deeper reading:**
- [pytest documentation](https://docs.pytest.org/) — the test runner this suite is built on
- [GitHub Actions — Python CI quickstart](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python) — wiring the pytest suite into CI on every push

---

### 1.18 Configuration Validation & Startup Invariants — `src/config/validate.py` *(v9 addition)*

**Purpose.** Four of the eight revisions to this document were caused by the same underlying problem: configuration lives across eleven independent YAML files with no schema and no cross-file invariant checking, so a value can drift out of sync with a related value elsewhere — silently, until it surfaces as a runtime bug or a misleading metric. v1 duplicated a confirmation-count field under two different names (§1.4); v6 discovered `kalman_sigma_accel` had been misnamed for its own units; v7 found the value derived from that naming error was itself arithmetically wrong (§1.4); v8 found `eval.yaml`'s split dates contradicted the milestone table (§1.11/§3). Each of these was only caught by a human re-reading multiple files side by side.

**How it works.** Add a validation step that runs before the ingestion thread starts (the same point in the startup sequence as the NPU device checks in §1.3 and the `startup_checks` list in §1.13), asserting invariants this document already states in prose but has never enforced in code:

```python
# src/config/validate.py (v9)
class ConfigError(Exception):
    """Raised at startup when a cross-file config invariant is violated. Never a warning — refuse to start."""

def validate_config(cfg) -> None:
    if cfg.detector.confidence_threshold != cfg.tracker.low_confidence_threshold:
        raise ConfigError(
            f"detector.confidence_threshold ({cfg.detector.confidence_threshold}) must equal "
            f"tracker.low_confidence_threshold ({cfg.tracker.low_confidence_threshold}) — "
            "the two-tier association in §1.4 depends on the detector never discarding "
            "boxes below the tracker's low-confidence tier."
        )
    for camera_id in cfg.cameras:
        zone_path = cfg.events.zone_rules.zone_definitions / f"{camera_id}.json"
        if not zone_path.exists():
            raise ConfigError(f"missing zone polygon file for camera '{camera_id}': {zone_path}")
```

This checks exactly the two invariants named above — `detector.confidence_threshold == tracker.low_confidence_threshold` (§1.3/§1.4) and that every configured camera has a zone-polygon file (§1.2/§1.5) — and raises rather than warns, matching the fail-fast philosophy already used for NPU and storage checks elsewhere in this document. It does not replace careful review of the eleven YAML files; it catches the specific class of drift that has already caused real bugs in this project's own history. Validation runs at startup only: a configuration change takes effect only after a service restart, and no hot-reload path exists or is planned for the MVP (see `docs/post-mvp-backlog.md`).

**Deeper reading:**
- [Pydantic Settings documentation](https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/) — a ready-made typed-settings/validation layer this module can be built on

---

## 2. How the Stages Fit Together

| Stage | Consumes | Produces | Failure mode it guards against |
|---|---|---|---|
| **Pixel budget (§0)** | Lens spec + zone distance | Go/no-go on model, lens, resolution | Building the whole pipeline around a person too small for YOLOv8n to see |
| Ingestion (threaded mailbox, reconnect, staleness) | Camera IP + ONVIF creds | Decoded frames, monotonic + UTC timestamps | RTSP path guessing, silent stream death, the "OpenCV smear", frozen-but-alive cameras |
| Motion gate (MOG2, `== 255`, Δt learning rate, zone mask) | Decoded frames | Forward/skip decision | Wasted NPU cycles; false triggers from trees/shadows/IR insects; load-dependent gate sensitivity |
| YOLOv8n detector (letterboxed, 0.15 floor, fail-loud) | Gated frames | Bounding boxes + class + confidence | Unsupported model, stretched-aspect distortion, starving the tracker's low-confidence tier, silent NPU death |
| Tracker (time-parameterized Kalman, cubic `Q(Δt)`, IoU) | Per-frame boxes | Persistent `track_id`s + `is_confirmed` | Flicker-driven false alerts; bad extrapolation and identity loss after long gate skips |
| Event state machine (entry-only for MVP) | Confirmed tracks + zone polygons | Emitted events | Duplicate alerts, premature triggering, config drift on confirmation thresholds |
| Pre-roll ring buffer (PyAV, keyframe-aligned) | Mainstream packets | Pre/post-roll clips | Clips starting after the action began; undecodable mid-GOP clip heads |
| SQLite store (async writer thread) | Emitted events | Durable rows, both timestamps | Event loss; a WAL checkpoint stalling the detection loop |
| FastAPI + `/metrics` + tri-state `/health` | Stored events | HTTP responses, latency metrics, health | API blocking detection; invisible budget overruns; "up" that isn't analyzing |
| Enrichment (bounded queue, drop-oldest) | Persisted event metadata | Natural-language summary | Cloud dependency; a wedged LLM backpressuring detection |
| Offline replay harness (matching rule, held-out split, Wilson CI) | Recorded video + labeled CSV | Precision/recall + honest CIs + per-stage traces | Calibrating thresholds with no ground truth; overfitting a tiny label set; an under-covering CI hiding real noise |
| Notification channel (v4) | Persisted event + optional enrichment | Push alert to a human | An alert history nobody is ever told about |
| Operator UI (v4) | Stored events + clips | Human-viewable event/clip list | An MVP that is demoable but not actually usable day to day |
| Tamper/obstruction detection (v4) | Decoded frames' luminance/variance | `tamper_suspected` health signal | A covered or re-aimed lens reading as "a quiet scene" |
| Process supervision + memory budget (v4) | systemd unit + `/metrics` host RAM | Auto-restart, hang detection, OOM warning | A crashed or wedged process that silently stops the whole system |
| Unit tests / CI (v4; written per-milestone as of v6) | Pure functions (IoU, `Q(Δt)`, polygon, homography) | Pass/fail on every commit | A cheap regression surfacing weeks later as an unexplained harness metric drop |

---

## 3. Milestone Checklist & Proposed Timeline

Proposed pacing assumes part‑time, incremental work (evenings/weekends), starting the week of **September 14, 2026**. Items deliberately deferred past the MVP are tracked in `docs/post-mvp-backlog.md` (v11), not in this guide.

**Reprioritized in v2:** the highest‑risk item (Hailo HEF compatibility) runs first as an isolated offline spike, RTSP ingestion moves to week 2, the replay/eval harness is built *before* any threshold tuning, and the event state machine ships entry‑only.

**Further corrected in v3:** a pre-flight pixel budget (§0) runs before milestone 1 because it can invalidate the model or lens choice; **continuous footage recording starts passively at the end of milestone 2**, since accumulating dusk/dawn/wind/delivery footage is measured in *calendar weeks* and is almost none of your working time; **zone polygons are authored at milestone 3**, because the motion gate's ROI mask (§1.2) consumes them and milestone 4 cannot otherwise proceed; and v2's milestone 8 is split, because it bundled the two riskiest remaining items (pre-roll and enrichment) into a single week immediately before the soak test.

**Refined further in v5:** milestone 1 now measures the actual §1.10 decode cost against recorded footage instead of carrying an assumed budget forward unverified; and **milestone 7 is split** — persistence/API/`/metrics` and the notification channel were bundled into one week in v4, which overloaded it, so notification now has its own slot (milestone 8), pushing every milestone after it back by one week.

| # | Milestone | Target window | Exit criteria |
|---|---|---|---|
| **0** | **Pixel budget & lens decision** | Sep 8 – Sep 13, 2026 | Measured pixel height of a person at each camera's zone boundary; written decision on lens (2.8 vs 4.0 mm), substream resolution, crop-vs-full-frame inference, and YOLOv8n vs YOLOv8s (§0) |
| 1 | YOLOv8n on Hailo‑10H — offline spike | Sep 14 – Sep 20, 2026 | `yolov8n_h10h.hef` loads and returns person detections against a **recorded** `.mp4` (no live camera needed); INT8 HEF output cross-checked against FP32 Ultralytics on the same frames (§1.3); startup fail-fast and mid-run NPU failure policy implemented; **v5: software H.264 decode cost of the target substream measured directly against the recorded `.mp4` (per-frame duration and core utilization), replacing the assumed `decode: 25 ms` budget with a measured value (§1.10)** |
| 2 | RTSP ingestion + threaded capture | Sep 21 – Sep 27, 2026 | Live substream via the single-slot mailbox (§1.10) with exponential-backoff reconnect and staleness detection; every frame carries both monotonic and UTC timestamps; camera GOP set to ~1 s (§1.1); **v11: ONVIF credentials loaded from environment variables; no credentials present anywhere in the repository (§1.1)**. **Start continuous 24/7 footage recording at the end of this week and leave it running through milestone 6** |
| 3 | Zone polygons, labeling & replay harness | Sep 28 – Oct 4, 2026 | Zone polygons authored to `configs/zones/` (§1.5); ground-truth CSV labeled from week-2 footage, **explicitly split `day` / `night_ir` (§1.3, v4)**; `replay_runner.py` (§1.11) replays through the real pipeline on a fake clock and reports precision/recall **with the matching rule and held-out split explicitly configured**, using a **Wilson score interval and per-stage traces rather than a Wald CI (§1.11, v4)** — built before any threshold is tuned; **v11: `night_ir_insect` negative test passes — zero emitted events on the labeled insect-activity windows (§1.11)** |
| 4 | Motion gate calibrated via harness | Oct 5 – Oct 11, 2026 | MOG2 + zone-derived mask (§1.2) with the `== 255` shadow test and Δt-scaled `learningRate`; `minimum_motion_ratio` tuned by re-running the harness, not by eyeballing a live feed |
| 5 | Time-aware Kalman/IoU tracker | Oct 12 – Oct 18, 2026 | Tracker owns `min_confirmed_hits` and exposes `track.is_confirmed`; `F(Δt)` rebuilt per predict step and `Q(Δt)` uses the continuous white-noise block (§1.4/§4.1); expiry is `max_missed_seconds`; `kalman_sigma_accel_sq` tuned via the harness; **v6: `IoU` and `Q(Δt)` pure-function unit tests written against §4.1's closed-form values** |
| 6 | Event state machine — entry-only | Oct 19 – Oct 25, 2026 | State machine reads `track.is_confirmed` (no duplicated hit-counting); only `entry_event_enabled` is on; FP rate validated on the **held-out** day before crossing/dwell are started (§1.5); **v6: point-in-polygon zone-membership unit test written against a known zone shape** |
| 7 | Persistence + FastAPI + `/metrics` | Oct 26 – Nov 1, 2026 | Async writer thread keeps SQLite off the frame path (§1.6); `/health` is tri-state; `/metrics` reports per-stage throughput utilization *and* separate end-to-end latency against §1.10's split budgets; **v4: Uvicorn bound to a specific interface with a bearer-token check on non-`/health` routes (§1.7)** |
| 8 | Alerting & notification channel (v5) | Nov 2 – Nov 8, 2026 | Notification consumer wired to a dedicated post-commit notification queue fed by the §1.6 writer thread — never a second consumer of the writer's own input queue — firing once each event commits (§1.14); delivery goes through webhook first, per the project's outbox decision, with Telegram wired as an additive second channel; delivery failures are logged and retried with backoff without blocking ingestion; synthetic end-to-end test event delivered through both channels |
| 9 | Pre-roll evidence buffer | Nov 9 – Nov 15, 2026 | PyAV packet ring buffer flushes keyframe-aligned pre/post-roll clips linked to event rows; achieved pre-roll duration recorded; second RTSP connection has its own reconnect and counters (§1.12) |
| 10 | Local enrichment (Hailo-Ollama) | Nov 16 – Nov 22, 2026 | Text-only summaries generated asynchronously; `/hailo/v1/list` reachable only on localhost; detector latency re-measured with the LLM resident on the NPU (§1.9) |
| 11 | Hardening, soak test & demo | Nov 23 – Nov 29, 2026 | DB/clips on USB SSD, RTC battery fitted, log rotation and disk-full eviction verified, startup checks in the systemd unit (§1.13); **v4: `Restart=on-failure` + `RestartSec` + `StartLimitBurst` + `WatchdogSec`/`sd_notify` heartbeat configured, and host-memory budget exposed on `/metrics` (§1.13)**; **v11: `OnFailure=camera-alert@%n.service` configured and verified by killing the pipeline process and receiving the webhook (§1.13); dead-man's switch verified by powering the Pi down and confirming the external alert arrives (§1.14)**; 24-hour soak with no crashes, hangs, or memory growth; full pipeline demoed on one camera |

**Phase-2 backlog (v4):** camera tamper/obstruction detection (§1.16) and the minimal operator UI (§1.15) are valuable but not on the critical path to a one-camera MVP demo — they belong alongside the existing §4 math refinements as post-MVP hardening, not as blockers for milestone 11 (renumbered in v5).

**v6 correction — the pure-function unit-test/CI suite (§1.17) is no longer part of this backlog.** Deferring these tests to a post-MVP cleanup pass meant every pure function they'd have caught bugs in — including the \(\Delta t^2\) approximation (§1.4/§4.1) and the \(\sigma\)-vs-\(\sigma^2\) units bug just corrected in §1.4 — shipped and ran uncaught for multiple milestones first. Each pure-function test is now written in the milestone that first touches that function, not bundled into a single later week: IoU and the \(Q(\Delta t)\) block are tested at milestone 5 (§1.4/§4.1, where the tracker is built), and point-in-polygon zone membership is tested at milestone 6 (§1.5, where zone logic is built). A \(Q(\Delta t)\) test against §4.1's closed-form values costs about an hour to write and would have caught both the \(\Delta t^2\) approximation and the units bug the moment either was introduced, rather than weeks later as an unexplained harness metric drop.

**Readiness checklist for the MVP demo:**

*Pre-flight*
- [ ] Person pixel height measured at each camera's zone boundary; lens/resolution/model decision written down (§0)

*Detection*
- [ ] `yolov8n_h10h.hef` runs end-to-end at expected latency, with INT8-vs-FP32 accuracy verified on recorded footage
- [ ] Detector emits at `confidence_threshold: 0.15`, matching the tracker's `low_confidence_threshold` exactly
- [ ] Letterbox padding with the inverse affine transform applied to returned boxes, not stretch-to-square
- [ ] NPU failure policy: fail-fast at startup, fail-loud mid-run, `/health` degraded, never silently continue

*Motion gate*
- [ ] MOG2 mask tested with `== 255`, **not** `> 0`, so `detectShadows: true` actually suppresses shadows
- [ ] `learningRate` passed explicitly and scaled by measured Δt, not derived from a frame-count history
- [ ] Motion ratio restricted to a zone-derived ROI mask, and thresholds calibrated via the harness

*Tracking*
- [ ] `min_confirmed_hits` defined once, in `tracker.yaml`, read everywhere as `track.is_confirmed`
- [ ] `F(Δt)` rebuilt from measured wall-clock elapsed time on every predict step
- [ ] `Q(Δt)` uses the continuous white-noise block (⅓Δt³ / ½Δt² / Δt), not the Δt² approximation
- [ ] Track expiry evaluated in seconds (`max_missed_seconds`), never frame counts
- [ ] A confirmed track survives brief detection gaps and short occlusions without losing identity

*Events*
- [ ] Zone events emitted from confirmed `track_id`s only; only the entry-zone rule enabled
- [ ] Same-track/same-zone cooldown prevents duplicate alerts during a single visit
- [ ] Distinct confirmed tracks entering the same zone produce distinct entry events (v11.2)
- [ ] Notifier throttles same camera/event type within 10 s and records every suppressed notification (v11.2)

*Ingestion & timing*
- [ ] Single-slot mailbox with exponential-backoff reconnect and a staleness signal, not a busy loop
- [ ] Both monotonic and UTC timestamps captured at read time and persisted; RTC coin cell fitted
- [ ] Throughput and end-to-end latency tracked as **separate** budgets, at ~65% frame-period utilization
- [ ] SQLite writes on a dedicated thread, off the per-frame path
- [ ] ONVIF credentials loaded from environment variables; no credentials present anywhere in the repository (v11)

*Evaluation*
- [ ] Harness matching rule (window, cardinality, unmatched handling) explicitly configured in `eval.yaml`
- [ ] Held-out day never used for tuning; reported numbers come from it, with sample size and CIs
- [ ] Confidence intervals computed with a Wilson score interval, not a Wald/normal approximation (v4)
- [ ] Ground-truth labels split `day` / `night_ir`, with per-frame detector recall reported separately from event-level metrics (v4)
- [ ] Replay harness emits per-stage traces (detection / confirmed-track / event) alongside final precision/recall (v4)
- [ ] `night_ir_insect` negative test passes — zero emitted events on the labeled insect-activity windows (v11)

*Evidence & operations*
- [ ] Pre-roll clips are keyframe-aligned via PyAV and start before the confirmed track's first detection
- [ ] Camera GOP interval lowered to ~1 s; achieved pre-roll duration recorded per event
- [ ] DB and clips on USB SSD; log rotation and disk-full eviction verified before the soak test
- [ ] `hailo-ollama` reachable only on `127.0.0.1`; VLM enrichment stays disabled until separately tested
- [ ] AGPL-3.0 implications reviewed before any decision to open-source the repository
- [ ] systemd unit configured with `Restart=on-failure`, `RestartSec`, `StartLimitIntervalSec`/`StartLimitBurst`, and `WatchdogSec`/`sd_notify` heartbeat (v4)
- [ ] Host-memory budget estimated with the enrichment model resident; exposed on `/metrics` (v4)
- [ ] `max_retention_days` purge defined independent of disk-full eviction; privacy stance on non-household visitors decided (v4)
- [ ] API bound to a specific interface with a bearer-token check on non-`/health` routes (v4)
- [ ] Notification channel fires on event commit without blocking ingestion (v4)
- [ ] `OnFailure=` alert unit verified by killing the process; dead-man's switch verified by powering the Pi down and receiving the external alert (v11)

Once this checklist is fully green for one camera, the next phase is generalizing to multiple cameras with bounded per-camera worker isolation, health checks, and backpressure handling.

**v6 correction — the H.265/HEVC decode change belongs here, not in milestone 2's exit criteria or the single-camera readiness checklist.** The guide's own cited measurement for the `-hwaccel drm` flag on Pi 5 is a saving of 13% → 9% of one CPU core (§1.10) — roughly 1% of total CPU on a single-camera pipeline, not a change worth planning around this early, and one that also touches the §1.12 PyAV remux path, since the pre-roll buffer's packet handling assumes whatever codec the mainstream is delivering. That calculus flips once cameras generalize: at four camera streams the same per-core saving is multiplied fourfold, at which point it can be the difference between fitting comfortably in the Pi 5's CPU budget and not. If the cameras support streaming H.265/HEVC substreams (most modern PoE cameras do), reconfigure the substream codec and decode through the Pi 5's actual hardware HEVC block instead of software H.264 as part of this multi-camera generalization work — a camera-side setting change, not a code change. Get the flag right: use **`-hwaccel drm`**, not `v4l2m2m` — the Pi 5 does not use the same V4L2 M2M decode path the Pi 4 did, and OpenCV's default FFmpeg/GStreamer backend does not pick a hardware decoder automatically just because one exists for the negotiated codec, so this still has to be set explicitly.

**v10 addition — labeling continues past milestone 3 on a rolling basis.** Milestone 3's exit criteria label a ground-truth CSV from one week of footage, which is unlikely by itself to reach `minimum_labeled_events_for_hard_gate: 100` (§1.11). Keep adding labeled events from the ongoing 24/7 recording (started at the end of milestone 2) at a regular cadence after milestone 3 ships, rather than treating that single week's labeling pass as the harness's final ground-truth set.

---

## 4. Phase 2 Mathematical Refinements

Two deeper optimizations to keep in your back pocket as the system scales. Neither blocks the MVP; both become worth the effort once you have the §1.11 harness in place to prove they helped.

### 4.1 The Physics of the Process Noise Covariance \(Q(\Delta t)\)

§1.4 notes that the process noise covariance must scale with elapsed time. The convenient shorthand \(Q(\Delta t) \propto \Delta t^2\) is an approximation; the physically grounded form comes from a continuous white noise acceleration model, in which unmodeled acceleration is treated as a zero-mean noise process integrated over the interval.

Under that model the two components grow at different rates. Velocity uncertainty accumulates the integral of acceleration noise, so it grows **linearly** in \(\Delta t\). Position uncertainty accumulates the integral of an already-growing velocity error, so it grows with the **cube** of \(\Delta t\). For each independent position/velocity pair the exact block is:

\[
Q = \begin{bmatrix} \tfrac{1}{3}\Delta t^{3} & \tfrac{1}{2}\Delta t^{2} \\[2pt] \tfrac{1}{2}\Delta t^{2} & \Delta t \end{bmatrix} \sigma_a^{2}
\]

where \(\sigma_a^2\) is the variance of the object's acceleration — physically, how abruptly you expect a person to change speed or direction. A jogger who might cut sideways warrants a larger \(\sigma_a^2\) than a delivery driver walking a straight path to a door.

The off-diagonal \(\tfrac{1}{2}\Delta t^2\) terms are not decoration: they encode the **correlation** between position and velocity error, which the naive \(\Delta t^2\) scaling discards entirely. A filter that ignores that correlation misjudges the shape of its own uncertainty ellipse, not just its size.

**Why this matters specifically in this pipeline.** The motion gate (§1.2) can produce \(\Delta t\) values of several seconds — a heartbeat inference after a quiet interval, or a track resuming after a long occlusion. At \(\Delta t = 3\) s the cubic term is roughly an order of magnitude larger than the quadratic approximation. Using the approximation leaves the filter **overconfident** about where the person is: it reports a tight covariance, the IoU association gate is correspondingly tight, the real detection falls outside it, the match fails, and the track dies and is reborn with a fresh `track_id`. That is precisely the identity break the tracker exists to prevent, and it will present as a mysterious duplicate-alert bug rather than as a filter problem.

**Implementation.** `filterpy.common.Q_continuous_white_noise(dim, dt, spectral_density)` produces this block directly, so the practical change is a config flag and one call site; `spectral_density` is exactly \(\sigma_a^2\), which is what `kalman_sigma_accel_sq` (§1.4, v6) now names accurately — pass it straight through, not its square root. Treat `kalman_sigma_accel_sq` as a tunable calibrated through the §1.11 harness rather than a constant guessed once — it is exactly the kind of parameter that looks arbitrary until you can measure track-identity persistence against labeled footage, and because the tracker runs in pixel space (§4.2), its units are px²/s⁴, not a metric acceleration variance.

The discrete white noise variant (`Q_discrete_white_noise`) assumes acceleration is constant *within* each interval and changes only between them, which is the better model when \(\Delta t\) is small and uniform. Given this pipeline's deliberately irregular and sometimes long intervals, the continuous form is the better fit.

### 4.2 Spatial Depth Invariance via Homography (Inverse Perspective Mapping)

`tracker.yaml` specifies `max_centroid_distance_px: 120` as an association sanity check. The problem is that a 2D camera image is a **perspective projection**, so a pixel is not a unit of distance: 120 px near the horizon might span 10 meters of real ground, while 120 px in the foreground might span half a meter. A single pixel threshold is therefore simultaneously too permissive for distant objects (letting the tracker swap identities between two people far away) and too restrictive for near ones (breaking the track of someone walking quickly past the camera). The same distortion affects the Kalman filter's velocity estimates: a person walking at constant speed appears to accelerate as they approach the lens.

**Inverse Perspective Mapping (IPM)** removes the distortion by projecting image coordinates onto the real ground plane. Because a plane-to-plane mapping under perspective projection is a homography, the transform is a single 3×3 matrix \(H\) applied in homogeneous coordinates:

\[
\begin{bmatrix} X' \\ Y' \\ w \end{bmatrix} = H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \qquad (X, Y) = \left(\tfrac{X'}{w},\ \tfrac{Y'}{w}\right)
\]

**Calibrating \(H\) is a one-time, low-tech procedure.** Pick four points on a flat surface in the camera's view whose real-world positions you can measure — corners of a driveway, paving slabs, or four markers you place with a tape measure. Note their pixel coordinates \((u_i, v_i)\) in a captured frame and their metric ground coordinates \((X_i, Y_i)\). Four correspondences fully determine \(H\):

```python
import cv2, numpy as np

image_pts  = np.float32([[412, 688], [905, 690], [1102, 431], [246, 428]])   # pixels
ground_pts = np.float32([[0.0, 0.0], [3.5, 0.0],  [3.5, 8.0],  [0.0, 8.0]])  # metres

H, _ = cv2.findHomography(image_pts, ground_pts)

def to_ground(u, v, H=H):
    p = H @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]      # (X, Y) in metres
```

Use the track's **bottom-center** point (the feet, where the person meets the ground plane) rather than the bounding-box centroid — the homography is only valid for points on the plane it was calibrated to, and a torso centroid floats above it.

**What this buys you.** Association gating, velocity, and dwell distance are all computed in metres instead of pixels, so the tracker becomes invariant to how close a person is to the lens. `max_centroid_distance_px: 120` becomes something like `max_association_distance_m: 1.5`, a threshold with a physical meaning you can reason about and that transfers unchanged between cameras and locations — which matters directly for the portability goal in the architecture doc, since a pixel threshold tuned at one house is meaningless at the next.

The same depth dependence applies to `kalman_sigma_accel_sq` (§1.4): the derivation there shows \(\sigma_a^2\) swinging roughly 9× between 5 m and 15 m in the same frame, precisely because it is a pixel-space quantity and pixels-per-metre changes with distance. Once the Kalman filter operates on ground-plane coordinates from this homography, \(\sigma_a^2\) becomes a physical acceleration variance in m\(^2\)/s\(^4\) — roughly 2–3 m\(^2\)/s\(^4\) for a walking person — a single constant across the whole zone rather than one that has to be re-derived per distance band. It carries the same portability payoff as `max_association_distance_m`: it transfers unchanged between cameras and locations, instead of being tied to one lens/distance combination.

It also unlocks event rules that pixels cannot express honestly: real walking speed (useful for separating a delivery walk-up from someone loitering), actual distance from the door, and zone areas in square metres.

**Caveats.** The homography is valid only for the plane it was calibrated on, so a sloped driveway, steps, or a raised patio each need their own \(H\) or a piecewise treatment. It also must be recalibrated whenever the camera is moved or re-aimed, which makes it a natural companion to a per-location config bundle rather than something baked into code.

**Deeper reading:**
- [OpenCV — `findHomography` and perspective transforms](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html) — the calibration call and the underlying geometry
- [`filterpy.common` discretization helpers](https://filterpy.readthedocs.io/en/latest/common/discretization.html) — `Q_continuous_white_noise` vs `Q_discrete_white_noise`

---

## Sources

**Core (v1):**

- [python-onvif-zeep](https://github.com/FalkTannhaeuser/python-onvif-zeep)
- [ONVIF Core Specification](https://www.onvif.org/specs/core/ONVIF-Core-Specification.pdf)
- [OpenCV VideoCapture documentation](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [PyImageSearch — Basic motion detection and tracking with Python and OpenCV](https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/)
- [OpenCV absdiff / core array operations](https://docs.opencv.org/4.x/d2/de8/group__core__array.html)
- [Detecting movement with pairwise frame subtraction](https://sam-low.com/opencv/frame-differencing.html)
- [Hailo Model Zoo (GitHub)](https://github.com/hailo-ai/hailo_model_zoo)
- [Hailo Model Zoo — yolov8n.yaml config](https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/networks/yolov8n.yaml)
- [Hailo RPi5 Examples — object detection pipeline](https://github.com/hailo-ai/hailo-rpi5-examples/blob/main/doc/basic-pipelines.md)
- [Hailo Application Code Examples (Python runtime)](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime/python)
- [Raspberry Pi AI accelerator documentation](https://www.raspberrypi.com/documentation/computers/ai.html)
- [ByteTrack (ECCV 2022, GitHub)](https://github.com/FoundationVision/ByteTrack)
- [SORT — Simple Online and Realtime Tracking (GitHub)](https://github.com/abewley/sort)
- [PyImageSearch — Intersection over Union (IoU) for object detection](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
- [kalmanfilter.net — Kalman Filter Explained Through Examples](https://www.kalmanfilter.net/)
- [Hikvision — Line Crossing Detection](https://enpinfo.hikvision.com/hkwsen/unzip/20230410194813_20373_doc/GUID-246BF07A-3F33-48FC-99D9-DE1AFC3E9144.html)
- [yas-sim/object-tracking-line-crossing-area-intrusion](https://github.com/yas-sim/object-tracking-line-crossing-area-intrusion)
- [Debounce design pattern (community.openhab.org)](https://community.openhab.org/t/design-pattern-debounce/101566)
- [Python sqlite3 module documentation](https://docs.python.org/3/library/sqlite3.html)
- [SQLite WAL mode documentation](https://sqlite.org/wal.html)
- [FastAPI official tutorial](https://fastapi.tiangolo.com/tutorial/)
- [FastAPI project homepage](https://fastapi.tiangolo.com/)
- [APScheduler documentation](https://apscheduler.readthedocs.io/)
- [Hailo Model Zoo GenAI (GitHub) — Hailo-Ollama](https://github.com/hailo-ai/hailo_model_zoo_genai)
- [Raspberry Pi AI HAT+ 2 — Hailo-10H local LLM walkthrough](https://raspberry.tips/en/raspberrypi-tutorials/raspberry-pi-ai-hat-2-hailo-10h-40-tops-local-llms)
- [Hailo GenAI Model Explorer — VLM models](https://hailo.ai/products/hailo-software/model-explorer/generative-ai/type/vlm/)

**Added in the v2 architectural review:**

- [OpenCV `BackgroundSubtractorMOG2` reference](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)
- [OpenCV background subtraction tutorial](https://docs.opencv.org/4.x/de/df4/tutorial_js_bg_subtraction.html)
- [Ultralytics `data.augment` API reference (LetterBox)](https://docs.ultralytics.com/reference/data/augment)
- [Ultralytics letterbox preprocessing discussion (GitHub issue)](https://github.com/ultralytics/ultralytics/issues/2580)
- [YOLOv5 letterbox PR reference (GitHub)](https://github.com/ultralytics/yolov5/pull/9213)
- [Stack Overflow — OpenCV VideoCapture lag due to the capture buffer](https://stackoverflow.com/questions/30032063/opencv-videocapture-lag-due-to-the-capture-buffer)
- [PyImageSearch — Faster video file FPS with cv2.VideoCapture and OpenCV](https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/)
- [PyImageSearch — Increasing webcam FPS with a threaded video stream](https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/)
- [Cross Validated — Kalman smoothing with irregular time steps](https://stats.stackexchange.com/questions/49300/how-does-one-apply-kalman-smoothing-with-irregular-time-steps)
- [filterpy issue — handling variable dt](https://github.com/rlabbe/filterpy/issues/196)
- [Ultralytics License page](https://www.ultralytics.com/license)
- [Ultralytics AGPL licensing discussion (GitHub issue)](https://github.com/ultralytics/ultralytics/issues/5691)
- [Ultralytics YOLOv8 model docs](https://docs.ultralytics.com/models/yolov8)
- [Ultralytics — Hailo export integration](https://docs.ultralytics.com/integrations/hailo)
- [VisioForge — Pre-event recording guide](https://www.visioforge.com/help/docs/dotnet/mediablocks/Guides/pre-event-recording/)
- [picamera circular streams (deepwiki)](https://deepwiki.com/waveform80/picamera/4.2-circular-streams)
- [Battleroid/seccam (GitHub)](https://github.com/Battleroid/seccam)
- [prometheus-fastapi-instrumentator (GitHub)](https://github.com/trallnag/prometheus-fastapi-instrumentator)
- [SQLite on a Raspberry Pi (Atomic Object)](https://spin.atomicobject.com/sqlite-raspberry-pi/)
- [SD card lifespan calculator (raspberry.tips)](https://raspberry.tips/en/sd-card-lifespan-calculator-how-long-will-your-storage-last)
- [Python logging cookbook](https://docs.python.org/3/howto/logging-cookbook.html)
- [Python `logging.handlers` reference](https://docs.python.org/3/library/logging.handlers.html)
- [Evaluating object detection models: methods and metrics (GeeksforGeeks)](https://www.geeksforgeeks.org/computer-vision/evaluating-object-detection-models-methods-and-metrics/)
- [Object detection metrics explained (Label Your Data)](https://labelyourdata.com/articles/object-detection-metrics)

**Added in the v3 review:**

- [PyAV documentation](https://pyav.org/docs/stable/) — packet-level demux/remux for the pre-roll ring buffer
- [`filterpy.common` discretization helpers](https://filterpy.readthedocs.io/en/latest/common/discretization.html) — `Q_continuous_white_noise` vs `Q_discrete_white_noise`
- [OpenCV — homography and perspective transform tutorial](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html) — `findHomography` for the IPM ground-plane projection
- [Raspberry Pi hardware documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html) — Pi 5 RTC and coin-cell battery connector

**Added in the v4 review:**

- [Hailo Model Zoo — DATA.rst](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/DATA.rst) — confirms COCO2017 as the calibration/evaluation set behind shipped model-zoo artifacts
- [Raspberry Pi Forums — Pi 5 has no hardware H.264 decoder](https://forums.raspberrypi.com/viewtopic.php?t=364180) — confirms the codec/decode gap driving the H.265 substream recommendation
- [Hardware-accelerated video decoding on Raspberry Pi with FFmpeg](https://salivity.github.io/ffmpeg/article/hardware-accelerated-video-decoding-on-raspberry-pi-with-ffmpeg) — decode capability comparison across Pi 4/5 generations
- [Comparative analysis of Wald, Wilson, and other proportion CIs](https://arxiv.org/html/2508.10223v1) — coverage behavior at small n and boundary proportions
- [systemd restart/watchdog patterns for Raspberry Pi services](https://forums.raspberrypi.com/viewtopic.php?t=376126) — `Restart=`, `WatchdogSec=`, and the crash-vs-hang distinction
- [Keeping software running on the Raspberry Pi (dzombak.com)](https://www.dzombak.com/blog/2023/12/keep-your-software-up-and-running-on-the-raspberry-pi/) — a practical systemd reliability walkthrough
- [ntfy.sh documentation](https://docs.ntfy.sh/) — self-hostable push notifications over a simple HTTP API
- [Pushover API](https://pushover.net/api) — a hosted push-notification service commonly used for home-automation alerts
- [FastAPI — serving static files](https://fastapi.tiangolo.com/tutorial/static-files/) — the minimal mechanism for the operator UI
- [OpenCV — image statistics and histogram basics](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html) — the building blocks behind the tamper heuristic
- [pytest documentation](https://docs.pytest.org/) — the test runner for the pure-function unit-test suite
- [GitHub Actions — Python CI quickstart](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python) — wiring the pytest suite into CI

**Added in the v5 review:**

- [Brown, Cai & DasGupta (2001) — Interval Estimation for a Binomial Proportion](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full) — replaces the v4 CASRAI citation with the canonical Wald-vs-Wilson coverage paper (Statistical Science 16(2):101–133)
- [YOLOv8 nighttime small-object surveillance benchmark](https://www.iieta.org/journals/ijsse/paper/10.18280/ijsse.140611) — replaces the v4 low-light citations with concrete published precision/recall/mAP numbers
- [Systematic review of low-light object detection — YOLOv8–v11 on ExDark](https://link.springer.com/article/10.1007/s42452-025-08051-5) — confirms the low-light accuracy gap across the YOLO model family, not just one version
- [Raspberry Pi Forums — Pi 5 software decode outperforms Pi 4 hardware decode](https://forums.raspberrypi.com/viewtopic.php?t=391283) — evidence the §1.10 decode cost is not automatically a hard ceiling
- [Raspberry Pi Forums — NEON-optimised software H.264 decode on Pi 5](https://forums.raspberrypi.com/viewtopic.php?t=357870) — same point, with implementation detail on why it's fast
- [Frigate GitHub discussion — Pi 5 `hwaccel drm` vs. `v4l2m2m`](https://github.com/blakeblackshear/frigate/discussions/18431) — the correct decode flag for Pi 5, with measured real-world CPU deltas
