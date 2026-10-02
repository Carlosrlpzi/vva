Video Surveillance Assistant

**Goal:** Build a private, edge-first smart camera system on a Raspberry Pi 5 — RTSP/ONVIF cameras feeding a local Hailo-accelerated detection pipeline, with the internet used only for outbound alerts (Telegram, etc.), not for core video processing.

## **Decisions already made**

**Hardware (finalized):**

* Raspberry Pi 5 \+ AI HAT+ 2 (Hailo-10H, 40 TOPS) — chosen over the older AI HAT+ because you want headroom for multi-camera, multi-model, and eventually local LLM/VLM enrichment, not just basic detection.  
* Official Pi 5 Active Cooler \+ the HAT's own heatsink for thermal management.  
* Cameras: 2× Uniarch IPC-B124-APF40K (4.0 mm) for narrow "choke points" (front door, driveway/gate) \+ 1× Uniarch IPC-B124-APF28K (2.8 mm) for wider coverage (backyard/patio) — all RTSP/ONVIF Profile S/T/G, PoE, dual-stream, no mandatory cloud subscription.  
* Network: Ruijie RG-ES209GC-P 9-port PoE switch, with the Pi on the switch's non-PoE uplink port; outdoor-rated solid-copper Cat6 for camera runs.

**Architecture/software philosophy (finalized):**

* No LLM in the real-time video path. The MVP is a deterministic pipeline: RTSP ingest → Hailo object detection → lightweight tracker \+ zone/time rules → event state machine (dedupe/cooldown) → SQLite event storage \+ evidence clips → FastAPI for status/history → alert outbox (webhook first, Telegram later).  
* LLM/VLM is reserved strictly for **post-event enrichment** (human-readable summaries, natural-language search, delivery/vehicle classification) — never for deciding whether an alert fires.  
* Use camera substreams (low-res, 5–10 fps) for inference, and the main stream only for recorded evidence — this is what makes multi-camera scaling realistic.  
* Project scaffold already defined: `src/ingest`, `src/inference`, `src/events`, `src/api`, `src/enrichment`, `configs/`, `data/`, `tests/`.

**Setup guide:** A full hardware→OS→network→Hailo→Python setup guide is saved in the project files (`docs/01-setup-guide.md`), covering physical assembly, OS flashing, PCIe Gen 3 enablement, the critical `hailo-h10-all` (not `hailo-all`) package distinction, and a troubleshooting table.

## **Current state**

You're through **Phase 7** of the setup guide: Python 3 is installed on the Pi and a virtual environment is created. Per the guide, this should have been created with `--system-site-packages` so the Hailo Python bindings (`hailo_platform`, installed system-wide by `hailo-h10-all`) are visible inside it — worth double-checking now if you haven't confirmed that flag was used, since it's the \#1 gotcha at this stage. You're working with the Pi over temporary home Wi-Fi (the PoE switch/fixed camera networking is deferred until that hardware arrives).

## **Decisions still to make**

* **Frigate vs. custom pipeline:** whether to spend a short experiment validating Frigate (open-source NVR) against your Hailo-10H, since existing Frigate/Hailo guides mostly target the older Hailo-8/8L — or go straight to the custom Python pipeline, which is the better long-term fit for your project.  
* **Event rule specifics:** confidence thresholds, minimum consecutive frames, zone polygons per camera, and cooldown windows (a starting policy was already sketched: `min_confidence: 0.60`, 3 consecutive frames, 45s cooldown).  
* **Alert channel order:** webhook-first with Telegram layered in after, or Telegram from day one.  
* **Storage plan:** where clips/retention live long-term (local disk vs. USB SSD/NAS) once volume grows past the microSD.  
* Final confirmation of the 2.8 mm vs 4.0 mm choice for camera \#3 depending on actual backyard/side-area width.

## **Suggested next steps**

1. Verify the venv: `python3 -c "import cv2, onvif, hailo_platform; print('all imports OK')"` — confirms the environment is truly ready.  
2. Install the remaining core packages if not yet done (`opencv-python-headless`, `numpy`, `onvif-zeep`, `fastapi`, `uvicorn`, `python-dotenv`, `pydantic`, `APScheduler`).  
3. Initialize the Git scaffold (`src/ingest`, `src/inference`, `src/events`, `src/api`, `src/enrichment`, `configs/`, `data/`, `tests/`) if not already committed.  
4. Build the single-camera MVP slice first: pull one RTSP substream → run a precompiled Hailo person detector → log structured JSON detections (bounding box, confidence, timestamp) — proven stable for 24 hours unattended before adding zones/tracking/alerts.  
5. Only after that: layer in zone rules, the event state machine, SQLite persistence, and the alert outbox — then scale to the remaining cameras one at a time, each as an isolated worker process so one dead RTSP feed can't take down the others.

