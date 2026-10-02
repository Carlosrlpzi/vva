# VVA project rules (apply to every agent)

Project: Smart Camera System, a private, edge-first smart video surveillance system.
Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H), PoE RTSP/ONVIF cameras. One repository:
- `src/vva_app/`: the application that runs on the Pi (built milestone by milestone).
- `src/vva_contracts/`: deterministic measurement tasks (no LLM inside).
Source of design truth: `docs/mvp-guide.md` (MVP Development Guide v11.2; Spanish copy
`docs/mvp-guide.es.md`). Every guide path `src/<x>/...` maps to `src/vva_app/<x>/...`.
Module contracts: `docs/module-contracts.md`.

## Non-negotiable invariants
1. No LLM or VLM decides whether an alert fires. The alert path is deterministic:
   detector -> tracker -> zone rule -> event state machine -> persistence -> notify.
   LLMs are for development and for post-event, asynchronous enrichment only.
2. Evidence is measured, never claimed. Never estimate, round, average or recompute a
   number produced by a tool. Never write "tests pass" without having run them.
3. Privacy. Frames, clips, RTSP URLs, credentials, `.env` content and real detection logs
   never enter the chat or a prompt. Timestamps of detections reveal when the house is
   occupied. Use synthetic fixtures (`tests/builders.py`) or aggregated numbers.
   Refer to a camera URL only by the NAME of its env var: `VVA_CAM_<ID>_<SUB|MAIN>_URL`.
   ONVIF credentials: `VVA_CAM_<ID>_ONVIF_HOST`, `VVA_CAM_<ID>_ONVIF_USER`,
   `VVA_CAM_<ID>_ONVIF_PASSWORD`. Never a literal value in code, tests or chat.
4. One source of truth per concept. A threshold lives in exactly one config file and is
   read everywhere else (example: `min_confirmed_hits` lives only in `tracker.yaml`).
5. Fail loud. Startup checks fail fast; mid-run hardware faults degrade `/health`; never
   continue silently.
6. Do not invent fields, defaults, thresholds, exit codes, file paths or model ids.
   Copy them from code or docs. If a fact is missing, say so and ask ONE question.

## Code standards (Python >= 3.11)
- Docstring on every public module/class/function stating its contract; comments say WHY.
- `ruff check`, `ruff format --check`, `mypy --strict src` and `pytest -q` must pass.
- McCabe complexity <= 10 per function. English identifiers, comments and docstrings.
- Pure functions (IoU, Q(dt), point-in-polygon, homography, percentiles) are tested
  against closed-form values in the same milestone that introduces them.
- No new dependency without the user's written approval.
- Do not edit a test to make it pass. Do not weaken or skip existing tests.
- Decided cooldown split (guide v11.2): M6 keeps 30 s per (track_id, zone_id) and the
  5 s re-arm, and emits one entry event per distinct confirmed track; the 10 s per
  (camera_id, event_type) limit is a NOTIFICATION throttle in M8 that records every
  suppression. Never suppress an event to limit notifications.
- RULE_REPLAY evaluates frame-level detection policies only. Its per-camera cooldown
  limitation is pinned by `test_rule_replay_tradeoff` (loose recall 0.0, default 1.0):
  do not change cooldown semantics in `src/vva_contracts/` to alter that test.
  Production events are validated by the M3 replay harness running the real M6 code.

## Repository boundaries
- `opencode.json`, `AGENTS.md`, `prompts/`, `.opencode/` and
  `tests/test_skills_consistency.py` are owned by the user. Agents never edit them.
- `src/vva_contracts/` changes only through the debug or optimize protocol with a
  regression test or an equivalence proof.
- Generated code runs with the user's privileges and is NOT sandboxed.
