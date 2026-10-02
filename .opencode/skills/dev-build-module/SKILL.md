---
name: dev-build-module
description: Contract-first procedure to implement one VVA MVP module (ingest, motion gate, detector, tracker, events, store, API, notify, pre-roll, enrichment, eval, hardening) from its card in docs/module-contracts.md, with tests and the full verification loop.
compatibility: opencode
metadata:
  role: vva-builder
  phase: build
---
# Build one MVP module

## 1. Scope
One module per request, named by the user (example: "M5 tracker"). If the user names
several, do them one at a time and stop after each for review.

## 2. Read before writing (in this order)
1. The module card in `docs/module-contracts.md` (inputs, outputs, invariants, config
   owner, required tests, verified-by, escalation tier).
2. The guide section the card cites in `docs/mvp-guide.md`, including every "vN
   correction" paragraph: corrections override the text above them. Guide paths
   `src/<x>/...` are written as `src/vva_app/<x>/...` in this repository.
3. Existing code the module consumes (its upstream card's outputs).
4. The config file(s) the module owns or reads under `configs/`.
If the card and the guide disagree, stop and ask ONE question. Do not pick one.

## 3. Plan (write it in the chat, then continue)
- Files to create or change, public functions/classes with signatures.
- Which invariants become tests, and which closed-form values they check.
- Acceptance commands.
- Escalation check: does any trigger T1..T4 apply? If yes, load `dev-escalate-pro`.

## 4. Implement, contract first
1. Types first: frozen dataclasses or Pydantic models for the module's inputs/outputs.
   Reuse `vva_contracts.contracts.logs` row models for anything written to JSONL.
2. Tests next, from the card's "required tests" (they should fail now).
3. Implementation: pure functions separated from I/O and threads, so they are testable
   without cameras or the NPU. Hardware access sits behind a small interface with a
   fake for tests.
4. Config: read thresholds from the owning YAML file through the startup validator
   (`src/vva_app/config/validate.py`); never hard-code a threshold or re-define it elsewhere.
5. Time: use the monotonic clock for durations and the UTC clock for persistence;
   express expiry and cooldowns in seconds, never in frame counts.

## 5. Verification loop (run all, report the real output)
```
ruff check src tests
ruff format --check src tests
mypy --strict src
pytest -q
```
If one fails: fix and re-run. After two failed attempts on the same failure, escalate
(trigger T2). Never edit a test to make it pass.

## 6. Done means
- All four commands pass, or the failure is reported honestly with its output.
- Every invariant on the card has a test or is listed as "not testable here" with the
  measurement that will check it on the Pi (verified-by column).
- Final message: files changed, command outputs, open questions, proposed commit
  message explaining WHY. The user makes the commit.

## 7. Never
- Put an LLM call in the alert path, or ask a model for confidence scores.
- Write credentials, RTSP URLs, IPs or real logs into code, tests, fixtures or chat.
- Add a dependency without written approval.
