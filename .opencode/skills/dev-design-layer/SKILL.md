---
name: dev-design-layer
description: Design-document protocol for complex analytics layers above raw detection (tracking/association, zone-time analytics, threshold optimisation from replays, correlated false positives, enrichment boundary). Produces design only unless the layer is approved.
compatibility: opencode
metadata:
  role: vva-pro
  phase: design
---
# Analytics-layer design

Gate: implement code ONLY for layers listed in `docs/approved-layers.md`. If the file is
missing or empty, produce the design and no code.

Derive candidate layers from `docs/mvp-guide.md`; drop what the guide does not support.
YOLO itself is out of scope.

Per layer, write in `docs/analytics-layers.md`:
1. Problem it solves and which guide section motivates it.
2. Input/output contract: a strict Pydantic report (extra="forbid", strict, frozen) whose
   validator recomputes its invariants, same pattern as `src/vva_contracts/contracts/`.
3. The math, with the assumption each formula needs.
4. Measurement: a synthetic case with a known analytical answer, and the test that checks it.
5. Failure modes and how the report exposes them (flags).
6. What it must NOT do: no LLM in the alert path; no model-produced confidence scores.
