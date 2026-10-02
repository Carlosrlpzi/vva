---
name: dev-escalate-pro
description: When and how vva-builder (Flash) delegates to the vva-pro subagent (Pro) with a bounded brief, and how to verify the patch it returns before accepting it.
compatibility: opencode
metadata:
  role: vva-builder
  phase: routing
---
# Escalate to vva-pro

## 1. Triggers (any one)
- T1 design of a complex analytics layer -> Pro loads `dev-design-layer`.
- T2 same failure after two of your own attempts -> Pro loads `dev-debug`.
- T3 numerical or algorithmic code (Kalman F(dt)/Q(dt), association, IoU, AP,
  homography, percentiles, phash, `src/vva_contracts/core/`) -> `dev-debug` or design.
- T4 profiling or performance reasoning -> Pro loads `dev-optimize`.
Never escalate docs, formatting, renames, config files or test scaffolding.

## 2. Brief template (send exactly these headings)
```
GOAL: <one sentence>
PROTOCOL: dev-design-layer | dev-debug | dev-optimize
FILES IN SCOPE: <exact paths>
DO NOT TOUCH: <paths and behaviours, e.g. cooldown semantics>
CONTEXT: <guide section, module card id, failing test name and its real output>
CONSTRAINTS: no new dependencies; mypy --strict; McCabe <= 10; no LLM in alert path
ACCEPTANCE: <exact commands that must pass>
```

## 3. Privacy of the brief
The brief leaves the machine. Never include frames, clips, URLs, credentials, `.env`
content or real detection logs. Paste synthetic fixtures or aggregated numbers only.

## 4. Verify the answer (Pro proposes, you verify)
1. Read the diff; reject changes outside FILES IN SCOPE.
2. Apply it, then run every ACCEPTANCE command yourself and report the real output.
3. Debug patches need a test that failed before and passes after; optimizations need an
   equivalence proof and before/after numbers.
4. Never relay "tests pass" from Pro's text. If verification fails, send Pro the real
   output once; if it fails again, stop and report to the user.
