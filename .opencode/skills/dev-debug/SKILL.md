---
name: dev-debug
description: Root-cause debugging protocol for VVA code (reproduce with a failing test, find the cause with evidence, fix at the cause, add a regression test). Used by vva-pro; vva-builder follows it before escalating.
compatibility: opencode
metadata:
  role: vva-pro
  phase: debug
---
# Debug protocol

1. Reproduce: write or identify a failing test with synthetic data. No reproduction, no fix.
2. Hypothesis -> evidence: state one hypothesis, then the check that would refute it
   (a print-free assertion, a smaller fixture, a bisect with `git log`/`git diff`).
   Repeat until one hypothesis survives.
3. Fix at the cause, never at the symptom (no extra sleeps, no widened tolerances, no
   try/except that hides the error).
4. Regression test: the reproduction test stays in the suite and passes after the fix.
5. Never edit an existing test to make it pass.
6. If the failure is a disagreement about intended behaviour, not a bug, STOP and hand it
   to the user with the options and their trade-offs.

Pinned behaviour (do not "fix"): `test_rule_replay_tradeoff` is a regression test of
frame-level RULE_REPLAY. A false-positive alert opens the 45 s per-camera cooldown and
suppresses the real event (loose recall 0.0, default 1.0). Production cooldowns live
in M6 (30 s track/zone, 5 s re-arm) and M8 (10 s notification throttle).

Report: cause, evidence, patch, regression test, commands run with real output.
