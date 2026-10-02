---
name: dev-optimize
description: Measurement-first optimization protocol (baseline benchmark, cProfile, one change at a time, equivalence proof, before/after numbers) for VVA Python code. Used by vva-pro.
compatibility: opencode
metadata:
  role: vva-pro
  phase: optimize
---
# Optimization protocol

1. Baseline: deterministic synthetic input, at least 5 runs; report median and spread
   (min-max or IQR) and the hardware it ran on.
2. Profile with `python -m cProfile -s cumtime`; optimise only the proven hotspot.
3. One change at a time; re-measure after each.
4. Equivalence: outputs identical, or inside a stated tolerance with the reason.
   Existing oracles: `tests/test_ap_crosscheck.py` (AP vs pycocotools) and the banded
   near-duplicate search vs brute force in `tests/test_core.py`.
5. Report before/after numbers with the same method.
6. Readability is a requirement: do not trade reviewability for speed; McCabe <= 10.
7. Raspberry Pi / Hailo performance cannot be verified here: label it UNVERIFIED and give
   the user the PIPELINE_BENCH request to run on Pi-side logs.
