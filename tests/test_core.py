"""Unit tests for the pure algorithms in vva_contracts.core."""

from __future__ import annotations

import random

import numpy as np
import pytest

from vva_contracts.core.ap import ImageDetections, evaluate_class
from vva_contracts.core.geometry import iou_matrix, point_in_polygon, xywhn_to_xyxyn
from vva_contracts.core.phash import cross_near_duplicates, hamming
from vva_contracts.core.state_machine import Detection, EventRule, EventStateMachine, Frame


def test_iou_known_values():
    a = np.array([[0.0, 0.0, 0.5, 0.5]])
    b = np.array([[0.0, 0.0, 0.5, 0.5], [0.25, 0.0, 0.75, 0.5], [0.6, 0.6, 0.9, 0.9]])
    # identical -> 1; half overlap -> 0.125 / 0.375 = 1/3; disjoint -> 0
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])


def test_iou_is_invariant_to_anisotropic_scaling():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 0.5, (20, 2))
    boxes = np.hstack([xy, xy + rng.uniform(0.05, 0.4, (20, 2))])
    scale = np.array([1920.0, 1080.0, 1920.0, 1080.0])
    np.testing.assert_allclose(iou_matrix(boxes, boxes[::-1]), iou_matrix(boxes * scale, boxes[::-1] * scale))


def test_xywhn_conversion():
    np.testing.assert_allclose(xywhn_to_xyxyn(np.array([[0.5, 0.5, 0.2, 0.4]])), [[0.4, 0.3, 0.6, 0.7]])


def test_point_in_polygon_concave():
    # A "U" shape: the notch at the top centre is outside.
    u_shape = [(0, 0), (1, 0), (1, 1), (0.7, 1), (0.7, 0.4), (0.3, 0.4), (0.3, 1), (0, 1)]
    assert point_in_polygon(0.1, 0.5, u_shape)
    assert not point_in_polygon(0.5, 0.7, u_shape)
    assert point_in_polygon(0.5, 0.2, u_shape)


def test_ap_perfect_duplicate_and_missing():
    gt = np.array([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]])
    perfect = ImageDetections(gt, gt.copy(), np.array([0.9, 0.8]))
    r = evaluate_class([perfect], operating_conf=0.5, max_dets=100)
    assert r.ap_per_iou == [1.0] * 10 and r.tp_at_operating_point == 2
    # A duplicate of a matched box is an FP, never a second TP.
    dup = ImageDetections(gt, np.vstack([gt, gt[:1]]), np.array([0.9, 0.8, 0.7]))
    r = evaluate_class([dup], operating_conf=0.5, max_dets=100)
    assert (r.tp_at_operating_point, r.fp_at_operating_point) == (2, 1)
    # No predictions at all -> AP 0, not undefined.
    none = ImageDetections(gt, np.zeros((0, 4)), np.zeros(0))
    assert evaluate_class([none], 0.5, 100).ap_per_iou == [0.0] * 10


def test_banded_search_equals_brute_force():
    rng = random.Random(0)
    left = {f"l{i}": rng.getrandbits(64) for i in range(300)}
    right = {f"r{i}": rng.getrandbits(64) for i in range(300)}
    # Plant near-duplicates at distances 0..7 by flipping random bits.
    for d in range(8):
        base = left[f"l{d}"]
        for bit in rng.sample(range(64), d):
            base ^= 1 << bit
        right[f"r{d}"] = base
    for max_d in (0, 3, 7):
        brute = sorted(
            (a, b, hamming(va, vb)) for a, va in left.items() for b, vb in right.items() if hamming(va, vb) <= max_d
        )
        assert cross_near_duplicates(left, right, max_d) == brute


def test_banded_search_refuses_unguaranteed_distance():
    with pytest.raises(ValueError):
        cross_near_duplicates({}, {}, 8)


def _frames(confidences, fps=10.0, start=0.0, camera="cam"):
    return [
        Frame(camera, i, start + i / fps, (Detection("person", c, (0.4, 0.4, 0.6, 0.9)),) if c else ())
        for i, c in enumerate(confidences)
    ]


RULE = EventRule(frozenset({"person"}), 0.6, 3, cooldown_s=5.0, max_gap_s=1.0)


def test_streak_needs_k_consecutive_positive_frames():
    machine = EventStateMachine(RULE)
    alerts = [machine.step(f) for f in _frames([0.9, 0.9, 0.0, 0.9, 0.9, 0.9])]
    assert [a is not None for a in alerts] == [False, False, False, False, False, True]


def test_low_confidence_does_not_count():
    machine = EventStateMachine(RULE)
    assert all(machine.step(f) is None for f in _frames([0.59] * 20))


def test_cooldown_deduplicates_continuous_presence():
    machine = EventStateMachine(RULE)
    alerts = [a for f in _frames([0.9] * 120) if (a := machine.step(f))]  # 12 s of presence
    # fires at frame 2 (t=0.2), then t>=5.2, then t>=10.2
    assert [round(a.ts, 1) for a in alerts] == [0.2, 5.2, 10.2]


def test_time_gap_restarts_the_streak():
    machine = EventStateMachine(RULE)
    frames = _frames([0.9, 0.9]) + _frames([0.9], start=10.0)  # 9.8 s hole
    assert all(machine.step(f) is None for f in frames)
    assert machine.streak == 1


def test_zone_uses_bottom_centre():
    zone = ((0.0, 0.8), (1.0, 0.8), (1.0, 1.0), (0.0, 1.0))  # bottom strip
    rule = EventRule(frozenset({"person"}), 0.6, 1, 0.0, 1.0, zone=zone)
    machine = EventStateMachine(rule)
    feet_in = Frame("c", 0, 0.0, (Detection("person", 0.9, (0.4, 0.2, 0.5, 0.9)),))
    feet_out = Frame("c", 1, 0.1, (Detection("person", 0.9, (0.4, 0.2, 0.5, 0.6)),))
    assert machine.step(feet_in) is not None
    assert machine.step(feet_out) is None


def test_backwards_time_fails_fast():
    machine = EventStateMachine(RULE)
    machine.step(Frame("c", 0, 10.0, ()))
    with pytest.raises(ValueError):
        machine.step(Frame("c", 1, 9.0, ()))
