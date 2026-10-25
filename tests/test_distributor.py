"""Distributor multi-turn math: the servo wraps ~3.26x per ring turn, so every
move is planned in ring counts and executed as a signed servo delta."""
import random

import pytest

from machine import (RING, MultiTurnTracker, bin_ring_pos, encode_signed, plan_ring_delta,
                     ring_pos_to_bin, wrap_delta)

BINS, RATIO, TOL = 7, 3.26, 20


def test_wrap_delta_across_boundary():
    assert wrap_delta(10, 4090) == 16
    assert wrap_delta(4090, 10) == -16
    assert wrap_delta(2000, 1000) == 1000
    assert wrap_delta(0, 2048) == -2048


def test_tracker_follows_many_wraps_both_directions():
    t = MultiTurnTracker(4000)
    pos, truth = 4000, 0
    rng = random.Random(1)
    for _ in range(5000):
        step = rng.randint(-600, 600)
        truth += step
        pos = (pos + step) % RING
        assert t.update(pos) == truth


def test_servo_counts_per_bin_matches_the_brief():
    # 4096/7 ring counts ≈ 585, x3.26 ≈ 1908 servo counts per bin
    ring_per_bin = bin_ring_pos(1, BINS)
    assert ring_per_bin == pytest.approx(585.14, abs=0.01)
    assert ring_per_bin * RATIO == pytest.approx(1907.6, abs=0.1)


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_round_trip_every_bin(direction):
    """Walk 0->1->...->6->0 (or the reverse) and land on each bin exactly."""
    order = list(range(BINS)) + [0]
    if direction == "backward":
        order = order[::-1]
    ring_abs = 0.0
    servo_abs = 0
    for target in order[1:]:
        delta = plan_ring_delta(ring_abs, target, BINS)
        servo_delta = round(delta * RATIO)
        servo_abs += servo_delta
        ring_abs += servo_delta / RATIO           # what the servo actually did
        assert ring_pos_to_bin(ring_abs, BINS, TOL) == target
        # the servo-side error of rounding to whole counts is far below tolerance
        err = (ring_abs - bin_ring_pos(target, BINS) + RING / 2) % RING - RING / 2
        assert abs(err) < 1.0


@pytest.mark.parametrize("src", range(BINS))
@pytest.mark.parametrize("dst", range(BINS))
def test_every_pair_takes_the_short_way_when_unwound(src, dst):
    ring_abs = bin_ring_pos(src, BINS)
    delta = plan_ring_delta(ring_abs, dst, BINS)
    assert abs(delta) <= RING / 2 + 1e-6
    assert ring_pos_to_bin(ring_abs + delta, BINS, TOL) == dst


def test_excursion_bound_holds_over_a_long_run():
    """Random bins for a long time: software wind-up stays inside the bound and
    we still land on the requested bin every time (going the long way if needed)."""
    rng = random.Random(7)
    ring_abs = 0.0
    limit = 1.5 * RING
    long_way = 0
    for _ in range(3000):
        target = rng.randrange(BINS)
        delta = plan_ring_delta(ring_abs, target, BINS, limit)
        if abs(delta) > RING / 2:
            long_way += 1
        ring_abs += round(delta * RATIO) / RATIO
        assert abs(ring_abs) <= limit + 1.0
        assert ring_pos_to_bin(ring_abs, BINS, TOL) == target
    assert long_way > 0  # the bound actually engaged at least once


def test_encode_signed_sign_magnitude():
    assert encode_signed(100) == 100
    assert encode_signed(-100) == 0x8000 | 100
    assert encode_signed(0) == 0
