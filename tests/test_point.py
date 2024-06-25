"""Test 2D point (P) implementation."""

import math
import sys

import numpy as np
from geom2d import const, point
from geom2d.const import float_eq
from geom2d.point import P

P1 = P(60, 40)
CCW_PT = [
    P(120, 90),
    P(20, 90),
    P(0, 100),
    P(-50, 140),
    P(-50, 40),
    P(-40, 10),
    P(-30, -10),
]
CW_PT = [
    P(-30, -30),
    P(30, -30),
    P(40, -10),
]


def test_misc() -> None:
    """Test misc point methods.

    P.max_point()
    P.min_point()
    P.from_polar()
    P.to_polar()
    P.length()
    P.length2()
    P.angle()
    P.is_zero()
    P.unit()

    """
    p = P.max_point()
    assert p.x == point.max_xy()
    assert p.y == point.max_xy()
    p = P.min_point()
    assert p.x == -point.max_xy()
    assert p.y == -point.max_xy()

    for angle in np.linspace(-math.pi, math.pi, 100):
        p = P.from_polar(const.EPSILON - sys.float_info.epsilon, angle)
        p2 = P.random()
        p3 = p2 + const.EPSILON * max(p2.x, p2.y)

        # test __eq__
        assert p.is_zero()
        assert p2 + p == p2
        assert p2 != p3

        # test to/from polar
        mag = np.random.default_rng().uniform(const.EPSILON, const.MAX_XY)
        p = P.from_polar(mag, angle)
        m, a = p.to_polar()
        assert float_eq(mag, p.length())
        assert float_eq(angle, p.angle())
        assert float_eq(m, mag)
        assert float_eq(a, angle)

        # test unit length
        p2u = p2.unit()
        ulen = p2u.length()
        assert ulen <= (1.0 + const.EPSILON)
        assert float_eq(ulen * p2.x / p2u.x, p2.length())


def test_is_ccw() -> None:
    """Test P.is_ccw()."""
    # All points in CCW_PT are CCW (left of) P1
    for p2 in CCW_PT:
        assert P1.is_ccw(p2)
    # All points in CW_PT are CW (right of) P1
    for p2 in CW_PT:
        assert not P1.is_ccw(p2)


def test_hash() -> None:
    """Test point hash for collisions."""
    # Big coordinate space
    _test_hash(10000, -const.MAX_XY, const.MAX_XY)
    # Small coordinate space
    _test_hash(10000, -1.0, 1.0)


def _test_hash(max_hashes: int, min_xy: float, max_xy: float) -> None:
    rng = np.random.default_rng()
    points = {
        P(rng.uniform(min_xy, max_xy), rng.uniform(min_xy, max_xy))
        for n in range(max_hashes)
    }
    # Test equality for reasonable rando collisions
    assert (max_hashes - len(points)) < (max_hashes * 0.01)

    hashes = set()
    for p in points:
        h = hash(p)
        hashes.add(h)

    assert len(hashes) == len(points)
