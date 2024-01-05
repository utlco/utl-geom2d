"""Test 2D point (P) implementation."""

import math
import sys

import numpy as np
from geom2d import point
from geom2d.const import EPSILON, MAX_XY, float_eq
from geom2d.point import P

# Epsilon minus a tiny amount
EPSILON_MINUS = EPSILON - sys.float_info.epsilon  # (EPSILON / 1000)


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
        p = P.from_polar(EPSILON_MINUS, angle)
        p2 = P.random()
        p3 = p2 + EPSILON * max(p2.x, p2.y)

        # test __eq__
        assert p.is_zero()
        assert p2 + p == p2
        assert p2 != p3

        # test to/from polar
        mag = np.random.default_rng().uniform(EPSILON, MAX_XY)
        p = P.from_polar(mag, angle)
        m, a = p.to_polar()
        assert float_eq(mag, p.length())
        assert float_eq(angle, p.angle())
        assert float_eq(m, mag)
        assert float_eq(a, angle)

        # test unit length
        p2u = p2.unit()
        ulen = p2u.length()
        assert ulen <= (1.0 + EPSILON)
        assert float_eq(ulen * p2.x / p2u.x, p2.length())


def test_hash() -> None:
    """Test point hash for collisions."""
    # Big coordinate space
    _test_hash(10000, -MAX_XY, MAX_XY)
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
        h = p.__hash__()
        hashes.add(h)

    assert len(hashes) == len(points)
