"""Test float equality test (geom2d.const.float_eq)."""

import sys

import numpy as np
from geom2d import const
from geom2d.const import (
    EPSILON,
    EPSILON2,
    float_eq,
    is_zero,
)

# Tolerence/epsilon minus a tiny bit (system float epsilon)
EPSILON_MINUS = EPSILON - sys.float_info.epsilon


def test_float_eq() -> None:
    """Test geom2d.const.float_eq."""
    for n in np.linspace(-EPSILON_MINUS, EPSILON_MINUS, num=1000):
        assert float_eq(n, 0)
        assert is_zero(n)

    for n in np.linspace(-10, 10, 1000):
        _float_eq_t(n)

    for n in np.linspace(-const.MAX_XY, const.MAX_XY, 1000):
        _float_eq_t(n)

    for n in np.random.default_rng().uniform(
        -const.MAX_XY, const.MAX_XY, 10000
    ):
        _float_eq_t(n)


def _float_eq_t(n: float) -> None:
    assert float_eq(n, n)
    if abs(n) < 1:
        assert float_eq(n, n + (EPSILON - EPSILON2))  # (n * EPSILON))
        assert not float_eq(n, n + (EPSILON + EPSILON / 10))
    else:
        scaled_epsilon = abs(n) * EPSILON
        assert float_eq(n, n + scaled_epsilon / 10)
        assert not float_eq(n, n + scaled_epsilon + scaled_epsilon / 10)
