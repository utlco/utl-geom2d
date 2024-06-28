"""Test float equality test (geom2d.const.float_eq)."""

import sys

import numpy as np
import pytest
from geom2d import const
from geom2d.const import (
    float_eq,
    is_zero,
)

EPSILONS = [1 / (10**p) for p in range(3, 9)]


@pytest.mark.parametrize('epsilon', EPSILONS)
def test_float_eq(epsilon: float) -> None:
    """Test geom2d.const.float_eq."""
    const.set_epsilon(epsilon)

    # Tolerence/epsilon minus a tiny bit (system float epsilon)
    epsilon_minus = epsilon - sys.float_info.epsilon

    # Verify tiny float values below EPSILON are zero
    for n in np.linspace(-epsilon_minus, epsilon_minus, num=1000):
        assert float_eq(n, 0)
        assert is_zero(n)

    # Verify medium range float values
    for n in np.linspace(-10, 10, 1000):
        _float_eq_t(n)

    # Verify range of larger float values
    for n in np.linspace(-const.MAX_XY, const.MAX_XY, 1000):
        _float_eq_t(n)

    # And a bunch of random floats
    for n in np.random.default_rng().uniform(
        -const.MAX_XY, const.MAX_XY, 10000
    ):
        _float_eq_t(n)


def _float_eq_t(n: float) -> None:
    assert float_eq(n, n)
    if abs(n) < 1:
        assert float_eq(
            n, n + (const.EPSILON - const.EPSILON2)
        )  # (n * const.EPSILON))
        assert not float_eq(n, n + (const.EPSILON + const.EPSILON / 10))
    else:
        scaled_epsilon = abs(n) * const.EPSILON
        assert float_eq(n, n + scaled_epsilon / 10)
        assert not float_eq(n, n + scaled_epsilon + scaled_epsilon / 10)
