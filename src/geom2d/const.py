"""Floating point comparison functions and miscellaneous constants.

For a discussion about floating point comparison see:
    http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/

The default value of :py:const:`EPSILON` should be fine for most applications
but if it is to be changed it should be set once (and only once)
with a call to :py:func:`set_epsilon()` before using the :py:mod:`geom2d` API.

:py:const:`EPSILON` (and related) should not be referenced by module globals
at import time due to this possible mutability.

"""

# pylint: disable=global-statement
# ruff: noqa: PLW0603
from __future__ import annotations

import math
import os
import sys

DEBUG = bool(os.environ.get('DEBUG', os.environ.get('GEOM2D_DEBUG')))

TAU: float = math.pi * 2.0
"""Commonly used constant 2 * *pi*."""

# TODO: see about using this technique to constrain constant refs
# https://www.safaribooksonline.com/library/view/python-cookbook/0596001673/ch05s16.html

EPSILON: float = 1e-08
"""Tolerance value used for floating point comparisons.

The maximum numeric distance between two floating point numbers for
them to be considered equal.
"""

EPSILON2: float = EPSILON * EPSILON
"""Handy for comparing distance**2 to avoid sqrt()."""

EPSILON_PRECISION: int = max(0, round(abs(math.log10(EPSILON))))
"""Number of significant digits after decimal point."""

REPSILON: float = 10**EPSILON_PRECISION
"""Reciprocal epsilon for int coordinate conversions, etc."""

EPSILON_MINUS: float = EPSILON - sys.float_info.epsilon
"""Tolerence/epsilon minus a tiny bit (system float epsilon).

Useful for testing float comparisons... But possibly not much else.
"""


MAX_XY: float = sys.maxsize * EPSILON
"""Maximum reasonable floating point coordinate value.

Numbers in this range will still work with coordinate hashing,
float comparisons, and integer conversions.
"""

HASH_PRIME_X = 73856093  # X
HASH_PRIME_Y = 19349663  # Y
HASH_PRIME_Z = 83492791  # Z (unused, but for reference)
HASH_SIZE = 2305843009213693951  # largest Mersenne prime < sys.maxsize
# See: https://oeis.org/A000043 for list of Marsenne exponents.
"""Hash constants for hashing integers and vectors/points."""


def set_epsilon(value: float) -> float:
    """Set the nominal tolerance for float comparisons.

    This updates the global constants:

        - :py:const:`EPSILON`
        - :py:const:`EPSILON2`
        - :py:const:`EPSILON_PRECISION`
        - :py:const:`REPSILON`
        - :py:const:`MAX_XY`

    Returns:
        Previous value of EPSILON
    """
    global EPSILON, EPSILON2, EPSILON_PRECISION, REPSILON, MAX_XY

    prev_epsilon = EPSILON

    # Recalculate "constants"
    EPSILON = float(value)
    EPSILON2 = EPSILON * EPSILON
    EPSILON_PRECISION = max(0, round(abs(math.log10(value))))
    REPSILON = 10**EPSILON_PRECISION
    MAX_XY = sys.maxsize * EPSILON

    return prev_epsilon


def float_eq1(a: float, b: float, tolerance: float | None = None) -> bool:
    """Compare floats.

    The two float are considered equal if the difference between them is
    less than a normalized tolerance value.
    """
    # Note: EPSILON can be initialized after import
    if tolerance is None:
        tolerance = EPSILON
    # This method tries to behave better when comparing
    # very large and very small numbers.
    norm = max(abs(a), abs(b))
    return (norm < EPSILON) or (abs(a - b) < (tolerance * norm))


def float_eq2(a: float, b: float, tolerance: float | None = None) -> bool:
    """Compare floats.

    This simpler and faster version works ok for small/reasonable values.
    It fails for larger numbers where floating point gaps are significant.
    """
    if tolerance is None:
        tolerance = EPSILON
    return abs(a - b) < tolerance


def float_eq3(a: float, b: float, tolerance: float | None = None) -> bool:
    """Compare floats.

    This is slightly faster than float_eq1 and does not scale
    tolerance for small numbers (< 1.0).
    """
    if tolerance is None:
        tolerance = EPSILON

    # Avoid function calls to max/abs
    aa = a if a >= 0 else -a
    bb = b if b >= 0 else -b
    ab_max = aa if aa > bb else bb  # noqa: FURB136

    if ab_max > 1.0:
        tolerance *= ab_max  # scale for larger numbers

    return (a - b if a > b else b - a) < tolerance  # * ab_max


float_eq = float_eq3


def angle_eq(a: float, b: float, tolerance: float | None = None) -> bool:
    """Special case of float_eq where angles close to +-PI are considered equal."""
    if tolerance is None:
        tolerance = EPSILON

    # Avoid function calls to max/abs
    aa = a if a >= 0 else -a
    bb = b if b >= 0 else -b
    ab_max = aa if aa > bb else bb  # noqa: FURB136

    if ab_max > 1.0:
        tolerance *= ab_max  # scale for larger numbers

    return (a - b if a > b else b - a) < tolerance or (
        abs(math.pi - aa) < tolerance and abs(math.pi - bb) < tolerance
    )


def is_zero(value: float) -> bool:
    """Determine if the float value is essentially zero."""
    return bool(-EPSILON < value < EPSILON)


def float_round(value: float) -> float:
    """Round the value to a rounding precision corresponding to EPSILON."""
    return float(round(value, EPSILON_PRECISION))
