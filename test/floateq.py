"""Time various float equality tests."""

# ruff: noqa: D103, B023, T201

from __future__ import annotations

import inspect
import math
import timeit
from typing import Callable

import geom2d
from geom2d.const import EPSILON


def float_eq11(a: float, b: float) -> bool:
    return abs(a - b) < EPSILON


def float_eq12(a: float, b: float) -> bool:
    norm = max(abs(a), abs(b), 1.0)
    return abs(a - b) < (EPSILON * norm)


def float_eq13(a: float, b: float) -> bool:
    norm = max(abs(a), abs(b), 1.0)
    return (norm < EPSILON) or (abs(a - b) < (EPSILON * norm))


def float_eq14(a: float, b: float) -> bool:
    return abs(a - b) < (EPSILON * max(abs(a), abs(b), 1.0))


def float_eq15(a: float, b: float, tolerance: float = EPSILON) -> bool:
    return abs(a - b) < (tolerance * max(abs(a), abs(b), 1.0))


def float_eq16(a: float, b: float, tolerance: float = EPSILON) -> bool:
    return math.fabs(a - b) < (tolerance * max(math.fabs(a), math.fabs(b), 1.0))


def float_eq17(a: float, b: float, tolerance: float = EPSILON) -> bool:
    aa = a if a >= 0 else -a  # abs(a)
    bb = b if b >= 0 else -b  # abs(b)
    norm = max(bb, aa)  # max(abs(a), abs(b))
    return abs(a - b) < (tolerance * (norm + 1))


def float_eq18(a: float, b: float, tolerance: float = EPSILON) -> bool:
    aa = a if a >= 0 else -a  # abs(a)
    bb = b if b >= 0 else -b  # abs(b)
    norm = max(bb, aa)  # max(abs(a), abs(b))
    d = a - b if a > b else b - a
    return d < (tolerance * (norm + 1))


def float_eq19(a: float, b: float, tolerance: float = EPSILON) -> bool:
    aa = -a if a < 0 else a  # abs(a)
    bb = -b if b < 0 else b  # abs(b)
    norm = max(aa, bb)  # max(abs(a), abs(b))
    d = b - a if a < b else a - b  # abs(a - b)
    return d < (tolerance * (norm + 1))


def float_eq9a(a: float, b: float, tolerance: float | None = None) -> bool:
    if tolerance is None:
        tolerance = EPSILON
    aa = a if a >= 0 else -a  # abs(a)
    bb = b if b >= 0 else -b  # abs(b)
    ab_max = max(bb, aa)
    if ab_max > 1.0:
        tolerance *= ab_max  # scale for larger numbers
    return (a - b if a > b else b - a) < tolerance


def float_eq9b(a: float, b: float, tolerance: float = EPSILON) -> bool:
    aa = a if a >= 0 else -a  # abs(a)
    bb = b if b >= 0 else -b  # abs(b)
    return (a - b if a > b else b - a) < (tolerance * ((max(bb, aa)) + 1))


N1, N2 = 1.4238764823764, 3487687.198278

FLOAT_EQ_FNS: tuple[Callable, ...] = (
    geom2d.const.float_eq,
    geom2d.const.float_eq1,
    geom2d.const.float_eq2,
    geom2d.const.float_eq3,
    float_eq11,
    float_eq12,
    float_eq13,
    float_eq14,
    float_eq15,
    float_eq16,
    float_eq17,
    float_eq18,
    float_eq19,
    float_eq9a,
    float_eq9b,
)


def main() -> None:
    results = []
    for feq in FLOAT_EQ_FNS:
        nargs = len(inspect.signature(feq).parameters)
        if nargs == 3:
            t = timeit.timeit(
                lambda: feq(N1, N2, EPSILON),
                number=1000000,
            )
        else:
            t = timeit.timeit(
                lambda: feq(N1, N2),
                number=1000000,
            )
        results.append((t, feq.__name__))
    for r in sorted(results):
        print(f'{r[1]:<10} {r[0]:.06f}')


if __name__ == '__main__':
    main()
