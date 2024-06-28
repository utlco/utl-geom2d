"""Test geom2d.bezier module."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import geom2d
import geom2d.const
import numpy as np
import pytest
from geom2d import bezier

if TYPE_CHECKING:
    from collections.abc import Sequence

CURVE1 = ((1.6, 3.3), (3.6, 1.8), (4.8, 2.2), (4.8, 2.8))
BIARC_COUNT = 12
BIARC_TOLERANCE = 0.001
BIARC_DEPTH = 3
BIARC_LINE_FLATNESS = 0


# Constants for Line-Bezier intersections
LB_C1 = ((60, 400), (150, 80), (500, 400), (700, 200))
LB_L1 = ((300, 50), (400, 450))
LB_PT1 = [(351.6515026666981, 256.6060106667958)]
LB_INFL1 = (0.5477102270093078, -1)

LB_C2 = ((60, 400), (1001, 406), (44, 658), (700, 200))
LB_L2 = ((300, 50), (512, 525))
LB_PT2 = [
    (461.88771127523523, 412.72010781007975),
    (473.52124814601206, 438.7858154214898),
    (489.04919776387135, 473.57721197093827),
]
LB_INFL2 = (0.3391249987179755, 0.6291958150963373)

LB_C3 = ((60, 400), (150, 80), (230, 280), (700, 200))
LB_L3 = ((416, 153), (36, 311))
LB_PT3 = [
    (104.33406367316012, 282.5874156306336),
    (275.6986635546605, 211.3358188377989),
]
LB_INFL3 = (0.6274423042781576, -1)

INTERSECTIONS_PARAMETRIZE = [
    (LB_C1, LB_L1, LB_PT1),
    (LB_C2, LB_L2, LB_PT2),
    (LB_C3, LB_L3, LB_PT3),
]

# this curve appriximates a 90deg circular arc
CC1 = (
    (5.2250878300000005, 9.1147451),
    (5.2250878300000005, 8.852063688317722),
    (5.438033188317722, 8.63911833),
    (5.7007146, 8.63911833),
)
CC1_INFL = (-1, -1)

# Curve with loop
RC1 = ((1.75, 6.5), (5, 5), (1, 5), (4.25, 6.5))
RC1_INFL = (0.3391831197743308, 0.6608168802256692)
# RC1 rotated 15deg
RC1_ROT = (
    (1.938238, 6.8043492),
    (4.6892684, 4.5142986),
    (0.82556505, 5.5495748),
    (4.3530526, 6.1573016),
)

# Curve with point cusp
RC2 = ((6, 2), (8, 0.5), (6, 0.5), (8, 2))
RC2_INFL = (0.5, -1)


# Circle to test circle-to-bezier approximation
CIRCLE1_R = 3.5
CIRCLE1_CENTER = (4.25, 5)

# Circle-Bezier approximations
CIRCLE1_B1 = (
    (4.25, 8.5001932),
    (6.186994, 8.4955755),
    (7.7455755, 6.936994),
    (7.7501932, 5),
)
CIRCLE1_B2 = (
    (7.7501932, 5),
    (7.7455755, 3.063006),
    (6.186994, 1.5044245),
    (4.25, 1.4998068),
)
CIRCLE1_B3 = (
    (4.25, 1.4998068),
    (2.313006, 1.5044245),
    (0.75442453, 3.063006),
    (0.74980683, 5),
)
CIRCLE1_B4 = (
    (0.74980683, 5),
    (0.75442453, 6.936994),
    (2.313006, 8.4955755),
    (4.25, 8.5001932),
)


def test_biarcs() -> None:
    curve = geom2d.CubicBezier(*CURVE1)
    biarcs = curve.biarc_approximation(
        tolerance=BIARC_TOLERANCE,
        max_depth=BIARC_DEPTH,
        line_flatness=BIARC_LINE_FLATNESS,
    )
    assert len(biarcs) == BIARC_COUNT
    assert biarcs[0].p1 == curve.p1
    assert biarcs[-1].p2 == curve.p2
    # TODO: verify Hausdorff distance
    _verify_biarc_hausdorff(curve, biarcs, BIARC_TOLERANCE)

def _verify_biarc_hausdorff(curve: geom2d.CubicBezier, biarcs: list[geom2d.Arc], tolerance: float) -> None:
    maxhd: float = 0
    for arc in biarcs:
        hd = curve.hausdorff_distance(arc, ndiv=100)
        #print(f'hd = {hd}')
        maxhd = max(hd, maxhd)

    #print(f'max hd: {maxhd}')
    assert hd < tolerance


@pytest.mark.parametrize(('curve', 'line', 'points'), INTERSECTIONS_PARAMETRIZE)
def test_line_intersection(curve: tuple, line: tuple, points: tuple) -> None:
    b = geom2d.CubicBezier(*curve)
    pts = b.line_intersection(line)
    assert pts == points


def test_find_roots():
    f1 = geom2d.CubicBezier(*LB_C1).roots()
    assert f1 == LB_INFL1
    f2 = geom2d.CubicBezier(*LB_C2).roots()
    assert f2 == LB_INFL2
    f3 = geom2d.CubicBezier(*LB_C3).roots()
    assert f3 == LB_INFL3

    # Verify rotated curve has same roots.
    f1 = geom2d.CubicBezier(*RC1).roots()
    assert f1 == RC1_INFL
    f2 = geom2d.CubicBezier(*RC1_ROT).roots()
    assert geom2d.float_eq(f1[0], f2[0])
    assert geom2d.float_eq(f1[1], f2[1])
    f1 = geom2d.CubicBezier(*CC1).roots()
    assert f1 == CC1_INFL

    f1 = geom2d.CubicBezier(*RC2).roots()
    assert f1 == RC2_INFL


def test_bezier_circle() -> None:
    r = CIRCLE1_R
    x, y = CIRCLE1_CENTER

    curves = bezier.bezier_circle((x, y), r)
    _verify_circle_hausdorff(x, y, r, curves, 0.0006863)

    curves = bezier.bezier_circle_2((x, y), r)
    _verify_circle_hausdorff(x, y, r, curves, 0.0001945)


def _verify_circle_hausdorff(
    x: float,
    y: float,
    r: float,
    curves: Sequence[geom2d.CubicBezier],
    max_hd: float,
) -> None:
    # Circle quadrant subdivisions
    p1 = (x, r + y)
    p2 = (r + x, y)
    p3 = (x, -r + y)
    p4 = (-r + x, y)
    arc_endpoints = ((p1, p2), (p2, p3), (p3, p4), (p4, p1))

    for curve, p in zip(curves, arc_endpoints):
        a = geom2d.Arc(p[0], p[1], r, math.pi / 2, (x, y))
        hd = curve.hausdorff_distance(a, ndiv=100)
        assert hd < (max_hd + geom2d.const.EPSILON)


def test_arc_bezier_h() -> None:
    assert geom2d.float_eq(bezier.arc_bezier_h(math.pi / 2), 0.55191497)
    assert geom2d.float_eq(bezier.arc_bezier_h((2 * math.pi) / 3), 0.76808599)
    for n in np.linspace(-math.pi, math.pi, num=100):
        assert bezier.arc_bezier_h(n) >= 0
