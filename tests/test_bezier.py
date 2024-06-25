"""Test geom2d.bezier module."""

from __future__ import annotations

import math

import geom2d
import geom2d.const
import pytest
from geom2d import bezier

CURVE1 = (
    (1.7205388, 8.4824794),
    (5.4513704, 6.8569157),
    (0.58214369, 5.854022),
    (3.8688956, 8.2376514),
)
BIARC_PATH = (
    (1.7205388, 8.4824794),
    (
        7.7133622647036715,
        7.7133622647036715,
        0,
        0,
        0,
        2.1625218720749353,
        8.272986567702915,
    ),
    (
        4.417260178011077,
        4.417260178011077,
        0,
        0,
        0,
        2.5138965383690364,
        8.071718003288789,
    ),
    (
        3.2854394339445125,
        3.2854394339445125,
        0,
        0,
        0,
        2.783500044955975,
        7.880732798411543,
    ),
    (
        1.7296134648164587,
        1.7296134648164587,
        0,
        0,
        0,
        2.979234179246683,
        7.700717073176886,
    ),
    (
        1.2365528141025641,
        1.2365528141025641,
        0,
        0,
        0,
        3.1100601562401966,
        7.533655146894853,
    ),
    (
        0.6639387810502014,
        0.6639387810502014,
        0,
        0,
        0,
        3.184008424992583,
        7.3806033705273375,
    ),
    (
        0.5061315836185065,
        0.5061315836185065,
        0,
        0,
        0,
        3.2094750529183615,
        7.2430993862934265,
    ),
    (
        0.384593944149876,
        0.384593944149876,
        0,
        0,
        0,
        3.1956759779663773,
        7.122503656203189,
    ),
    (
        0.38473726716288403,
        0.38473726716288403,
        0,
        0,
        0,
        3.142040851944804,
        7.007869885888827,
    ),
    (
        0.4354441551553535,
        0.4354441551553535,
        0,
        0,
        0,
        3.0603449123854376,
        6.918775038155422,
    ),
    (
        0.46387066523207166,
        0.46387066523207166,
        0,
        0,
        0,
        2.9629309569716007,
        6.85704646636593,
    ),
    (
        0.384094222829155,
        0.384094222829155,
        0,
        0,
        0,
        2.8617773138928184,
        6.82555323198668,
    ),
    (
        0.29087511560817814,
        0.29087511560817814,
        0,
        0,
        0,
        2.7695337701246645,
        6.825162197598666,
    ),
    (
        0.13677324050799555,
        0.13677324050799555,
        0,
        0,
        0,
        2.698624267466431,
        6.859110409223775,
    ),
    (
        0.11355246916250542,
        0.11355246916250542,
        0,
        0,
        0,
        2.675045999731159,
        6.889280372631843,
    ),
    (
        0.13591182823609743,
        0.13591182823609743,
        0,
        0,
        0,
        2.661156665333671,
        6.928516170221641,
    ),
    (
        0.17678058839751204,
        0.17678058839751204,
        0,
        0,
        0,
        2.6588139174695717,
        6.977267456208635,
    ),
    (
        0.3271009635564713,
        0.3271009635564713,
        0,
        0,
        0,
        2.6695368580841854,
        7.035718741393522,
    ),
    (
        0.47956879210699327,
        0.47956879210699327,
        0,
        0,
        0,
        2.7041109713198708,
        7.124388319319448,
    ),
    (
        1.0464849346772587,
        1.0464849346772587,
        0,
        0,
        0,
        2.7656005700090214,
        7.229482078894049,
    ),
    (
        1.465852686268859,
        1.465852686268859,
        0,
        0,
        0,
        2.856607029401953,
        7.351809066016947,
    ),
    (
        2.731563503199021,
        2.731563503199021,
        0,
        0,
        0,
        2.980689094986611,
        7.491616601604698,
    ),
    (
        3.571814896862915,
        3.571814896862915,
        0,
        0,
        0,
        3.140711910160394,
        7.649669615289186,
    ),
    (
        5.876705351536765,
        5.876705351536765,
        0,
        0,
        0,
        3.3400411869919413,
        7.826285358861378,
    ),
    (
        7.325615402054509,
        7.325615402054509,
        0,
        0,
        0,
        3.5816783844870956,
        8.022152731169065,
    ),
    (11.057347795264578, 11.057347795264578, 0, 0, 0, 3.8688956, 8.2376514),
)
BIARC_TOLERANCE = 0.001
BIARC_DEPTH = 4
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
CIRCLE1_CX = 4.25
CIRCLE1_CY = 5

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
    p1 = BIARC_PATH[0]
    for biarc, biarc_ref in zip(biarcs, BIARC_PATH[1:]):
        rx, ry, phi, large_arc, sweep_flag, x, y = biarc_ref
        p2 = (x, y)
        elarc = geom2d.EllipticalArc.from_endpoints(
            p1, p2, rx, ry, phi, large_arc, sweep_flag
        )
        assert elarc
        assert isinstance(biarc, geom2d.Arc)
        assert biarc.p1 == p1
        assert biarc.p2 == p2
        assert geom2d.float_eq(biarc.radius, rx)
        assert geom2d.float_eq(biarc.angle, elarc.sweep_angle)
        assert biarc.center == elarc.center
        p1 = p2


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


def test_bezier_circle():
    prev_epsilon = geom2d.const.set_epsilon(1e-7)
    curves = bezier.bezier_circle_2((CIRCLE1_CX, CIRCLE1_CY), CIRCLE1_R)
    assert curves[0] == CIRCLE1_B1
    assert curves[1] == CIRCLE1_B2
    assert curves[2] == CIRCLE1_B3
    assert curves[3] == CIRCLE1_B4
    geom2d.const.set_epsilon(prev_epsilon)
    p1 = (CIRCLE1_CX, CIRCLE1_R + CIRCLE1_CY)
    p2 = (CIRCLE1_R + CIRCLE1_CX, CIRCLE1_CY)
    a1 = geom2d.Arc(p1, p2, CIRCLE1_R, math.pi / 2, (CIRCLE1_CX, CIRCLE1_CY))
    hd = curves[0].hausdorff_distance(a1)
    assert hd < 0.00018
    curves = bezier.bezier_circle((CIRCLE1_CX, CIRCLE1_CY), CIRCLE1_R)
    hd = curves[0].hausdorff_distance(a1)
    assert hd < 0.00063
