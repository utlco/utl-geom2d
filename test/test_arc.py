"""Test geom2d.arc module."""

from __future__ import annotations

import math

import geom2d
from geom2d import const, ellipse
from geom2d.arc import Arc
from geom2d.ellipse import EllipticalArc
from geom2d.point import P

# ruff: noqa: T201

# Two arcs that maintain sequential G1 continuity
G1_ARCS = [
    Arc(
        (2.9339552, 4.5527481),
        (2.796207118678724, 5.066830803567288),
        1.0281651740545676,
        0.5235989064807819,
        (1.9057900259454326, 4.5527481),
    ),
    Arc(
        (2.796207118678724, 5.066830803567288),
        (2.4198724, 5.443165499999999),
        1.028165930499886,
        0.5235985729456468,
        (1.9057893708446176, 4.5527477217772585),
    ),
]

ARC_3 = Arc(
    (2.73197710, 3.19424020),
    (3.17520630, 2.75101100),
    0.35000000,
    -4.06426634,
    (3.06375961, 3.08279351),
)

ARC_5 = Arc(
    (2.96989410, 4.80045950),
    (3.41312330, 4.35723030),
    0.31341037,
    -math.pi,
    (3.19150870, 4.57884490),
)

ELARC_2_1 = EllipticalArc(
    P(1.145833333333, 2.239583333333),
    P(1.569895833333, 1.773854166667),
    0.625,
    0.375,
    1.5707963267948966,
    0,
    1,
    1.8150841461749252,
    1.4877295260771353,
    P(1.5096995168996439, 2.3907491772911187),
)
ELARC_2_2 = EllipticalArc(
    P(1.797395833333, 1.588020833333),
    P(2.240625, 1.1447917),
    0.52235058,
    0.31341035,
    0.7853981633974483,
    0,
    1,
    1.5710778241926737,
    3.141029703918528,
    P(2.0191143815439614, 1.366510231546776),
)


def test_arc_g1() -> None:
    """Test G1 (tangential connection)."""
    assert geom2d.float_eq(
        G1_ARCS[0].end_tangent_angle(), G1_ARCS[1].start_tangent_angle()
    )


def test_arc_subdivide() -> None:
    """Test Arc subdivision."""
    arcs = ARC_3.subdivide_at(0.5)
    assert len(arcs) == 2
    assert const.angle_eq(arcs[0].angle, arcs[1].angle)
    assert const.angle_eq(arcs[0].angle + arcs[1].angle, ARC_3.angle)
    assert arcs[0].length() == arcs[1].length()


def test_arc_center() -> None:
    """Test Arc center calculation."""
    print('ARC_3')
    c = geom2d.arc.calc_center(ARC_3.p1, ARC_3.p2, ARC_3.radius, ARC_3.angle)
    assert c == ARC_3.center
    print('ARC_4')
    c = geom2d.arc.calc_center(ARC_5.p1, ARC_5.p2, ARC_5.radius, ARC_5.angle)
    assert c == ARC_5.center


def test_arc_elliptical() -> None:
    """Test EllipticalArc."""
    ea1 = EllipticalArc.from_endpoints(
        ELARC_2_1.p1,
        ELARC_2_1.p2,
        ELARC_2_1.rx,
        ELARC_2_1.ry,
        ELARC_2_1.phi,
        ELARC_2_1.large_arc,
        ELARC_2_1.sweep_flag,
    )
    assert ea1 == ELARC_2_1
    ea1 = EllipticalArc.from_center(
        ELARC_2_1.center,
        ELARC_2_1.rx,
        ELARC_2_1.ry,
        ELARC_2_1.phi,
        ELARC_2_1.start_angle,
        ELARC_2_1.sweep_angle,
    )
    assert ea1 == ELARC_2_1

    ea2 = EllipticalArc.from_endpoints(
        ELARC_2_2.p1,
        ELARC_2_2.p2,
        ELARC_2_2.rx,
        ELARC_2_2.ry,
        ELARC_2_2.phi,
        ELARC_2_2.large_arc,
        ELARC_2_2.sweep_flag,
    )
    assert ea2 == ELARC_2_2
    ea2 = EllipticalArc.from_center(
        ELARC_2_2.center,
        ELARC_2_2.rx,
        ELARC_2_2.ry,
        ELARC_2_2.phi,
        ELARC_2_2.start_angle,
        ELARC_2_2.sweep_angle,
    )
    assert ea2 == ELARC_2_2

    p = ea1.point_at(.5)
    assert const.float_eq(ea1.point_to_theta(p), .5)
    p = ea2.point_at(.5)
    assert const.float_eq(ea2.point_to_theta(p), .5)

    assert ea1.point_inside(ea1.center)
    p = ea1.center + P(ea1.rx - const.EPSILON * 10, 0).rotate(ea1.phi)
    assert ea1.point_inside(p)
    p = ea1.center + P(0, ea1.ry - const.EPSILON * 10).rotate(ea1.phi)
    assert ea1.point_inside(p)

    assert ea2.point_inside(ea2.center)

