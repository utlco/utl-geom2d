"""Test geom2d.polygon module."""

from __future__ import annotations

import geom2d
from geom2d import const, point, polygon
from geom2d.point import P

TOLERANCE = 1e-6
POLY1 = (
    (2.86579, 7.19138),
    (2.60321, 5.60556),
    (2.66237, 5.80965),
    (2.66237, 5.6006),
    (2.75099, 5.96077),
    (2.89142, 5.55078),
    (2.85855, 6.4492),
    (3.43943, 6.91374),
    (4.4335, 6.14409),
    (3.50829, 7.40838),
    (4.81146, 7.12855),
    (3.43943, 8.18852),
    (3.43943, 8.42778),
    (3.25346, 8.03314),
    (2.82143, 7.93781),
    (3.22212, 7.68965),
    (1.74289, 7.45716),
    (2.95283, 7.5525),
    (2.86579, 7.19138),
)
POLY1_AREA = 1.766671
POLY1_CENTROID = (3.427692, 7.231737)

POLY2 = ((2, 2), (4, 2), (5, 1), (6, 4), (4, 5), (1, 4))
POLY2_AREA = 11.5

POLY3 = (
    (5,2),
    (6,3),
    (6,4),
    (4,5),
    (3,4),
    (3,2),
    (2,2),
    (2,4),
    (3,6),
    (6,6),
    (7,5),
    (7,4),
    (6,2),
)
POLY4 = (
    (5,2),
    (5,0),
    (4,0),
    (4,1),
    (3,1),
    (3,2),
    (2,2),
    (2,4),
    (3,6),
    (6,6),
    (7,5),
    (7,4),
    (6,2),
)
POLY4_AREA = -20.5

SIMPOLY1 = [
    (2.86579, 7.19138),
    (2.60321, 5.60556),
    (2.75099, 5.96077),
    (2.89142, 5.55078),
    (2.85855, 6.4492),
    (3.43943, 6.91374),
    (4.4335, 6.14409),
    (3.50829, 7.40838),
    (4.81146, 7.12855),
    (3.43943, 8.18852),
    (2.82143, 7.93781),
    (3.22212, 7.68965),
    (1.74289, 7.45716),
    (2.95283, 7.5525),
    (2.86579, 7.19138),
]

LINE = [(1.6504, 1.8223), (5.2148, 0.8864)]
STROKE_TO_PATH = [[
    P(5.151309919, 0.644596341),
    P(5.278290081, 1.128203659),
    P(1.713890081, 2.064103659),
    P(1.586909919, 1.580496341),
    P(5.151309919, 0.644596341),
]]


def test_polygon_turn() -> None:
    assert polygon.turn((1, 1), (2, 4), (-1, 7)) == polygon.TURN_LEFT
    assert polygon.turn((2, 2), (4, 2), (5, 1)) == polygon.TURN_RIGHT
    assert polygon.turn((2, 2), (4, 2), (8, 2)) == 0
    assert polygon.turn((8, 3), (2, 4), (1, 1)) == polygon.TURN_LEFT

def test_polygon_winding() -> None:
    assert polygon.winding(POLY3) == polygon.CW
    assert polygon.winding(POLY3, close=False) == polygon.CW
    assert polygon.winding(reversed(POLY3)) == polygon.CCW
    assert polygon.winding(POLY4) == polygon.CW
    assert polygon.winding(reversed(POLY4)) == polygon.CCW

def test_polygon_area() -> None:
    """Test polygon.area function."""
    assert const.float_eq(POLY1_AREA, polygon.area(POLY1), tolerance=TOLERANCE)
    assert const.float_eq(POLY1_AREA, polygon.area(POLY1[:-1]), tolerance=TOLERANCE)

    assert const.float_eq(-POLY1_AREA, polygon.area(reversed(POLY1)), tolerance=TOLERANCE)

    assert polygon.area(POLY2) == POLY2_AREA
    assert polygon.area(POLY4) == POLY4_AREA


def test_polygon_centroid() -> None:
    """Test polygon.centroid function."""
    c = polygon.centroid(POLY1)
    assert point.almost_equal(c, POLY1_CENTROID, tolerance=TOLERANCE)

    c = polygon.centroid(POLY1[:1])
    assert point.almost_equal(c, POLY1[0], tolerance=TOLERANCE)

    c = polygon.centroid(POLY1[:2])
    mp = geom2d.Line(POLY1[0], POLY1[1]).midpoint()
    assert point.almost_equal(c, mp, tolerance=TOLERANCE)


def test_simplify_vw() -> None:
    """Test polygon.simplify_polyline_vw function."""
    simpoly = polygon.simplify_polyline_vw(POLY1, min_area=0.05)
    assert simpoly == SIMPOLY1
    # TODO: calculate proper test results


def test_stroke_to_path() -> None:
    """Test polygon.stroke_to_path function."""
    paths = polygon.poly_stroke_to_path(LINE, 0.5)
    assert len(paths) == 1
    assert paths == STROKE_TO_PATH
