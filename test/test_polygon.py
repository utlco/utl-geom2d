
import geom2d
from geom2d import const, polygon, point
from inkext import geomsvg

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
POLY1_CENTROID = (3.427692485125855, 7.23173736316392)


def test_polygon_area():
    area = polygon.area(POLY1)
    assert(const.float_eq(area, polygon.area(POLY1[:-1]), tolerance=TOLERANCE))
    area_r = polygon.area(reversed(POLY1))
    assert(area < 0) # winding direction
    assert(area_r > 0) # winding direction
    assert(const.float_eq(area_r, polygon.area(list(reversed(POLY1))[:-1]), tolerance=TOLERANCE))
    assert(const.float_eq(abs(area), area_r, tolerance=TOLERANCE))
    assert(const.float_eq(abs(area), POLY1_AREA, tolerance=TOLERANCE))


def test_polygon_centroid():
    c = polygon.centroid(POLY1)
    assert(point.P(c) == point.P(POLY1_CENTROID))

    c = polygon.centroid(POLY1[:1])
    assert(point.P(c) == point.P(POLY1[0]))

    c = polygon.centroid(POLY1[:2])
    assert(point.P(c) == geom2d.Line(POLY1[0], POLY1[1]).midpoint())

