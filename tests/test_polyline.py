"""Test geom2d.polyline module."""

from __future__ import annotations

from geom2d import polyline, triangle
from geom2d.point import P

POLY1 = (
    (3.860001, 1.6309768),
    (3.2514605, 2.1073057),
    (4.8289231, 2.568105),
    (4.2550619, 5.1915081),
    (5.5861492, 6.0517228),
    (7.1606736, 5.4181565),
    (7.6051564, 2.8175232),
    (5.7761771, 3.036092),
    (8.1331108, 4.4706927),
    (7.5812848, 7.3190128),
)
P1 = (4.2630358, 6.5547555)

POLY2 = (
    (2.9727669, 3.2002023),
    (2.0094587, 4.6535853),
    (4.4769359, 5.5000018),
    (5.3745057, 4.2477692),
    (4.8416196, 3.0795855),
    (2.9727669, 3.2002023),
)
P2_A = (4.267737974968339, 5.428240822629039)
D_P2_A = 4.131091039124777
P2_B = (4.4769359, 5.5000018)
D_P2_B = 4.352254801325523

TRIANGLE = (
    P(2.3496753, 5.8597622),
    P(1.840792, 4.4450859),
    P(4.2500014, 4.0902167),
)
INCIRCLE_CENTER = P(2.5846470413588323, 4.8905934584730995)
INCIRCLE_RADIUS = 0.5491498205604967


def test_closest_point() -> None:
    c1 = polyline.closest_point(POLY1, P1)
    c2 = polyline.closest_point(POLY1, P1, vertices_only=True)
    assert c1 == (4.88214136, 5.59675798)
    assert c2 == (4.25506190, 5.19150810)


def test_polyline_length_to() -> None:
    d1 = polyline.polyline_length_to(POLY2, P2_A)
    d2 = polyline.polyline_length_to(POLY2, P2_B)
    assert d1 == D_P2_A
    assert d2 == D_P2_B


def test_triangle_incircle() -> None:
    c, r = triangle.incircle(*TRIANGLE)
    assert c == INCIRCLE_CENTER
    assert r == INCIRCLE_RADIUS
