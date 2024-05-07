"""Test transform2d module."""

from __future__ import annotations

import math

from geom2d import transform2d as t

POINT = (3, 4)
ANGLE = math.pi / 3
OFFSET = (3.328768, 1.394879)
SCALE = (2.56, 1.1)
ORIGIN = (130.72368, 3.31092)

M1 = (
    (0.5000000000000001, -0.8660254037844386, 0.0),
    (0.8660254037844386, 0.5000000000000001, 0.0),
)
M2 = (
    (0.5000000000000001, -0.8660254037844386, 2.8723846492054337),
    (0.8660254037844386, 0.5000000000000001, -2.1853581513047184),
)
M3 = ((1.0, 0.0, 3.328768), (0.0, 1.0, 1.394879))
M4 = (
    (0.5000000000000001, -0.8660254037844386, 0.4563833507945667),
    (0.8660254037844386, 0.5000000000000001, 3.5802371513047184),
)
M5 = ((2.56, 0.0, 0.0), (0.0, 1.1, 0.0))
M6 = ((2.56, 0.0, -203.92894080000002), (0.0, 1.1, -0.3310920000000004))
M7 = ((2.56, 0.0, 3.328768), (0.0, 1.1, 1.394879))
M8 = (
    (1.2800000000000002, -2.217025033688163, 177.99547092453878),
    (0.9526279441628825, 0.5500000000000002, -121.31514553180654),
)
P1 = (172.96737078978612, -116.2572616993179)
P2 = (-144.92173661113185, -211.23368797941617)


def test_transforms() -> None:
    """Test geom2d.transform2d module functions."""
    assert t.is_identity_transform(t.IDENTITY_MATRIX)
    m1 = t.matrix_rotate(ANGLE)
    m2 = t.matrix_rotate(ANGLE, origin=(0, 0))
    assert t.matrix_equals(m1, m2)
    assert t.matrix_equals(m1, M1)

    m2 = t.matrix_rotate(ANGLE, origin=OFFSET)
    assert t.matrix_equals(m2, M2)

    m3 = t.matrix_translate(OFFSET[0], OFFSET[1])
    assert t.matrix_equals(m3, M3)
    m4 = t.compose_transform(m1, m3)
    assert t.matrix_equals(m4, M4)

    m5 = t.matrix_scale(SCALE[0], SCALE[1])
    m6 = t.matrix_scale(SCALE[0], SCALE[1], (ORIGIN[0], ORIGIN[1]))
    assert t.matrix_equals(m5, M5)
    assert t.matrix_equals(m6, M6)

    m7 = t.matrix_scale_translate(SCALE[0], SCALE[1], OFFSET[0], OFFSET[1])
    assert t.matrix_equals(m7, M7)
    m8 = t.compose_transform(m3, m5)
    assert t.matrix_equals(m7, m8)

    m8 = t.create_transform(SCALE, OFFSET, ANGLE, rotate_origin=ORIGIN)
    assert t.matrix_equals(m8, M8)

    p1 = t.matrix_apply_to_point(m8, POINT)
    assert p1[0] == P1[0]
    assert p1[1] == P1[1]

    p2 = t.canonicalize_point(p1, ORIGIN, ANGLE)
    assert p2[0] == P2[0]
    assert p2[1] == P2[1]
