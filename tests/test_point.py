"""Test 2D point (geom2d.point.P) implementation."""

import math
import sys
import warnings

import numpy as np
import pytest

from geom2d import const, line, point, transform2d
from geom2d.const import float_eq
from geom2d.point import P, TPoint

# ruff: file-ignore[no-explicit-stacklevel]

P1 = P(60, 40)
CCW_PT = [
    P(120, 90),
    P(20, 90),
    P(0, 100),
    P(-50, 140),
    P(-50, 40),
    P(-40, 10),
    P(-30, -10),
]
CW_PT = [
    P(-30, -30),
    P(30, -30),
    P(40, -10),
]


def test_random() -> None:
    """Test P.random method with various parameters."""
    # Default range
    for _ in range(10):
        p = P.random()
        assert abs(p.x) <= const.MAX_XY
        assert abs(p.y) <= const.MAX_XY

    # Custom range
    min_x, max_x = -10.0, 10.0
    min_y, max_y = -5.0, 5.0
    for _ in range(10):
        p = P.random(min_x, max_x, min_y, max_y)
        assert min_x <= p.x <= max_x
        assert min_y <= p.y <= max_y

    # Normal distribution (statistical test)
    min_x, max_x = -10.0, 10.0
    points = [P.random(min_x, max_x, normal=True) for _ in range(100)]
    x_values = [p.x for p in points]

    # Mean should be close to center
    assert abs(sum(x_values) / len(x_values)) < 2.0


def test_normal() -> None:
    """Test normal vector calculation."""
    p = P(3.0, 4.0)

    # Left normal (default)
    left_normal = p.normal()
    assert float_eq(left_normal.x, -4.0)
    assert float_eq(left_normal.y, 3.0)

    # Test that it's perpendicular (dot product should be zero)
    assert float_eq(p.dot(left_normal), 0.0)

    # Right normal
    right_normal = p.normal(left=False)
    assert float_eq(right_normal.x, 4.0)
    assert float_eq(right_normal.y, -3.0)

    # Test the alias
    assert p.perpendicular(left=True) == p.normal(left=True)
    assert p.perpendicular(left=False) == p.normal(left=False)


def test_mirror() -> None:
    """Test mirror method."""
    p = P(3.0, 4.0)
    mirror_p = p.mirror()

    assert float_eq(mirror_p.x, -3.0)
    assert float_eq(mirror_p.y, -4.0)

    # Mirror of mirror should be the original
    assert mirror_p.mirror() == p


def test_dot_product() -> None:
    """Test dot product calculation."""
    p1 = P(3.0, 4.0)
    p2 = P(2.0, 1.0)

    dot = p1.dot(p2)
    assert float_eq(dot, 3.0 * 2.0 + 4.0 * 1.0)
    assert float_eq(dot, 10.0)

    # Perpendicular vectors should have dot product of 0
    assert float_eq(p1.dot(p1.normal()), 0.0)

    # Parallel vectors with same direction
    assert float_eq(p1.dot(p1.unit() * 5), 5 * p1.length())


def test_cross_product() -> None:
    """Test cross product calculation."""
    p1 = P(3.0, 4.0)
    p2 = P(2.0, 1.0)

    cross = p1.cross(p2)
    assert float_eq(cross, 3.0 * 1.0 - 4.0 * 2.0)
    assert float_eq(cross, -5.0)

    # Parallel vectors should have cross product of 0
    assert float_eq(p1.cross(p1 * 2), 0.0)


def test_angle2() -> None:
    """Test angle2 method."""
    center = P(0.0, 0.0)
    p1 = P(1.0, 0.0)  # point on positive x-axis
    p2 = P(0.0, 1.0)  # point on positive y-axis

    # 90 degree angle (π/2)
    angle = center.angle2(p1, p2)
    assert float_eq(angle, math.pi / 2)

    # Negative angle when points are reversed
    angle = center.angle2(p2, p1)
    assert float_eq(angle, -math.pi / 2)

    # 180 degree angle (π)
    p3 = P(-1.0, 0.0)
    angle = center.angle2(p1, p3)
    assert float_eq(angle, math.pi)

    # Zero angle for coincident points
    angle = center.angle2(p1, p1)
    assert float_eq(angle, 0.0)


def test_ccw_angle2() -> None:
    """Test ccw_angle2 method."""
    center = P(0.0, 0.0)
    p1 = P(1.0, 0.0)
    p2 = P(0.0, 1.0)

    # 90 degree angle (π/2)
    angle = center.ccw_angle2(p1, p2)
    assert float_eq(angle, math.pi / 2)

    # 270 degree angle (3π/2) when points are reversed,
    # but converted to CCW representation
    angle = center.ccw_angle2(p2, p1)
    assert float_eq(angle, 3 * math.pi / 2)


def test_distance_methods() -> None:
    """Test distance and distance2 methods."""
    p1 = P(1.0, 2.0)
    p2 = P(4.0, 6.0)

    # Euclidean distance
    dist = p1.distance(p2)
    expected = math.sqrt((4.0 - 1.0) ** 2 + (6.0 - 2.0) ** 2)
    assert float_eq(dist, expected)
    assert float_eq(dist, 5.0)

    # Squared distance
    dist2 = p1.distance2(p2)
    assert float_eq(dist2, 25.0)
    assert float_eq(dist2, dist**2)


def test_distance_to_line() -> None:
    """Test distance_to_line method."""
    line_start = P(0.0, 0.0)
    line_end = P(10.0, 0.0)  # Horizontal line along x-axis

    # Point directly above the line
    p = P(5.0, 3.0)
    dist = p.distance_to_line(line_start, line_end)
    assert float_eq(dist, 3.0)

    # Point on the line
    p = P(5.0, 0.0)
    dist = p.distance_to_line(line_start, line_end)
    assert float_eq(dist, 0.0)

    # Degenerate line case (start = end)
    p = P(5.0, 3.0)
    dist = p.distance_to_line(line_start, line_start)
    assert float_eq(dist, p.distance(line_start))


def test_normal_projection() -> None:
    """Test normal_projection method."""
    v = P(10.0, 0.0)  # Vector along x axis

    # Point at unit distance along vector
    p = P(10.0, 0.0)
    projection = v.normal_projection(p)
    assert float_eq(projection, 1.0)

    # Point at twice the distance
    p = P(20.0, 0.0)
    projection = v.normal_projection(p)
    assert float_eq(projection, 2.0)

    # Point perpendicular to vector
    p = P(0.0, 5.0)
    projection = v.normal_projection(p)
    assert float_eq(projection, 0.0)

    # Degenerate case (zero vector)
    v = P(0.0, 0.0)
    projection = v.normal_projection(p)
    assert float_eq(projection, 0.0)


# triangle, inside, on_edge, on_vertex, outside
_TRIANGLES = (
    ((0.0, 0.0), (10.0, 0.0), (5.0, 10.0)),
    ((2.2605511, 4.9247722), (4.8440382, 2.2000007), (5.530277, 6.9834885)),
    ((2.3810714, 3.8914228), (1.454636, 1.0875541), (3.7954127, 7.4871229)),
)


@pytest.mark.parametrize("triangle", _TRIANGLES)
def test_inside_triangle(triangle: tuple[TPoint, TPoint, TPoint]) -> None:
    """Test inside_triangle method."""
    for offset in ((0, 0), (2.5, 3.7), (-5239879.287349, 32765.2983)):
        for rotation in np.linspace(-math.pi, math.pi, 7):
            t = [
                transform2d.transform_point(p, offset=offset, rotate_angle=rotation)
                for p in triangle
            ]
            _do_test_inside_triangle(t)


def _do_test_inside_triangle(triangle: tuple[TPoint, TPoint, TPoint]) -> None:
    a, b, c = triangle

    # Points inside triangle
    centroid = P(((a[0] + b[0] + c[0]) / 3), ((a[1] + b[1] + c[1]) / 3))
    on_edge = P(((a[0] + b[0]) / 2), ((a[1] + b[1]) / 2))  # midpoint of A-B
    on_vert = P(a)

    # Outside point (just extend centroid->on_edge)
    x0, y0 = centroid
    x1, y1 = on_edge
    dx, dy = (x1 - x0), (y1 - y0)
    outside = P((x1 + dx), (y1 + dy))

    for p in (centroid, on_edge, on_vert):
        assert p.inside_triangle(a, b, c)
        assert p.inside_triangle(c, b, a)

    assert not outside.inside_triangle(a, b, c)
    assert not outside.inside_triangle(c, b, a)


_WINDINGS = (
    ((4.7223558, 1.9336246), (1.454636, 1.0875541), (3.7954127, 7.4871229)),
    ((0.0, 0.0), (5.0, 10.0), (10.0, 0.0)),
)


def test_winding() -> None:
    """Test winding method."""
    for w in _WINDINGS:
        p0, p1, p2 = w
        assert P(p0).winding(p1, p2) < 0  # CW
        assert P(p0).winding(p2, p1) > 0  # CCW
        # Colinear (extend p0->p1)
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = (x1 - x0), (y1 - y0)
        p3 = ((x1 + dx), (y1 + dy))
        assert float_eq(P(p0).winding(p1, p3), 0)


def test_transform_and_rotate() -> None:
    """Test transform and rotate methods."""
    p = P(1.0, 0.0)

    # Rotation by 90 degrees should give (0, 1)
    rotated = p.rotate(math.pi / 2)
    assert float_eq(rotated.x, 0.0)
    assert float_eq(rotated.y, 1.0)

    # Test with origin specified
    origin = P(1.0, 1.0)
    rotated = p.rotate(math.pi / 2, origin)
    assert float_eq(rotated.x, 2.0)
    assert float_eq(rotated.y, 1.0)

    # Translation transform
    matrix = transform2d.matrix_translate(2.0, 3.0)
    transformed = p.transform(matrix)
    assert float_eq(transformed.x, 3.0)
    assert float_eq(transformed.y, 3.0)


def test_colinear() -> None:
    """Test colinear method."""
    p1 = P(2, 3)
    assert p1.colinear((4, 6), (6, 9))
    assert p1.colinear((2, 3), (6, 9))
    p1 = P(6.99677, 3.80186)
    p2 = (6.9795412, 4.2044431)
    p3 = (7.6313128, 4.5303551)
    assert not p1.colinear(p2, p3)

    p1 = P(15.70487, 2.03458)
    p2 = 16.4475, 2.43712
    p3 = 17.77478, 3.15755
    assert p1.colinear(p2, p3, tolerance=0.001)


def test_to_svg() -> None:
    """Test to_svg method."""
    p = P(1.5, 2.5)

    # Default scale
    svg = p.to_svg()
    assert svg == "1.5,2.5"

    # With scale factor
    svg = p.to_svg(scale=2.0)
    assert svg == "3,5"


def test_copysign() -> None:
    """Test copysign method."""
    p = P(3.0, 4.0)

    # Keep signs the same
    result = p.copysign((1.0, 1.0))
    assert float_eq(result.x, 3.0)
    assert float_eq(result.y, 4.0)

    # Flip x sign
    result = p.copysign((-1.0, 1.0))
    assert float_eq(result.x, -3.0)
    assert float_eq(result.y, 4.0)

    # Flip both signs
    result = p.copysign((-1.0, -1.0))
    assert float_eq(result.x, -3.0)
    assert float_eq(result.y, -4.0)


def test_operators() -> None:
    """Test arithmetic operators."""
    p1 = P(3.0, 4.0)
    p2 = P(1.0, 2.0)

    # Addition with point
    result = p1 + p2
    assert result == P(4.0, 6.0)

    # Addition with scalar
    result = p1 + 2.0
    assert result == P(5.0, 6.0)

    # Subtraction with point
    result = p1 - p2
    assert result == P(2.0, 2.0)

    # Subtraction with scalar
    result = p1 - 1.0
    assert result == P(2.0, 3.0)

    # Multiplication with scalar
    result = p1 * 2.0
    assert result == P(6.0, 8.0)

    # Right multiplication
    result = 2.0 * p1
    assert result == P(6.0, 8.0)

    # Division by scalar
    result = p1 / 2.0
    assert result == P(1.5, 2.0)

    # Floor division
    result = p1 // 2.0
    assert result == P(1.0, 2.0)

    # Negation
    result = -p1
    assert result == P(-3.0, -4.0)

    # Absolute value (magnitude)
    assert float_eq(abs(p1), 5.0)


def test_special_methods() -> None:
    """Test special methods."""
    p = P(3.0, 4.0)

    # String representation
    assert (
        str(p)
        == f"(3.{0:0<{const.EPSILON_PRECISION}}, 4.{0:0<{const.EPSILON_PRECISION}})"
    )

    # Repr
    assert repr(p) == "P(3.0, 4.0)"

    # Length (abs)
    assert float_eq(abs(p), 5.0)


def test_error_cases() -> None:
    """Test error cases and edge conditions."""
    # Division by zero
    p = P(3.0, 4.0)
    with pytest.raises(ZeroDivisionError):
        _ = p / 0

    # Multiplication by non-scalar
    with pytest.raises(ValueError):
        _ = p * "not a number"

    # Near max coordinate values
    big_xy = const.MAX_XY - 1.0
    p = P(big_xy, big_xy)
    assert p.x == big_xy
    assert p.y == big_xy

    # Very small values
    tiny = sys.float_info.epsilon
    p = P(tiny, tiny)
    assert p.is_zero()  # Should be considered zero since below EPSILON


def test_aliases() -> None:
    """Test method aliases."""
    p = P(3.0, 4.0)

    # mag = length
    assert p.mag() == p.length()

    # normalized = unit
    assert p.normalized() == p.unit()

    # perpendicular = normal
    assert p.perpendicular() == p.normal()


def test_misc() -> None:
    """Test misc point methods.

    P.max_point()
    P.min_point()
    P.from_polar()
    P.to_polar()
    P.length()
    P.length2()
    P.angle()
    P.is_zero()
    P.unit()

    """
    p = P.max_point()
    assert p.x == point.max_xy()
    assert p.y == point.max_xy()
    p = P.min_point()
    assert p.x == -point.max_xy()
    assert p.y == -point.max_xy()

    epsilon = const.EPSILON

    for angle in np.linspace(-math.pi, math.pi, 100):
        # test to/from polar
        mag = np.random.default_rng().uniform(epsilon, const.MAX_XY)
        p = P.from_polar(mag, angle)
        m, a = p.to_polar()
        assert float_eq(mag, p.length())
        assert float_eq(angle, p.angle())
        assert float_eq(m, mag)
        assert float_eq(a, angle)

        # test unit length
        p2 = P.random()
        p2u = p2.unit()
        ulen = p2u.length()
        assert ulen <= (1.0 + epsilon)
        assert float_eq(ulen * p2.x / p2u.x, p2.length())


def test_eq() -> None:
    """Test point.almost_equal(), P.__eq__()."""
    epsilon = const.EPSILON
    epsilon_plus = math.nextafter(epsilon, math.inf)
    epsilon_minus = math.nextafter(epsilon, -math.inf)

    assert point.almost_equal((0, 0), (epsilon_minus, 0))
    assert not point.almost_equal((0, 0), (epsilon, 0))
    assert not point.almost_equal((0, 0), (epsilon_plus, 0))
    p1 = P(epsilon * 0.5, epsilon * 0.5)
    p2 = P(epsilon, epsilon)
    assert p1.is_zero()
    assert not p2.is_zero()
    assert p1 == p2

    p0 = P(0, 0)
    h = epsilon_plus
    a = math.sqrt((h * h) / 2)
    p1 = P(a, a)
    assert not p1.is_zero()
    assert p1 != p0
    assert p0 + p1 != p0

    h = epsilon_minus
    a = math.sqrt((h * h) / 2)
    p2 = P(a, a)
    assert p2.is_zero()
    assert p2 == p0
    assert p1 == p2
    assert p0 + p2 == p0

    p3 = P(1, 2)

    # Due to float drift this is too close
    # assert p3 + p1 != p3
    a = math.sqrt(epsilon_plus * epsilon_plus / 2) * (1 + (epsilon / 2))
    assert p3 + P(a, a) != p3

    assert p3 + p2 == p3

    for angle in np.linspace(-math.pi, math.pi, 100):
        # test __eq__
        p1 = P.from_polar(epsilon_minus, angle)
        p2 = P.from_polar(epsilon_plus, angle)
        assert p1.is_zero()
        assert not p2.is_zero()
        assert p1 == p2
        assert point.almost_equal((0, 0), p1)
        assert not point.almost_equal((0, 0), p2)
        assert p0 + p1 == p1
        assert p1 + epsilon != p1


def test_is_ccw() -> None:
    """Test P.is_ccw()."""
    # All points in CCW_PT are CCW (left of) P1
    for p2 in CCW_PT:
        assert P1.is_ccw(p2)
        assert P(0, 0).winding(P1, p2) > 0
    # All points in CW_PT are CW (right of) P1
    for p2 in CW_PT:
        assert not P1.is_ccw(p2)
        assert P(0, 0).winding(P1, p2) < 0


def test_bisector() -> None:
    """Test bisector method."""
    center = P(1, 3)
    p1 = P(5, 7)
    p2 = P(3, 1)
    bisector = P(2, 3)

    assert bisector == center.bisector(p1, p2)

    # Test range of rotations and magnitudes
    rng = np.random.default_rng()
    for angle in np.linspace(-math.pi, math.pi, 10):
        pp1 = p1.rotate(angle, center)
        pp2 = p2.rotate(angle, center)
        bb = bisector.rotate(angle, center)
        assert bb == center.bisector(pp1, pp2)
        assert bb == center.bisector(pp2, pp1)
        mag = rng.uniform(0.5, 10)
        bb = center.bisector(pp1, pp2, mag)
        assert float_eq((bb - center).length(), mag)

    # Test 0-pi cardinals
    center = P(0, 0)
    assert center.bisector((-3, -3), (-3, 3)) == (-1, 0)
    assert center.bisector((3, 3), (3, -3)) == (1, 0)
    assert center.bisector((-3, 3), (3, 3)) == (0, 1)
    assert center.bisector((-3, -3), (3, -3)) == (0, -1)

    # Test directionality
    assert center.bisector((-3, -3), (-3, 3), winding=1) == (1, 0)
    assert center.bisector((-3, -3), (-3, 3), winding=-1) == (-1, 0)
    assert center.bisector((3, -3), (3, 3), winding=1) == (1, 0)
    assert center.bisector((3, -3), (3, 3), winding=-1) == (-1, 0)
    assert center.bisector((-3, 3), (3, 3), winding=1) == (0, -1)
    assert center.bisector((-3, 3), (3, 3), winding=-1) == (0, 1)
    assert center.bisector((-3, -3), (3, -3), winding=1) == (0, -1)
    assert center.bisector((-3, -3), (3, -3), winding=-1) == (0, 1)


def test_hash() -> None:
    """Test point.hash()."""
    # Collision test - big coordinate space
    _test_hash(100000, -const.MAX_XY, const.MAX_XY)
    # Collision test - small coordinate space
    _test_hash(100000, -1.0, 1.0)

    epsilon = const.EPSILON
    # p0 rotates around the origin
    for angle in np.linspace(-math.pi, math.pi, 12):
        p0 = P.from_polar(5.0, angle)
        assert hash(p0) == point.hash_xy(p0[0], p0[1])
        for angle2 in np.linspace(-math.pi, math.pi, 12):
            # These two points rotate around the primary point p0.
            # One slightly closer than EPSILON, the other slightly
            # further than EPSILON.
            p1 = p0 + P.from_polar(0.5 * epsilon, angle2)
            p2 = p0 + P.from_polar(0.49 * epsilon, angle2 + math.pi)
            p3 = p0 + P.from_polar(11 * epsilon, angle2)
            assert hash(p0) == hash(p1)
            assert p0 == p1
            assert p1.distance(p2) < const.EPSILON
            assert hash(p1) == hash(p2)
            assert p1 == p2
            assert hash(p0) != hash(p3)
            assert p0 != p3
            assert line.hash_line(p0[0], p0[1], p3[0], p3[1]) == hash(
                line.Line(p3, p0)
            )
            assert line.hash_line(p0[0], p0[1], p3[0], p3[1]) == hash(
                line.Line(p0, p3)
            )


def _test_hash(max_hashes: int, min_xy: float, max_xy: float) -> None:
    rng = np.random.default_rng()
    points = {
        P(rng.uniform(min_xy, max_xy), rng.uniform(min_xy, max_xy))
        for n in range(max_hashes)
    }
    # A hash collision shouldn't really be a test failure since there
    # is a non-zero (but very unlikely) chance of a collision.
    collisions = max_hashes - len(points)
    if collisions > 0:
        warnings.warn(f"{collisions} hash collisions - try running again.")

    hashes = {hash(p) for p in points}
    assert len(hashes) == len(points)
