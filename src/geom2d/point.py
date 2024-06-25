"""Basic 2D point/vector."""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, SupportsFloat

# from collections import namedtuple
from . import const, transform2d, util

if TYPE_CHECKING:
    from transform2d import TMatrix
    from typing_extensions import Self

# Generic point input type.
# A point is generally a sequence containing
# at least two floating point values.
TPoint = Sequence[float]

# Moved to const.py
# _HASH_PRIME_X = 73856093  # X
# _HASH_PRIME_Y = 19349663  # Y
# _HASH_PRIME_Z = 83492791  # Z (unused, but for reference)
# _HASH_SIZE = 2305843009213693951  # largest Mersenne prime < sys.maxsize
# See: https://oeis.org/A000043 for list of Marsenne exponents.
# _HASH_SIZE = 2147483647 # for 32bit python


def max_xy() -> float:
    """Max absolute value for X or Y.

    This is a fairly large number (well over 100 digits)
    that still works with the hash function and fast float
    comparisons.

    This is a function not a constant because const.EPSILON is
    potentially mutable.
    """
    # h = max(_HASH_PRIME_X, _HASH_PRIME_Y)
    # return sys.float_info.max / (h * const.REPSILON)
    return const.MAX_XY


# def big_xy() -> float:
#    """A reasonably big value for X or Y.
#
#    This is a function not a constant because const.EPSILON is
#    potentially mutable.
#    """
#    return 10.0 ** max(const.EPSILON_PRECISION, 16)
#    # h = max(_HASH_PRIME_X, _HASH_PRIME_Y)
#    # return h * (10.0 ** const.EPSILON_PRECISION)


def almost_equal(
    p1: TPoint, p2: TPoint, tolerance: float | None = None
) -> bool:
    """Compare points for geometric equality.

    Args:
        p1: First point. A 2-tuple (x, y).
        p2: Second point. A 2-tuple (x, y).
        tolerance: Max distance between the two points.
            Default is ``EPSILON``.

    Returns:
        True if distance between the points < `tolerance`.
    """
    if tolerance is None:
        tolerance = const.EPSILON  # Note: EPSILON is mutable
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    h2 = dx * dx + dy * dy
    # return bool(h2 < (h2 * tolerance * tolerance))
    return bool(h2 < (tolerance * tolerance))


# pylint: disable=invalid-name
class P(tuple[float, float]):  # namedtuple('P', 'x, y')):
    """Two dimensional immutable Cartesion point (vector).

    Represented as a simple tuple (x, y) so that it is compatible with many
    other libraries.
    """

    __slots__ = ()

    def __new__(cls, x: TPoint | float, y: float | None = None) -> Self:
        """Create a new P instance.

        If ``y`` is None ``x`` is assumed to be a tuple or list containing
        both x and y.

        Args:
            x: X coordinate or (x, y) sequence.
            y: Y coordinate. None if x is a sequence.
        """
        if y is None:
            assert isinstance(x, Sequence)
            # assume x is a tuple[float, float]
            return tuple.__new__(
                cls, ((float(x[0]), float(x[1])))
            )  # type: ignore [type-var]
        # assume x is float, y is float
        return tuple.__new__(
            cls, ((float(x), float(y)))  # type: ignore [arg-type]
        )  # type: ignore [type-var]

    @property
    def x(self) -> float:
        """The X axis coordinate."""
        return self[0]

    @property
    def y(self) -> float:
        """The Y axis coordinate."""
        return self[1]

    @staticmethod
    def max_point() -> P:
        """Create a point with maximum X and Y values."""
        return P(max_xy(), max_xy())

    @staticmethod
    def min_point() -> P:
        """Create a point with minimum negative X and Y values."""
        return P(-max_xy(), -max_xy())

    @staticmethod
    def from_polar(mag: float, angle: float) -> P:
        """Create a Cartesian point from polar coordinates.

        See http://en.wikipedia.org/wiki/Polar_coordinate_system

        Args:
            mag: Magnitude (radius)
            angle: Angle in radians

        Returns:
            A point.
        """
        return P(mag * math.cos(angle), mag * math.sin(angle))

    @staticmethod
    def random(
        min_x: float = -sys.float_info.max,
        max_x: float = sys.float_info.max,
        min_y: float | None = None,
        max_y: float | None = None,
        normal: bool = False,
    ) -> P:
        """Create a random point within a range.

        The default range is -big_xy() to big_xy().

        Args:
            min_x: Minimum X coordinate.
            max_x: Maximum X coordinate.
            min_y: Minimum Y coordinate.
                Default is value of min_x.
            max_y: Maximum Y coordinate.
                Default is value of max_x.
            normal: Use a normal/Gaussian instead of uniform
                distribution. Mu is midpoint of min/max span,
                sigma is half span divided by 3.
        """
        big_x = const.MAX_XY  # big_xy()
        min_x = max(min_x, -big_x)
        max_x = min(max_x, big_x)
        if min_y is None:
            min_y = min_x
        if max_y is None:
            max_y = max_x
        if normal:
            # TODO: calculate truncated normal to avoid out of bounds
            # long tail values.
            # See: https://en.wikipedia.org/wiki/Truncated_normal_distribution
            mu_x = (max_x + min_x) / 2
            mu_y = (max_y + min_y) / 2
            sigma_x = (max_x - mu_x) / 3
            sigma_y = (max_y - mu_y) / 3
            # Note: not thread safe. See docs.
            return P(random.gauss(mu_x, sigma_x), random.gauss(mu_y, sigma_y))
        return P(random.uniform(min_x, max_x), random.uniform(min_y, max_y))

    def to_polar(self) -> tuple[float, float]:
        """Convert this point to polar coordinates.

        ReturnsL
            A tuple containing the radius/magnitude
            and angle respectively (r, a).
        """
        return (self.length(), self.angle())

    def is_zero(self) -> bool:
        """Check if this point is within EPSILON distance to zero/origin."""
        # return self.length2() < (const.EPSILON * const.EPSILON)
        x, y = self
        return (x * x + y * y) < const.EPSILON2

    # def almost_equal(
    #    self, other: TPoint, tolerance: float | None = None
    # ) -> bool:
    #    """Compare points for geometric equality.

    #    Args:
    #        other: Vector (point) being compared. A 2-tuple (x, y).
    #        tolerance: Max distance between the two points.
    #            Default is ``EPSILON``.

    #    Returns:
    #        True if distance between the points < `tolerance`.
    #    """
    #    if tolerance is None:
    #        tolerance = const.EPSILON  # Note: EPSILON is mutable
    #    dx = self[0] - other[0]
    #    dy = self[1] - other[1]
    #    return bool((dx * dx + dy * dy) < (tolerance * tolerance))

    def length(self) -> float:
        """The length or scalar magnitude of the vector.

        Returns:
            Distance from (0, 0).
        """
        return math.hypot(self[0], self[1])

    def length2(self) -> float:
        """The square of the length of the vector."""
        x, y = self
        return x * x + y * y

    def unit(self) -> P:
        """The vector scaled to unit length.

        If the vector length is zero, a null (0, 0) vector is returned.

        Returns:
            A copy of this vector scaled to unit length.
        """
        if not self.is_zero():
            ln = self.length()
            return P(self[0] / ln, self[1] / ln)
        # x, y = self
        # vlen2 = x * x + y * y
        # if vlen2 > const.EPSILON2:
        #    vlen = math.sqrt(vlen2)
        #    return P(x / vlen, y / vlen)
        return P(0.0, 0.0)

    def normal(self, left: bool = True) -> P:
        """Return a vector perpendicular to this one.

        Args:
            left: Normal is left of vector if True, otherwise right.
                Default is True.
        """
        return P(-self[1], self[0]) if left else P(self[1], -self[0])

    def mirror(self) -> P:
        """This vector flipped 180d."""
        return P(-self[0], -self[1])

    def dot(self, other: TPoint) -> float:
        r"""Compute the dot product with another vector.

        Equivalent to \|p1\| * \|p2\| * cos(theta) where theta is the
        angle between the two vectors.

        See:
            http://en.wikipedia.org/wiki/Dot_product

        Args:
            other: The vector with which to compute the dot product.
                A 2-tuple (x, y).

        Returns:
            A scalar dot product.
        """
        x2, y2 = other
        return self[0] * x2 + self[1] * y2

    def cross(self, other: TPoint) -> float:
        """Compute the cross product with another vector.

        Also called the perp-dot product for 2D vectors.
        Also called determinant for 2D matrix.

        See:
            http://mathworld.wolfram.com/PerpDotProduct.html
            http://johnblackburne.blogspot.com/2012/02/perp-dot-product.html
            http://www.gamedev.net/topic/441590-2d-cross-product/

        From Woodward:
        The cross product generates a new vector perpendicular to the two
        that are being multiplied, with a length equal to the (ordinary)
        product of their lengths.

        Args:
            other: The vector with which to compute the cross product.

        Returns:
            A scalar cross product.
        """
        x2, y2 = other
        return self[0] * y2 - self[1] * x2

    def is_ccw(self, other: TPoint) -> bool:
        """Return True if the other vector is to the left of this vector.

        That would be counter-clockwise with respect to the origin as
        long as the sector angle is less than PI (180deg).
        """
        return self.cross(other) > 0

    def angle(self) -> float:
        """The angle of this vector relative to the x axis in radians.

        Returns:
            A float value between -pi and pi.
        """
        return math.atan2(self[1], self[0])

    def angle2(self, p1: TPoint, p2: TPoint) -> float:
        """The angle formed by p1->self->p2.

        The angle is negative if p1 is to the left of p2.

        Args:
            p1: First point as 2-tuple (x, y).
            p2: Second point as 2-tuple( x, y).

        Returns:
            The angle in radians between -pi and pi.
            Returns 0 if points are coincident.
        """
        v1 = P(p1) - self
        v2 = P(p2) - self
        if v1 == v2:
            return 0.0
        #    return math.acos(v1.dot(v2))
        # Apparently this is more accurate for angles near 0 or PI:
        # See:
        #   http://www.mathworks.com/matlabcentral/newsreader/view_thread/151925
        return math.atan2(v1.cross(v2), v1.dot(v2))

    def ccw_angle2(self, p1: TPoint, p2: TPoint) -> float:
        """The counterclockwise angle formed by p1->self->p2.

        Args:
            p1: First point as 2-tuple (x, y).
            p2: Second point as 2-tuple( x, y).

        Returns:
            An angle in radians between 0 and 2*math.pi.
        """
        a = self.angle2(p1, p2)
        return util.normalize_angle(a, center=math.pi)

    def bisector(self, p1: TPoint, p2: TPoint, mag: float = 1.0) -> P:
        """The bisector between the angle formed by p1->self->p2.

        Args:
            p1: First point as 2-tuple (x, y).
            p2: Second point as 2-tuple (x, y).
            mag: Optional magnitude. Default is 1.0 (unit vector).

        Returns:
            A vector with origin at `self` with magnitude `mag`.
        """
        a1 = (P(p1) - self).angle()
        a2 = (P(p2) - self).angle()
        a3 = (a1 + a2) / 2
        return self + P.from_polar(mag, a3)

    def distance(self, p: TPoint) -> float:
        """Euclidean distance from this point to another point.

        Args:
            p: The other point as a 2-tuple (x, y).

        Returns:
            The Euclidean distance.
        """
        return math.hypot(self[0] - p[0], self[1] - p[1])

    def distance2(self, p: TPoint) -> float:
        """Euclidean distance squared to other point.

        This can be used to compare distances without the
        expense of a sqrt.
        """
        a: float = self[0] - p[0]
        b: float = self[1] - p[1]
        return (a * a) + (b * b)

    def distance_to_line(self, p1: TPoint, p2: TPoint) -> float:
        """Distance to line.

        Euclidean distance from this point to it's normal projection
        on a line that intersects the given points.

        See:
            http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
            http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/

        Args:
            p1: First point on line.
            p2: Second point on line.

        Returns:
            Normal distance to line.
        """
        p1 = P(p1)
        p2 = P(p2)
        v1 = p2 - p1  # Normalize the line segment
        seglen = v1.length()  # Segment length
        if seglen < const.EPSILON:  # Degenerate line segment...?
            return self.distance(
                p1
            )  # TBD: This should probably be undefined...

        v2 = p1 - self
        return v1.cross(v2) / seglen

    def normal_projection(self, p: TPoint) -> float:
        """Unit distance to normal projection of point.

        The unit distance `mu` from the origin that corresponds to
        the projection of the specified point on to the line described by
        this vector.

        Args:
            p: A point as 2-tuple (x, y).
        """
        vlen2 = self.length2()
        if vlen2 < const.EPSILON2:
            return 0  # Degenerate case where the vector has zero length
        return self.dot(p) / vlen2

    def inside_triangle(self, a: TPoint, b: TPoint, c: TPoint) -> bool:
        """Test if this point lies inside the triangle A->B->C.

        Where ABC is clockwise or counter-clockwise.

        See:
            http://www.sunshine2k.de/stuff/Java/PointInTriangle/PointInTriangle.html

        Args:
            a: First point of triangle as 2-tuple (x, y)
            b: Second point of triangle as 2-tuple (x, y)
            c: Third point of triangle as 2-tuple (x, y)

        Returns:
            True if this point lies within the triangle ABC.
        """
        # Using barycentric coordinates
        v1 = P(b[0] - a[0], b[1] - a[1])
        v2 = P(c[0] - a[0], c[1] - a[1])
        v3 = P(self[0] - a[0], self[1] - a[1])
        det = v1.cross(v2)
        s = v1.cross(v3) / det
        t = v2.cross(v3) / det
        return bool(s >= 0 and t >= 0 and (s + t) <= 1)

    def winding(self, p2: TPoint, p3: TPoint) -> float:
        """Winding direction.

        Determine the direction defined by the three points
        p1->p2->p3. `p1` being this point.

        Args:
            p2: Second point as 2-tuple (x, y).
            p3: Third point as 2-tuple (x, y).

        Returns:
            Positive if self->p2->p3 is clockwise (right),
            negative if counterclockwise (left),
            zero if points are colinear.
        """
        return (P(p2) - self).cross(P(p3) - self)

    def transform(self, matrix: TMatrix) -> P:
        """Apply transform matrix to this vector.

        Returns:
            A copy of this point with the transform matrix applied to it.
        """
        return P(transform2d.matrix_apply_to_point(matrix, self))

    def rotate(self, angle: float, origin: TPoint | None = None) -> P:
        """Return a copy of this point rotated about the origin by `angle`."""
        if const.is_zero(angle):
            return P(self)  # just return a copy if no actual rotation
        return self.transform(transform2d.matrix_rotate(angle, origin))

    def to_svg(self, scale: float = 1) -> str:
        """SVG string representation."""
        ff = util.float_formatter()
        return f'{ff(self.x * scale)},{ff(self.y * scale)}'

    def copysign(self, v: tuple[float, float]) -> P:
        """Return a new point with x,y having the same sign as `v`.

        Where p.x value is magnitude self.x with sign of v[0],
        and p.y value is magnitude self.y with sign of v[1].
        """
        return P(math.copysign(self.x, v[0]), math.copysign(self.y, v[1]))

    def __eq__(self, other: object) -> bool:
        """Compare for equality.

        Uses EPSILON to compare point values so that spatial hash tables
        and other geometric comparisons work as expected.
        There may be cases where an exact compare is necessary but for
        most purposes (like collision detection) this works better.

        See:
            almost_equal()
        """
        return isinstance(other, Sequence) and almost_equal(self, other)

    # def __ne__(self, other: object) -> bool:
    #    """Compare for inequality."""
    #    return not self == other

    def __bool__(self) -> bool:
        """Return True if this is not a null vector.

        See:
            P.is_zero()
        """
        return not self.is_zero()

    def __neg__(self) -> P:
        """Return the unary negation of the vector (-x, -y)."""
        return P(-self[0], -self[1])

    def __add__(self, other: TPoint | float) -> P:  # type: ignore [override]
        """Add a scalar or another vector to this vector.

        Args:
            other: The vector or scalar to add.

        Returns:
            A vector (point).
        """
        if isinstance(other, Sequence):
            return P(self[0] + float(other[0]), self[1] + float(other[1]))

        n = float(other)
        return P(self[0] + n, self[1] + n)

    __iadd__ = __add__

    def __sub__(self, other: TPoint | float) -> P:
        """Subtract a scalar or another vector from this vector.

        Args:
            other: The vector or scalar to subtract.

        Returns:
            A vector (point).
        """
        if isinstance(other, Sequence):
            return P(self[0] - float(other[0]), self[1] - float(other[1]))

        n = float(other)
        return P(self[0] - n, self[1] - n)

    __isub__ = __sub__

    def __mul__(self, other: object) -> P:
        """Multiply the vector by a scalar.

        This operation is undefined for any other type
        besides float since it doesn't make geometric sense.
        Otherwise see dot() or cross() instead.

        Args:
            other: The scalar to multiply by.
        """
        if isinstance(other, SupportsFloat):
            t = float(other)
            return P(self[0] * t, self[1] * t)
        raise ValueError

    __rmul__ = __mul__  # type: ignore [assignment]
    __imul__ = __mul__  # type: ignore [assignment]

    def __truediv__(self, other: object) -> P:
        """Divide the vector by a scalar.

        Args:
            other: A scalar value to divide by.
        """
        if isinstance(other, SupportsFloat):
            t = float(other)
            return P(self[0] / t, self[1] / t)
        raise ValueError

    __idiv__ = __div__ = __itruediv__ = __truediv__

    def __floordiv__(self, other: object) -> P:
        """Divide the vector by a scalar, rounding down.

        Args:
            other: The value to divide by.
        """
        if isinstance(other, SupportsFloat):
            t = float(other)
            return P(self[0] // t, self[1] // t)
        raise ValueError

    __ifloordiv__ = __floordiv__

    def __pos__(self) -> P:
        """Identity."""
        return self

    def __abs__(self) -> float:
        """Compute the absolute magnitude of the vector."""
        return self.length()

    def __str__(self) -> str:
        """Concise string representation."""
        return (
            f'({self[0]:.{const.EPSILON_PRECISION}f},'
            f' {self[1]:.{const.EPSILON_PRECISION}f})'
        )

    def __repr__(self) -> str:
        """Precise string representation."""
        return f'P({self[0]!r}, {self[1]!r})'

    def __hash__(self) -> int:
        """Calculate a spatial hash value for this point.

        Can be used for basic collision detection.
        Uses the precision specified by EPSILON to round off coordinate values.

        See:
            http://www.beosil.com/download/CollisionDetectionHashing_VMV03.pdf
        """
        # The coordinate values are first rounded down to the current
        # level of precision (see EPSILON) so that floating point
        # artifacts and small differences in spatial distance
        # (spatial jitter) are filtered out.
        # This seems to work pretty well in practice.
        # a = int(round(self[0], const.EPSILON_PRECISION)) * 73856093
        # b = int(round(self[1], const.EPSILON_PRECISION)) * 83492791
        # a = int(round(self[0], const.EPSILON_PRECISION) * _HASH_PRIME_X)
        # b = int(round(self[1], const.EPSILON_PRECISION) * _HASH_PRIME_Y)

        repsilon: float = const.REPSILON
        a: int = round(self[0] * repsilon * const.HASH_PRIME_X)
        b: int = round(self[1] * repsilon * const.HASH_PRIME_Y)

        # Modulo a large prime that is less than max signed int.
        # The intent is to minimize collisions by creating a better
        # distribution over the long integer range.
        return (a ^ b) % const.HASH_SIZE

        # TODO: Revisit Rob Jenkins or Thomas Wang's integer hash functions
        # See:
        # https://web.archive.org/web/20071223173210/http://www.concentric.net/~Ttwang/tech/inthash.htm
        # http://burtleburtle.net/bob/hash/doobs.html
        # http://burtleburtle.net/bob/c/lookup3.c
        # http://burtleburtle.net/bob/hash/integer.html

    mag = None
    normalized = None
    perpendicular = None


# Make some method aliases to be compatible with various Point implementations
#: Alias of :method:`P.length()`
P.mag = P.length  # type: ignore [assignment]
#: Alias of :method:`P.unit()`
P.normalized = P.unit  # type: ignore [assignment]
#: Alias of :method:`P.normal()`
P.perpendicular = P.normal  # type: ignore [assignment]
