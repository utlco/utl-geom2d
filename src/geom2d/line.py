"""Basic 2D line/segment geometry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from . import const, util

# from .box import Box
from .point import P, TPoint

if TYPE_CHECKING:
    from typing_extensions import Self

    from .transform2d import TMatrix

TLine = Sequence[Sequence[float]]  # Generic input type


# namedtuple('Line', 'p1, p2')):
class Line(tuple[P, P]):  # noqa: SLOT001
    """Two dimensional immutable line segment defined by two points.

    Args:
        p1: Start point as 2-tuple (x, y).
        p2: End point as 2-tuple (x, y).
    """

    # __slots__ = ()

    def __new__(
        cls, p1: TPoint | Sequence[TPoint], p2: TPoint | None = None
    ) -> Self:
        """Create new line segment from points."""
        if p2 is None:
            return super().__new__(
                cls, (P(p1[0]), P(p1[1]))  # type: ignore [type-var, arg-type]
            )
        return super().__new__(
            cls, (P(p1), P(p2))  # type: ignore [type-var, arg-type]
        )

    @staticmethod
    def from_polar(startp: TPoint, length: float, angle: float) -> Line:
        """Create a Line given a start point, magnitude (length), and angle."""
        return Line(startp, P.from_polar(length, angle) + startp)

    def to_polar(self) -> tuple[float, float]:
        """Convert this line segment to polar coordinates.

        Returns:
            A tuple containing (length, angle)
        """
        return (self.length(), self.angle())

    @property
    def p1(self) -> P:
        """The start point of this line segment."""
        return self[0]

    @property
    def p2(self) -> P:
        """The end point of this line segment."""
        return self[1]

    def length(self) -> float:
        """Return the length of this line segment."""
        return (self.p2 - self.p1).length()

    def slope(self) -> float:
        """Return the slope of this line.

        If the line is vertical a NaN is returned.
        """
        dy = self.p2.y - self.p1.y
        dx = self.p2.x - self.p1.x
        if const.is_zero(dy):
            # Horizontal line
            return 0.0
        if const.is_zero(dx):
            # Vertical line
            return float('nan')
        return dy / dx

    def slope_intercept(self) -> tuple[float, float]:
        """The slope-intercept equation for this line.

        Where the equation is of the form:
        `y = mx + b`, where `m` is the slope and `b` is the `y` intercept.

        Returns:
            Slope and intercept as 2-tuple (m, b)
        """
        m = self.slope()
        b = (m * -self.p1.x) + self.p1.y
        return (m, b)

    def general_equation(self) -> tuple[float, float, float]:
        """Compute the coefficients of the general equation of this line.

        Where the equation is of the form
        `ax + by - c = 0`.

        See:
        http://www.cut-the-knot.org/Curriculum/Calculus/StraightLine.shtml

        Returns:
            A 3-tuple (a, b, c)
        """
        a = self.p1.y - self.p2.y
        b = self.p2.x - self.p1.x
        # c = self.p1.x * self.p2.y - self.p2.x * self.p1.y
        c = self.p1.cross(self.p2)
        return (a, b, c)

    def angle(self) -> float:
        """The angle of this line segment in radians."""
        return (self.p2 - self.p1).angle()

    start_tangent_angle = angle
    end_tangent_angle = angle

    def transform(self, matrix: TMatrix) -> Line:
        """A copy of this line with the transform matrix applied to it."""
        return Line(self[0].transform(matrix), self[1].transform(matrix))

    def midpoint(self) -> P:
        """The midpoint of this line segment."""
        # return P((self.p1.x + self.p2.x) / 2, (self.p1.y + self.p2.y) / 2)
        return (self.p1 + self.p2) * 0.5

    def bisector(self) -> Line:
        """Perpendicular bisector line.

        Essentially this line segment rotated 90deg about its midpoint.

        Returns:
            A line that is perpendicular to and passes through
            the midpoint of this line.
        """
        midp = self.midpoint()
        p1 = self.p1 - midp
        p2 = self.p2 - midp
        bp1 = midp + P(p1.y, -p1.x)
        bp2 = midp + P(p2.y, -p2.x)
        return Line(bp1, bp2)

    #    def angle_bisector(self, line2, length):
    #        """Return a line that bisects the angle formed by two lines that
    #        share a start point.
    #        This will raise an exception if the lines do not intersect at the
    #        start point.
    #        """
    #        if self.p1 != line2.p1:
    #            raise Exception('Line segments must share a start point.')
    #        angle = self.p2.angle2(self.p1, line2.p2) / 2
    #        return Line.from_polar(self.p2, length, angle)

    def offset(self, distance: float) -> Line:
        """Offset of this line segment.

        Args:
            distance: The distance to offset the line by.

        Returns:
            A line segment parallel to this one and offset by `distance`.
            If offset is < 0 the offset line will be to the right of this line,
            otherwise to the left. If offset is zero or the line segment length
            is zero then this line is returned.
        """
        length = self.length()
        if const.is_zero(distance) or const.is_zero(length):
            return self
        u = distance / length
        v1 = (self.p2 - self.p1) * u
        p1 = v1.normal() + self.p1
        v2 = (self.p1 - self.p2) * u
        p2 = v2.normal(left=False) + self.p2
        return Line(p1, p2)

    def mu(self, p: TPoint) -> float:
        """Unit distance of colinear point.

        The unit distance from the first end point of this line segment
        to the specified collinear point. It is assumed that the
        point is collinear, but this is not checked.
        """
        return self.p1.distance(p) / self.length()

    def subdivide(self, mu: float) -> tuple[Line, Line]:
        """Subdivide this line.

        Creates two lines at the given unit distance from the start point.

        Args:
            mu: location of subdivision, where 0.0 < `mu` < 1.0

        Returns:
            A tuple containing two Lines.
        """
        assert 0.0 < mu < 1.0
        p = self.point_at(mu)
        return (Line(self.p1, p), Line(p, self.p2))

    def point_at(self, mu: float) -> P:
        """Point at unit distance.

        The point that is unit distance `mu` from this segment's
        first point. The segment's first point would be at `mu=0.0` and the
        second point would be at `mu=1.0`.

        Args:
            mu: Unit distance from p1

        Returns:
            The point at `mu`
        """
        return self.p1 + ((self.p2 - self.p1) * mu)

    def normal_projection(self, p: P) -> float:
        """Unit distance to normal projection of point.

        The unit distance `mu` from this segment's first point that
        corresponds to the projection of the specified point on to this line.

        Args:
            p: point to project on to line

        Returns:
            A value between 0.0 and 1.0 if the projection lies between
            the segment endpoints.
            The return value will be < 0.0 if the projection lies south of the
            first point, and > 1.0 if it lies north of the second point.
        """
        # Check for degenerate case where endpoints are coincident
        if self.p1 == self.p2:
            return 0
        v1 = self.p2 - self.p1
        return v1.normal_projection(p - self.p1)

    def normal_projection_point(self, p: TPoint, segment: bool = False) -> P:
        """Normal projection of point to line.

        Args:
            p: point to project on to line
            segment: if True and if the point projection lies outside
                the two end points that define this line segment then
                return the closest endpoint. Default is False.

        Returns:
            The point on this line segment that corresponds to the projection
            of the specified point.
        """
        v1 = self.p2 - self.p1
        x, y = p
        u = v1.normal_projection(
            (x - self.p1[0], y - self.p1[1])
        )  # p - self.p1
        if segment:
            if u <= 0:
                return self.p1
            if u >= 1.0:
                return self.p2
        return self.p1 + v1 * u

    def distance_to_point(self, p: TPoint, segment: bool = False) -> float:
        """Distance from line to point.

        The Euclidean distance from the specified point and
        its normal projection on to this line or segment.

        See http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
        http://paulbourke.net/geometry/pointlineplane/

        Args:
            p: point to project on to line
            segment: if True and if the point projection lies outside
                the two end points that define this line segment then
                return the shortest distance to either of the two endpoints.
                Default is False.
        """
        return self.normal_projection_point(p, segment).distance(p)

    def intersection_mu(
        self,
        other: TLine,
        segment: bool = False,
        seg_a: bool = False,
        seg_b: bool = False,
    ) -> float | None:
        """Line intersection.

        http://paulbourke.net/geometry/pointlineplane/
        and http://mathworld.wolfram.com/Line-LineIntersection.html
        and
        https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
        (second SO answer)

        Args:
            other: line to test for intersection. A 4-tuple containing
                line endpoints.
            segment: if True then the intersection point must lie on both
                segments.
            seg_a: If True the intersection point must lie on this
                line segment.
            seg_b: If True the intersection point must lie on the other
                line segment.

        Returns:
            The unit distance from the segment starting point to the
            point of intersection if they intersect. Otherwise None
            if the lines or segments do not intersect.
        """
        if segment:
            seg_a = True
            seg_b = True

        x1, y1 = self[0]
        x2, y2 = self[1]
        x3, y3 = other[0]
        x4, y4 = other[1]

        a = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
        b = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

        if abs(denom) < const.EPSILON:  # Lines are parallel ?
            # # Lines are coincident? return segment midpoint by default
            # if abs(a) < const.EPSILON and abs(b) < const.EPSILON:
            #    return 0.5
            return None

        mu_a = a / denom
        mu_b = b / denom
        # if segment and (mua < 0.0 or mua > 1.0 or mub < 0.0 or mub > 1.0):
        #    if segment and (mua < -const.EPSILON or mua > 1.0 + const.EPSILON
        #                 or mub < -const.EPSILON or mub > 1.0 + const.EPSILON):
        mu_min = -const.EPSILON
        mu_max = 1.0 + const.EPSILON
        if (seg_a and (mu_a < mu_min or mu_a > mu_max)) or (  # noqa: PLR0916
            seg_b and (mu_b < mu_min or mu_b > mu_max)
        ):
            # The intersection lies outside the line segments
            return None
        return mu_a

    def intersection(
        self,
        other: TLine,
        segment: bool = False,
        seg_a: bool = False,
        seg_b: bool = False,
    ) -> P | None:
        """Intersection point (if any) of this line and another line.

        See:
            <http://paulbourke.net/geometry/pointlineplane/>
            and <http://mathworld.wolfram.com/Line-LineIntersection.html>
            and <http://geomalgorithms.com/a05-_intersect-1.html>

        Args:
            other: line to test for intersection. A 4-tuple containing
                line endpoints.
            segment: if True then the intersection point must lie on both
                segments.
            seg_a: If True the intersection point must lie on this
                line segment.
            seg_b: If True the intersection point must lie on the other
                line segment.

        Returns:
            A point if they intersect otherwise None.
        """
        mu = self.intersection_mu(other, segment, seg_a, seg_b)
        if mu is None:
            return None
        return self.point_at(mu)

    def intersects(self, other: TLine, segment: bool = False) -> bool:
        """Return True if this segment intersects another segment."""
        return self.intersection_mu(other, segment=segment) is not None
        # See also: http://algs4.cs.princeton.edu/91primitives/
        # for slightly more efficient method.

    # Use __eq__
    # def is_coincident(self, other, tolerance=None):
    #     """Return True if this line segment is
    #     coincident with another segment.
    #
    #     Args:
    #         other (tuple): Another line segment as a 2-tuple
    #             of end point tuples.
    #         tolerance (float): A floating point comparison tolerance.
    #             Default is geom.const.EPSILON.
    #     """
    #     if tolerance is None:
    #         tolerance = const.EPSILON
    #     return self.p1.almost_equal(
    #         other[0], tolerance
    #     ) and self.p2.almost_equal(other[1], tolerance)

    def is_parallel(self, other: TLine, inline: bool = False) -> bool:
        """Determine if this line segment is parallel with another line.

        Args:
            other (tuple): The other line as a tuple of two points.
            inline (bool): The other line must also be inline
                with this segment.

        Returns:
            bool: True if the other segment is parallel. If `inline` is
                True then the other segment must also be inline.
                Otherwise False.
        """
        x1, y1 = self[0]
        x2, y2 = self[1]
        x3, y3 = other[0]
        x4, y4 = other[1]

        a = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
        b = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

        if abs(denom) < const.EPSILON:
            if inline:
                return abs(a) < const.EPSILON and abs(b) < const.EPSILON
            return True
        return False

    def extend(self, amount: float, from_midpoint: bool = False) -> Line:
        """Extend/shrink line.

        A line segment that is longer (or shorter) than this line by
        `amount` amount.

        Args:
            amount: The distance to extend the line. The line length will
                be increased by this amount. If `amount` is less than zero
                the length will be decreased.
            from_midpoint: Extend the line an equal amount on both ends
                relative to the midpoint. The amount length on both ends
                will be `amount`/2. Default is False.

        Returns:
            A new Line.
        """
        length = self.length()
        if length == 0.0:  # const.is_zero(length):
            raise ValueError('Cannot extend line of zero length.')
        #         x1, y1 = self[0]
        #         x2, y2 = self[1]
        #         if from_midpoint:
        #             amount /= 2
        #         dx = (x2 - x1) / length * amount
        #         dy = (y2 - y1) / length * amount
        #         if from_midpoint:
        #             x1 -= dx
        #             y1 -= dy
        #         x2 += dx
        #         y2 += dy
        #         return Line((x1, y1), (x2, y2))
        if from_midpoint:
            amount /= 2
        dxdy = (self.p2 - self.p1) * (amount / length)
        if from_midpoint:
            return Line(self.p1 - dxdy, self.p2 + dxdy)
        return Line(self.p1, self.p2 + dxdy)

    def shift(self, amount: float) -> Line:
        """Shift this segment forward or backwards by `amount`.

        Forward means shift in the direction P1->P2.

        Args:
            amount: The distance to shift the line.
                If `amount` is less than zero
                the segment will be shifted backwards.

        Returns:
            A copy of this Line shifted by the specified amount.
        """
        dxdy = (self.p2 - self.p1) * (amount / self.length())
        return Line(self.p1 + dxdy, self.p2 + dxdy)

    def which_side(self, p: TPoint, inline: bool = False) -> int:
        """Determine which side of this line a point lies.

        Args:
            p: Point to test
            inline: If True return 0 if the point is inline.
                Default is False.

        Returns:
            1 if the point lies to the left of this line else -1.
            If ``inline`` is True and the point is inline then 0.
        """
        v1 = self.p2 - self.p1
        # v2 = p - self.p1
        v2 = p[0] - self.p1.x, p[1] - self.p1.y
        cp = v1.cross(v2)
        if inline and const.is_zero(cp):
            return 0
        return 1 if cp >= 0 else -1

    def which_side_angle(self, angle: float, inline: bool = False) -> int:
        """Find which side of line determined by angle.

        Determine which side of this line lies a vector from the
        second end point with the specified direction angle.

        Args:
            angle: Angle in radians of the vector
            inline: If True return 0 if the point is inline.

        Returns:
            1 if the vector direction is to the left of this line else -1.
            If ``inline`` is True and the point is inline then 0.
        """
        # Unit vector from endpoint
        vector = P.from_polar(1.0, angle) + self.p2
        return self.which_side(vector, inline)

    def same_side(self, pt1: P, pt2: P) -> bool:
        """True if the given points lie on the same side of this line."""
        # Normalize the points first
        v1 = self.p2 - self.p1
        v2 = pt1 - self.p1
        v3 = pt2 - self.p1
        # The sign of the perp-dot product determines which side the point lies.
        c1 = v1.cross(v2)
        c2 = v1.cross(v3)
        return (c1 >= 0 and c1 >= 0) or (c1 < 0 and c2 < 0)

    def point_on_line(self, p: TPoint, segment: bool = False) -> bool:
        """True if the point lies on the line defined by this segment.

        Args:
            p: the point to test
            segment: If True, the point must be collinear
                AND lie between the two endpoints.
                Default is False.
        """
        v1 = self.p2 - self.p1
        v2 = P(p[0] - self.p1[0], p[1] - self.p1[1])
        is_collinear = const.is_zero(v1.cross(v2))
        if segment and is_collinear:
            x1, y1 = self.p1
            x2, y2 = self.p2
            return (min(x1, x2) - const.EPSILON) <= p[0] <= (
                max(x1, x2) + const.EPSILON
            ) and (min(y1, y2) - const.EPSILON) <= p[1] <= (
                max(y1, y2) + const.EPSILON
            )
        return is_collinear

    def path_reversed(self) -> Line:
        """Line segment with start and end points reversed."""
        return Line(self.p2, self.p1)

    def flipped(self) -> Line:
        """Return a Line segment flipped 180deg around the first point."""
        p2 = -(self.p2 - self.p1) + self.p1
        return Line(self.p1, p2)

    def __add__(self, other: object) -> Line:
        """Add a scalar or another vector to this line.

        This translates the line.

        Args:
            other: The vector or scalar to add.

        Returns:
            A line.
        """
        if isinstance(other, (Sequence, float, int)):
            return Line(self.p1 + other, self.p2 + other)
        raise ValueError

    __iadd__ = __add__

    def __eq__(self, other: object) -> bool:
        """Compare for segment equality in a geometric sense.

        Returns:
            True if the two line segments are coicindent otherwise False.
        """
        # Compare both directions
        if isinstance(other, Sequence) and len(self) == len(other):
            return bool(
                (self.p1 == other[0] and self.p2 == other[1])
                or (self.p1 == other[1] and self.p2 == other[0])
            )
        return False

    def __hash__(self) -> int:
        """Create a hash value for this line segment.

        The hash value will be the same if p1 and p2 are reversed.
        """
        return hash(self.p1) ^ hash(self.p2)

    def __str__(self) -> str:
        """Concise string representation."""
        return f'Line({self.p1}, {self.p2})'

    def __repr__(self) -> str:
        """Precise string representation."""
        return f'Line({self.p1!r}, {self.p2!r})'

    def to_svg_path(
        self, scale: float = 1, add_prefix: bool = True, add_move: bool = False
    ) -> str:
        """Line to SVG path string.

        Args:
            scale: Scale factor. Default is 1.
            add_prefix: Prefix with the command prefix if True.
            add_move: Prefix with M command if True.

        Returns:
            A string with the SVG path 'd' attribute
            that corresponds with this line.
        """
        ff = util.float_formatter()

        prefix = 'L ' if add_prefix or add_move else ''
        if add_move:
            p1 = self.p1 * scale
            prefix = f'M {ff(p1.x)},{ff(p1.y)} {prefix}'

        p2 = self.p2 * scale
        return f'{prefix}{ff(p2.x)},{ff(p2.y)}'
