"""Basic 2D arc geometry.

.. autosummary::
    Arc

====
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

# from geom2d import debug
from . import const, ellipse
from .const import TAU
from .line import Line
from .point import P

if TYPE_CHECKING:
    from typing_extensions import Self

    from .point import TPoint
    from .transform2d import TMatrix

if const.DEBUG:
    from . import debug


# TODO: Refactor to generalized EllipticalArc
class Arc(tuple[P, P, float, float, P]):  # noqa: SLOT001
    """Two dimensional immutable circular arc segment.

    Args:
        p1: Start point.
        p2: End point.
        radius: Arc radius.
        angle: Arc angle described by p1->center->p2.
        center: Optional center of arc. Will be computed if not specified.
            Default is None.
    """

    # __slots__ = ()

    def __new__(  # pylint: disable=too-many-arguments
        cls,
        p1: TPoint,
        p2: TPoint,
        radius: float,
        angle: float,
        center: TPoint | None = None,
    ) -> Self:
        """Create a new Arc object.

        Args:
            p1: Start point of arc.
            p2: End point of arc.
            radius: Radius of arc.
            angle: Sweep angle.
            center: Arc center. Will be calculated if None.
        """
        p1 = P(p1)
        p2 = P(p2)
        center = P(center) if center else Arc.calc_center(p1, p2, radius, angle)

        if const.DEBUG:
            # Perform a sanity check
            d1 = p1.distance(center)
            d2 = p2.distance(center)
            # Check for consistent radius
            if not const.float_eq(d1, d2):
                raise ValueError(
                    'Bad arc: '
                    f'd1={d1} != d2={d2}, '
                    f'p1={p1} p2={p2} radius={radius} center={center} '
                    f'angle={angle} center={center}'
                )
            assert const.float_eq(d1, radius)
            assert -TAU < angle < TAU
            if not const.float_eq(abs(angle), abs(center.angle2(p1, p2))):
                debug.draw_point(p1, color='#ff0000')
                debug.draw_point(p2, color='#00ff00')
                debug.draw_point(center, color='#ffff00')
                raise ValueError(
                    'Bad arc: '
                    f'angle={angle} != {center.angle2(p1, p2)} '
                    f'p1={p1} p2={p2} radius={radius} center={center} '
                    f'd1={d1} d2={d2}'
                )

        return super().__new__(
            cls,
            (p1, p2, radius, angle, center),  # type: ignore [arg-type, type-var]
        )

    @staticmethod
    def from_two_points_and_center(
        p1: TPoint, p2: TPoint, center: TPoint, large_arc: bool = False
    ) -> Arc:
        """Create an Arc given two end points and a center point.

        Since this would be ambiguous, a hint must be given as
        to which way the arc goes.

        Args:
            p1: Start point.
            p2: End point.
            center: Center of arc.
            large_arc: If True the Arc will be on the
                large side (angle > pi). Default is False.
        """
        p1, p2, center = P(p1), P(p2), P(center)
        radius = p1.distance(center)
        angle = center.angle2(p1, p2)
        if large_arc:
            angle = (-TAU if angle < 0 else TAU) - angle
        return Arc(p1, p2, radius, angle, center)

    @staticmethod
    def from_two_points_and_tangent(
        p1: TPoint, ptan: TPoint, p2: TPoint, reverse: bool = False
    ) -> Arc | None:
        """Create an Arc given two points and a tangent vector from p1->ptan.

        Args:
            p1: Start point.
            ptan: Tangent vector with origin at p1.
            p2: End point.
            reverse: Reverse the resulting arc direction if True.
                Default is False.

        Returns:
            An Arc or None if the arc parameters are degenerate
            (i.e. if the endpoints are coincident or the
            start point and tangent vector are coincident.)
        """
        p1 = P(p1)
        p2 = P(p2)

        # if p1 == p2 or p1 == p1 + ptan:
        if p1 in {p2, p1 + ptan}:
            # degenerate arc
            return None

        # The arc angle is 2 * the angle defined by the tangent and the secant.
        # See http://en.wikipedia.org/wiki/Tangent_lines_to_circles
        angle = 2 * p1.angle2(ptan, p2)
        chord_len = p1.distance(p2)
        radius = abs(chord_len / (2 * math.sin(angle / 2)))
        if reverse:
            return Arc(p2, p1, radius, -angle)
        return Arc(p1, p2, radius, angle)

    @staticmethod
    def calc_center(p1: TPoint, p2: TPoint, radius: float, angle: float) -> P:
        """Calculate the center point of an arc.

        Given two endpoints, the radius, and a central angle.

        This method is static so that it can be used by __new__.

        Args:
            p1: Start point
            p2: End point
            radius: Radius of arc
            angle: The arc's central angle

        Returns:
            The center point as a tuple (x, y).

        See:
            Thanks to Christian Blatter for this elegant solution:
            - https://people.math.ethz.ch/~blatter/
            - http://math.stackexchange.com/questions/27535/how-to-find-center-of-an-arc-given-start-point-end-point-radius-and-arc-direc
        """
        p1, p2 = P(p1), P(p2)
        if p1 == p2:  # Points coinciedent?
            return p1
        chord = Line(p1, p2)
        # distance between start and endpoint
        chord_len = chord.length()
        # Determine which side the arc center is
        sign = 1 if angle > 0 else -1
        # determine mid-point
        midp = chord.midpoint()
        # distance from center to midpoint
        c2m = math.sqrt((radius * radius) - ((chord_len * chord_len) / 4))
        # calculate the center point
        center_x = midp.x - (sign * c2m * ((p2.y - p1.y) / chord_len))
        center_y = midp.y + (sign * c2m * ((p2.x - p1.x) / chord_len))
        return P(center_x, center_y)

    @property
    def p1(self) -> P:
        """The start point of the arc."""
        return self[0]

    @property
    def p2(self) -> P:
        """The end point of the arc."""
        return self[1]

    @property
    def radius(self) -> float:
        """The radius of the arc."""
        return self[2]

    @property
    def angle(self) -> float:
        """The central angle (AKA sweep angle) of this arc.

        The sign of the angle determines its direction.
        """
        return self[3]

    @property
    def center(self) -> P:
        """The center point of this arc."""
        return self[4]

    def start_angle(self) -> float:
        """The angle from the arc center between the x axis and the first point."""
        return self.center.angle2(P(1.0, 0.0), self.p1)

    def length(self) -> float:
        """The length of this arc segment."""
        return abs(self.radius * self.angle)

    def area(self) -> float:
        """The area inside the central angle between the arc and the center."""
        radius_squared = self.radius * self.radius
        return radius_squared * abs(self.angle) / 2

    def segment_area(self) -> float:
        """The area of the arc and a straight line.

        The area of the shape limited by the arc and a straight line
        forming a chord between the two end points.
        """
        radius_squared = self.radius * self.radius
        return radius_squared * abs(self.angle - math.sin(self.angle)) / 2

    def is_clockwise(self) -> bool:
        """True if arc direction is clockwise from first point to end point."""
        return self.angle < 0

    def direction(self) -> int:
        """Return -1 if clockwise or 1 if counter-clockwise."""
        return -1 if self.angle < 0 else 1

    def start_tangent_angle(self) -> float:
        """The start direction of this arc segment in radians.

        This is the angle of a tangent vector at the arc segment's
        first point. Unlike a chord tangent angle this angle is
        from the x axis. Value is between -PI and PI.
        """
        x, y = self.center - self.p1
        if self.is_clockwise():
            return math.atan2(x, -y)
        return math.atan2(-x, y)

        # vector = (
        #    self.center - self.p1
        #    if self.is_clockwise()
        #    else self.p1 - self.center
        # )
        # print(f'vector {vector} {vector.angle()}')
        # return util.normalize_angle(vector.angle() + (math.pi / 2), center=0.0)

    def end_tangent_angle(self) -> float:
        """The end direction of this arc segment in radians.

        This is the angle of a tangent vector at the arc segment's
        end point. Value is between -PI and PI.
        """
        x, y = self.center - self.p2
        if self.is_clockwise():
            return math.atan2(x, -y)
        return math.atan2(-x, y)
        # vector = (
        #    self.center - self.p2
        #    if self.is_clockwise()
        #    else self.p2 - self.center
        # )
        # return util.normalize_angle(vector.angle() + (math.pi / 2), center=0.0)

    def height(self) -> float:
        """The distance between the chord midpoint and the arc midpoint.

        Essentially the Hausdorff distance between the chord and the arc.
        """
        chord_midpoint = Line(self.p1, self.p2).midpoint()
        return self.radius - chord_midpoint.distance(self.center)

    def transform(self, matrix: TMatrix) -> Arc:
        """Apply transform to this Arc.

        Args:
            matrix: An affine transform matrix. The arc will remain
                circular.

        Returns:
            A copy of this arc with the transform matrix applied to it.
        """
        # TODO: return an Ellipse if the scaling is not regular ?
        new_p1 = self.p1.transform(matrix)
        new_p2 = self.p2.transform(matrix)
        scale_x = matrix[0][0]
        scale_y = matrix[1][1]
        # Make sure this won't make an ellipse.
        assert abs(scale_x) == abs(scale_y)
        angle = self.angle
        # If arc is mirrored then swap direction of angle
        if scale_x * scale_y < 0:
            angle = -angle
        # TODO: possibly find a more efficient way to scale radius...
        chord_len2 = self.p1.distance2(self.p2)
        new_chord_len2 = new_p1.distance2(new_p2)
        new_radius = self.radius * (new_chord_len2 / chord_len2)
        # Center will be recomputed...
        return Arc(new_p1, new_p2, new_radius, angle)

    def offset(self, distance: float, preserve_center: bool = True) -> Arc:
        """Return a copy of this Arc that is offset by `distance`.

        If offset is < 0 the offset line will be towards the center
        otherwise to the other side of this arc.
        The central angle will be preserved.

        Args:
            distance: The distance to offset the line by.
            preserve_center: If True the offset arc will have the same
                center point as this one. Default is True.

        Returns:
            An Arc offset by `distance` from this one.
        """
        if preserve_center:
            new_radius = self.radius + distance
            if new_radius < 0 or const.is_zero(new_radius):
                raise ValueError(
                    f'Cannot offset arc of radius {self.radius} by {distance}.'
                )
            line1 = Line(self.center, self.p1).extend(distance)
            line2 = Line(self.center, self.p2).extend(distance)
            return Arc(
                line1.p2,
                line2.p2,
                self.radius + distance,
                self.angle,
                self.center,
            )

        # Just copy and translate.
        midp = Line(self.p1, self.p2).midpoint()
        dxdy = (midp - self.center).unit() * distance
        offset_p1 = self.p1 + dxdy
        offset_p2 = self.p2 + dxdy
        offset_center = self.center + dxdy
        return Arc(
            offset_p1, offset_p2, self.radius, self.angle, center=offset_center
        )

    def distance_to_point(self, p: TPoint, segment: bool = True) -> float:
        """Distance from this arc to point.

        Args:
            p: The point to measure distance to
            segment: The point normal projection
                must lie on this the arc segment if True.
                Default is True.

        Returns:
            The minimum distance from this arc segment to the specified point,
            or -1 if `segment` is True and the point normal projection
            does not lie on this arc segment.
        """
        p = P(p)
        # Check for degenerate arc case
        if self.radius < const.EPSILON or self.p1 == self.p2:
            return self.p1.distance(p)
        aangle = abs(self.angle)
        if const.float_eq(aangle, math.pi):
            which_side = Line(self.p1, self.p2).which_side(p)
            is_inside_arc = (which_side == 1 and self.angle < 0) or (
                which_side == -1 and self.angle > 0
            )
        elif aangle > math.pi:
            # TODO: test this...
            phi = self.center.ccw_angle2(self.p1, p)
            if self.angle < 0.0:
                phi = TAU - phi
            is_inside_arc = (abs(self.angle) - abs(phi)) > 0.0
        else:
            # If the point->circle projection is outside the arc segment
            # then return the distance closest to either endpoint.
            # Note: see http://www.blackpawn.com/texts/pointinpoly/default.html
            # http://www.sunshine2k.de/stuff/Java/PointInTriangle/PointInTriangle.html
            # http://blogs.msdn.com/b/rezanour/archive/2011/08/07/barycentric-coordinates-and-point-in-triangle-tests.aspx
            # Using barycentric coordinates
            v1 = self.p1 - self.center
            v2 = self.p2 - self.center
            v3 = p - self.center
            determinant = v1.cross(v2)
            s = v1.cross(v3) / determinant
            t = v3.cross(v2) / determinant
            is_inside_arc = s >= 0.0 and t >= 0.0
        if is_inside_arc:
            # Distance from arc center to point.
            p2center = self.center.distance(p)
            # Distance from point to edge of arc.
            distance = abs(p2center - self.radius)
        elif segment:
            return -1
        else:
            # Otherwise distance to closest arc segment endpoint.
            distance = min(self.p1.distance(p), self.p2.distance(p))
        return distance

    def which_side_angle(self, angle: float, inline: bool = False) -> int:
        """Which side of endpoint tangent is vector.

        Determine which side of a line tangent to the end point of
        this arc lies a vector from the
        end point with the specified direction angle.

        Args:
            angle: Angle in radians of the vector
            inline: If True return 0 if the point is inline.

        Returns:
            1 if the vector direction is to the left of arc tangent else -1.
            If ``inline`` is True and the point is inline then 0.
        """
        vector1 = P.from_polar(1.0, self.end_tangent_angle()) + self.p2
        vector2 = P.from_polar(1.0, angle) + self.p2
        return Line(self.p2, vector1).which_side(vector2, inline)

    def mu(self, p: TPoint) -> float:
        """Unit distance from first point to point on arc.

        The unit distance from the first point of this arc segment
        to the specified point on the arc segment.

        Args:
            p: A point on this arc segment.

        Returns:
            The unit distance `mu` where `mu` >=0 and <= 1.0.
            If `p` is does not lie on
            this arc segment `mu` may be < 0 or > 1.
        """
        return self.center.angle2(self.p1, p) / self.angle

    def point_at(self, mu: float) -> P:
        """Point on arc at given unit distance.

        Args:
            mu: Unit distance along central arc from first point.

        Returns:
            The point at unit distance :mu: along this arc
            from the start point.
        """
        return self.point_at_angle(abs(self.angle) * mu)  # type: ignore [return-value]

    def midpoint(self) -> P:
        """The point at the middle of the arc segment."""
        return self.point_at(0.5)

    def subdivide(self, mu: float) -> tuple[Arc, Arc] | tuple[Arc]:
        """Subdivide this arc at unit distance :mu: from the start point.

        Args:
            mu: Unit distance along central arc from first point.

        Returns:
            A tuple containing one or two Arc objects.
            If `mu` is zero or 1 then a tuple containing just
            this arc is returned.
        """
        if mu < const.EPSILON or mu >= 1.0:
            return (self,)
        return self.subdivide_at_angle(abs(self.angle) * mu)

    def subdivide_at_angle(self, angle: float) -> tuple[Arc, Arc] | tuple[Arc]:
        """Split this arc into two arcs at angle.

        At the point on this arc given
        by the specified positive arc angle (0-2pi) from the start point.

        Args:
            angle: A central angle the arc start point, in radians.

        Returns:
            A tuple containing one or two Arc objects. If the
            angle is zero or greater than this arc's angle then
            a tuple containing just this arc will be returned.
        """
        if angle < const.EPSILON or angle >= self.angle:
            return (self,)
        angle2 = abs(self.angle) - angle
        p: P = self.point_at_angle(angle)  # type: ignore [assignment]
        if self.angle < 0:
            angle = -angle
            angle2 = -angle2
        arc1 = Arc(self.p1, p, self.radius, angle, self.center)
        arc2 = Arc(p, self.p2, self.radius, angle2, self.center)
        return (arc1, arc2)

    def subdivide_at_point(self, p: TPoint) -> tuple[Arc, Arc] | tuple[Arc]:
        """Split this arc into two arcs at the specified point.

        Args:
            p: A point on this arc.

        Returns:
            A tuple containing one or two Arc objects.
        """
        angle = self.center.angle2(self.p1, p)
        if const.is_zero(angle) or const.float_eq(angle, self.angle):
            return (self,)
        arc1 = Arc(self.p1, p, self.radius, angle, self.center)
        arc2 = Arc(p, self.p2, self.radius, self.angle - angle, self.center)
        return (arc1, arc2)

    def point_at_angle(self, angle: float, segment: bool = False) -> P | None:
        """Get a point on this arc given an angle.

        Args:
            angle: A central angle from start point.
            segment: The point must lie on the arc segment if True.
                Default is False.

        Returns:
            The point on this arc given the specified angle from
            the start point of the arc segment. If ``segment`` is True
            and the point would lie outside the segment then None.
            Otherwise,
            if `angle` is negative return the first point, or
            if `angle` is greater than the central angle then return the
            end point.
        """
        if segment and (angle < 0.0 or angle > abs(self.angle)):
            return None
        if angle <= 0.0:
            return self.p1
        if angle >= abs(self.angle):
            return self.p2
        p1_angle = (self.p1 - self.center).angle()
        angle = p1_angle - angle if self.angle < 0 else p1_angle + angle
        x = self.center.x + self.radius * math.cos(angle)
        y = self.center.y + self.radius * math.sin(angle)
        return P(x, y)

    def point_on_arc(self, p: TPoint) -> bool:
        """Determine if a point lies on this arc.

        Args:
            p: Point to test.

        Returns:
            True if the point lies on this arc, otherwise False.
        """
        # Distance from center to point
        distance_c2p = self.center.distance(p)
        # First test if the point lies on a circle defined by this arc.
        if const.float_eq(self.radius, distance_c2p):
            # Then see if it lies between the two end points.
            # By checking if this arcs chord intersects with
            # a line from the center to the point.
            # TODO: probably a more efficient way...
            chord = Line(self.p1, self.p2)
            pline = Line(self.center, p)
            intersection = chord.intersection(pline, segment=True)
            angle_is_major = abs(self.angle) > math.pi
            return (angle_is_major and intersection is None) or (
                not angle_is_major and intersection is not None
            )
        return False

    def normal_projection_point(
        self, p: TPoint, segment: bool = False
    ) -> P | None:
        """Normal project of point p to this arc."""
        ray = Line(self.center, p)
        intersections = self.intersect_line(ray, on_arc=segment)
        if intersections:
            return intersections[0]
        return None

    def intersect_line(  # too-many-locals
        self, line: Line, on_arc: bool = False, on_line: bool = False
    ) -> list[P]:
        """Find the intersection (if any) of this Arc and a Line.

        See:
            http://mathworld.wolfram.com/Circle-LineIntersection.html

        Args:
            line: A line defined by two points (as a 2-tuple of 2-tuples).
            on_arc: If True the intersection(s) must lie on the arc
                between the two end points. Default is False.
            on_line: If True the intersection(s) must lie on the line
                segment between its two end points. Default is False.

        Returns:
            A list containing zero, one, or two intersections as point
            (x, y) tuples.
        """
        # pylint: disable=too-many-locals
        lp1 = line.p1 - self.center
        lp2 = line.p2 - self.center
        dx = lp2.x - lp1.x
        dy = lp2.y - lp1.y
        dr2 = dx * dx + dy * dy
        # Determinant
        det = lp1.cross(lp2)
        # Discrimanant
        dsc = ((self.radius * self.radius) * dr2) - (det * det)
        intersections = []
        if const.is_zero(dsc):
            # Line is tangent so one intersection
            intersections.append(line.normal_projection_point(self.center))
        elif dsc > 0:
            # Two intersections - find them
            sgn = -1 if dy < 0 else 1
            dscr = math.sqrt(dsc)
            x1 = ((det * dy) + ((sgn * dx) * dscr)) / dr2
            x2 = ((det * dy) - ((sgn * dx) * dscr)) / dr2
            y1 = ((-det * dx) + (abs(dy) * dscr)) / dr2
            y2 = ((-det * dx) - (abs(dy) * dscr)) / dr2
            p1 = P(x1, y1) + self.center
            p2 = P(x2, y2) + self.center
            if (not on_arc or self.point_on_arc(p1)) and (
                not on_line or line.point_on_line(p1)
            ):
                intersections.append(p1)
            if (not on_arc or self.point_on_arc(p2)) and (
                not on_line or line.point_on_line(p2)
            ):
                intersections.append(p2)
        return intersections
        # pylint: enable=too-many-locals

    def intersect_arc(self, arc: Arc, on_arc: bool = False) -> list[P]:
        """The intersection (if any) of this Arc and another Arc.

        See:
            http://mathworld.wolfram.com/Circle-CircleIntersection.html

        Args:
            arc: An Arc.
            on_arc: If True the intersection(s) must lie on both arc
                segments, otherwise the arcs are treated as circles for
                purposes of computing the intersections. Default is False.

        Returns:
            A list containing zero, one, or two intersections.
        """
        intersections = list(
            ellipse.intersect_circle(
                self.center, self.radius, arc.center, arc.radius
            )
        )
        # Delete intersections that don't lie on the arc segments.
        if on_arc and intersections:
            if not (
                self.point_on_arc(intersections[0])
                and arc.point_on_arc(intersections[0])
            ):
                del intersections[0]
            if intersections and not (
                self.point_on_arc(intersections[-1])
                and arc.point_on_arc(intersections[-1])
            ):
                del intersections[-1]
        return intersections

    def path_reversed(self) -> Arc:
        """A copy of this Arc with direction reversed."""
        return Arc(self.p2, self.p1, self.radius, -self.angle, self.center)

    def __str__(self) -> str:
        """Convert this Arc to a readable string."""
        return (
            f'Arc({self.p1}, {self.p2}, '
            f'{self.radius:.{const.EPSILON_PRECISION}f}, '
            f'{self.angle:.{const.EPSILON_PRECISION}f}, '
            f'{self.center})'
        )

    def __repr__(self) -> str:
        """Convert this Arc to a string."""
        return (
            f'Arc({self.p1!r}, {self.p2!r}, {self.radius}, '
            f'{self.angle}, {self.center!r})'
        )

    def to_svg_path(self, mpart: bool = True) -> str:
        """Arc to SVG path.

        A string with the SVG path 'd' attribute value
        that corresponds to this arc.
        """
        sweep_flag = 0 if self.angle < 0 else 1
        dpart = (
            f'A {self.radius} {self.radius} 0.0 0'
            f' {sweep_flag} {self.p2.x} {self.p2.y}'
        )
        if mpart:
            return f'M {self.p1.x} {self.p1.y} ' + dpart
        return dpart
