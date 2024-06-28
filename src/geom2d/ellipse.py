"""Two dimensional ellipse and elliptical arc."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from . import const, util
from .line import Line
from .point import P, TPoint

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from inkext.svg import SVGContext
    from transform2d import TMatrix

# TODO: refactor to named tuple in order to be orthogonal to other geom types


class Ellipse:
    """Two dimensional ellipse.

    For the parametric function the parameter `t` is the
    parametric angle (aka eccentric anomaly) from
    the semi-major axis of the ellipse before stretch and rotation
    (i.e. as if this ellipse were a circle.)

    The ellipse will be normalized so that the semi-major axis
    is aligned with the X axis (i.e. `rx` >= `ry`.) The ellipse
    rotation (phi) will be adjusted 90deg to compensate if
    necessary.

    See:
        - https://en.wikipedia.org/wiki/Ellipse
        - http://www.spaceroots.org/documents/ellipse/
        - http://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes
        - https://en.wikipedia.org/wiki/Eccentric_anomaly
    """

    p1: P
    p2: P
    rx: float
    ry: float
    phi: float
    center: P

    def __init__(
        self,
        center: TPoint,
        rx: float,
        ry: float,
        phi: float = 0.0,
    ) -> None:
        """Create a new Ellipse.

        Args:
            center: The center of the ellipse.
            rx: Semi-major axis length.
            ry: Semi-minor axis length.
            phi: Rotation angle of the ellipse.

        Raises:
            ValueError: If rx or ry is zero.
        """
        if const.is_zero(rx) or const.is_zero(ry):
            raise ValueError

        # Note: This is possibly redundant...
        rx, ry, phi = _normalize_axis(rx, ry, phi)

        self.rx = rx
        self.ry = ry
        self.phi = phi
        self.center = P(center)

        # Defaults for paths
        self.p1 = self.point_at(0)
        self.p2 = self.p1

    #    @property
    #    def p1(self) -> P:
    #        """Virtual start point.
    #
    #        This is just to be orthogonal with the rest
    #        of the geometric objects.
    #        """
    #        return self.point_at(0)
    #
    #    @property
    #    def p2(self) -> P:
    #        """Virtual end point.
    #
    #        This is just to be orthogonal with the rest
    #        of the geometric objects.
    #        """
    #        return self.point_at(0)

    def is_circle(self) -> bool:
        """True if this ellipse is a circle."""
        return const.float_eq(self.rx, self.ry)

    def angle_to_theta(self, angle: float) -> float:
        """Compute parametric angle from geometric angle.

        Args:
            angle: The geometrical angle from
                the semi-major axis and a point on the ellipse.

        Returns:
            `t` - the parametric angle - 0 < `t` < 2*PI.
        """
        if self.is_circle():
            return angle
        return math.atan2(math.sin(angle) / self.ry, math.cos(angle) / self.rx)

    def point_to_theta(self, p: TPoint) -> float:
        """Compute `t` given a point on the ellipse.

        Args:
            p: A point on the ellipse.

        Returns:
            `t` - the parametric angle - 0 < `t` < 2*PI.
        """
        theta = self.center.angle2(self.point_at(0), p)
        return self.angle_to_theta(theta)

    def point_at(self, t: float) -> P:
        """Return the point on the ellipse at `t`.

        This is the parametric function for this ellipse.

        Args:
            t: Parametric angle - 0 < t < 2*PI.

        Returns:
            A point at `t`
        """
        return _point_at(self.center, self.rx, self.ry, self.phi, t)
        p = P(self.rx * math.cos(t), self.ry * math.sin(t))
        return p.rotate(self.phi) + self.center
        cos_theta = math.cos(self.phi)
        sin_theta = math.sin(self.phi)
        cos_t = math.cos(t)
        sin_t = math.sin(t)
        x = (self.rx * cos_theta * cos_t) - (self.ry * sin_theta * sin_t)
        y = (self.rx * sin_theta * cos_t) + (self.ry * cos_theta * sin_t)
        return self.center + P(x, y)

    def point_inside(self, p: TPoint) -> bool:
        """Test if point is inside ellipse or not.

        Args:
            p: Point (x, y) to test.

        Returns:
            True if the point is inside the ellipse, otherwise False.
        """
        if self.is_circle() or const.is_zero(self.phi):
            x, y = P(p) - self.center
        else:
            # Canonicalize the point by rotating it back clockwise by phi
            x, y = (P(p) - self.center).rotate(-self.phi)
        # Point is inside if the result sign is negative
        xrx = x / self.rx
        yry = y / self.ry
        return ((xrx * xrx) + (yry * yry) - 1) < 0

    def all_points_inside(self, points: Iterable[TPoint]) -> bool:
        """Return True if all the given points are inside this ellipse."""
        return all(self.point_inside(p) for p in points)

    def focus(self) -> float:
        """The focus of this ellipse.

        Returns:
            Distance from center to focus points.
        """
        return math.sqrt(self.rx * self.rx - self.ry * self.ry)

    def focus_points(self) -> tuple[P, P]:
        """Return the two focus points.

        Returns:
            A tuple of two focus points along major axis.
        """
        d = self.focus()
        fp = P(d * math.cos(self.phi), d * math.sin(self.phi))
        return (self.center - fp, self.center + fp)

    def area(self) -> float:
        """The area of this ellipse."""
        return math.pi * self.rx * self.ry

    def eccentricity(self) -> float:
        """The eccentricity `e` of this ellipse."""
        return self.focus() / self.rx

    def curvature(self, p: TPoint) -> float:
        """The curvature at a given point."""
        x, y = p
        rx2 = self.rx * self.rx
        ry2 = self.ry * self.ry
        tmp1 = 1 / (rx2 * ry2)
        tmp2 = ((x * x) / (rx2 * rx2)) + ((y * y) / (ry2 * ry2))
        return tmp1 * math.pow(tmp2, -1.5)

    def derivative(self, t: float, d: int = 1) -> P:
        """First and second derivatives of the parametric ellipse function.

        Args:
            t: Parametric angle - 0 < t < 2*PI.
            d: 1 => First derivative, 2 => Second derivative.
                Default is 1.

        Returns:
            A 2-tuple: (dx, dy)
        """
        cos_theta = math.cos(self.phi)
        sin_theta = math.sin(self.phi)
        cos_t = math.cos(t)
        sin_t = math.sin(t)
        if d == 1:
            dx = -(self.rx * cos_theta * sin_t) - (self.ry * sin_theta * cos_t)
            dy = -(self.rx * sin_theta * sin_t) + (self.ry * cos_theta * cos_t)
        else:
            dx = -(self.rx * cos_theta * cos_t) + (self.ry * sin_theta * sin_t)
            dy = -(self.rx * sin_theta * cos_t) - (self.ry * cos_theta * sin_t)
        return P(dx, dy)

    def transform(self, _matrix: TMatrix) -> EllipticalArc:
        """Transform this using the specified affine transform matrix."""
        # TODO: implement this.
        # See:
        # http://atrey.karlin.mff.cuni.cz/projekty/vrr/doc/man/progman/Elliptic-arcs.html
        raise RuntimeError('not implemented.')


def _normalize_axis(
    rx: float, ry: float, phi: float
) -> tuple[float, float, float]:
    """Normalize radii and axis so rx is always semi-major."""
    rx = abs(rx)
    ry = abs(ry)

    if const.float_eq(rx, ry):
        rx = ry  # Mitigate possible float artifacts
    elif rx < ry:
        # Normalize semi-major axis
        rx, ry = ry, rx
        phi += math.pi / 2

    if const.is_zero(phi):
        phi = 0  # Remove possible float artifacts

    return rx, ry, phi


def _point_at(center: TPoint, rx: float, ry: float, phi: float, t: float) -> P:
    """Return the point on the ellipse at `t`.

    This is the parametric function for this ellipse.

    Args:
        center: Ellipse center point.
        rx: Semi-major radius.
        ry: Semi-minor radius.
        phi: Semi-major axis angle.
        t: Parametric angle - 0 < t < 2*PI.

    Returns:
        A point at `t`
    """
    p = P(rx * math.cos(t), ry * math.sin(t))
    return p.rotate(phi) + center


class EllipticalArc(Ellipse):
    """Two dimensional elliptical arc. A section of an ellipse.

    See:
        http://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes
    """

    start_angle: float
    sweep_angle: float
    large_arc: int
    sweep_flag: int

    def __init__(
        self,
        p1: TPoint,
        p2: TPoint,
        rx: float,
        ry: float,
        phi: float,
        large_arc: int,
        sweep_flag: int,
        start_angle: float,
        sweep_angle: float,
        center: TPoint,
    ) -> None:
        """Create an elliptical arc.

        If only center parameters or endpoint parameters are known
        the static factory methods can be used instead.

        Args:
            center: The center of the ellipse.
            p1: The start point of the arc.
            p2: The end point of the arc.
            rx: Semi-major axis length.
            ry: Semi-minor axis length.
            phi: Rotation angle from coordinate system X axis, in radians,
                of the semi-major axis.  Default is 0.
            large_arc: The large arc flag.
                0 if the arc span is less than or equal to 180 degrees,
                or 1 if the arc span is greater than 180 degrees.
            sweep_flag: The sweep flag.
                0 if the line joining center to arc sweeps through
                decreasing angles, or 1 if it sweeps through increasing angles.
            start_angle: Parametric start angle of the arc.
            sweep_angle: Parametric sweep angle of the arc.
        """
        super().__init__(center, rx, ry, phi)
        # self.center = center
        self.p1 = P(p1)
        self.p2 = P(p2)
        # self.rx = abs(rx)
        # self.ry = abs(ry)
        self.start_angle = start_angle
        self.sweep_angle = sweep_angle
        self.large_arc = large_arc
        self.sweep_flag = sweep_flag
        # self.phi = phi

    @staticmethod
    def from_center(
        center: TPoint,
        rx: float,
        ry: float,
        phi: float,
        start_angle: float,
        sweep_angle: float,
    ) -> EllipticalArc:
        """Create an elliptical arc from center parameters.

        Args:
            center: The center point of the arc.
            rx: Semi-major axis length.
            ry: Semi-minor axis length.
            start_angle: Start angle of the arc.
            sweep_angle: Sweep angle of the arc.
            phi: The angle from the X axis to the
                semi-major axis of the ellipse.

        Returns:
            An EllipticalArc
        """
        rx, ry, phi = _normalize_axis(rx, ry, phi)
        p1 = _point_at(center, rx, ry, phi, start_angle)
        p2 = _point_at(center, rx, ry, phi, start_angle + sweep_angle)
        large_arc = 1 if abs(sweep_angle) > math.pi else 0
        sweep_flag = 1 if sweep_angle > 0.0 else 0
        return EllipticalArc(
            p1,
            p2,
            rx,
            ry,
            phi,
            large_arc,
            sweep_flag,
            start_angle,
            sweep_angle,
            center,
        )

    @staticmethod
    def from_endpoints(
        p1: TPoint,
        p2: TPoint,
        rx: float,
        ry: float,
        phi: float,
        large_arc: int,
        sweep_flag: int,
    ) -> EllipticalArc | None:
        """Create an elliptical arc from SVG-style endpoint parameters.

        This will correct out of range parameters as per SVG spec.
        The center, start angle, and sweep angle will also be
        calculated.

        See:
            https://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes

        Args:
            p1: The start point of the arc.
            p2: The end point of the arc.
            rx: Semi-major axis length.
            ry: Semi-minor axis length.
            phi: The angle in radians from the X axis to the
                semi-major axis of the ellipse.
            large_arc: The large arc flag (0 or 1).
            sweep_flag: The sweep flag (0 or 1).

        Returns:
            An EllipticalArc or None if the parameters do not
            describe a valid arc.
        """
        # print(p1, p2, rx, ry, large_arc, sweep_flag, math.degrees(phi))
        p1 = P(p1)
        p2 = P(p2)

        # If the semi-major or semi-minor axes are 0 then
        # this should really be a straight line.
        if p1 == p2 or const.is_zero(rx) or const.is_zero(ry):
            return None

        # Relative midpoint of chord
        midp = (p1 - p2) / 2

        # Normalize radii and axis so rx is always semi-major
        rx, ry, phi = _normalize_axis(rx, ry, phi)

        if const.is_zero(phi):
            phi = 0  # Remove possible float artifacts
            sin_phi = 0.0
            cos_phi = 1.0
            xprime = midp[0]
            yprime = midp[1]
        else:
            # F.6.5.1 rotate chord midpoint coordinates to line up with phi
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            xprime = cos_phi * midp[0] + sin_phi * midp[1]
            yprime = -sin_phi * midp[0] + cos_phi * midp[1]

        # Precompute squares
        rx2 = rx * rx
        ry2 = ry * ry
        xprime2 = xprime * xprime
        yprime2 = yprime * yprime

        # F.6.6 Ensure radii are large enough and correct if not.
        lam2 = xprime2 / rx2 + yprime2 / ry2
        if lam2 > 1.0:
            # Arc radii too small, so scale up
            lam = math.sqrt(lam2)
            rx = lam * rx
            ry = lam * ry
            rx2 = rx * rx
            ry2 = ry * ry

        # F.6.5.2 Compute center-prime
        t1 = (rx2 * ry2) - (rx2 * yprime2) - (ry2 * xprime2)
        t2 = (rx2 * yprime2) + (ry2 * xprime2)
        t3 = math.sqrt(t1 / t2)
        sign = -1 if large_arc == sweep_flag else 1
        cxprime = sign * t3 * ((rx * yprime) / ry)
        cyprime = sign * t3 * -((ry * xprime) / rx)

        # F.6.5.3 Compute center
        if phi == 0:
            center = P(cxprime, cyprime) + (p1 + p2) / 2
        else:
            # Rotate to semi-major axis
            cx = cos_phi * cxprime + -sin_phi * cyprime
            cy = sin_phi * cxprime + cos_phi * cyprime
            center = P(cx, cy) + (p1 + p2) / 2

        # F.6.5.4 Compute start angle and sweep angles
        # Create unit vectors
        v1 = P(1, 0)
        v2 = P((xprime - cxprime) / rx, (yprime - cyprime) / ry)
        v3 = P((-xprime - cxprime) / rx, (-yprime - cyprime) / ry)
        origin = P(0.0, 0.0)

        # F.6.5.5 Compute start angle (theta)
        # start_angle = math.acos(v1.dot(v2))
        start_angle = origin.angle2(v1, v2)

        # F.6.5.6 Compute sweep angle (lambda)
        # sweep_angle = math.acos(v2.dot(v3))
        sweep_angle = util.normalize_angle(origin.angle2(v2, v3))
        if not sweep_flag and sweep_angle > 0:
            sweep_angle -= math.pi * 2
        elif sweep_flag and sweep_angle < 0:
            sweep_angle += math.pi * 2

        return EllipticalArc(
            p1,
            p2,
            rx,
            ry,
            phi,
            large_arc,
            sweep_flag,
            start_angle,
            sweep_angle,
            center,
        )

    def transform(self, _matrix: TMatrix) -> EllipticalArc:
        """Transform this using the specified affine transform matrix."""
        # TODO: implement this.
        # See:
        # http://atrey.karlin.mff.cuni.cz/projekty/vrr/doc/man/progman/Elliptic-arcs.html
        raise RuntimeError('not implemented.')

    def to_svg_path(
        self, scale: float = 1, add_prefix: bool = True, add_move: bool = False
    ) -> str:
        """EllipticalArc to SVG path string.

        See:
            https://www.w3.org/TR/SVG11/paths.html#PathDataEllipticalArcCommands

        Args:
            scale: Scale factor. Default is 1.
            add_prefix: Prefix with the command prefix if True.
                Default is True.
            add_move: Prefix with M command if True.
                Default is False.

        A string with the SVG path 'd' attribute value
        that corresponds to this arc.
        """
        ff = util.float_formatter()

        prefix = 'A ' if add_prefix or add_move else ''
        if add_move:
            p1 = self.p1 * scale
            prefix = f'M {ff(p1.x)},{ff(p1.y)} {prefix}'

        rx = self.rx * scale
        ry = self.ry * scale
        p2 = self.p2 * scale
        return (
            f'{prefix}{ff(rx)},{ff(ry)} {ff(math.degrees(self.phi))}'
            f' {self.large_arc} {self.sweep_flag} {ff(p2.x)},{ff(p2.y)}'
        )

    def __str__(self) -> str:
        """Convert this EllipticalArc to a readable string."""
        ff = util.float_formatter()
        return (
            f'EllipticalArc({self.p1}, {self.p2}, '
            f'{ff(self.rx)}, {ff(self.ry)}, {ff(self.phi)}, '
            f'{self.large_arc}, {self.sweep_flag}, '
            f'{ff(self.start_angle)}, {ff(self.sweep_angle)}, '
            f'{self.center})'
        )

    def __repr__(self) -> str:
        """Convert this EllipticalArc to a readable string."""
        return (
            f'EllipticalArc({self.p1!r}, {self.p2!r}, '
            f'{self.rx!r}, {self.ry!r}, {self.phi!r}, '
            f'{self.large_arc}, {self.sweep_flag}, '
            f'{self.start_angle!r}, {self.sweep_angle!r}, '
            f'{self.center!r})'
        )

    def __eq__(self, other: object) -> bool:
        """Compare arcs for geometric equality.

        Returns:
            True if the two arcs are the same.
        """
        if isinstance(other, EllipticalArc):
            return bool(
                self.p1 == other.p1
                and self.p2 == other.p2
                and const.float_eq(self.rx, other.rx)
                and const.float_eq(self.ry, other.ry)
                and const.angle_eq(self.phi, other.phi)
                and self.center == other.center
                and self.large_arc == other.large_arc
                and self.sweep_flag == other.sweep_flag
                and const.angle_eq(self.start_angle, other.start_angle)
                and const.angle_eq(self.sweep_angle, other.sweep_angle)
            )
        return False

    def __hash__(self) -> int:
        """Create a hash value for this arc."""
        # Just use SVG components
        rxh = round(self.rx * const.REPSILON * const.HASH_PRIME_X)
        ryh = round(self.ry * const.REPSILON * const.HASH_PRIME_Y)
        ah = round(self.phi * const.REPSILON * const.HASH_PRIME_X)
        fh = self.large_arc << 1 & self.sweep_flag
        rehash = (rxh ^ ryh ^ ah ^ fh) % const.HASH_SIZE
        return hash(self.p1) ^ hash(self.p2) ^ hash(self.center) ^ rehash


def ellipse_in_parallelogram(
    vertices: Sequence[TPoint], eccentricity: float = 1.0
) -> Ellipse:
    """Inscribe a parallelogram with an ellipse.

    See: Horwitz 2008, http://arxiv.org/abs/0808.0297

    :vertices: The four vertices of a parallelogram as a list of 2-tuples.
    :eccentricity: The eccentricity of the ellipse.
        Where 0.0 >= `eccentricity` <= 1.0.
        If `eccentricity` == 1.0 then a special eccentricity value will
        be calculated to produce an ellipse of maximal area.
        The minimum eccentricity of 0.0 will produce a circle.

    :return: A tuple containing the semi-major and semi-minor axes
        respectively.
    """
    # Determine the angle of the ellipse major axis
    axis = Line(vertices[0], vertices[2])
    major_angle = axis.angle()
    center = axis.midpoint()
    # The parallelogram is defined as having four vertices
    # O = (0,0), P = (l,0), Q = (d,k), R = (l+d,k),
    # where l > 0, k > 0, and d >= 0.
    # Diff Horwitz: h is used instead of l because it's easier to read.
    # Determine the acute corner angle of the parallelogram.
    theta = abs(P(vertices[0]).angle2(vertices[1], vertices[3]))
    nfirst = 0  # index of first point
    if theta > (math.pi / 2):
        # First corner was obtuse, use the next corner...
        theta = math.pi - theta
        nfirst = 1
        # Rotate the major angle
        major_angle += math.pi / 2
    h2 = P(vertices[nfirst]).distance(vertices[nfirst + 1])
    h = P(vertices[nfirst + 1]).distance(vertices[nfirst + 2])
    k = math.sin(theta) * h2
    d = math.cos(theta) * h2
    # Use a nice default for degenerate eccentricity values
    if eccentricity >= 1.0 or eccentricity < 0.0:
        # This seems to produce an ellipse of maximal area
        # but I don't have proof.
        v = k / 2
    else:
        # Calculate v for minimal eccentricity (a circle)
        v = k / 2 * ((d + h) ** 2 + k * k) / (k * k + d * d + h * h)
        # Then add the desired eccentricity.
        v *= 1.0 - eccentricity
    # pylint: disable=invalid-name
    A = k**3
    B = k * (d + h) ** 2 - (4 * d * h * v)
    C = -k * (k * d - 2 * h * v + k * h)
    D = -2 * k * k * h * v
    E = 2 * k * h * v * (d - h)
    F = k * h * h * v * v
    T1 = (
        (A * E * E)
        + (B * D * D)
        + (4 * F * C * C)
        - (2 * C * D * E)
        - (4 * A * B * F)
    )
    T2 = 2 * (A * B - C * C)
    T3 = math.sqrt((B - A) * (B - A) + (4 * C * C))
    # Calculate semi-major axis
    a = math.sqrt(T1 / (T2 * ((A + B) - T3)))
    # Calculate semi-minor axis
    b = math.sqrt(T1 / (T2 * ((A + B) + T3)))
    # pylint: enable=invalid-name
    return Ellipse(center, a, b, major_angle)


def intersect_circle(
    c1_center: TPoint, c1_radius: float, c2_center: TPoint, c2_radius: float
) -> tuple:
    """The intersection (if any) of two circles.

    See:
        <http://mathworld.wolfram.com/Circle-CircleIntersection.html>

    Args:
        c1_center: Center of first circle.
        c1_radius: Radius of first circle.
        c2_center: Center of second circle.
        c2_radius: Radius of second circle.

    Returns:
        A tuple containing two intersection points if the circles
        intersect. A tuple containing a single point if the circles
        are only tangentially connected. An empty tuple if the circles
        do not intersect or if they are coincident (infinite intersections).
    """
    line_c1c2 = Line(c1_center, c2_center)

    # Distance between the two centers
    dist_c1c2 = line_c1c2.length()
    if dist_c1c2 > (c1_radius + c2_radius):
        # Circles too far apart - do not intersect.
        return ()
    if dist_c1c2 < (c1_radius - c2_radius):
        # Circle inside another - do not intersect.
        return ()
    # Check for degenerate cases
    if const.is_zero(dist_c1c2):
        # Circles are coincident so the number of intersections is infinite.
        return ()  # For now this means no intersections...
    if const.float_eq(dist_c1c2, c1_radius + c2_radius):
        # Circles are tangentially connected at a single point.
        return (line_c1c2.midpoint(),)
    # Radii ** 2
    rr1 = c1_radius * c1_radius
    rr2 = c2_radius * c2_radius
    # The distance from circle centers to the radical line
    # This is the X distance from C1 to the intersections.
    dist_c1rad = ((dist_c1c2 * dist_c1c2) - rr2 + rr1) / (2 * dist_c1c2)
    # Half the length of the radical line segment.
    # I.e. half the distance between the two intersections.
    # This is the Y distance from C1 to the intersections.
    hr2 = rr1 - (dist_c1rad * dist_c1rad)
    if hr2 < 0:
        # TODO: handle this correctly - find the cases that cause this
        # print('WTF? rr1 %f  dist_c1rad %f', rr1, dist_c1rad)
        return ()
    half_rad = math.sqrt(hr2)
    # Intersection points.
    # Rotate the points so that they are normal to c1->c2
    # TODO: optimize this...
    angle_c1c2 = line_c1c2.angle()
    ip1 = P(dist_c1rad, half_rad).rotate(angle_c1c2)
    ip2 = P(dist_c1rad, -half_rad).rotate(angle_c1c2)
    c1_center = P(c1_center)
    p1 = c1_center + ip1
    p2 = c1_center + ip2
    return (p1, p2)


def intersect_circle_line(  # too-many-locals
    center: TPoint, radius: float, line: Line, on_line: bool = False
) -> list[P]:
    """Find the intersection (if any) of a circle and a Line.

    See:
        http://mathworld.wolfram.com/Circle-LineIntersection.html

    Args:
        center: Center of circle.
        radius: Radius of circle.
        line: A line defined by two points (as a 2-tuple of 2-tuples).
        on_line: If True the intersection(s) must lie on the line
            segment between its two end points. Default is False.

    Returns:
        A list containing zero, one, or two intersections as point
        (x, y) tuples.
    """
    # pylint: disable=too-many-locals
    lp1 = line.p1 - center
    lp2 = line.p2 - center
    dx = lp2.x - lp1.x
    dy = lp2.y - lp1.y
    dr2 = dx * dx + dy * dy
    # Determinant
    det = lp1.cross(lp2)
    # Discrimanant
    dsc = ((radius * radius) * dr2) - (det * det)
    intersections = []
    if const.is_zero(dsc):
        # Line is tangent so one intersection
        intersections.append(line.normal_projection_point(center))
    elif dsc > 0:
        # Two intersections - find them
        sgn = -1 if dy < 0 else 1
        dscr = math.sqrt(dsc)
        x1 = ((det * dy) + ((sgn * dx) * dscr)) / dr2
        x2 = ((det * dy) - ((sgn * dx) * dscr)) / dr2
        y1 = ((-det * dx) + (abs(dy) * dscr)) / dr2
        y2 = ((-det * dx) - (abs(dy) * dscr)) / dr2
        p1 = P(x1, y1) + center
        p2 = P(x2, y2) + center
        if not on_line or line.point_on_line(p1):
            intersections.append(p1)
        if not on_line or line.point_on_line(p2):
            intersections.append(p2)
    return intersections
    # pylint: enable=too-many-locals


def center_to_endpoints(
    center: P,
    rx: float,
    ry: float,
    phi: float,
    start_angle: float,
    sweep_angle: float,
) -> tuple[P, P, float, float]:
    """Convert center parameterization to endpoint parameterization.

    Returns:
        A tuple containing:
        (p1, p2, large_arc_flag, sweep_flag)
    """
    t = P(rx * math.cos(start_angle), ry * math.sin(start_angle))
    if not const.is_zero(phi):
        t = t.rotate(phi)
    p1 = t + center
    t = P(
        rx * math.cos(start_angle + sweep_angle),
        ry * math.sin(start_angle + sweep_angle),
    )
    if not const.is_zero(phi):
        t = t.rotate(phi)
    p2 = t + center
    large_arc_flag = 1 if abs(sweep_angle) > math.pi else 0
    sweep_flag = 1 if sweep_angle > 0 else 0
    return p1, p2, large_arc_flag, sweep_flag


if const.DEBUG or TYPE_CHECKING:
    from . import debug


def draw_ellipse(
    ellipse: Ellipse,
    color: str = '#cccc99',
    width: str | float = '1px',
    opacity: float = 1,
    verbose: bool = False,
    svg_context: SVGContext | None = None,
) -> None:
    """Draw an SVG arc for debugging/testing."""
    if not svg_context:
        svg_context = debug.svg_context

    if not const.DEBUG or not svg_context:
        return

    style = debug.linestyle(color=color, width=width, opacity=opacity)
    svg_context.create_ellipse(
        ellipse.center,
        ellipse.rx,
        ellipse.ry,
        phi=ellipse.phi,
        style=style,
    )
    if verbose:
        debug.draw_point(ellipse.center, color=color)
