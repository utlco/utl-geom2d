"""Basic 2D bounding box geometry."""

from __future__ import annotations

import math
import typing
from collections.abc import Sequence

from . import const
from .line import Line, TLine
from .point import P, TPoint

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

    from .arc import Arc
    from .transform2d import TMatrix

# Generic box type
TBox = Sequence[TPoint]


def bounding_box(points: Iterable[TPoint]) -> Box:
    """Simple bounding box of a collection of points.

    :param points: an iterable collection of point 2-tuples (x,y).
    """
    xmin, ymin = typing.cast(tuple[float, float], map(min, zip(*points)))
    xmax, ymax = typing.cast(tuple[float, float], map(max, zip(*points)))
    return Box(P(xmin, ymin), P(xmax, ymax))


class Box(tuple[P, P]):
    """Bounding box.

    Two dimensional immutable rectangle defined by two points,
    the lower left corner and the upper right corner respectively.

    The sides are always assumed to be aligned with the X and Y axes.

    Useful as clipping rectangle or bounding box.
    """

    __slots__ = ()

    def __new__(cls, p1: TBox | TPoint, p2: TPoint | None = None) -> Self:
        """Create a new Box."""
        # Canonicalize the point order so that p1 is
        # always lower left.
        if p2 is None:
            p1, p2 = p1[0], p1[1]  # type: ignore [assignment]
        x1 = min(p1[0], p2[0])  # type: ignore [index, type-var]
        y1 = min(p1[1], p2[1])  # type: ignore [index, type-var]
        x2 = max(p1[0], p2[0])  # type: ignore [index, type-var]
        y2 = max(p1[1], p2[1])  # type: ignore [index, type-var]
        return tuple.__new__(
            cls, (P(x1, y1), P(x2, y2))  # type: ignore [type-var, arg-type]
        )

    @staticmethod
    def from_points(points: Iterable[TPoint]) -> Box:
        """Create a Box from the bounding box of the given points.

        Returns:
            A geom.Box or None if there are zero points.
        """
        if not points:
            raise ValueError
        x_values, y_values = zip(*points)
        xmin = min(x_values)
        xmax = max(x_values)
        ymin = min(y_values)
        ymax = max(y_values)
        #         xmin = sys.float_info.max
        #         ymin = sys.float_info.max
        #         xmax = sys.float_info.min
        #         ymax = sys.float_info.min
        #         for p in points:
        #             x, y = p
        #             xmin = min(xmin, x)
        #             ymin = min(ymin, y)
        #             xmax = max(xmax, x)
        #             ymax = max(ymax, y)
        return Box(P(xmin, ymin), P(xmax, ymax))

    @staticmethod
    def from_path(path: Iterable[tuple[tuple[float, ...]]]) -> Box | None:
        """Create a Box from the bounding box of path segments.

        This does not compute the actual bounding box of Arc
        or Bezier segments, just the naive box around the endpoints.
        TODO: compute actual bounding box.

        Returns:
            A geom.Box or None if there are zero segments.
        """
        if not path:
            return None
        x_values, y_values = zip(*[p for seg in path for p in seg])
        xmin = min(x_values)
        xmax = max(x_values)
        ymin = min(y_values)
        ymax = max(y_values)
        return Box(P(xmin, ymin), P(xmax, ymax))

    @property
    def p1(self) -> P:
        """The lower left corner of the box rectangle."""
        return self[0]

    @property
    def p2(self) -> P:
        """The upper right corner of the box rectangle."""
        return self[1]

    @property
    def topleft(self) -> P:
        """The upper left corner of the box rectangle."""
        return P(self[0][0], self[1][1])

    @property
    def bottomright(self) -> P:
        """The bottom right corner of the box rectangle."""
        return P(self[1][0], self[0][1])

    @property
    def xmin(self) -> float:
        """Minimum X value of bounding box."""
        return self[0][0]

    @property
    def xmax(self) -> float:
        """Maximum X value of bounding box."""
        return self[1][0]

    @property
    def ymin(self) -> float:
        """Minimum Y value of bounding box."""
        return self[0][1]

    @property
    def ymax(self) -> float:
        """Maximum X value of bounding box."""
        return self[1][1]

    @property
    def center(self) -> P:
        """Return the center point of this rectangle."""
        return self.p1 + ((self.p2 - self.p1) / 2)

    @property
    def height(self) -> float:
        """Height of rectangle. (along Y axis)."""
        return self[1][1] - self[0][1]

    @property
    def width(self) -> float:
        """Width of rectangle. (along X axis)."""
        return self[1][0] - self[0][0]

    def diagonal(self) -> float:
        """Length of diagonal."""
        return (self.p2 - self.p1).length()

    def vertices(self) -> tuple[P, P, P, P]:
        """Get the four vertices of the box as a tuple of four points."""
        return (
            P(self.xmin, self.ymin),
            P(self.xmin, self.ymax),
            P(self.xmax, self.ymax),
            P(self.xmax, self.ymin),
        )

    def point_inside(self, p: TPoint) -> bool:
        """True if the point is inside this rectangle."""
        return (
            p[0] > self[0][0]
            and p[0] < self[1][0]
            and p[1] > self[0][1]
            and p[1] < self[1][1]
        )

    def line_inside(self, ln: TLine) -> bool:
        """True if the line segment is inside this rectangle."""
        return self.point_inside(ln[0]) and self.point_inside(ln[1])

    def all_points_inside(self, points: Iterable[TPoint]) -> bool:
        """Return True if the given set of points lie inside this rectangle."""
        return all(self.point_inside(p) for p in points)

    def buffered(self, distance: float) -> Box:
        """Expand or shrink box boundaries by `distance`.

        Args:
            distance: The distance to offset.
                The box will shrink if the distance is negative.

        Returns:
            A copy of this box with it's boundaries expanded or shrunk
            by the specified distance. Also known as buffering.
        """
        return Box(self.p1 - distance, self.p2 + distance)

    def transform(self, matrix: TMatrix) -> Box:
        """Apply transform to this Box.

        Note: rotations just scale since a Box is always aligned to
            the X and Y axes.
        """
        return Box(self[0].transform(matrix), self[1].transform(matrix))

    def clip_line(self, ln: TLine) -> Line | None:
        """Use this box to clip a line segment.

        If the given line segment is clipped by this rectangle then
        return a new line segment with clipped end-points.

        If the line segment is entirely within the rectangle this
        returns the same (unclipped) line segment.

        If the line segment is entirely outside the rectangle this
        returns None.

        Uses the Liang-Barsky line clipping algorithm. Translated C++ code
        from: http://hinjang.com/articles/04.html

        Args:
            ln: The line segment to clip.

        Returns:
            A new clipped line segment or None if the segment
            is outside this clipping rectangle.
        """
        if self.line_inside(ln):
            return Line(ln)
        x1, y1 = ln[0]  # ln.p1.x, ln.p1.y
        x2, y2 = ln[1]  # ln.p2.x, ln.p2.y
        dx = x2 - x1
        dy = y2 - y1
        u_minmax = [0.0, 1.0]
        if (
            _lbclip_helper(self.xmin - x1, dx, u_minmax)
            and _lbclip_helper(x1 - self.xmax, -dx, u_minmax)
            and _lbclip_helper(self.ymin - y1, dy, u_minmax)
            and _lbclip_helper(y1 - self.ymax, -dy, u_minmax)
        ):
            if u_minmax[1] < 1.0:
                x2 = x1 + u_minmax[1] * dx
                y2 = y1 + u_minmax[1] * dy
            if u_minmax[0] > 0.0:
                x1 += u_minmax[0] * dx
                y1 += u_minmax[0] * dy
            return Line(P(x1, y1), P(x2, y2))
        return None

    def clip_line_stdeq(self, a: float, b: float, c: float) -> Line | None:
        """Clip a line defined by the standard form equation ax + by = c.

        The endpoints of the line will lie on the box edges.

        The clipped line endpoints will be oriented bottom->top, left->right

        Returns:
            A Line segment or None if no intersection.
        """
        if const.is_zero(b):
            # Line is vertical
            if const.is_zero(a):
                raise ValueError('both a and b cannot be zero')
            x = c / a
            if self.p1.x <= x <= self.p2.x:
                return Line((x, self.p1.y), (x, self.p2.y))
        elif const.is_zero(a):
            # Line is horizontal
            if const.is_zero(b):
                raise ValueError('both a and b cannot be zero')
            y = c / b
            if self.p1.y <= y <= self.p2.y:
                return Line((self.p1.x, y), (self.p2.x, y))
        else:
            # left and right points
            p1 = (self.p1.x, (c - a * self.p1.x) / b)
            p2 = (self.p2.x, (c - a * self.p2.x) / b)
            return self.clip_line((p1, p2))
        return None

    def clip_arc(self, _arc: Arc) -> Arc | None:
        """Use this Box to clip an Arc.

        If the given circular arc is clipped by this rectangle then
        return a new arc with clipped end-points.

        This only returns a single clipped arc even if the arc could
        be clipped into two sub-arcs... For now this is considered
        a pathological condition.

        Args:
            arc: The arc segment to clip.

        Returns:
            A new clipped arc or None if the arc segment
            is entirely outside this clipping rectangle.
            If the arc segment is entirely within the rectangle this
            returns the same (unclipped) arc segment.
        """
        # TODO: implement clip_arc...
        raise NotImplementedError

    def start_tangent_angle(self) -> float:
        """Tangent at start point.

        The angle in radians of a line tangent to this shape
        beginning at the first point. It's pretty obvious this will
        always be PI/2...

        This is just to provide an orthogonal interface for geometric shapes...

        The corner point order for rectangles is clockwise from lower left.
        """
        return math.pi / 2

    def bounding_box(self) -> Box:
        """Bounding box - self."""
        return self

    def intersection(self, other: TBox) -> Box | None:
        """The intersection of this Box rectangle and another.

        Returns None if the rectangles do not intersect.
        """
        other = Box(other)
        xmin = max(self.xmin, other.xmin)
        xmax = min(self.xmax, other.xmax)
        ymin = max(self.ymin, other.ymin)
        ymax = min(self.ymax, other.ymax)
        if xmin > xmax or ymin > ymax:
            return None
        return Box((xmin, ymin), (xmax, ymax))

    def union(self, other: TBox) -> Box:
        """Return a Box that is the union of this rectangle and another."""
        other = Box(other)
        xmin = min(self.xmin, other.xmin)
        xmax = max(self.xmax, other.xmax)
        ymin = min(self.ymin, other.ymin)
        ymax = max(self.ymax, other.ymax)
        return Box((xmin, ymin), (xmax, ymax))


# pylint: disable=invalid-name
def _lbclip_helper(
    nQ: float, nP: float, u_minmax: list[float]  # noqa: N803
) -> bool:
    """Lian-Barsky helper."""
    if const.is_zero(nP):
        # line is parallel to box edge - is it outside the box?
        return nQ <= 0.0 + const.EPSILON
    u = nQ / nP
    if nP > 0.0:
        # line goes from inside box to outside
        if u > u_minmax[1]:
            return False
        u_minmax[0] = max(u, u_minmax[0])
    else:
        # line goes from outside to inside
        if u < u_minmax[0]:
            return False
        u_minmax[1] = min(u, u_minmax[1])
    return True


# pylint: enable=invalid-name

# def rectangle(self):
#     """Return an equivalent shape as a rectangular polygon."""
#     return parallelogram.Parallelogram(
#         P(self.xmin, self.ymin), P(self.xmin, self.ymax),
#         P(self.xmax, self.ymax), P(self.xmax, self.ymin))


# __and__ = Box.intersection
# __or__ = Box.union
