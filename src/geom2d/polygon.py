"""Some handy polygon tools.

Includes convex hull, area, and centroid calculations.

Some references:

    - http://paulbourke.net/geometry/
    - http://geomalgorithms.com/index.html

====
"""

from __future__ import annotations

import contextlib
import functools
import heapq
import math
from typing import TYPE_CHECKING

from contrib import clipper

from . import const, point, util
from .line import Line, TLine
from .point import P, TPoint

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# pylint: disable=ungrouped-imports
try:
    from itertools import (  # type: ignore [attr-defined]
        pairwise,
    )
except ImportError:
    from itertools import tee

    def pairwise(iterable: Iterable) -> Iterable:  # type: ignore [no-redef]
        """Implement itertools.pairwise for python < 3.10."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


# TODO: refactor functions to allow iterable instead of Sequence


TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)


def turn(p: TPoint, q: TPoint, r: TPoint) -> int:
    """Determine relative direction of point.

    Args:
        p: Point from which initial direction is determined.
        q: Point from which turn is determined.
        r: End point which determines turn direction.
            I.e. from q this point is to the left or right
            of p.

    Returns:
        -1, 0, 1 if p,q,r forms a right, straight, or left turn.
    """
    a = (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])
    return (a > const.EPSILON) - (a < -const.EPSILON)
    # or easier to read:
    # if const.is_zero(a):
    #    return 0
    # return TURN_LEFT if a > const.EPSILON else TURN_RIGHT


def convex_hull(points: Iterable[TPoint]) -> list[P]:
    """Returns points on convex hull of an array of points in CCW order.

    Uses the Graham Scan algorithm.

    :param points: a list of 2-tuple (x, y) points.
    :return: The convex hull as a list of 2-tuple (x, y) points.
    """
    # TODO: correctly type points for sorted
    sorted_points = sorted(points)  # type: ignore [type-var]
    lh: list[P] = functools.reduce(_keep_left, sorted_points, [])
    uh: list[P] = functools.reduce(_keep_left, reversed(sorted_points), [])
    # lh.extend(uh[i] for i in range(1, len(uh) - 1))
    lh.extend(uh[1:-1])
    return lh


def _keep_left(hull: list[P], r: TPoint) -> list[P]:
    while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
        hull.pop()
    if not hull or hull[-1] != r:
        hull.append(P(r))
    return hull


# ==============================================================================
# Chan's Convex Hull O(n log h) - Tom Switzer <thomas.switzer@gmail.com>
# See http://tomswitzer.net/2010/12/2d-convex-hulls-chans-algorithm/
# ==============================================================================
def convex_hull_chan(points: list[TPoint]) -> list[P]:
    """Returns the points on the convex hull of points in CCW order.

    Uses Chan's algorithm. May be faster than Graham scan on
    large point collections.

    See http://tomswitzer.net/2010/12/2d-convex-hulls-chans-algorithm/

    :param points: a list of 2-tuple (x, y) points.
    :return: The convex hull as a list of 2-tuple (x, y) points.
    """
    # for m in (1 << (1 << t) for t in range(len(points))):
    hull: list[P] = []
    for m in ((1 << t) for t in range(len(points))):
        hulls = [
            convex_hull(points[i : i + m]) for i in range(0, len(points), m)
        ]
        hull_pairs = [_min_hull_pt_pair(hulls)]
        for _ in range(m):
            h, i = _next_hull_pt_pair(hulls, hull_pairs[-1])
            if h == hull_pairs[0][0] and i == hull_pairs[0][1]:
                return [hulls[h][i] for h, i in hull_pairs]
            hull.append(P(hulls[h][i]))
    return hull


def _rtangent(hull: list[P], p: TPoint) -> int:
    """Right tangent point.

    Index of the point on hull that the right tangent line from p
    to hull touches.
    """
    k, r = 0, len(hull)
    k_prev = turn(p, hull[0], hull[-1])
    k_next = turn(p, hull[0], hull[(k + 1) % r])
    while k < r:
        c = int((k + r) / 2)
        c_prev = turn(p, hull[c], hull[(c - 1) % len(hull)])
        c_next = turn(p, hull[c], hull[(c + 1) % len(hull)])
        c_side = turn(p, hull[k], hull[c])
        if TURN_RIGHT not in (c_prev, c_next):
            return c
        if (c_side == TURN_LEFT and k_next in (TURN_RIGHT, k_prev)) or (
            c_side == TURN_RIGHT and c_prev == TURN_RIGHT
        ):
            r = c  # Tangent touches left chain
        else:
            k = c + 1  # Tangent touches right chain
            k_prev = -c_next  # Switch sides
            k_next = turn(p, hull[k], hull[(k + 1) % len(hull)])
    return k


def _min_hull_pt_pair(hulls: list[list[P]]) -> tuple[int, int]:
    """Returns the hull, point index pair that is minimal."""
    h, p = 0, 0
    for i in range(len(hulls)):
        # j = min(range(len(hulls[i])), key=lambda k, ii=i: hulls[ii][k])
        def sortkey(k: int, i: int = i) -> tuple[float, float]:
            return hulls[i][k]

        j = min(range(len(hulls[i])), key=sortkey)
        if hulls[i][j] < hulls[h][p]:
            h, p = i, j

    return (h, p)


def _dist2(p1: TPoint, p2: TPoint) -> float:
    """Euclidean distance squared between two points."""
    a = p1[0] - p2[0]
    b = p1[1] - p2[1]
    return (a * a) + (b * b)


def _next_hull_pt_pair(
    hulls: list[list[P]], pair: tuple[int, int]
) -> tuple[int, int]:
    """(hull, point) index pair of the next point in the convex hull."""
    p = hulls[pair[0]][pair[1]]
    nextpair = (pair[0], (pair[1] + 1) % len(hulls[pair[0]]))
    for h in (i for i in range(len(hulls)) if i != pair[0]):
        s = _rtangent(hulls[h], p)
        q, r = hulls[nextpair[0]][nextpair[1]], hulls[h][s]
        t = turn(p, q, r)
        if t == TURN_RIGHT or (t == TURN_NONE and _dist2(p, r) > _dist2(p, q)):
            nextpair = (h, s)
    return nextpair


# def bounding_box(points):
#    """Simple bounding box of a collection of points.
#
#    :param points: an iterable collection of point 2-tuples (x,y).
#    """
#    xmin = sys.float_info.max
#    ymin = sys.float_info.max
#    xmax = sys.float_info.min
#    ymax = sys.float_info.min
#    for x, y in points:
#        xmin = min(xmin, x)
#        ymin = min(ymin, y)
#        xmax = max(xmax, x)
#        ymax = max(ymax, y)
#    return box.Box(P(xmin, ymin), P(xmax, ymax))


# def bounding_box(points):
#    """Simple bounding box of a collection of points.
#
#    :param points: an iterable collection of point 2-tuples (x,y).
#    """
#    xmin, ymin = typing.cast(tuple[float, float], map(min, zip(*points)))
#    xmax, ymax = typing.cast(tuple[float, float], map(max, zip(*points)))
#    return Box(P(xmin, ymin), P(xmax, ymax))


# ==============================================================================
# Area and centroid calculations for non self-intersecting closed polygons.
# See http://paulbourke.net/geometry/polygonmesh/
# ==============================================================================


def winding(vertices: Sequence[TPoint]) -> int:
    """Determine polygon winding.

    Returns:
        1 if CCW else -1 if CW
    """
    a = area(vertices)
    return (a >= const.EPSILON) - (a < const.EPSILON)


def area(vertices: Iterable[TPoint]) -> float:
    """Area of a simple polygon.

    Also determines winding (area >= 0 ==> CCW, area < 0 ==> CW).

    Args:
        vertices: the polygon vertices. A list of 2-tuple (x, y) points.

    Returns (float):
        The area of the polygon. The area will be negative if the
        vertices are ordered clockwise.
    """
    # This works for non-closed polygons as well.
    # a = 0.0
    # for n in range(-1, len(vertices) - 1):
    #    p2 = vertices[n]
    #    p1 = vertices[n + 1]
    #    # Accumulate the cross product of each pair of vertices
    #    a += (p1[0] * p2[1]) - (p2[0] * p1[1])
    # return a / 2

    a = 0.0
    startp: TPoint | None = None
    for p1, p2 in pairwise(vertices):
        # Keep track of the start point in case polygon is not closed
        if not startp:
            startp = p1
        # Accumulate the cross product of each pair of vertices
        a += (p1[0] * p2[1]) - (p2[0] * p1[1])

    # Close polygon if necessary
    if startp and (
        not const.float_eq(p2[0], startp[0])
        or not const.float_eq(p2[1], startp[1])
    ):
        a += (p2[0] * startp[1]) - (startp[0] * p2[1])

    return -a / 2


def area_triangle(
    a: TPoint,  # | Sequence[TPoint],
    b: TPoint | None = None,
    c: TPoint | None = None,
) -> float:
    """Area of a triangle.

    This is just a slightly more efficient specialization of
    the more general polygon area.

    Args:
        a: The first vertex of a triangle or an iterable of three vertices.
        b: The second vertex or None if `a` is iterable.
        c: The third vertex or None if `a` is iterable.

    Returns (float):
        The area of the triangle.
    """
    if not b:
        a, b, c = a  # type: ignore [assignment]
    if not (b and c):
        raise ValueError
    # See: http://mathworld.wolfram.com/TriangleArea.html
    ux = b[0] - a[0]
    uy = b[1] - a[1]
    vx = c[0] - a[0]
    vy = c[1] - a[1]
    det = (ux * vy) - (uy * vx)
    return abs(det) / 2


def centroid(vertices: Sequence[TPoint]) -> P:
    """Return the centroid of a simple polygon.

    See http://paulbourke.net/geometry/polygonmesh/

    :param vertices: The polygon vertices. A list of 2-tuple (x, y) points.
    :return: The centroid point as a 2-tuple (x, y)
    """
    # TODO: allow iterable for vertices
    num_vertices = len(vertices)
    # Handle degenerate cases for point and single segment
    if num_vertices == 1:
        # if it's just one point return the same point
        return P(vertices[0])
    if num_vertices == 2:
        # if it's a single segment just return the midpoint
        return Line(vertices[0], vertices[1]).midpoint()
    x = 0.0
    y = 0.0
    a = 0.0  # area
    for n in range(-1, num_vertices - 1):
        p2 = vertices[n]
        p1 = vertices[n + 1]
        cross = (p1[0] * p2[1]) - (p2[0] * p1[1])
        a += cross
        x += (p1[0] + p2[0]) * cross
        y += (p1[1] + p2[1]) * cross
    t = a * 3
    return P(x / t, y / t)


# pylint: disable=line-too-long
# ==============================================================================
# Portions of this code (point in polygon test) are derived from:
# https://wrfranklin.org/Research/Short_Notes/pnpoly.html
# and uses the following license:
# Copyright (c) 1970-2003, Wm. Randolph Franklin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimers.
# Redistributions in binary form must reproduce the above copyright notice in the
# documentation and/or other materials provided with the distribution.
# The name of W. Randolph Franklin may not be used to endorse or promote products
# derived from this Software without specific prior written permission.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
# pylint: enable=line-too-long


def point_inside(
    polygon: Sequence[TPoint], p: TPoint, edge_ok: bool = False
) -> bool:
    """Test if point is inside a closed polygon.

    See:
        - https://wrfranklin.org/Research/Short_Notes/pnpoly.html
        - http://erich.realtimerendering.com/ptinpoly/
        - http://paulbourke.net/geometry/polygonmesh/

    The original C code::

        int pnpoly(int nvert, float *x, float *y, float testx, float testy)
        {
          int i, j, c = 0;
          for (i = 0, j = nvert-1; i < nvert; j = i++) {
            if (
              ((y[i] > testy) != (y[j] > testy)) &&
              (testx < (x[j] - x[i]) * (testy - y[i]) / (y[j] - y[i]) + x[i])
            )
               c = !c;
          }
          return c;
        }



    Args:
        polygon: polygon vertices. A list of 2-tuple (x, y) points.
        p: Point to test.
        edge_ok: Point is considered inside if it lies on a vertex
            or an edge segment.

    Returns:
        True if the point lies inside the polygon, else False.
    """
    p_is_inside = False
    x, y = p
    j = -1
    for i in range(len(polygon)):
        # Special case test for point on a vertex or on an edge,
        # in which case it's considered "inside".
        if edge_ok and (
            point.almost_equal(p, polygon[i])
            or Line(polygon[i], polygon[j]).point_on_line(p)
        ):
            return True

        x1, y1 = polygon[i]
        x2, y2 = polygon[j]
        # This is a tricky conditional - see W. R. Franklin's web page
        if (y1 > y) != (y2 > y) and x < ((x2 - x1) * (y - y1) / (y2 - y1)) + x1:
            p_is_inside = not p_is_inside
        j = i
    return p_is_inside


def all_points_inside(
    polygon: Sequence[TPoint], points: Iterable[TPoint], edge_ok: bool = False
) -> bool:
    """True if all points are within the polygon."""
    return all(point_inside(polygon, p, edge_ok=edge_ok) for p in points)


def intersect_line(  # noqa: PLR0912 pylint: disable=too-many-branches
    polygon: Sequence[TPoint], lineseg: TLine, edge_ok: bool = False
) -> list[Line]:
    """Compute the intersection(s) of a polygon/polyline and a line segment.

    Args:
        polygon: Polygon vertices.
        lineseg: A line possibly intersecting the polygon.
            A sequence of two line end points, of the form
            ``((x1, y1), (x2, y2))``.
        edge_ok: Intersection considered if segment endpoint lies
            on polygon edge.

    Returns:
        A list of one or more interior line segments that intersect
        the polygon or that lie completely within the polygon.
        Returns an empty list if there are no intersections.
    """
    if not isinstance(lineseg, Line):
        lineseg = Line(lineseg)

    # automatically close the polygon
    start = 0 if polygon[0] == polygon[-1] else -1
    # Find all the intersections of the line segment with the polygon
    intersections = []
    for i in range(start, len(polygon) - 1):
        edge = Line(polygon[i], polygon[i + 1])
        # if lineseg.p1 == edge.p1 or lineseg.p1 == edge.p2:
        #    if point_on_vertex(polygon, lineseg.p2):
        #        pass
        # elif lineseg.p2 == edge.p1 or lineseg.p2 == edge.p1:
        #    if point_on_vertex(polygon, lineseg.p2):
        #        pass
        if edge_ok and lineseg == edge:
            # Line is coincident with polygon edge
            # debug.draw_line(lineseg, color='#0000ff')
            return [lineseg]
        # Find the intersection unit distance (mu) from the line start point
        mu = lineseg.intersection_mu(edge, segment=True)
        if mu and const.EPSILON < mu < (1 - const.EPSILON):
            intersections.append(mu)

    tline = lineseg
    if lineseg.p1 in polygon and lineseg.p2 in polygon:
        tline = lineseg.extend(const.EPSILON * -2, from_midpoint=True)

    p1_inside: bool = point_inside(polygon, tline[0])  # , edge_ok=edge_ok)
    p2_inside: bool = point_inside(polygon, tline[1])  # , edge_ok=edge_ok)
    segments = []
    if intersections:
        # Sort intersections in mu order
        intersections.sort()
        i = 0
        # Determine the starting point
        if p1_inside:
            p1 = lineseg[0]
        elif p2_inside:
            p1 = lineseg[1]
            intersections.reverse()
        else:
            p1 = lineseg.point_at(intersections[0])
            i = 1
        num_intersections = len(intersections)
        while i < num_intersections:
            p2 = lineseg.point_at(intersections[i])
            if p1 != p2:  # ignore degenerate lines - may not be necessary...
                # debug.draw_line(Line(p1, p2), color='#00ff00')
                segments.append(Line(p1, p2))
            if (i + 1) == num_intersections:
                break
            p1 = lineseg.point_at(intersections[i + 1])
            i += 2
    elif p1_inside and p2_inside:
        # Line segment is completely contained by the polygon
        # debug.draw_line(lineseg, color='#ff0000')
        return [lineseg]
    return segments


def is_closed(polygon: Sequence[TPoint]) -> bool:
    """Test if the polygon is closed.

    I.e. if the first vertice matches the last vertice.
    """
    x1, y1 = polygon[0]
    xn, yn = polygon[-1]
    return const.float_eq(x1, xn) and const.float_eq(y1, yn)


# This doesn't work of course...
# def coincident_triangle_inside(polygon, triangle: Sequence[P]) -> bool:
#    """Return True if the triangle that has at least one edge
#    or vertex that is coincident with a polygon edge or vertex
#    is inside the polygon.
#    """
#    midp1 = Line(triangle[0], triangle[1]).midpoint()
#    midp2 = Line(triangle[2], midp1).midpoint()
#    return point_inside(polygon, midp2)

# class ClipPolygon(object):
#    """Clipping polygon."""
#
#    def __init__(self, polygon):
#        """
#        :param polygon: the polygon vertices.
#            An iterable of 2-tuple (x, y) points.
#        """
#        self.polygon = polygon
#
#    def point_inside(self, p):
#        """Return True if the point is inside this polygon."""
#        return point_inside(self.polygon, p)
#
#    def clip_line(self, line):
#        """Compute the intersection(s), if any, of this polygon
#        and a line segment.
#
#        :param line: the line to test for intersections.
#        :return: a list of one or more line segments that intersect the polygon
#            or that lie completely within the polygon. Returns None if there
#            are no intersections.
#        """
#        return intersect_line(self.polygon, line)
#


def poly_stroke_to_path(
    poly: Sequence[TPoint],
    stroke_width: float,
    jointype: clipper.JoinType = clipper.JoinType.Miter,
    endtype: clipper.EndType = clipper.EndType.Butt,
    limit: float = 0.0,
) -> list[list[P]]:
    """Convert a stroke (line + width) to a path.

    Args:
        poly: A polyline as a list of 2-tuple vertices.
        stroke_width: Stroke width.
        offset: The amount to offset (can be negative).
        jointype: The type of joins for offset vertices.
        endtype: The type of end caps.
        limit: The max distance to a offset vertice before it
            will be squared off.

    If the stroke is a closed polygon then two closed sub-paths will be returned,
    allowing a fillable SVG entity defined by an inner and outer polygon..
    Otherwise a single closed path.
    """
    return offset_polyline(
        poly, stroke_width / 2, jointype=jointype, endtype=endtype, limit=limit
    )


def offset_polygons(
    poly: Sequence[TPoint],
    offset: float,
    jointype: clipper.JoinType = clipper.JoinType.Miter,
    limit: float = 0.0,
) -> list[list[P]]:
    """Offset a polygon by *offset* amount.

    This is also called polygon buffering.

    See:
        http://www.angusj.com/delphi/clipper.php

    Args:
        poly: A polygon as a list of 2-tuple vertices.
        offset: The amount to offset (can be negative).
        jointype: The type of joins for offset vertices.
        limit: The max distance to a offset vertice before it
            will be squared off.

    Returns:
        Zero or more offset polygons as a list of 2-tuple vertices.
        If the specified offset cannot be performed for the input polygon
        an empty list will be returned.
    """
    mult = 10**const.EPSILON_PRECISION
    offset *= mult
    limit *= mult
    clipper_poly = poly2clipper(poly)
    clipper_offset_polys = clipper.OffsetPolygons(
        [
            clipper_poly,
        ],
        offset,
        jointype=jointype,
        limit=limit,
    )
    return [clipper2poly(p) for p in clipper_offset_polys]


def offset_polyline(
    poly: Sequence[TPoint],
    offset: float,
    jointype: clipper.JoinType = clipper.JoinType.Miter,
    endtype: clipper.EndType = clipper.EndType.Butt,
    limit: float = 0.0,
) -> list[list[P]]:
    """Offset a polyline by *offset* amount.

    This is also called polygon buffering.

    See:
        http://www.angusj.com/delphi/clipper.php

    Args:
        poly: A polyline as a list of 2-tuple vertices.
        offset: The amount to offset (can be negative).
        jointype: The type of joins for offset vertices.
        endtype: The type of end caps for polylines.
        limit: The max distance to a offset vertice before it
            will be squared off.

    Returns:
        Zero or more offset polygons as a list of 2-tuple vertices.
        If the specified offset cannot be performed for the input polygon
        an empty list will be returned.
    """
    mult = 10**const.EPSILON_PRECISION
    offset *= mult
    limit *= mult
    clipper_poly = poly2clipper(poly)
    clipper_offset_polys = clipper.OffsetPolyLines(
        [
            clipper_poly,
        ],
        offset,
        jointype=jointype,
        endtype=endtype,
        limit=limit,
    )
    return [clipper2poly(p) for p in clipper_offset_polys]


def poly2clipper(poly: Iterable[TPoint]) -> list[clipper.Point]:
    """Convert a polygon to a Clipper polygon.

    Args:
        poly: An iterable of floating point coordinates.

    Returns:
        A Clipper polygon which is a list of integer coordinates.
    """
    clipper_poly = []
    mult = 10**const.EPSILON_PRECISION
    for p in poly:
        x = int(p[0] * mult)
        y = int(p[1] * mult)
        clipper_poly.append(clipper.Point(x, y))
    return clipper_poly


def clipper2poly(clipper_poly: Iterable[clipper.Point]) -> list[P]:
    """Convert a Clipper polygon to a float tuple polygon.

    Convert a Clipper polygon (a list of integer 2-tuples) to
    a polygon (as a list of float 2-tuple vertices).

    Args:
        clipper_poly: An interable of integer coordinates.

    Returns:
        A list of floating point coordinates.
    """
    poly = []
    mult = 10**const.EPSILON_PRECISION
    for p in clipper_poly:
        x = float(p.x) / mult
        y = float(p.y) / mult
        poly.append(P(x, y))
    # Close the polygon
    if len(poly) > 2 and poly[0] != poly[-1]:
        poly.append(poly[0])
    return poly


def simplify_polyline_rdp(
    points: Sequence[TPoint], tolerance: float
) -> list[P]:
    """Simplify a polyline.

    A polyline is a sequence of vertices.

    Uses Ramer-Douglas-Peucker algorithm.

    See:
        https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm

    Args:
        points: A sequence of polyline vertices.
        tolerance (float): Line flatness tolerance.

    Returns:
        A list of points defining the vertices of the simplified polyline.
    """
    num_points = len(points)
    if num_points < 3:
        # Nothing to simplify
        return [P(p) for p in points]

    # Find the index of the point that's farthest from a chord
    # connecting the endpoints of the polyline.
    dmax = 0.0
    dmax_index = 0
    chord = Line(points[0], points[-1])

    for i, p in enumerate(points[1:-1]):
        d = chord.distance_to_point(p, segment=True)
        if d > dmax:
            dmax_index = i + 1
            dmax = d

    if dmax > tolerance:
        if num_points == 3:
            # Can't sub-divide any further
            return [P(p) for p in points]
        # Divide the polyline at the max distance point and
        # recursively get the simplified sub-polylines.
        simplified1 = simplify_polyline_rdp(
            points[: (dmax_index + 1)], tolerance
        )
        simplified2 = simplify_polyline_rdp(points[dmax_index:], tolerance)
        simplified1.extend(simplified2[1:])
        return simplified1

    # All points in between chord endpoints are within tolerance
    # so they are eliminated.
    return [chord.p1, chord.p2]


def simplify_polyline_vw(
    points: Iterable[TPoint], min_area: float
) -> list[TPoint]:
    """Simplify a polyline.

    Uses Visvalingam-Whyatt algorithm.

    See:
        [1] Visvalingam, M., and Whyatt, J.D. (1992)
        "Line Generalisation by Repeated Elimination of Points",
        Cartographic J., 30 (1), 46 - 51

    Args:
        points: A sequence of polyline vertices.
        min_area: Minimum point triplet triangle area to filter for.

    Returns:
        A list of points defining the vertices of the simplified polyline.
    """
    # https://archive.fo/Tzq2#selection-91.0-91.89
    # https://hull-repository.worktribe.com/preview/376364/000870493786962263.pdf
    # https://bost.ocks.org/mike/simplify/

    minheap: list = []

    class _Triangle:
        p1: TPoint
        p2: TPoint
        p3: TPoint
        area: float
        index: int = 0
        tprev: _Triangle | None = None  # pylint: disable=undefined-variable
        tnext: _Triangle | None = None  # pylint: disable=undefined-variable

        def __init__(
            self,
            p1: TPoint,
            p2: TPoint,
            p3: TPoint,
            i: int,
            tprev: _Triangle | None,
        ) -> None:
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3
            self.index = i
            self.area = area_triangle(self.p1, self.p2, self.p3)
            if tprev:
                tprev.tnext = self
                self.tprev = tprev

        def update_area(self) -> None:
            with contextlib.suppress(ValueError):
                minheap.remove(self)
            self.area = area_triangle(self.p1, self.p2, self.p3)
            heapq.heappush(minheap, self)

        def __lt__(self, other: _Triangle) -> bool:
            return self.area < other.area

    triangles: list[_Triangle] = []

    # Populate a min-heap based on triangle areas and create
    # a linked list of triangles.
    previous_triangle: _Triangle | None = None
    for i, (p1, p2, p3) in enumerate(util.triplepoints(points)):
        triangle = _Triangle(p1, p2, p3, i + 1, previous_triangle)
        triangles.append(triangle)
        heapq.heappush(minheap, triangle)
        previous_triangle = triangle

    max_area: float = 0

    while minheap:
        triangle = heapq.heappop(minheap)
        # Ensure that the current point cannot be eliminated without
        # eliminating previously eliminated points.
        # See [1] Visvalingam
        if triangle.area < max_area:
            triangle.area = max_area
        else:
            max_area = triangle.area

        # Unlink the triangle
        if triangle.tprev:
            triangle.tprev.tnext = triangle.tnext
            triangle.tprev.p3 = triangle.p3
            triangle.tprev.update_area()

        if triangle.tnext:
            triangle.tnext.tprev = triangle.tprev
            triangle.tnext.p1 = triangle.p1
            triangle.tnext.update_area()

    simpoly: list[TPoint] = [triangles[0].p1]
    simpoly.extend([t.p2 for t in triangles if t.area > min_area])
    simpoly.append(triangles[-1].p3)
    return simpoly


# def path_to_polyline(path: Sequence[Line]) -> list[P]:
#    """Convert a path (list of line segments)
#    to a polyline (list of vertices).
#    """
#    polyline = [path[0].p1]  # [seg.p1 for seg in path]
#    for seg in path:
#        polyline.append(seg.p2)
#    return polyline


def length(polyline: Iterable[TPoint]) -> float:
    """The total length of a polyline/polygon."""
    return float(
        sum(
            math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            for p1, p2 in pairwise(polyline)
        )
    )


def is_inside(polygon1: Sequence[TPoint], polygon2: Iterable[TPoint]) -> bool:
    """Is polygon2 inside polygon1?

    It is assumed that the polygons
    are non-intersecting and non-self-intersecting.
    """
    return all(point_inside(polygon1, p) for p in polygon2)
