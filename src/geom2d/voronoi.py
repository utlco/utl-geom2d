"""Voronoi diagram / Delaunay triangulation.

Compute a Voronoi diagram and optional Delaunay triangulation for a set of
2D input points.

Based on Steve Fortune's original code:
    http://ect.bell-labs.com/who/sjf/

Derek Bradley's fixes for memory leaks:
    http://zurich.disneyresearch.com/derekbradley/voronoi.html

Shane O'Sullivan's translation to C++:
    http://mapviewer.skynet.ie/voronoi.html

Translated to Python by Bill Simons September, 2005:
    (not sure where this original translation can be found anymore)

Nicely refactored version by Manfred Moitzi at:
    https://bitbucket.org/mozman/geoalg

This version was based on the Bill Simons version,
refactored with some of Moitzi's cleanups, and comments
incorporated from Shane O'Sullivan's update.

Derived from code bearing the following notice::

    The author of this software is Steven Fortune. Copyright (c) 1994 by AT&T
    Bell Laboratories.

    Permission to use, copy, modify, and distribute this software for any
    purpose without fee is hereby granted, provided that this entire notice
    is included in all copies of any software which is or includes a copy
    or modification of this software and in all copies of the supporting
    documentation for such software.
    THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
    WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR AT&T MAKE ANY
    REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
    OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.

This module has no dependencies besides standard Python libraries.

====
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Sequence
from typing import NamedTuple

#: Tolerance for floating point comparisons
EPSILON = 1e-9

TPoint = Sequence[float]  # Union[tuple[float, float], list[float]]
TLine = Sequence[TPoint]
TLineEq = tuple[float, float, float]

MIN_POINTS = 5


class DelaunayEdge(NamedTuple):
    """A Delaunay edge.

    The dual of a corresponding Voronoi segment
    that bisects this Delaunay segment.
    This is a line segment between nearest neighbor sites.
    """

    p1: TPoint
    p2: TPoint


class VoronoiEdge(NamedTuple):
    """A Voronoi edge.

    The dual of a corresponding Delauney edge.
    This is a line segment that bisects a line
    between nearest neighbor sites.

    If one end point of the edge is None it means
    the line extends to infinity. If the first
    end point is None the edge extends to the left.
    If the second end point is None the edge extends
    to the right.
    """

    p1: TPoint | None
    p2: TPoint | None
    # The line equation for this segment in the form `a*x + b*y = c`
    # as a 3-tuple (a, b, c)
    equation: TLineEq
    # The dual of this Voronoi edge
    delaunay_edge: DelaunayEdge


class DelaunayTriangle(NamedTuple):
    """A Delaunay triangle.  This a 3-tuple of 2-tuple (x, y) points."""

    p1: TPoint
    p2: TPoint
    p3: TPoint


class VoronoiDiagram:
    """Voronoi diagram and Delaunay triangulation."""

    _voronoi_edges: list[VoronoiEdge]
    _delaunay_edges: list[DelaunayEdge]
    _triangles: list[DelaunayTriangle]
    _vertices: list[TPoint]
    _lines: list[TLineEq]

    # Useful for determining direction of vertical lines
    input_bbox: tuple[tuple[float, float], tuple[float, float]]

    def __init__(
        self,
        input_points: Sequence[TPoint],
        delaunay: bool = False,
        jiggle_points: bool = False,
    ) -> None:
        """Create a VoronoiDiagram.

        Args:
            input_points: An indexable collection of points as (x, y) 2-tuples
            delaunay: Generate Delaunay edges and triangles if True.
            jiggle_points: Jiggle the input points by a small random
                distance to mitigate problems caused by degenerate
                point sets (such as collinear or coincident points).
                Default is False.
        """
        self._voronoi_edges = []
        self._delaunay_edges = []
        self._triangles = []
        self._vertices = []
        self._lines = []

        if len(input_points) >= MIN_POINTS:
            if jiggle_points:
                input_points = [jiggle(p) for p in input_points]
            self._compute_voronoi(input_points, delaunay)

    @property
    def vertices(self) -> list[TPoint]:
        """List of Voronoi diagram vertices."""
        return self._vertices

    @property
    def lines(self) -> list[TLineEq]:
        """List of Voronoi edges as line equations.

        A line is a 3-tuple (a, b, c) for the
        line equation of the form `a*x + b*y = c`.
        """
        return self._lines

    @property
    def edges(self) -> list[VoronoiEdge]:
        """List of VoronoiEdges."""
        return self._voronoi_edges

    @property
    def triangles(self) -> list[DelaunayTriangle]:
        """List of DelaunayTriangles."""
        return self._triangles

    @property
    def delaunay_edges(self) -> list[DelaunayEdge]:
        """List of DelaunayEdges."""
        return self._delaunay_edges

    def _add_vertex(self, site: _Site) -> None:
        site.sitenum = len(self._vertices)
        self._vertices.append((site.x, site.y))

    def _add_triangle(self, p1: TPoint, p2: TPoint, p3: TPoint) -> None:
        self._triangles.append(DelaunayTriangle(p1, p2, p3))

    def _add_bisector(self, edge: _Edge, delaunay: bool) -> None:
        edge.edgenum = len(self._lines)
        self._lines.append((edge.a, edge.b, edge.c))
        if delaunay:
            segment = DelaunayEdge(
                (edge.dsegment[0].x, edge.dsegment[0].y),
                (edge.dsegment[1].x, edge.dsegment[1].y),
            )
            self._delaunay_edges.append(segment)

    def _add_edge(self, edge: _Edge) -> None:
        p1 = None
        left_edge = edge.endpoints[_Edge.LEFT]
        if left_edge:
            p1 = self._vertices[left_edge.sitenum]

        p2 = None
        right_edge = edge.endpoints[_Edge.RIGHT]
        if right_edge:
            p2 = self._vertices[right_edge.sitenum]

        assert p1 or p2
        self._voronoi_edges.append(
            VoronoiEdge(
                p1,
                p2,
                self._lines[edge.edgenum],
                self._delaunay_edges[edge.edgenum],
            )
        )

    def _compute_voronoi(
        self, input_points: Sequence[TPoint], delaunay: bool
    ) -> None:
        """Create the Voronoi diagram.

        Args:
            input_points: A list of points as (x, y) 2-tuples
            delaunay: Create Delaunay triangulation
        """
        sites = _SiteList(input_points)
        nsites = len(sites)
        edges = _EdgeList(sites.xmin, sites.xmax, nsites)
        priority_queue = _PriorityQueue(sites.ymin, sites.ymax, nsites)
        itersites = iter(sites)

        bottomsite = next(itersites)
        newsite: _Site | None = next(itersites)
        min_point = _Site(sys.float_info.min, sys.float_info.min)

        while True:
            if not priority_queue.is_empty():
                min_point = priority_queue.get_min_point()
            if newsite and (priority_queue.is_empty() or newsite < min_point):
                self._handle_event1(
                    priority_queue, edges, bottomsite, newsite, delaunay
                )
                try:
                    newsite = next(itersites)
                except StopIteration:
                    newsite = None
            elif not priority_queue.is_empty():
                # intersection is smallest - this is a vector (circle) event
                self._handle_event2(
                    input_points, priority_queue, edges, bottomsite, delaunay
                )
            else:
                break

        halfedge = edges.leftend.right
        while halfedge is not edges.rightend:
            if halfedge:
                if halfedge.edge:
                    self._add_edge(halfedge.edge)
                halfedge = halfedge.right

        self.input_bbox = (sites.xmin, sites.ymin), (sites.xmax, sites.ymax)

    def _handle_event1(
        self,
        priority_queue: _PriorityQueue,
        edges: _EdgeList,
        bottomsite: _Site,
        newsite: _Site,
        delaunay: bool,
    ) -> None:
        # get first HalfEdge to the LEFT and RIGHT of the new site
        lbnd = edges.pop_leftbnd(newsite)
        assert lbnd
        rbnd = lbnd.right
        assert rbnd

        # if this halfedge has no edge, bot = bottom site
        # create a new edge that bisects
        bot = lbnd.right_site(bottomsite)
        edge = _Edge(bot, newsite, len(self._lines))
        self._add_bisector(edge, delaunay)

        # create a new HalfEdge, setting its orientation to LEFT and insert
        # this new bisector edge between the left and right vectors in
        # a linked list
        bisector = _HalfEdge(edge, _Edge.LEFT)
        edges.insert(lbnd, bisector)

        # if the new bisector intersects with thalfedge left edge,
        # remove the left edge's vertex, and put in the new one
        site = lbnd.intersect(bisector)
        if site is not None:
            priority_queue.delete(lbnd)
            priority_queue.insert(lbnd, site, newsite.distance(site))

        # create a new HalfEdge, setting its orientation to RIGHT
        # insert the new HalfEdge to the right of the original bisector
        lbnd = bisector
        bisector = _HalfEdge(edge, _Edge.RIGHT)
        edges.insert(lbnd, bisector)

        # if this new bisector intersects with the right HalfEdge
        site = bisector.intersect(rbnd)
        if site is not None:
            # push the HalfEdge into the ordered linked list
            # of vertices
            priority_queue.insert(bisector, site, newsite.distance(site))

    def _handle_event2(
        self,
        input_points: Sequence[TPoint],
        priority_queue: _PriorityQueue,
        edges: _EdgeList,
        bottomsite: _Site,
        delaunay: bool,
    ) -> None:
        # Pop the HalfEdge with the lowest vector off the ordered list
        # of vectors.
        # Get the HalfEdge to the left and right of the above HalfEdge
        # and also the HalfEdge to the right of the right HalfEdge
        lbnd = priority_queue.pop_min_halfedge()
        llbnd = lbnd.left
        rbnd = lbnd.right
        rrbnd = rbnd.right

        # get the Site to the left of the left HalfEdge and
        # to the right of the right HalfEdge which it bisects
        bot = lbnd.left_site(bottomsite)
        top = rbnd.right_site(bottomsite)
        orientation = _Edge.LEFT
        # If the site to the left of the event is higher than the Site
        # to the right of it, then swap the half edge orientation.
        if bot.y > top.y:
            bot, top = top, bot
            orientation = _Edge.RIGHT

        # Output the triple of sites (a Delaunay triangle)
        # stating that a circle goes through them
        if delaunay:
            mid = lbnd.right_site(bottomsite)
            self._add_triangle(
                input_points[bot.sitenum],
                input_points[top.sitenum],
                input_points[mid.sitenum],
            )

        # Add the vertex that caused this event to the Voronoi diagram.
        vertex = lbnd.vertex
        assert vertex
        self._add_vertex(vertex)
        # set the endpoint of the left and right HalfEdge to be
        # this vector.
        assert lbnd.edge
        assert rbnd.edge
        if lbnd.edge.set_endpoint(lbnd.orientation, vertex):
            self._add_edge(lbnd.edge)
        if rbnd.edge.set_endpoint(rbnd.orientation, vertex):
            self._add_edge(rbnd.edge)

        # delete the lowest HalfEdge, remove all vertex events to do with the
        # right HalfEdge and delete the right HalfEdge
        edges.delete(lbnd)
        priority_queue.delete(rbnd)
        edges.delete(rbnd)

        # Create an Edge (or line) that is between the two Sites.
        # This creates the formula of the line, and assigns
        # a line number to it
        edge = _Edge(bot, top, len(self._lines))
        self._add_bisector(edge, delaunay)

        # create a HalfEdge from the edge
        bisector = _HalfEdge(edge, orientation)

        # insert the new bisector to the right of the left HalfEdge
        # set one endpoint to the new edge to be the vector point
        # 'vertex'.
        # If the site to the left of this bisector is higher than
        # the right Site, then this endpoint is put in position 0;
        # otherwise in pos 1.
        edges.insert(llbnd, bisector)
        if edge.set_endpoint(_Edge.RIGHT - orientation, vertex):
            self._add_edge(edge)

        # if left HalfEdge and the new bisector don't intersect, then delete
        # the left HalfEdge, and reinsert it
        site = llbnd.intersect(bisector)
        if site is not None:
            priority_queue.delete(llbnd)
            priority_queue.insert(llbnd, site, bot.distance(site))

        # if right HalfEdge and the new bisector don't intersect,
        # then reinsert it
        site = bisector.intersect(rrbnd)
        if site is not None:
            priority_queue.insert(bisector, site, bot.distance(site))


class _Site:  # noqa: PLW1641 # no __hash__
    """Enumerated input points."""

    x: float
    y: float
    sitenum: int

    def __init__(self, x: float, y: float, sitenum: int = 0) -> None:
        self.x = x
        self.y = y
        # Index to original array of input points
        self.sitenum = sitenum

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _Site):
            return self.y == other.y and self.x == other.x
        return False

    def __lt__(self, other: object) -> bool:
        if isinstance(other, _Site):
            if self.y == other.y:
                return self.x < other.x
            return self.y < other.y
        return False

    def distance(self, other: _Site) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class _SiteList(list[_Site]):
    """A sorted list of sites with min/max point values.

    Sites will be ordered by (Y, X) but the site number will
    correspond to the initial order.
    """

    xmin = sys.float_info.max
    ymin = sys.float_info.max
    xmax = sys.float_info.min
    ymax = sys.float_info.min

    def __init__(self, input_points: Sequence[TPoint]) -> None:
        """Points should be 2-tuples with x and y value."""
        super().__init__()
        for i, p in enumerate(input_points):
            site = _Site(p[0], p[1], i)
            self.append(site)
            self.xmin = min(site.x, self.xmin)
            self.ymin = min(site.y, self.ymin)
            self.xmax = max(site.x, self.xmax)
            self.ymax = max(site.y, self.ymax)
        self.sort(key=lambda site: (site.y, site.x))


class _Edge:
    """A Voronoi diagram edge.

    This contains the line equation and endpoints of the Voronoi segment
    as well as the endpoints of the Delaunay segment this line is bisecting.
    """

    # Voronoi segment line equation
    a: float
    b: float
    c: float

    # Left and right end points of Voronoi segment.
    # By default there are no endpoints - they go to infinity.
    # Mutable
    endpoints: list[_Site | None]
    # The Delaunay segment this line is bisecting
    dsegment: tuple[_Site, _Site]
    # Index of line and delaunay segment
    edgenum: int

    LEFT = 0
    RIGHT = 1

    def __init__(self, site1: _Site, site2: _Site, edgenum: int) -> None:
        """Create a new Voronoi edge bisecting the two sites."""
        dx = site2.x - site1.x
        dy = site2.y - site1.y
        # get the slope of the line
        slope = (site1.x * dx) + (site1.y * dy) + ((dx * dx + dy * dy) / 2)
        if abs(dx) > abs(dy):
            # set formula of line, with x fixed to 1
            self.a = 1.0
            self.b = dy / dx
            self.c = slope / dx
        else:
            # set formula of line, with y fixed to 1
            self.b = 1.0
            self.a = dx / dy
            self.c = slope / dy
        self.endpoints = [None, None]
        self.dsegment = (site1, site2)
        self.edgenum = edgenum

    def set_endpoint(self, index: int, site: _Site) -> bool:
        """Set the value of one of the end points.

        Returns True if the other endpoint is not None.
        """
        assert index in (0, 1)
        self.endpoints[index] = site
        return self.endpoints[1 - index] is not None


class _HalfEdge:
    """Oriented edge."""

    # Orientation of edge endpoints, can be 0 or 1
    orientation: int
    ystar: float = sys.float_info.max
    # left HalfEdge in the edge list
    left: _HalfEdge
    # right HalfEdge in the edge list
    right: _HalfEdge
    # priority queue linked list pointer
    qnext: _HalfEdge | None = None
    # edge list Edge
    edge: _Edge | None = None
    vertex: _Site | None = None
    deleted: bool = False

    def __init__(
        self, edge: _Edge | None = None, orientation: int = _Edge.LEFT
    ) -> None:
        self.edge = edge
        self.orientation = orientation
        self.left = self
        self.right = self

    def __gt__(self, other: object) -> bool:
        return bool(
            isinstance(other, _HalfEdge)
            and self.vertex
            and other.vertex
            and (
                self.ystar > other.ystar
                or (
                    self.vertex.x > other.vertex.x and self.ystar == other.ystar
                )
            )
        )

    def left_site(self, default_site: _Site) -> _Site:
        """Site to the left of this half edge."""
        if not self.edge:
            return default_site
        if self.orientation == _Edge.LEFT:
            return self.edge.dsegment[_Edge.LEFT]
        return self.edge.dsegment[_Edge.RIGHT]

    def right_site(self, default_site: _Site) -> _Site:
        """Site to the right of this half edge."""
        if not self.edge:
            return default_site
        if self.orientation == _Edge.LEFT:
            return self.edge.dsegment[_Edge.RIGHT]
        return self.edge.dsegment[_Edge.LEFT]

    def is_left_of_site(self, site: _Site) -> bool:
        """Returns True if site is to right of this half edge."""
        assert self.edge
        edge: _Edge = self.edge
        topsite = edge.dsegment[1]
        right_of_site = site.x > topsite.x

        if right_of_site and self.orientation == _Edge.LEFT:
            return True

        if not right_of_site and self.orientation == _Edge.RIGHT:
            return False

        if _float_eq(edge.a, 1.0):
            dyp = site.y - topsite.y
            dxp = site.x - topsite.x
            fast = False
            if (not right_of_site and edge.b < 0) or (
                right_of_site and edge.b >= 0
            ):
                above = dyp >= edge.b * dxp
                fast = above
            else:
                above = site.x + site.y * edge.b > edge.c
                if edge.b < 0:
                    above = not above
                if not above:
                    fast = True
            if not fast:
                dxs = topsite.x - (edge.dsegment[0]).x
                above = edge.b * (dxp * dxp - dyp * dyp) < dxs * dyp * (
                    1.0 + 2.0 * dxp / dxs + edge.b * edge.b
                )
                if edge.b < 0:
                    above = not above
        else:  # edge.b == 1.0
            y_int = edge.c - edge.a * site.x
            t1 = site.y - y_int
            t2 = site.x - topsite.x
            t3 = y_int - topsite.y
            above = (t1 * t1) > (t2 * t2 + t3 * t3)

        return above if self.orientation == _Edge.LEFT else not above

    def intersect(self, other: _HalfEdge) -> _Site | None:
        """Create a new site where the HalfEdges el1 and el2 intersect."""
        edge1 = self.edge
        edge2 = other.edge

        # bail if the two edges bisect the same parent
        if not edge1 or not edge2 or edge1.dsegment[1] is edge2.dsegment[1]:
            return None

        dst = edge1.a * edge2.b - edge1.b * edge2.a
        if _float_eq(dst, 0.0):
            return None

        xint = (edge1.c * edge2.b - edge2.c * edge1.b) / dst
        yint = (edge2.c * edge1.a - edge1.c * edge2.a) / dst
        if edge1.dsegment[1] < edge2.dsegment[1]:
            half_edge = self
            edge = edge1
        else:
            half_edge = other
            edge = edge2

        right_of_site = xint >= edge.dsegment[1].x
        if (right_of_site and half_edge.orientation == _Edge.LEFT) or (
            not right_of_site and half_edge.orientation == _Edge.RIGHT
        ):
            return None

        # create a new site at the point of intersection - this is a new
        # vector event waiting to happen
        return _Site(xint, yint)


class _EdgeList:
    """A double linked list of _HalfEdges."""

    hashsize: int
    hashtable: list[_HalfEdge | None]
    hashscale: float
    xmin: float
    leftend: _HalfEdge
    rightend: _HalfEdge

    def __init__(self, xmin: float, xmax: float, nsites: int) -> None:
        self.hashsize = int(2 * math.sqrt(nsites + 4))
        self.hashtable = [None] * self.hashsize
        self.hashscale = (xmax - xmin) * self.hashsize
        self.xmin = xmin
        self.leftend = _HalfEdge()
        self.rightend = _HalfEdge()
        self.leftend.right = self.rightend
        self.rightend.left = self.leftend
        self.hashtable[0] = self.leftend
        self.hashtable[-1] = self.rightend

    def insert(self, left: _HalfEdge, half_edge: _HalfEdge) -> None:
        """Insert halfedge."""
        half_edge.left = left
        half_edge.right = left.right
        left.right.left = half_edge
        left.right = half_edge

    def delete(self, half_edge: _HalfEdge) -> None:
        """Delete halfedge."""
        half_edge.left.right = half_edge.right
        half_edge.right.left = half_edge.left
        half_edge.deleted = True

    def pop_leftbnd(self, site: _Site) -> _HalfEdge | None:
        # Use hash table to get close to desired halfedge
        bucket = int((site.x - self.xmin) / self.hashscale)
        if bucket < 0:
            bucket = 0
        elif bucket >= self.hashsize:
            bucket = self.hashsize - 1

        half_edge = self._get_bucket_entry(bucket)
        i = 1
        while not half_edge:
            half_edge = self._get_bucket_entry(bucket - i)
            if half_edge is None:
                half_edge = self._get_bucket_entry(bucket + i)
            i += 1
            if (bucket - i) < 0 or (bucket + i) >= self.hashsize:
                break

        if half_edge:
            # Now search linear list of halfedges for the correct one
            if half_edge is self.leftend or (
                half_edge is not self.rightend
                and half_edge.is_left_of_site(site)
            ):
                half_edge = half_edge.right
                while (
                    half_edge is not self.rightend
                    and half_edge.is_left_of_site(site)
                ):
                    half_edge = half_edge.right
                half_edge = half_edge.left
            else:
                half_edge = half_edge.left
                while (
                    half_edge is not self.leftend
                    and not half_edge.is_left_of_site(site)
                ):
                    half_edge = half_edge.left

            if 0 < bucket < self.hashsize - 1:
                self.hashtable[bucket] = half_edge

        return half_edge

    # Get the bucket entry from hash table, pruning any deleted nodes
    def _get_bucket_entry(self, b: int) -> _HalfEdge | None:
        half_edge = self.hashtable[b]
        if half_edge and half_edge.deleted:
            self.hashtable[b] = None
            half_edge = None
        return half_edge


class _PriorityQueue:
    """Priority queue of halfedges."""

    ymin: float
    deltay: float
    hashsize: int
    count: int
    minidx: int
    hashtable: list[_HalfEdge]

    def __init__(self, ymin: float, ymax: float, nsites: int) -> None:
        self.ymin = ymin
        self.deltay = ymax - ymin
        self.hashsize = int(4 * math.sqrt(nsites))
        self.count = 0
        self.minidx = 0
        self.hashtable = [_HalfEdge() for dummy in range(self.hashsize)]

    def __len__(self) -> int:
        return self.count

    def is_empty(self) -> bool:
        return self.count == 0

    def insert(self, half_edge: _HalfEdge, site: _Site, offset: float) -> None:
        half_edge.vertex = site
        half_edge.ystar = site.y + offset
        last = self.hashtable[self._get_bucket(half_edge)]
        qnext = last.qnext
        while qnext and half_edge > qnext:
            last = qnext
            qnext = last.qnext
        half_edge.qnext = last.qnext
        last.qnext = half_edge
        self.count += 1

    def delete(self, half_edge: _HalfEdge) -> None:
        if half_edge.vertex:
            last = self.hashtable[self._get_bucket(half_edge)]
            while last.qnext is not half_edge:
                assert last.qnext
                last = last.qnext
            last.qnext = half_edge.qnext
            half_edge.vertex = None
            self.count -= 1

    def get_min_point(self) -> _Site:
        assert not self.is_empty()
        while self.hashtable[self.minidx].qnext is None:
            self.minidx += 1
        halfedge = self.hashtable[self.minidx].qnext
        assert halfedge
        assert halfedge.vertex
        return _Site(halfedge.vertex.x, halfedge.ystar)

    def pop_min_halfedge(self) -> _HalfEdge:
        curr = self.hashtable[self.minidx].qnext
        assert curr
        self.hashtable[self.minidx].qnext = curr.qnext
        self.count -= 1
        return curr

    def _get_bucket(self, halfedge: _HalfEdge) -> int:
        hashval = (halfedge.ystar - self.ymin) / self.deltay
        bucket = max(int(hashval * self.hashsize), 0)
        if bucket >= self.hashsize:
            bucket = self.hashsize - 1
        self.minidx = min(bucket, self.minidx)
        return bucket


def _float_eq(a: float, b: float) -> bool:
    """Compare two floats for relative equality.

    See:
        http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition
        for a discussion of floating point comparisons.
    """
    norm = max(abs(a), abs(b))
    return (norm < EPSILON) or (abs(a - b) < (EPSILON * norm))
    # return abs(a - b) < EPSILON


def jiggle(point: TPoint) -> TPoint:
    """Move a point in a random direction by a very small random distance.

    Useful for when input is degenerate (i.e. when points are collinear.)
    avoiding divide by zero exceptions.

    Args:
        point: The point as a 2-tuple of the form (x, y)

    Returns:
        A new jiggled point as a 2-tuple
    """
    x, y = point
    norm_x = EPSILON * abs(x)
    norm_y = EPSILON * abs(y)
    sign = random.choice((-1, 1))
    return (
        x + random.uniform(norm_x * 10, norm_x * 100) * sign,
        y + random.uniform(norm_y * 10, norm_y * 100) * sign,
    )
