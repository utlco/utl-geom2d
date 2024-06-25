"""Polyline functions.

A polyline being a series of connected straight line segments.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from itertools import starmap
from typing import TYPE_CHECKING

from . import polygon
from .line import Line, TLine
from .point import P, TPoint
from .util import pairwise

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

TPolyLine: TypeAlias = Sequence[TPoint]
TPolyPath: TypeAlias = Sequence[TLine]


def polypath_to_polyline(polypath: Iterable[TLine]) -> Iterator[P]:
    """Convert a polypath to a polyline.

    Args:
        polypath: An iterable of line segments.

    Returns:
        A polyline as an iterable of vertices.
    """
    p2: TPoint | None = None
    for seg in polypath:
        # Despite typing, this allows CubicBeziers and Arcs
        # by treating them as line segments defined by endpoints.
        p2 = seg[-1]
        yield P(seg[0])
    if p2:
        yield P(p2)


def polyline_to_polypath(polyline: Iterable[TPoint]) -> Iterator[Line]:
    """Convert a polyline to a polypath.

    Args:
        polyline: An iterable of vertices.

    Returns:
        An iterable of line segments.
    """
    return starmap(Line, pairwise(polyline))


def polypath_is_closed(polypath: Sequence[TLine]) -> bool:
    """True if polypath is a closed polygon."""
    return bool(polypath[0][0] == polypath[-1][1])


def polypath_reversed(polypath: Sequence[Line]) -> list[Line]:
    """Reverse a polypath."""
    return [seg.path_reversed() for seg in reversed(polypath)]


def polypath_length(polypath: Iterable[TLine]) -> float:
    """Total cumulative length of polypath."""
    return float(
        sum(math.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]) for s in polypath)
    )


def polyline_length(polyline: Iterable[TPoint]) -> float:
    """Total cumulative length of polyline."""
    return float(
        sum(
            math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            for p1, p2 in pairwise(polyline)
        )
    )


def polypath_length_to(polypath: Iterable[Line], p: TPoint) -> float:
    """Distance along polypath from start point to a point on the polypath."""
    length: float = 0
    for seg in polypath:
        if seg.point_on_line(p, segment=True):
            return length + seg.p1.distance(p)
        length += seg.length()
    return length  # Default is total length of polyline


def polyline_length_to(polyline: Iterable[TPoint], p: TPoint) -> float:
    """Distance along polyline from start point to a point on the polyline."""
    length: float = 0
    for p1, p2 in pairwise(polyline):
        seg = Line(p1, p2)
        if seg.point_on_line(p, segment=True):
            return length + seg.p1.distance(p)
        length += seg.length()
    return length  # Default is total length of polyline


def closest_point(
    polyline: Iterable[TPoint], p: TPoint, vertices_only: bool = False
) -> P:
    """Get the closest point on a polyline to point `p`.

    Args:
        polyline: An iterable of vertices.
        p: Reference point.
        vertices_only: Closest vertices only.
            Otherwise normal projection on closest segment.
    """
    p = P(p)
    polyiter = iter(polyline)
    p1 = next(polyiter)
    dmin = p.distance2(p1)
    closest_p = P(p1)
    for p2 in polyiter:
        p_mu = p2
        if not vertices_only:
            segment = Line(p1, p2)
            mu = segment.normal_projection(p)
            if mu <= 0:
                p_mu = p1
            elif mu < 1:
                p_mu = segment.point_at(mu)
        d = p.distance2(p_mu)
        if d < dmin:
            dmin = d
            closest_p = P(p_mu)
        p1 = p2

    return closest_p


def is_inside(polypath1: Iterable[TLine], polypath2: Iterable[TLine]) -> bool:
    """Is polypath1 inside polypath2?"""
    # TODO: they should have no intersections
    polygon2 = list(polypath_to_polyline(polypath2))
    segment: TLine | None = None
    for segment in polypath1:
        if not polygon.point_inside(polygon2, segment[0]):
            return False
    if segment:
        return polygon.point_inside(polygon2, segment[1])
    return False


def segment_intersects(polyline: Iterable[TPoint], segment: Line) -> bool:
    """Return True if the segment intersects polyline."""
    for polyseg in polyline_to_polypath(polyline):
        if segment.intersection(polyseg, segment=True):
            return True
    return False
