"""Polyline functions.

A polyline being a series of connected straight line segments.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Iterator, Sequence
from itertools import starmap
from typing import TYPE_CHECKING

from . import polygon
from .line import Line, TLine
from .point import P, TPoint

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

TPolyline: TypeAlias = Iterable[TPoint]
TPolypath: TypeAlias = Iterable[TLine]

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


def polypath_to_polyline(polypath: Iterable[TLine]) -> Iterator[TPoint]:
    """Convert a polypath to a polyline.

    Args:
        polypath: An iterable of line segments.

    Returns:
        A polyline as an iterable of vertices.
    """
    p2: TPoint | None = None
    for p1, p2 in polypath:  # noqa: B007
        yield p1
    if p2:
        yield p2


def polyline_to_polypath(polyline: Iterable[TPoint]) -> Iterator[Line]:
    """Convert a polyline to a polypath.

    Args:
        polyline: An iterable of vertices.

    Returns:
        An iterable of line segments.
    """
    return starmap(Line, pairwise(polyline))


def is_closed(polypath: Sequence[TLine]) -> bool:
    """True if polypath is a closed polygon."""
    return bool(polypath[0][0] == polypath[-1][1])


def length(polypath: Iterable[TLine]) -> float:
    """Total cumulative length of polypath."""
    return float(
        sum(math.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]) for s in polypath)
    )


def closest_point(
    polypath: Iterable[Line], p: P, prefer_normal: bool = False
) -> P | None:
    """Get the closest point on a polyline to point `p`.

    Args:
        polypath: An iterable of line segments.
        p: Reference point.
        prefer_normal: Give preference to normal projection
            on line segment over closest endpoint.
    """
    if not polypath:
        return None
    closest_p = None
    closest_p_alt = None
    d = sys.float_info.max
    d_alt = sys.float_info.max
    p_normal_alt = None
    for segment in polypath:
        if prefer_normal:
            p_normal, p_normal_alt = _normal_projection_point(
                segment.p1, segment.p2, p
            )
        else:
            p_normal = segment.normal_projection_point(p, segment=True)
        if p_normal:
            dnorm = p.distance(p_normal)
            if dnorm < d:
                closest_p = p_normal
                d = dnorm
        elif not closest_p and p_normal_alt:
            dnorm = p.distance(p_normal_alt)
            if dnorm < d_alt:
                closest_p_alt = p_normal_alt
                d_alt = dnorm
    return closest_p if closest_p else closest_p_alt


def _normal_projection_point(p1: P, p2: P, p: P) -> tuple[P | None, P | None]:
    """Projection on line segment.

    Returns:
        The point on this line segment that corresponds to
        the projection of the specified point.
    """
    v1 = p2 - p1
    u = v1.normal_projection(p - p1)
    if u <= 0:
        return (None, p1)
    if u >= 1.0:
        return (None, p2)
    return (p1 + v1 * u, None)


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


def segment_intersects(polyline: TPolyline, segment: Line) -> bool:
    """Return True if the segment intersects polyline."""
    for polyseg in polyline_to_polypath(polyline):
        if segment.intersection(polyseg, segment=True):
            return True
    return False
