"""Connect Line/Arc segments with a fillet arc."""

from __future__ import annotations

import collections
import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, TypeVar

from . import const, polyline
from .arc import Arc
from .bezier import CubicBezier
from .line import Line
from .point import P

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .point import TPoint

_DEBUG = False  # const.DEBUG

if _DEBUG:
    from . import debug, plotpath

_T = TypeVar('_T')


def fillet_path(
    path: Iterable[Line | Arc | CubicBezier],
    radius: float,
    fillet_close: bool = True,
    adjust_radius: bool = False,
) -> Sequence[Line | Arc]:
    """Fillet a path of connected Line and Arc segments.

    Attempt to insert a circular arc of the specified radius
    to connect adjacent path segments.

    Args:
        path: An iterable of connected Line or Arc segments.
        radius: The radius of the fillet arc.
        fillet_close: If True and the path is closed then
            add a terminating fillet.
        adjust_radius: Shrink radius to fit segments if they
            are too short to accommodate the specified radius.
            This only works on line->line fillets.

    Returns:
        A new path with fillet arcs. If no fillets are created then
        the original path will be returned.
    """
    return _fillet_path_iter(
        _path_iter(path), radius, fillet_close, adjust_radius
    )


def fillet_polygon(
    poly: Iterable[TPoint],
    radius: float,
    fillet_close: bool = True,
    adjust_radius: bool = False,
) -> Sequence[Line | Arc]:
    """Fillet polygon edges.

    Attempt to insert a circular arc of the specified radius
    connecting adjacent polygon segments.

    Args:
        poly: A list of polygon vertices.
        radius: The radius of the fillet arc.
        fillet_close: If True and the path is closed then
            add a terminating fillet. Default is True.
        adjust_radius: Shrink radius to fit segments if they
            are too short to accommodate the specified radius.
            This only works on line->line fillets.

    Returns:
        A new path with fillet arcs as a list of Line and Arc segments.
        If no fillets are created then the original path will be returned.
    """
    return _fillet_path_iter(
        polyline.polyline_to_polypath(poly), radius, fillet_close, adjust_radius
    )


def _path_iter(
    path: Iterable[Line | Arc | CubicBezier],
) -> Iterator[Line | Arc]:
    """A segment path iterator.

    Converts CubicBezier curves to biarc segments.
    """
    for seg in iter(path):
        if isinstance(seg, CubicBezier):
            yield from seg.biarc_approximation()
        else:
            yield seg


def _fillet_path_iter(
    path_iter: Iterator[Line | Arc],
    radius: float,
    fillet_close: bool,
    adjust_radius: bool,
) -> Sequence[Line | Arc]:
    """Fillet a path of connected Line and Arc segments."""
    new_path: list[Line | Arc] = []
    path_iter = PathLookahead(path_iter)
    seg1 = next(path_iter)
    if _DEBUG:
        debug.draw_point(seg1.p1, color='#00ff00')
        plotpath.draw_segment(seg1, color='#00ffff')
    for seg2 in path_iter:
        if _DEBUG:
            plotpath.draw_segment(seg2, color='#00ffff')
        # Already G1?
        if const.float_eq(seg1.end_tangent_angle(), seg2.start_tangent_angle()):
            new_path.append(seg1)
            seg1 = seg2
            continue
        new_segs = fillet_segments(seg1, seg2, radius, adjust_radius)
        # Try to connect to forward adjacent segments
        while not new_segs:
            seg2_a = path_iter.lookahead()
            if not seg2_a:
                break
            new_segs = fillet_segments(seg1, seg2_a, radius, adjust_radius)
            if new_segs:
                path_iter.reset()
        if new_segs:
            new_path.extend(new_segs[:-1])
            seg1 = new_segs[-1]
        else:
            new_path.append(seg1)
            seg1 = seg2
    new_path.append(seg1)

    # Close the path with a fillet (if it is a closed path)
    if fillet_close and new_path[0].p1 == new_path[-1].p2:
        new_segs = fillet_segments(
            new_path[-1], new_path[0], radius, adjust_radius
        )
        if new_segs:
            new_path[-1] = new_segs[0]
            new_path.append(new_segs[1])
            new_path[0] = new_segs[2]

    return new_path


def fillet_segments(
    seg1: Line | Arc,
    seg2: Line | Arc,
    radius: float,
    adjust_radius: bool = False,
) -> tuple[Line | Arc, Arc, Line | Arc] | None:
    """Try to create a fillet between two segments.

    Args:
        seg1: First segment, an Arc or a Line.
        seg2: Second segment, an Arc or a Line.
        radius: Fillet radius.
        adjust_radius: Shrink radius to fit segments if they
            are too short to accommodate the specified radius.

    Returns:
        A tuple containing the adjusted segments and fillet arc:
        (seg1, fillet_arc, seg2)
        Returns None if the segments cannot be connected
        with a fillet arc (either they are too small
        or somehow degenerate.)
    """
    farc = create_fillet_arc(seg1, seg2, radius, adjust_radius=adjust_radius)
    if farc:
        return connect_fillet(seg1, farc, seg2)
    return None


def connect_fillet(
    seg1: Line | Arc,
    farc: Arc,
    seg2: Line | Arc,
) -> tuple[Line | Arc, Arc, Line | Arc] | None:
    """Connect a fillet arc between two segments.

    Args:
        seg1: First segment, an Arc or a Line.
        farc: The fillet arc.
        seg2: Second segment, an Arc or a Line.

    Returns:
        A tuple containing the adjusted segments and fillet arc:
        (seg1, fillet_arc, seg2)
    """
    new_seg1: Line | Arc
    new_seg2: Line | Arc

    if isinstance(seg1, Line):
        new_seg1 = Line(seg1.p1, farc.p1)
        if isinstance(seg2, Line):
            # Connect Line->Fillet->Line
            new_seg2 = Line(farc.p2, seg2.p2)
        else:
            # Connect Line->Fillet->Arc
            new_angle = seg2.angle - seg2.center.angle2(seg2.p1, farc.p2)
            new_seg2 = Arc(
                farc.p2, seg2.p2, seg2.radius, new_angle, seg2.center
            )
    else:
        new_angle = seg1.angle - seg1.center.angle2(farc.p1, seg1.p2)
        new_seg1 = Arc(seg1.p1, farc.p1, seg1.radius, new_angle, seg1.center)
        if isinstance(seg2, Line):
            # Connect Arc->Fillet->Line
            new_seg2 = Line(farc.p2, seg2.p2)
        else:
            # Connect Arc->Fillet->Arc
            new_angle = seg2.angle - seg2.center.angle2(seg2.p1, farc.p2)
            new_seg2 = Arc(
                farc.p2, seg2.p2, seg2.radius, new_angle, seg2.center
            )

    return (new_seg1, farc, new_seg2)


def create_fillet_arc(
    seg1: Line | Arc,
    seg2: Line | Arc,
    radius: float,
    adjust_radius: bool = False,
) -> Arc | None:
    """Create a fillet arc between two Line/Arc segments.

    Args:
        seg1: A Line or Arc.
        seg2: A Line or Arc connected to seg1.
        radius: The radius of the fillet.
        adjust_radius: Shrink radius to fit segments if they
            are too short to accommodate the specified radius.
            This only works on line->line fillets.

    Returns:
        An Arc, or None if the fillet radius is too big to fit or
        if the two segments are already tangentially (G1) connected.
    """
    # Already G1? Then bail
    if const.float_eq(seg1.end_tangent_angle(), seg2.start_tangent_angle()):
        return None

    if adjust_radius:
        # Shrink radius to fit segment lengths and angle
        radius = _adjusted_radius(seg1, seg2, radius)

    fillet_center = _fillet_center(seg1, seg2, radius)
    if fillet_center:
        return _fillet_arc(seg1, seg2, fillet_center)

    return None


def _fillet_arc(
    seg1: Line | Arc, seg2: Line | Arc, fillet_center: P
) -> Arc | None:
    """Construct the fillet arc."""
    # debug.draw_point(fillet_center, color='#ffff00')
    fp1 = seg1.normal_projection_point(fillet_center, segment=True)
    fp2 = seg2.normal_projection_point(fillet_center, segment=True)
    if _DEBUG:
        if fp1:
            debug.draw_point(fp1, color='#ff0080')
        if fp2:
            debug.draw_point(fp2, color='#ff0080')

    # fillet_arc: Arc | None = None
    if fp1 and fp2 and fp1 != fp2:
        fillet_arc = Arc.from_two_points_and_center(fp1, fp2, fillet_center)
        if _DEBUG and fillet_arc:
            debug.draw_arc(fillet_arc, color='#ff0080')
        return fillet_arc

    return None


def _fillet_center(
    seg1: Line | Arc, seg2: Line | Arc, radius: float
) -> P | None:
    """Find the fillet arc center point."""
    if isinstance(seg1, Line):
        if isinstance(seg2, Line):
            return _fillet_center_line_line(seg1, seg2, radius)
        return _fillet_center_line_arc(seg1, seg2, radius)
    if isinstance(seg2, Line):
        return _fillet_center_arc_line(seg1, seg2, radius)
    return _fillet_center_arc_arc(seg1, seg2, radius)


def _fillet_center_line_line(
    seg1: Line,
    seg2: Line,
    radius: float,
) -> P | None:
    offset = radius * seg1.which_side(seg2.p2)
    offset_seg1 = seg1.offset(offset)
    offset_seg2 = seg2.offset(offset)
    fillet_center = offset_seg1.intersection(offset_seg2, segment=True)
    if _DEBUG:
        _debug_draw_offsets(offset_seg1, offset_seg2, fillet_center, radius)
    return fillet_center


def _fillet_center_line_arc(
    seg1: Line,
    seg2: Arc,
    radius: float,
) -> P | None:
    if seg2.radius < radius:
        return None
    p = P.from_polar(1, seg2.start_tangent_angle()) + seg1.p2
    offset = radius * seg1.which_side(p)
    offset_seg1 = seg1.offset(offset)
    offset_seg2 = seg2.offset(offset * -seg2.direction())
    intersections = offset_seg2.intersect_line(
        offset_seg1, on_arc=True, on_line=True
    )
    if intersections:
        fillet_center = intersections[0]
        if _DEBUG:
            _debug_draw_offsets(offset_seg1, offset_seg2, fillet_center, radius)
        return fillet_center
    return None


def _fillet_center_arc_line(
    seg1: Arc,
    seg2: Line,
    radius: float,
) -> P | None:
    if seg1.radius < radius:
        return None
    p = P.from_polar(1, seg1.end_tangent_angle()) + seg1.p2
    offset = radius * seg2.which_side(p)
    offset_seg1 = seg1.offset(offset * seg1.direction())
    offset_seg2 = seg2.offset(-offset)
    # debug.draw_arc(seg1, color='#0080ff')
    # debug.draw_line(seg2, color='#0080ff')
    intersections = offset_seg1.intersect_line(
        offset_seg2, on_arc=True, on_line=True
    )
    if intersections:
        fillet_center = intersections[0]
        if _DEBUG:
            _debug_draw_offsets(offset_seg1, offset_seg2, fillet_center, radius)
        return fillet_center
    return None


def _fillet_center_arc_arc(
    seg1: Arc,
    seg2: Arc,
    radius: float,
) -> P | None:
    if seg1.radius < radius or seg2.radius < radius:
        return None
    offset = radius * -seg1.which_side_angle(seg2.start_tangent_angle())
    offset_seg1 = seg1.offset(offset * seg1.direction())
    offset_seg2 = seg2.offset(offset * seg2.direction())
    intersections = offset_seg1.intersect_arc(offset_seg2, on_arc=True)
    if intersections:
        fillet_center = intersections[0]
        if _DEBUG:
            _debug_draw_offsets(offset_seg1, offset_seg2, fillet_center, radius)
        return fillet_center
    return None


def _adjusted_radius(
    seg1: Line | Arc, seg2: Line | Arc, radius: float
) -> float:
    # For now this only works for line segments...
    if isinstance(seg1, Line) and isinstance(seg2, Line):
        # First see if the segments are long enough to
        # accommodate the fillet radius.
        # And if not, adjust the radius to fit.
        t = abs(math.tan(seg1.p2.angle2(seg1.p1, seg2.p2) / 2))
        if t > 0:
            # Distance from seg1.p2 to fillet arc intersection
            d = radius / t
            lmin = min(seg1.length() / 2, seg2.length() / 2)
            if d > lmin:
                return lmin * t
    return radius


def _debug_draw_offsets(
    offset_seg1: Line | Arc,
    offset_seg2: Line | Arc,
    fillet_center: P | None,
    radius: float,
) -> None:
    plotpath.draw_segment(offset_seg1, color='#ff8000')
    plotpath.draw_segment(offset_seg2, color='#ff8000')
    if fillet_center:
        debug.draw_circle(fillet_center, radius, color='#ff80ff')
        debug.draw_point(fillet_center)


def _debug_draw_farc_endpoints(fp1: TPoint | None, fp2: TPoint | None) -> None:
    if fp1:
        debug.draw_point(fp1, color='#ff0080')
    if fp2:
        debug.draw_point(fp2, color='#ff0080')


_marker = object()


class PathLookahead(Iterator[_T]):
    """Wrap an iterator to allow lookahead."""

    _source: Iterator[_T]
    _cache: collections.deque[_T]
    _maxlen: int | None

    def __init__(self, iterable: Iterable, maxlen: int | None = None) -> None:
        """Constructor."""
        self._source = iter(iterable)
        self._cache = collections.deque(maxlen=maxlen)
        self._maxlen = maxlen

    def __iter__(self) -> PathLookahead:
        """iter()."""
        return self

    def __bool__(self) -> bool:
        """bool()."""
        try:
            self.lookahead()
        except StopIteration:
            return False
        return True

    def __next__(self) -> _T:
        """Advance iterator and return next object."""
        if self._cache:
            return self._cache.popleft()

        return next(self._source)

    def lookahead(self) -> _T | None:
        """Return the next item and cache it.

        Return ``default`` if there are no items left. If ``default`` is not
        provided, raise ``StopIteration``.

        """
        if len(self._cache) == self._maxlen:
            return None
        try:
            self._cache.append(next(self._source))
        except StopIteration:
            return None
        return self._cache[-1]

    def reset(self) -> None:
        """Reset lookahead cache."""
        self._cache.clear()
