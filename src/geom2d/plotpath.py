"""Debug output support for geometry package."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Union

from . import arc, bezier, debug, ellipse, line, point

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

TSeg: TypeAlias = Union[
    Sequence[float],
    Sequence[Sequence[float]],
    point.P,
    line.Line,
    arc.Arc,
    ellipse.Ellipse,
    bezier.CubicBezier,
]

_POINT_LEN = 2
_LINE_LEN = 2
_ARC_LEN = 5


def _is_point(seg: TSeg) -> bool:
    return isinstance(seg, point.P) or (
        isinstance(seg, Sequence)
        and len(seg) == _POINT_LEN
        and isinstance(seg[0], float)
    )


def draw_path(
    path: Iterable[TSeg],
    color: str = '#ff0000',
    width: str | float = '1px',
    opacity: float = 1,
    verbose: bool = False,
) -> None:
    """Debug SVG output for paths consisting of segments or points."""
    lastp: point.TPoint | None = None
    for seg in path:
        if lastp and _is_point(seg):
            debug.draw_line(
                (lastp, typing.cast(point.TPoint, seg)),
                color=color,
                width=width,
                opacity=opacity,
            )
        # TODO: mark gaps in non-G0 paths
        lastp = draw_segment(
            seg, color=color, width=width, opacity=opacity, verbose=verbose
        )


def draw_segment(
    seg: TSeg,
    color: str = '#c00000',
    width: str | float = '1px',
    opacity: float = 1,
    verbose: bool = False,
) -> point.TPoint | None:
    """Draw a geom object, returns the last segment point."""
    if isinstance(seg, arc.Arc) or (
        isinstance(seg, Sequence)
        and len(seg) == _ARC_LEN
        and isinstance(seg[0], Sequence)
    ):
        debug.draw_arc(
            typing.cast(arc.Arc, seg),
            color=color,
            width=width,
            opacity=opacity,
            verbose=verbose,
        )
        return typing.cast(point.TPoint, seg[1])

    if isinstance(seg, line.Line) or (
        isinstance(seg, Sequence)
        and len(seg) == _LINE_LEN
        and isinstance(seg[0], Sequence)
    ):
        debug.draw_line(
            typing.cast(line.TLine, seg),
            color=color,
            width=width,
            opacity=opacity,
            verbose=verbose,
        )
        return typing.cast(point.TPoint, seg[1])

    if _is_point(seg):
        # Not actually a segment, but just in case...
        debug.draw_point(
            typing.cast(point.TPoint, seg), color=color, opacity=opacity
        )
        return typing.cast(point.TPoint, seg)

    if isinstance(seg, ellipse.EllipticalArc):
        ellipse.draw_ellipse(
            seg, color=color, width=width, opacity=opacity, verbose=verbose
        )
        return seg.p2

    if isinstance(seg, bezier.CubicBezier):
        bezier.draw_bezier(
            seg, color=color, width=width, opacity=opacity, verbose=verbose
        )
        return seg.p2

    return None
