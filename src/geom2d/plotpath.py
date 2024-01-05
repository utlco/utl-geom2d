# -----------------------------------------------------------------------------
# Copyright 2012-2016 Claude Zervas
# email: claude@utlco.com
# -----------------------------------------------------------------------------
"""Debug output support for geometry package."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from . import arc, bezier, debug, ellipse, line, point

if TYPE_CHECKING:
    from collections.abc import Iterable

    import etree
    from typing_extensions import TypeAlias

TSeg: TypeAlias = Union[
    point.P, line.Line, arc.Arc, ellipse.Ellipse, bezier.CubicBezier
]


def plot_path(
    path: Iterable[TSeg],
    color: str = '#000000',
    parent: etree.Element | None = None,
) -> None:
    """Debug output for paths."""
    #     prev_seg = None
    segnum = 1
    for seg in path:
        # logger.debug('\nSegment %d: %s' % (segnum, str(seg)))
        # if prev_seg is not None and prev_seg.p2 != seg.p1:
        #     logger.debug(
        #         'path not continuous: p1=%s, p2=%s' % (
        #         str(prev_seg.p2), str(seg.p1)))
        #     prev_seg.p2.svg_plot(color='#0000ff')
        #     seg.p1.svg_plot(color='#0000ff')
        #     for name in inline_hint_attrs(seg):
        #         logger.debug('%s=%s' % str(getattr(seg, name)))
        draw_obj(seg, color=color, parent=parent)
        # prev_seg = seg
        segnum += 1


def draw_obj(
    obj: TSeg, color: str = '#c00000', parent: etree.Element = None
) -> None:
    """Draw a geom object."""
    if isinstance(obj, point.P):
        debug.draw_point(obj, color=color, parent=parent)
    elif isinstance(obj, line.Line):
        debug.draw_line(obj, color=color, parent=parent)
    elif isinstance(obj, arc.Arc):
        debug.draw_arc(obj, color=color, parent=parent)
    elif isinstance(obj, ellipse.Ellipse):
        debug.draw_ellipse(obj, color=color, parent=parent)
    elif isinstance(obj, bezier.CubicBezier):
        debug.draw_bezier(obj, color=color, parent=parent)
