# -----------------------------------------------------------------------------
# Copyright 2012-2016 Claude Zervas
# email: claude@utlco.com
# -----------------------------------------------------------------------------
"""Debug SVG output support for geometry package."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import etree
    from inkext.svg import SVGContext

    from .arc import Arc
    from .bezier import CubicBezier
    from .ellipse import Ellipse
    from .line import TLine
    from .point import TPoint

# svg.SVGContext singleton for drawing debug output.
# Debug drawing is effectively disabled if this is None (default).
svg_context: SVGContext = None  # pylint: disable=invalid-name


def set_svg_context(context: SVGContext) -> None:
    """Initialize this module with an SVGContext.

    The SVGContext will be used for debug output by draw...() methods.
    """
    global svg_context  # noqa: PLW0603 pylint: disable=global-statement
    svg_context = context


def draw_point(
    point: TPoint,
    radius: str = '2px',
    color: str = '#000000',
    parent: etree.Element | None = None,
) -> None:
    """Draw a dot. Useful for debugging and testing."""
    if svg_context:
        svg_context.create_circle(
            point,
            svg_context.unit2uu(radius),
            style=f'fill:{color};stroke:none',
            parent=parent,
        )


def draw_line(
    line: TLine,
    color: str = '#c00000',
    width: str = '1px',
    opacity: float = 1,
    verbose: bool = False,
    parent: etree.Element | None = None,
) -> None:
    """Draw an SVG line segment for debugging/testing."""
    if svg_context:
        style = _linestyle(color, width, opacity)
        svg_context.create_line(line[0], line[1], style, parent=parent)
        if verbose:
            draw_point(line[0], color=color)
            draw_point(line[1], color=color)


def draw_poly(
    vertices: Sequence[TPoint],
    color: str = '#c00000',
    width: str = '1px',
    verbose: bool = False,
    parent: etree.Element | None = None,
    close_poly: bool = True,
    style: str | None = None,
) -> None:
    """Draw an SVG polygon."""
    if svg_context:
        if not style:
            style = _linestyle(color, width)
        svg_context.create_polygon(
            vertices, close_polygon=close_poly, style=style, parent=parent
        )
        if verbose:
            for p in vertices:
                draw_point(p, color=color)


def draw_arc(
    arc: Arc,
    color: str = '#cccc99',
    width: str = '1px',
    verbose: bool = False,
    parent: etree.Element | None = None,
) -> None:
    """Draw an SVG arc for debugging/testing."""
    if svg_context:
        style = _linestyle(color, width)
        attrs = {'d': arc.to_svg_path(), 'style': style}
        svg_context.create_path(attrs, parent=parent)
        if verbose:
            # Draw the center-arc wedge
            draw_point(arc.center, color=color, radius='2px')
            draw_line((arc.center, arc.p1), color=color, parent=parent)
            draw_line((arc.center, arc.p2), color=color, parent=parent)
            draw_point(arc.p1, color='#cc99cc', radius='2px')
            draw_point(arc.p2, color='#99cccc', radius='2px')


def draw_circle(
    center: TPoint,
    radius: float,
    color: str = '#cccc99',
    width: str = '1px',
    verbose: bool = False,
    parent: etree.Element | None = None,
) -> None:
    """Draw an SVG circle."""
    if svg_context:
        style = _linestyle(color, width)
        svg_context.create_circle(center, radius, style=style, parent=parent)
        if verbose:
            draw_point(center, color=color, parent=parent)


def draw_ellipse(
    ellipse: Ellipse,
    color: str = '#cccc99',
    width: str = '1px',
    verbose: bool = False,
    parent: etree.Element | None = None,
) -> None:
    """Draw an SVG arc for debugging/testing."""
    if svg_context:
        style = _linestyle(color, width)
        svg_context.create_ellipse(
            ellipse.center,
            ellipse.rx,
            ellipse.ry,
            angle=ellipse.phi,
            style=style,
            parent=parent,
        )
        if verbose:
            draw_point(ellipse.center, color=color, parent=parent)


def draw_bezier(
    curve: CubicBezier,
    color: str = '#cccc99',
    width: str = '1px',
    verbose: bool = False,
    parent: etree.Element | None = None,
) -> None:
    """Draw an SVG version of this curve for debugging/testing.

    Draws control points, inflection points, and tangent lines.
    """
    if svg_context:
        style = _linestyle(color, width)
        attrs = {'d': curve.to_svg_path(), 'style': style}
        svg_context.create_path(attrs, parent=parent)
        if verbose:
            # Draw control points and tangents
            draw_point(curve.c1, color='#0000c0', parent=parent)
            draw_point(curve.c2, color='#0000c0', parent=parent)
            draw_line((curve.p1, curve.c1), parent=parent)
            draw_line((curve.p2, curve.c2), parent=parent)
            # Draw inflection points if any
            t1, t2 = curve.find_inflections()
            if t1 > 0.0:
                # ip1 = curve.controlpoints_at(t1)[2]
                ip1 = curve.point_at(t1)
                draw_point(ip1, color='#c00000', parent=parent)
            if t2 > 0.0:
                # ip2 = curve.controlpoints_at(t2)[2]
                ip2 = curve.point_at(t2)
                draw_point(ip2, color='#c00000', parent=parent)
            # Draw midpoint
            mp = curve.point_at(0.5)
            draw_point(mp, color='#00ff00', parent=parent)


def _linestyle(color: str, width: str = '1px', opacity: float = 1.0) -> str:
    """Create an SVG line style using the specified attributes."""
    assert svg_context
    uuwidth = svg_context.unit2uu(width)
    return (
        f'fill:none;stroke:{color};'
        f'stroke-width:{uuwidth:.4f};'
        f'stroke-opacity:{opacity}'
    )
