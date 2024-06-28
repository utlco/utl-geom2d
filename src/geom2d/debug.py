"""Debug SVG output support for geometry package."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from inkext.svg import SVGContext

from . import const

# svg.SVGContext singleton for drawing debug output.
# Debug drawing is effectively disabled if this is None (default).
svg_context: SVGContext | None = None  # pylint: disable=invalid-name

_DEFAULT_COLOR = '#f00'


def set_svg_context(context: SVGContext) -> None:
    """Initialize this module with an SVGContext.

    The SVGContext will be used for debug output by draw...() methods.
    """
    global svg_context  # noqa: PLW0603 pylint: disable=global-statement
    svg_context = context
    if context is not None:
        # global const.DEBUG
        const.DEBUG = True


def debug(*args) -> None:  # noqa: ANN002
    """Just write a message to stderr.

    Dumb but useful when debugging in Inkscape.
    """
    print(*args, file=sys.stderr)  # noqa: T201


def draw_point(
    point: Sequence[float],
    color: str = _DEFAULT_COLOR,
    width: str | float = '4px',
    opacity: float = 1,
) -> None:
    """Draw a dot. Useful for debugging and testing."""
    if not svg_context:
        return

    svg_context.create_circle(
        point,
        svg_context.unit2uu(width) / 2,
        style=f'fill:{color};stroke:none;opacity:{opacity:.3f}',
    )


def draw_line(
    line: Sequence[Sequence[float]],
    color: str = _DEFAULT_COLOR,
    width: str | float = '1px',
    opacity: float = 1,
    verbose: bool = False,
) -> None:
    """Draw an SVG line segment for debugging/testing."""
    if not svg_context:
        return

    style = linestyle(color=color, width=width, opacity=opacity)
    svg_context.create_line(line[0], line[1], style=style)
    if verbose:
        draw_point(line[0], color=color)
        draw_point(line[1], color=color)


def draw_poly(
    vertices: Sequence[Sequence[float]],
    color: str = _DEFAULT_COLOR,
    width: str | float = '1px',
    opacity: float = 1,
    verbose: bool = False,
    close_poly: bool = True,
) -> None:
    """Draw an SVG polygon."""
    if not svg_context:
        return

    style = linestyle(color=color, width=width, opacity=opacity)
    svg_context.create_polygon(vertices, close_polygon=close_poly, style=style)
    if verbose:
        for p in vertices:
            draw_point(p, color=color)


def draw_arc(
    arc: tuple[Sequence[float], Sequence[float], float, float, Sequence[float]],
    color: str = _DEFAULT_COLOR,
    width: str | float = '1px',
    opacity: float = 1,
    verbose: bool = False,
) -> None:
    """Draw an SVG arc for debugging/testing."""
    if not svg_context:
        return

    # p1: Sequence[float]
    # p2: Sequence[float]
    # radius: float
    # angle: float
    # center: Sequence[float]
    p1, p2, radius, angle, center = arc
    sweep_flag = 0 if angle < 0 else 1

    style = linestyle(color=color, width=width, opacity=opacity)
    svg_context.create_circular_arc(p1, p2, radius, sweep_flag, style=style)

    if verbose:
        # Draw the center-arc wedge
        draw_point(center, color=color)
        draw_line((center, p1), color='#ffc0e0')
        draw_line((center, p2), color='#ffc0e0')
        draw_point(p1, color='#cc99cc')
        draw_point(p2, color='#99cccc')


def draw_circle(
    center: Sequence[float],
    radius: float,
    color: str = _DEFAULT_COLOR,
    width: str | float = '1px',
    opacity: float = 1,
    verbose: bool = False,
) -> None:
    """Draw an SVG circle."""
    if not svg_context:
        return

    style = linestyle(color=color, width=width, opacity=opacity)
    svg_context.create_circle(center, radius, style=style)
    if verbose:
        draw_point(center, color=color)


def linestyle(
    color: str = _DEFAULT_COLOR,
    width: str | float = '1px',
    opacity: float = 1.0,
) -> str:
    """Create an SVG line style using the specified attributes."""
    assert svg_context
    uuwidth = svg_context.unit2uu(width)
    return (
        f'fill:none;stroke:{color};'
        f'stroke-width:{uuwidth:.3f};'
        f'stroke-opacity:{opacity:.3f}'
    )
