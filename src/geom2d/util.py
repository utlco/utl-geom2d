"""Basic 2D utility functions."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Callable

from . import const, point

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from .arc import Arc
    from .bezier import CubicBezier
    from .line import Line
    from .point import TPoint


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


def float_formatter(
    scale: float = 1, precision: float | None = None
) -> Callable[[float], str]:
    """Get a float formatter for a specified precision.

    Args:
        scale: Scaling factor (for SVG). Default is 1.
        precision: The max number of digits after the decimal point.

    Returns:
        A function that formats a float to the specified precision
        and strips off trailing zeros and decimal point when necessary.
    """
    if precision is None:
        # Assign here instead of kwarg because theoretically mutable
        precision = const.EPSILON_PRECISION
    fmt = f'{{:.{precision}f}}'
    return lambda x: fmt.format(x * scale).rstrip('0').rstrip('.')


def normalize_angle(angle: float, center: float = math.pi) -> float:
    """Normalize ``angle`` about a 2*PI interval centered at ``center``.

    For angle between 0 and 2*PI (default):
        normalize_angle(angle, center=math.pi)
    For angle between -PI and PI:
        normalize_angle(angle, center=0.0)

    Args:
        angle: Angle in radians to normalize
        center: Center value about which to normalize.
            Default is math.pi.

    Returns:
        An angle value in radians between 0 and 2 * PI if center == PI,
        otherwise a value between -PI and PI if center == 0.
    """
    return angle - (
        const.TAU * math.floor((angle + math.pi - center) / const.TAU)
    )


def calc_rotation(start_angle: float, end_angle: float) -> float:
    """Calculate the amount of rotation between two angles.

    Args:
        start_angle: Start angle in radians.
        end_angle: End angle in radians.

    Returns:
        Rotation amount in radians where -PI <= rotation <= PI.
    """
    if const.float_eq(start_angle, end_angle):
        return 0.0
    start_angle = normalize_angle(start_angle, 0)
    end_angle = normalize_angle(end_angle, 0)
    rotation = end_angle - start_angle
    if rotation < -math.pi:
        rotation += const.TAU
    elif rotation > math.pi:
        rotation -= const.TAU
    return rotation


def segments_are_g1(
    seg1: Line | Arc | CubicBezier,
    seg2: Line | Arc | CubicBezier,
    tolerance: float | None = None,
) -> bool:
    """Determine if two segments have G1 continuity.

    G1 continuity is when two  segments are tangentially connected.
    G1 implies G0 continuity.

    Args:
        seg1: First segment. Can be geom.Line, geom2d.Arc, geom.CubicBezier.
        seg2: Second segment. Can be geom.Line, geom2d.Arc, geom.CubicBezier.
        tolerance: G0/G1 tolerance. Default is geom2d.const.EPSILON.

    Returns:
        True if the two segments have G1 continuity within the
        specified tolerance. Otherwise False.
    """
    if tolerance is None:
        tolerance = const.EPSILON
    # G0 continuity - end points are connected
    if point.almost_equal(seg1.p2, seg2.p1, tolerance):
        # G1 continuity -> G0 + segment end points share tangent
        # td = seg1.end_tangent_angle() - seg2.start_tangent_angle()
        # return abs(td) < tolerance
        return const.float_eq(
            seg1.end_tangent_angle(), seg2.start_tangent_angle()
        )
    return False


def reverse_path(
    path: Sequence[Line | Arc | CubicBezier],
) -> list[Line | Arc | CubicBezier]:
    """Reverse the order and direction of path segments."""
    rpath: list[Line | Arc | CubicBezier] = []
    for i, segment in enumerate(reversed(path)):
        rpath[i] = segment.path_reversed()
    return rpath


def triplepoints(
    points: Iterable[TPoint],
) -> Iterator[tuple[TPoint, TPoint, TPoint]]:
    """Return overlapping point triplets from *points*.

    >>> list(triplewise('ABCDE'))
    [('A', 'B', 'C'), ('B', 'C', 'D'), ('C', 'D', 'E')]

    See:
        https://github.com/more-itertools/more-itertools
    """
    for (a, _), (b, c) in itertools.pairwise(itertools.pairwise(points)):
        yield a, b, c
