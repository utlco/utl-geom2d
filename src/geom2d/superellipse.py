"""Create an approximated superellipse (Lamé curve) using Bezier curves.

Note: This was partly built using ChatGPT/Codex 5.6-Sol-high
    then significantly cleaned up.
"""

import math
from collections.abc import Sequence
from operator import itemgetter
from typing import TypeAlias

from . import TPoint
from .polygon import rect_midpoints

_HALF_PI = math.pi / 2.0
_EPSILON = 1.0e-9

# Min/max number of curves for approximation
_MIN_CURVES = 4
_MAX_CURVES = 40

# Shape range
_MIN_K = -3.0
_MAX_K = 3.0


# Cubic Bezier curve type
TCubicBezier: TypeAlias = tuple[TPoint, TPoint, TPoint, TPoint]

# Quadratic Bezier curve type
TQuadBezier: TypeAlias = tuple[TPoint, TPoint, TPoint]


def _is_zero(n: float) -> bool:
    return abs(n) < _EPSILON


def _is_gt(a: float, b: float) -> bool:
    return a > (b - _EPSILON)


def rhombus_superellipse(
    points: Sequence[TPoint],
    k: float,
    num_curves: int = 16,
    scale: float = 1.0,
    gap: float = 0.0,
    use_vertice_endpoints: bool = False,
    fit_inside: bool = True,
    use_lame_max: bool = False,
) -> list[tuple[TPoint, TPoint, TPoint, TPoint]]:
    """Approximate a superellipse inscribed in a rhombus/parallelogram.

    Uses a fixed number of cubic Beziers to approximate a Lamé curve.

    Args:
        points: Sequence of 4 (x, y) rhombus vertices
            in clockwise or counter-clockwise order.
        k: CSS-superellipse-like parameter::

            k < 0   concave / scooped
            k = 0   diamond / straight lines
            k = 1   ellipse
            k > 1   increasingly square

        num_curves: Total number of Bezier segments. Must be a multiple of 4.
            The default is 16.
        scale: Scale superellipse.
        gap: Endpoint gap.
        use_vertice_endpoints: Use vertices as quadrant curve endpoints
            instead of midpoints.
        fit_inside: Constrain convex superellipse within rhombus when
            `use_vertice_endpoints` is True.
        use_lame_max: Use Lamé curve maximum for fitting if True,
            otherwise use Bezier curve maximum. Default is False (use Bezier).

    Returns:
        List of (P0, C1, C2, P1) Cubic Bezier segments.
    """
    if len(points) != 4:
        raise ValueError("Exactly four vertices are required")

    # Make sure num curves is a multiple of 4 within a reasonable range.
    num_curves = (num_curves // 4) * 4
    num_curves = max(min(num_curves, _MAX_CURVES), _MIN_CURVES)

    # Clamp K to reasonable limits.
    k = max(min(k, _MAX_K), _MIN_K)

    # This would be a simple diamond of four straight line Beziers
    # connected at rhombus edge midpoints.
    if _is_zero(k):  # n == 1.0
        p0, p1, p2, p3 = points if use_vertice_endpoints else rect_midpoints(points)
        return [
            (p0, p0, p1, p1),
            (p1, p1, p2, p2),
            (p2, p2, p3, p3),
            (p3, p3, p0, p0),
        ]

    # Lamé equation exponent.
    # See: https://en.wikipedia.org/wiki/Superellipse
    n = 2.0**k

    # Bezier curves approximating Lamé curve for one quadrant
    base_curves: list[TQuadBezier] = []

    curves_per_quadrant = num_curves // 4
    step = _HALF_PI / curves_per_quadrant

    # Create normalized curves for one quadrant
    for i in range(curves_per_quadrant):
        theta = i * step
        p0 = _lame_point(theta, n)
        p1 = _lame_point(theta + step, n)
        t0 = _lame_tangent(p0, n)
        t1 = _lame_tangent(p1, n)
        q = _tangent_intersection(p0, t0, p1, t1)
        base_curves.append((p0, q, p1))

    # Rhombus center
    cx, cy = (
        0.25 * sum(x for x, _y in points),
        0.25 * sum(y for _x, y in points),
    )

    # TODO: add transform for endpoint gap

    # Create transform vectors to skew curves to fit rhombus
    if use_vertice_endpoints:
        if n > 1 and fit_inside:
            # Scale endpoint overshoot to fit inside rhombus
            scale *= (
                _fit_scale_lame(n)
                if use_lame_max
                else _fit_scale_bezier(base_curves)[0]
            )
        ux = points[0][0] - cx
        uy = points[0][1] - cy
        vx = points[1][0] - cx
        vy = points[1][1] - cy
    else:  # Use midpoints
        ux = 0.5 * (points[0][0] - points[1][0])
        uy = 0.5 * (points[0][1] - points[1][1])
        vx = 0.5 * (points[0][0] - points[3][0])
        vy = 0.5 * (points[0][1] - points[3][1])

    # Affine map from normalized square to rhombus
    def _transform(point: TPoint, quadrant: int) -> TPoint:
        x, y = point
        # Rotate normalized point to quadrant
        if quadrant == 1:
            x, y = -y, x
        elif quadrant == 2:
            x, y = -x, -y
        elif quadrant == 3:
            x, y = y, -x
        x *= scale
        y *= scale
        # Affine transform to rhombus
        return (
            cx + x * ux + y * vx,
            cy + x * uy + y * vy,
        )

    cubic_curves: list[TCubicBezier] = []

    # Create transformed curves for each quadrant
    for quadrant in range(4):
        for curve in base_curves:
            p0, q, p1 = curve
            p0 = _transform(p0, quadrant)
            q = _transform(q, quadrant)
            p1 = _transform(p1, quadrant)
            cubic_curves.append(_quadratic_to_cubic(p0, q, p1))

    return cubic_curves


def _quadratic_to_cubic(p0: TPoint, q: TPoint, p1: TPoint) -> TCubicBezier:
    """Elevate Bezier from quadratic to cubic."""
    c1 = (
        (p0[0] + 2.0 * q[0]) / 3.0,
        (p0[1] + 2.0 * q[1]) / 3.0,
    )
    c2 = (
        (p1[0] + 2.0 * q[0]) / 3.0,
        (p1[1] + 2.0 * q[1]) / 3.0,
    )
    return p0, c1, c2, p1


def _lame_point(theta: float, n: float) -> TPoint:
    """Quadrant Lamé curve endpoint at `theta`."""
    # Clamp to axes
    if _is_zero(theta):
        return (1.0, 0.0)
    if _is_gt(theta, _HALF_PI):
        return (0.0, 1.0)

    exp = 2.0 / n
    return (
        math.cos(theta) ** exp,
        math.sin(theta) ** exp,
    )


def _lame_tangent(point: TPoint, n: float) -> TPoint:
    """Tangent direction to the normalized Lamé curve."""
    # For
    #
    #     F(x,y) = x^n + y^n - 1
    #
    # the gradient is normal to the curve, so a tangent direction
    # in the first quadrant is:
    #
    #     T = (-y^(n-1), x^(n-1))
    #
    x, y = point

    # Check for vertical/horizontal tangents
    if _is_zero(y):
        return (0.0, 1.0) if n > 1.0 else (-1.0, 0.0)
    if _is_zero(x):
        return (-1.0, 0.0) if n > 1.0 else (0.0, 1.0)

    return -(y ** (n - 1.0)), x ** (n - 1.0)


def _tangent_intersection(p0: TPoint, t0: TPoint, p1: TPoint, t1: TPoint) -> TPoint:
    """Intersection of tangent lines (quadratic Bezier control point)."""

    def _cross(a: TPoint, b: TPoint) -> float:
        return a[0] * b[1] - a[1] * b[0]

    denominator = _cross(t0, t1)
    delta = (p1[0] - p0[0], p1[1] - p0[1])

    assert not _is_zero(denominator)  # can't happen, already checked for n == 1
    s = _cross(delta, t1) / denominator

    return (
        p0[0] + s * t0[0],
        p0[1] + s * t0[1],
    )


def _fit_scale_lame(n: float) -> float:
    """Find the quadrant maximum using ideal Lamé curve."""
    # Exact analytical scaling of the ideal Lamé curve.
    # Maximum x+y occurs at x=y=2^(-1/n), therefore:
    #     max(x+y) = 2^(1 - 1/n)
    # and
    #     scale = 2^(1/n - 1)
    return float(2.0 ** (1.0 / n - 1.0))


def _fit_scale_bezier(
    base_curves: list[TQuadBezier],
) -> tuple[float, int, float, TPoint]:
    """Find the maximum extent of the curves in the base quadrant."""
    extent, t, p, i = max(
        ((*_bezier_max(curve), i) for i, curve in enumerate(base_curves)),
        key=itemgetter(0),
    )
    scale = 1.0 / extent
    return scale, i, t, p


def _bezier_max(curve: TQuadBezier) -> tuple[float, float, TPoint]:
    """Get x+y maximum of quadratic Bezier curve."""
    p0, q, p1 = curve

    a = p0[0] + p0[1]
    b = q[0] + q[1]
    c = p1[0] + p1[1]

    denominator = a - 2.0 * b + c
    if not _is_zero(denominator):
        t = (a - b) / denominator
        if 0.0 < t < 1.0:
            point = _bezier_point(p0, q, p1, t)
            extent = point[0] + point[1]
            if extent > a and extent > c:
                return extent, t, point

    if a > c:
        return a, 0.0, p0
    return c, 1.0, p1


def _bezier_point(p0: TPoint, q: TPoint, p1: TPoint, t: float) -> TPoint:
    """Point on quadratic Bezier curve at `t`."""
    s = 1.0 - t
    a = s * s
    b = 2.0 * s * t
    c = t * t

    return (
        a * p0[0] + b * q[0] + c * p1[0],
        a * p0[1] + b * q[1] + c * p1[1],
    )
