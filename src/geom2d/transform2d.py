# -----------------------------------------------------------------------------
# Copyright 2012-2023 Claude Zervas
# email: claude@utlco.com
# -----------------------------------------------------------------------------
"""Basic 2D affine transform matrix operations.

====
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import TypeAlias

# from . import const

TMatrix: TypeAlias = tuple[
    tuple[float, float, float], tuple[float, float, float]
]

# :2D transform identity matrix
IDENTITY_MATRIX = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def is_identity_transform(m: TMatrix) -> bool:
    """Return True if the matrix is the identity matrix."""
    # return (const.float_eq(m[0][0], IDENTITY_MATRIX[0][0])
    #     and const.float_eq(m[0][1], IDENTITY_MATRIX[0][1])
    #     and const.float_eq(m[0][2], IDENTITY_MATRIX[0][2])
    #     and const.float_eq(m[1][0], IDENTITY_MATRIX[1][0])
    #     and const.float_eq(m[1][1], IDENTITY_MATRIX[1][1])
    #     and const.float_eq(m[1][2], IDENTITY_MATRIX[1][2]))
    return (
        m[0][0] == IDENTITY_MATRIX[0][0]
        and m[0][1] == IDENTITY_MATRIX[0][1]
        and m[0][2] == IDENTITY_MATRIX[0][2]
        and m[1][0] == IDENTITY_MATRIX[1][0]
        and m[1][1] == IDENTITY_MATRIX[1][1]
        and m[1][2] == IDENTITY_MATRIX[1][2]
    )


def compose_transform(m1: TMatrix, m2: TMatrix) -> TMatrix:
    """Combine two matrices by multiplying them.

    Args:
        m1: 2X3 2D transform matrix.
        m2: 2X3 2D transform matrix.

    Note:
        `m2` is applied before (to) `m1`
    """
    m100 = m1[0][0]
    m101 = m1[0][1]
    m110 = m1[1][0]
    m111 = m1[1][1]
    return (
        (
            m100 * m2[0][0] + m101 * m2[1][0],
            m100 * m2[0][1] + m101 * m2[1][1],
            m100 * m2[0][2] + m101 * m2[1][2] + m1[0][2],
        ),
        (
            m110 * m2[0][0] + m111 * m2[1][0],
            m110 * m2[0][1] + m111 * m2[1][1],
            m110 * m2[0][2] + m111 * m2[1][2] + m1[1][2],
        ),
    )


def matrix_rotate(
    angle: float, origin: Sequence[float] = (0.0, 0.0)
) -> TMatrix:
    """Create a transform matrix to rotate about the origin.

    Args:
        angle: Rotation angle in radians.
        origin: Optional rotation origin. Default is (0,0).

    Returns:
        A transform matrix as 2x3 tuple
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    m1 = ((cos_a, -sin_a, origin[0]), (sin_a, cos_a, origin[1]))
    m2 = matrix_translate(-origin[0], -origin[1])
    return compose_transform(m1, m2)


def matrix_translate(x: float, y: float) -> TMatrix:
    """Create a transform matrix to translate (move).

    Args:
        x: translation along X axis
        y: translation along Y axis

    Returns:
        A transform matrix as 2x3 tuple
    """
    return ((1.0, 0.0, x), (0.0, 1.0, y))


def matrix_scale(
    scale_x: float, scale_y: float, origin: Sequence[float] | None = None
) -> TMatrix:
    """Create a transform matrix to scale.

    Args:
        scale_x: X axis scale factor
        scale_y: Y axis scale factor
        origin: Optional scale origin. Default is (0,0).

    Returns:
        A transform matrix as 2x3 tuple
    """
    m = ((scale_x, 0.0, 0.0), (0.0, scale_y, 0.0))
    if origin:
        ms1 = matrix_translate(-origin[0], -origin[1])
        ms2 = matrix_translate(origin[0], origin[1])
        m = compose_transform(ms2, compose_transform(m, ms1))
    return m


def matrix_scale_translate(
    scale_x: float, scale_y: float, offset_x: float, offset_y: float
) -> TMatrix:
    """Create a transform matrix to scale and translate.

    Args:
        scale_x: X axis scale factor
        scale_y: Y axis scale factor
        offset_x: translation along X axis
        offset_y: translation along Y axis

    Returns:
        A transform matrix as 2x3 tuple
    """
    return ((scale_x, 0.0, offset_x), (0.0, scale_y, offset_y))


def matrix_skew_x(angle: float) -> TMatrix:
    """Create a transform matrix to skew along X axis by `angle`.

    Args:
        angle: Angle in radians to skew.

    Returns:
        A transform matrix as 2x3 tuple
    """
    return ((1.0, math.tan(angle), 0.0), (0.0, 1.0, 0.0))


def matrix_skew_y(angle: float) -> TMatrix:
    """Create a transform matrix to skew along Y axis by `angle`.

    Args:
        angle: Angle in radians to skew.

    Returns:
        A transform matrix as 2x3 tuple
    """
    return ((1.0, 0.0, 0.0), (math.tan(angle), 1.0, 0.0))


def matrix_apply_to_point(
    matrix: TMatrix, p: Sequence[float]
) -> tuple[float, float]:
    """Return a copy of `p` with the transform matrix applied to it."""
    return (
        matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2],
        matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2],
    )


def canonicalize_point(
    p: Sequence[float], origin: Sequence[float], theta: float
) -> tuple[float, float]:
    """Canonicalize the point.

    This just rotates then translates the point so that
    the origin is (0, 0) and axis rotation is zero.


    Args:
        p: The point to canonicalize (x, y).
        origin: The origin offset as a 2-tuple (X, Y).
        theta: The axis rotation angle.

    Returns:
        A point as 2-tuple
    """
    p = matrix_apply_to_point(matrix_rotate(-theta), p)
    return (p[0] - origin[0], p[1] - origin[1])
