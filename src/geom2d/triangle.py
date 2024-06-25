"""Triangle geometry."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .point import P


def incircle(p1: P, p2: P, p3: P) -> tuple[P, float]:
    """Calculate triangle inscribed circle center and radius.

    See:
        https://en.wikipedia.org/wiki/Incircle_and_excircles

    Args:
        p1: First point of triangle
        p2: Second point of triangle
        p3: Third point of triangle

    Returns:
        A tuple containing the incenter point and radius.
    """
    a = p2.distance(p3)
    b = p1.distance(p3)
    c = p1.distance(p2)
    perim = a + b + c
    incenter = (p1 * a + p2 * b + p3 * c) / perim
    s = perim / 2
    radius = math.sqrt(((s - a) * (s - b) * (s - c)) / s)
    return incenter, radius
