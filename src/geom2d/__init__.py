# -----------------------------------------------------------------------------
# Copyright 2012-2016 Claude Zervas
# email: claude@utlco.com
# -----------------------------------------------------------------------------
"""2D geometry package.

Parts of this library where inspired by planar, a 2D geometry library for
python gaming:
    https://bitbucket.org/caseman/planar/
"""

# Expose package-wide constants and functions
from .arc import Arc
from .bezier import CubicBezier
from .box import Box
from .const import (
    TAU,
    float_eq,
    float_round,
    is_zero,
    set_epsilon,
)
from .ellipse import Ellipse, EllipticalArc
from .line import Line
from .point import P, TPoint

# Expose some basic geometric classes and types at package level
from .transform2d import TMatrix
from .util import calc_rotation, normalize_angle, segments_are_g1
