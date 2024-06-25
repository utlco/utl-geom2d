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
    DEBUG,
    TAU,
    float_eq,
    float_round,
    is_zero,
    set_epsilon,
)
from .ellipse import Ellipse, EllipticalArc
from .line import Line, TLine
from .point import P, TPoint

# Expose some basic geometric classes and types at package level
from .transform2d import TMatrix
from .util import calc_rotation, normalize_angle, segments_are_g1
