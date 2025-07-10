"""Test 2D line (geom2d.line.Line) implementation."""

import math
import sys

import pytest

import numpy as np
from geom2d import const, line, point, transform2d
from geom2d.const import float_eq
from geom2d.line import Line
from geom2d.point import P


