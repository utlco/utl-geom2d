"""Tests for axis-aligned bounding boxes and line clipping."""

from __future__ import annotations

import math

import pytest

from geom2d import transform2d
from geom2d.arc import Arc
from geom2d.box import Box, bounding_box
from geom2d.line import Line, TLine
from geom2d.point import P, TPoint


@pytest.mark.parametrize(
    ("p1", "p2"),
    [((1, 2), (4, 6)), ((4, 6), (1, 2)), ((1, 6), (4, 2)), ((4, 2), (1, 6))],
)
def test_constructor(p1: TPoint, p2: TPoint) -> None:
    for box in (Box(p1, p2), Box((p1, p2)), Box(Box(p1, p2))):
        assert box == ((1, 2), (4, 6))
        assert isinstance(box.p1, P)
        assert isinstance(box.p2, P)


def test_properties() -> None:
    box = Box((-2, -1), (1, 3))
    assert box.p1 == (-2, -1)
    assert box.p2 == (1, 3)
    assert box.topleft == (-2, 3)
    assert box.bottomright == (1, -1)
    assert (box.xmin, box.ymin, box.xmax, box.ymax) == (-2, -1, 1, 3)
    assert box.center == (-0.5, 1)
    assert box.width == 3
    assert box.height == 4
    assert box.diagonal() == pytest.approx(5)
    assert box.vertices() == ((-2, -1), (-2, 3), (1, 3), (1, -1))
    assert box.bounding_box() is box
    assert box.start_tangent_angle() == pytest.approx(math.pi / 2)


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([(3, -4), (-5, 2), (1, 7), (8, 0)], Box((-5, -4), (8, 7))),
        ([(-8, -9), (-2, -3)], Box((-8, -9), (-2, -3))),
        ([(2, 3)], Box((2, 3), (2, 3))),
        ([(2, 3), (2, 3)], Box((2, 3), (2, 3))),
    ],
)
def test_from_points(points: list[TPoint], expected: Box) -> None:
    assert bounding_box(points) == expected
    assert Box.from_points(points) == expected
    assert Box.from_points(iter(points)) == expected


def test_bounding_box_iterator() -> None:
    assert bounding_box(iter([(3, -4), (-5, 2)])) == Box((-5, -4), (3, 2))


def test_empty_points() -> None:
    with pytest.raises(ValueError):
        bounding_box([])
    with pytest.raises(ValueError):
        Box.from_points([])
    with pytest.raises(ValueError):
        Box.from_points(iter([]))


def test_from_path() -> None:
    path = [Line((3, -4), (-5, 2)), Line((-5, 2), (8, 7))]
    assert Box.from_path(path) == Box((-5, -4), (8, 7))
    assert Box.from_path(iter(path)) == Box((-5, -4), (8, 7))
    assert Box.from_path([]) is None


@pytest.mark.parametrize(
    ("point", "strict", "inclusive"),
    [
        ((2, 3), True, True),
        ((0, 3), False, True),
        ((4, 3), False, True),
        ((2, 0), False, True),
        ((2, 6), False, True),
        ((0, 0), False, True),
        ((4, 6), False, True),
        ((0, 6), False, True),
        ((4, 0), False, True),
        ((-1, 3), False, False),
        ((5, 3), False, False),
        ((2, -1), False, False),
        ((2, 7), False, False),
    ],
)
def test_point_inside(point: TPoint, strict: bool, inclusive: bool) -> None:
    box = Box((0, 0), (4, 6))
    assert box.point_inside(point) is strict
    assert box.point_inside(point, edge_ok=True) is inclusive


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (((1, 1), (3, 5)), True),
        (((0, 1), (3, 5)), False),
        (((1, 1), (5, 5)), False),
        (((-1, 3), (5, 3)), False),
    ],
)
def test_line_inside(line: TLine, expected: bool) -> None:
    box = Box((0, 0), (4, 6))
    assert box.line_inside(line) is expected
    assert box.line_inside(Line(line[1], line[0])) is expected


def test_all_points_inside() -> None:
    box = Box((0, 0), (4, 6))
    assert box.all_points_inside(iter([(1, 1), (3, 5)]))
    assert box.all_points_inside([])
    assert not box.all_points_inside([(1, 1), (4, 5)])
    assert not box.all_points_inside([(1, 1), (5, 5)])


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (2, Box((-2, -2), (6, 8))),
        (0, Box((0, 0), (4, 6))),
        (-1, Box((1, 1), (3, 5))),
        (-2, Box((2, 2), (2, 4))),
    ],
)
def test_buffered(distance: float, expected: Box) -> None:
    box = Box((0, 0), (4, 6))
    assert box.buffered(distance) == expected
    assert box == ((0, 0), (4, 6))


def test_transform() -> None:
    box = Box((1, 2), (4, 6))
    assert box.transform(transform2d.IDENTITY_MATRIX) == box
    assert box.transform(transform2d.matrix_translate(-3, 5)) == ((-2, 7), (1, 11))
    assert box.transform(transform2d.matrix_scale(-2, 3)) == ((-8, 6), (-2, 18))
    assert box.transform(transform2d.matrix_rotate(math.pi / 2)) == ((-6, 1), (-2, 4))
    assert box == ((1, 2), (4, 6))


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (((1, 1), (3, 5)), Line((1, 1), (3, 5))),
        (((-2, 3), (6, 3)), Line((0, 3), (4, 3))),
        (((2, -2), (2, 8)), Line((2, 0), (2, 6))),
        (((-2, -2), (8, 8)), Line((0, 0), (4, 4))),
        (((-2, 8), (8, -2)), Line((0, 6), (4, 2))),
        (((2, 3), (6, 3)), Line((2, 3), (4, 3))),
        (((-2, 0), (6, 0)), Line((0, 0), (4, 0))),
        (((4, -2), (4, 8)), Line((4, 0), (4, 6))),
        (((-1, 1), (1, -1)), Line((0, 0), (0, 0))),
        (((2, 3), (2, 3)), Line((2, 3), (2, 3))),
        (((0, 0), (0, 0)), Line((0, 0), (0, 0))),
        (((-1, 3), (-1, 3)), None),
        (((-2, -1), (6, -1)), None),
        (((-2, 7), (6, 7)), None),
        (((-1, -2), (-1, 8)), None),
        (((5, -2), (5, 8)), None),
        (((-3, 1), (1, -3)), None),
        (((5, 5), (6, 6)), None),
    ],
)
def test_clip_line(line: TLine, expected: Line | None) -> None:
    box = Box((0, 0), (4, 6))
    result = box.clip_line(line)
    assert result == expected
    if expected is not None:
        assert isinstance(result, Line)
    reverse = Line(line[1], line[0])
    assert box.clip_line(reverse) == (
        None if expected is None else Line(expected[1], expected[0])
    )


@pytest.mark.parametrize(
    ("a", "b", "c", "expected"),
    [
        (1, 0, 2, Line((2, 0), (2, 6))),
        (1, 0, 0, Line((0, 0), (0, 6))),
        (1, 0, 4, Line((4, 0), (4, 6))),
        (1, 0, -1, None),
        (1, 0, 5, None),
        (0, 1, 3, Line((0, 3), (4, 3))),
        (0, 1, 0, Line((0, 0), (4, 0))),
        (0, 1, 6, Line((0, 6), (4, 6))),
        (0, 1, -1, None),
        (0, 1, 7, None),
        (-1, 1, 0, Line((0, 0), (4, 4))),
        (1, 1, 6, Line((0, 6), (4, 2))),
        (-2, 1, -2, Line((1, 0), (4, 6))),
        (1, 1, 0, Line((0, 0), (0, 0))),
        (1, 1, -1, None),
    ],
)
def test_clip_line_stdeq(a: float, b: float, c: float, expected: Line | None) -> None:
    box = Box((0, 0), (4, 6))
    assert box.clip_line_stdeq(a, b, c) == expected
    assert box.clip_line_stdeq(-2 * a, -2 * b, -2 * c) == expected


def test_clip_line_stdeq_invalid() -> None:
    with pytest.raises(ValueError, match="both a and b cannot be zero"):
        Box((0, 0), (4, 6)).clip_line_stdeq(0, 0, 1)


def test_clip_arc_unimplemented() -> None:
    arc = Arc((1, 0), (0, 1), 1, math.pi / 2, (0, 0))
    with pytest.raises(NotImplementedError):
        Box((-2, -2), (2, 2)).clip_arc(arc)


@pytest.mark.parametrize(
    ("other", "intersection", "union"),
    [
        (Box((2, 3), (6, 8)), Box((2, 3), (4, 6)), Box((0, 0), (6, 8))),
        (Box((1, 1), (3, 5)), Box((1, 1), (3, 5)), Box((0, 0), (4, 6))),
        (Box((0, 0), (4, 6)), Box((0, 0), (4, 6)), Box((0, 0), (4, 6))),
        (Box((4, 1), (7, 5)), Box((4, 1), (4, 5)), Box((0, 0), (7, 6))),
        (Box((4, 6), (7, 8)), Box((4, 6), (4, 6)), Box((0, 0), (7, 8))),
        (Box((5, 1), (7, 5)), None, Box((0, 0), (7, 6))),
        (Box((1, 7), (3, 8)), None, Box((0, 0), (4, 8))),
    ],
)
def test_intersection_and_union(
    other: Box, intersection: Box | None, union: Box
) -> None:
    box = Box((0, 0), (4, 6))
    assert box.intersection(other) == intersection
    assert other.intersection(box) == intersection
    assert box.intersection((other.p2, other.p1)) == intersection
    assert box.union(other) == union
    assert other.union(box) == union
    assert box.union((other.p2, other.p1)) == union


@pytest.mark.parametrize("p2", [(2, 3), (2, 6), (4, 3)])
def test_degenerate_box(p2: TPoint) -> None:
    box = Box((2, 3), p2)
    assert not box.point_inside(box.center)
    assert box.point_inside(box.center, edge_ok=True)
    assert box.intersection(Box((0, 0), (10, 10))) == box
    assert box.clip_line(Line((0, 3), (6, 3))) == Line((2, 3), (p2[0], 3))
