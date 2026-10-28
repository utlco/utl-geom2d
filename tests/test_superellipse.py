"""Test approximated superellipse.

Note: This was mostly built using ChatGPT/Codex 5.6/6.1-Sol-high
"""

import math
import pathlib
import xml.etree.ElementTree as ET

import pytest

from geom2d.superellipse import rhombus_superellipse


def _svg_path_from_curves(curves: list[tuple]) -> str:
    """Convert the four cubic Bezier curves into one closed SVG path."""
    start = curves[0][0]
    parts = [f"M {start[0]:.3f},{start[1]:.3f}"]

    for _, c1, c2, end in curves:
        parts.append(
            f"C "
            f"{c1[0]:.3f},{c1[1]:.3f} "
            f"{c2[0]:.3f},{c2[1]:.3f} "
            f"{end[0]:.3f},{end[1]:.3f}"
        )

    parts.append("Z")

    return " ".join(parts)


def _write_test_svg(filename: str = "superellipse.svg") -> None:
    # A deliberately skewed/rotated rhombus.
    base_rhombus = [(30, 20), (220, 65), (250, 225), (60, 180)]

    # Negative, zero, circular, and increasingly square-ish cases.
    # Concave to convex.
    k_values = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]

    cell_width = 300
    cell_height = 300

    width = cell_width * 3
    rows = (len(k_values) + 2) // 3
    height = cell_height * rows

    svg = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        )
    ]

    svg.append("""
    <style>
        text {
            font-family: sans-serif;
            font-size: 16px;
        }

        .rhombus {
            fill: none;
            stroke: #2bb673;
            stroke-width: 1;
        }

        .curve {
            fill: none;
            stroke: #087bea;
            stroke-width: 1;
        }

        .midpoint {
            fill: red;
        }

        .control-line {
            stroke: #999;
            stroke-width: 1;
            stroke-dasharray: 4 4;
        }

        .control-point {
            fill: #666;
        }
    </style>
    """)

    for index, k in enumerate(k_values):
        col = index % 3
        row = index // 3

        ox = col * cell_width + 20
        oy = row * cell_height + 45

        # translate rhombus
        rhombus = [(x + ox, y + oy) for x, y in base_rhombus]

        curves = rhombus_superellipse(
            rhombus, k
        )  # , use_vertice_endpoints=True, fit_inside=True, use_bezier_max=True)

        path = _svg_path_from_curves(curves)

        # Label
        n = 2.0**k

        svg.append(
            f'<text x="{ox + 5}" y="{oy - 15}">'
            f"K={k:g}, n={2.0**k:.4g}, curves={len(curves)}"
            f"</text>\n"
            f'<text x="{ox + 5}" y="{oy - 15}">'
            f"K={k:g}, n={n:.4g}"
            f"</text>"
        )

        # Draw rhombus and Bezier curve.
        point_string = " ".join(f"{x:.3f},{y:.3f}" for x, y in rhombus)

        svg.append(
            f'<polygon class="rhombus" points="{point_string}" />\n'
            f'<path class="curve" d="{path}" />'
        )

        # Draw midpoint and control-points
        #        for start, c1, c2, end in curves:
        #            svg.append(
        #                f'<line class="control-line" '
        #                f'x1="{start[0]}" y1="{start[1]}" '
        #                f'x2="{c1[0]}" y2="{c1[1]}" />'
        #            )
        #
        #            svg.append(
        #                f'<line class="control-line" '
        #                f'x1="{end[0]}" y1="{end[1]}" '
        #                f'x2="{c2[0]}" y2="{c2[1]}" />'
        #            )
        #
        #            svg.append(
        #                f'<circle class="control-point" '
        #                f'cx="{c1[0]}" cy="{c1[1]}" r="3" />'
        #            )
        #
        #            svg.append(
        #                f'<circle class="control-point" '
        #                f'cx="{c2[0]}" cy="{c2[1]}" r="3" />'
        #            )

        # Four edge midpoints.
        # mids = [
        #    (
        #        0.5 * (rhombus[i][0] + rhombus[(i + 1) % 4][0]),
        #        0.5 * (rhombus[i][1] + rhombus[(i + 1) % 4][1]),
        #    )
        #    for i in range(4)
        # ]

        # for x, y in mids:
        #    svg.append(
        #        f'<circle class="midpoint" '
        #        f'cx="{x}" cy="{y}" r="4" />'
        #    )

    svg.append("</svg>")

    pathlib.Path(filename).write_text("\n".join(svg), encoding="utf-8")

    print(f"Wrote {filename}")


if __name__ == "__main__":
    _write_test_svg()


@pytest.mark.parametrize("k", [-3, 0, 1, 3])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("gap", [0.0, 12.0])
def test_endpoint_gap(k: float, scale: float, gap: float) -> None:
    """Quadrant seams have a drawing-unit gap parallel to their edges."""
    points = [(30, 20), (220, 65), (250, 225), (60, 180)]
    curves = rhombus_superellipse(points, k, scale=scale, gap=gap)
    continuous = rhombus_superellipse(points, k, scale=scale)
    count = len(curves) // 4
    for quadrant in range(4):
        end_index = (quadrant + 1) * count - 1
        start_index = (end_index + 1) % len(curves)
        end = curves[end_index][-1]
        start = curves[start_index][0]
        assert math.dist(end, start) == pytest.approx(gap)
        seam = continuous[end_index][-1]
        assert ((end[0] + start[0]) / 2, (end[1] + start[1]) / 2) == pytest.approx(seam)
        # Midpoint-mode axes alternate between the two rhombus edges.
        edge_end = points[1] if quadrant % 2 == 0 else points[3]
        dx, dy = edge_end[0] - points[0][0], edge_end[1] - points[0][1]
        assert (end[0] - start[0]) * dy - (end[1] - start[1]) * dx == pytest.approx(
            0, abs=1e-9
        )
        for index in range(quadrant * count, end_index):
            assert curves[index][-1] == curves[index + 1][0]


_SQUARE = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
_K_VALUES = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]


@pytest.mark.parametrize("size", [0, 1, 2, 3, 5])
def test_invalid_vertex_count(size: int) -> None:
    with pytest.raises(ValueError, match="Exactly four vertices are required"):
        rhombus_superellipse([(0.0, 0.0)] * size, 1.0)


@pytest.mark.parametrize(
    ("k", "requested", "expected"),
    [
        (k, requested, expected)
        for k in _K_VALUES
        for requested, expected in [
            (-4, 4),
            (0, 4),
            (3, 4),
            (4, 4),
            (7, 4),
            (19, 16),
            (40, 40),
            (100, 40),
        ]
    ],
)
def test_curve_count(k: float, requested: int, expected: int) -> None:
    curves = rhombus_superellipse(_SQUARE, k, num_curves=requested)
    assert len(curves) == (4 if k == 0 else expected)
    assert all(len(curve) == 4 for curve in curves)
    assert all(
        math.isfinite(value) for curve in curves for point in curve for value in point
    )


@pytest.mark.parametrize(
    ("k", "clamped"), [(-100.0, -3.0), (-4.0, -3.0), (4.0, 3.0), (100.0, 3.0)]
)
def test_k_clamping(k: float, clamped: float) -> None:
    assert rhombus_superellipse(_SQUARE, k) == rhombus_superellipse(_SQUARE, clamped)


@pytest.mark.parametrize("k", _K_VALUES)
@pytest.mark.parametrize("vertices", [False, True])
def test_closed_endpoints_and_symmetry(k: float, vertices: bool) -> None:
    curves = rhombus_superellipse(
        _SQUARE, k, use_vertice_endpoints=vertices, fit_inside=False
    )
    count = len(curves) // 4
    expected = _SQUARE if vertices else [(-1, 0), (0, -1), (1, 0), (0, 1)]
    for quadrant, endpoint in enumerate(expected):
        assert curves[quadrant * count][0] == pytest.approx(endpoint)
    for index, curve in enumerate(curves):
        assert curve[-1] == pytest.approx(curves[(index + 1) % len(curves)][0])
        opposite = curves[(index + len(curves) // 2) % len(curves)]
        for point, other in zip(curve, opposite):
            assert point == pytest.approx((-other[0], -other[1]))
    if k == 0:
        assert all(start == c1 and c2 == end for start, c1, c2, end in curves)


@pytest.mark.parametrize(
    ("k", "num_curves"),
    [
        (k, count)
        for k in [-3.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]
        for count in [4, 16, 40]
    ],
)
def test_endpoints_satisfy_lame_equation(k: float, num_curves: int) -> None:
    curves = rhombus_superellipse(_SQUARE, k, num_curves=num_curves)
    exponent = 2.0**k
    for curve in curves:
        for x, y in (curve[0], curve[-1]):
            assert abs(x) ** exponent + abs(y) ** exponent == pytest.approx(1.0)


@pytest.mark.parametrize("k", [-3.0, -1.0, 0.5, 1.0, 3.0])
def test_tangent_continuity(k: float) -> None:
    curves = rhombus_superellipse(_SQUARE, k)
    for index, curve in enumerate(curves):
        following = curves[(index + 1) % len(curves)]
        incoming = (curve[-1][0] - curve[-2][0], curve[-1][1] - curve[-2][1])
        outgoing = (
            following[1][0] - following[0][0],
            following[1][1] - following[0][1],
        )
        lengths = math.hypot(*incoming) * math.hypot(*outgoing)
        assert lengths > 0
        assert (
            incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        ) / lengths == pytest.approx(0.0, abs=1e-9)
        dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
        if k < 0 and (index + 1) % (len(curves) // 4) == 0:
            assert dot < 0  # Concave quadrants meet at cusps.
        else:
            assert dot > 0


@pytest.mark.parametrize("k", _K_VALUES)
@pytest.mark.parametrize("vertices", [False, True])
@pytest.mark.parametrize("scale", [0.0, 0.25, 2.0, -1.0])
def test_scaling_about_center(k: float, vertices: bool, scale: float) -> None:
    points = [(30, 20), (220, 65), (250, 225), (60, 180)]
    center = (140.0, 122.5)
    original = rhombus_superellipse(points, k, use_vertice_endpoints=vertices)
    scaled = rhombus_superellipse(
        points, k, scale=scale, use_vertice_endpoints=vertices
    )
    for curve, scaled_curve in zip(original, scaled):
        for point, actual in zip(curve, scaled_curve):
            assert actual == pytest.approx(
                tuple(c + scale * (v - c) for c, v in zip(center, point))
            )


@pytest.mark.parametrize("k", _K_VALUES)
@pytest.mark.parametrize("vertices", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_affine_mapping(k: float, vertices: bool, reverse: bool) -> None:
    # Includes translation, unequal axis lengths, skew, and both windings.
    points = _SQUARE[::-1] if reverse else _SQUARE
    mapped = [(3 * x + y + 10, -x + 2 * y - 7) for x, y in points]
    original = rhombus_superellipse(points, k, use_vertice_endpoints=vertices)
    actual = rhombus_superellipse(mapped, k, use_vertice_endpoints=vertices)
    for curve, mapped_curve in zip(original, actual):
        for (x, y), point in zip(curve, mapped_curve):
            assert point == pytest.approx((3 * x + y + 10, -x + 2 * y - 7))


@pytest.mark.parametrize("k", [0.5, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("num_curves", [4, 16, 40])
def test_bezier_fit_inside(k: float, num_curves: int) -> None:
    curves = rhombus_superellipse(
        _SQUARE, k, num_curves=num_curves, use_vertice_endpoints=True
    )
    maximum = 0.0
    for curve in curves:
        for step in range(101):
            t = step / 100
            weights = ((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t**3)
            point = tuple(
                sum(weight * p[axis] for weight, p in zip(weights, curve))
                for axis in (0, 1)
            )
            maximum = max(maximum, abs(point[0]), abs(point[1]))
    assert maximum <= 1.0 + 1e-9
    assert maximum == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("k", [0.5, 1.0, 2.0, 3.0])
def test_lame_fit_scale(k: float) -> None:
    curves = rhombus_superellipse(
        _SQUARE, k, use_vertice_endpoints=True, use_lame_max=True
    )
    expected_scale = 2.0 ** (1.0 / (2.0**k) - 1.0)
    assert curves[0][0] == pytest.approx((-expected_scale, -expected_scale))
    unfitted = rhombus_superellipse(
        _SQUARE, k, use_vertice_endpoints=True, fit_inside=False
    )
    for curve, raw in zip(curves, unfitted):
        for point, original in zip(curve, raw):
            assert point == pytest.approx(
                tuple(expected_scale * value for value in original)
            )
    assert any(
        abs(value) > 1.0
        for curve in unfitted
        for point in (curve[0], curve[-1])
        for value in point
    )


@pytest.mark.parametrize("k", [-3.0, -1.0, 0.0])
def test_fitting_leaves_concave_and_diamond_unchanged(k: float) -> None:
    expected = rhombus_superellipse(
        _SQUARE, k, use_vertice_endpoints=True, fit_inside=False
    )
    for lame in (False, True):
        assert (
            rhombus_superellipse(
                _SQUARE, k, use_vertice_endpoints=True, use_lame_max=lame
            )
            == expected
        )


@pytest.mark.parametrize("k", [-3.0, 0.0, 1.0, 3.0])
@pytest.mark.parametrize("vertices", [False, True])
@pytest.mark.parametrize("gap", [0.01, 0.25, 0.75])
def test_gap_under_rigid_motion(k: float, vertices: bool, gap: float) -> None:
    # A quarter turn plus translation preserves a gap measured in drawing units.
    moved = [(5 - y, x - 3) for x, y in _SQUARE]
    original = rhombus_superellipse(_SQUARE, k, gap=gap, use_vertice_endpoints=vertices)
    actual = rhombus_superellipse(moved, k, gap=gap, use_vertice_endpoints=vertices)
    count = len(actual) // 4
    for quadrant in range(4):
        end = actual[(quadrant + 1) * count - 1][-1]
        start = actual[((quadrant + 1) * count) % len(actual)][0]
        assert math.dist(end, start) == pytest.approx(gap)
    for curve, moved_curve in zip(original, actual):
        for (x, y), point in zip(curve, moved_curve):
            assert point == pytest.approx((5 - y, x - 3))


def test_svg_sample_output(tmp_path: pathlib.Path) -> None:
    """The existing sample generator still emits all nine shape examples."""
    filename = tmp_path / "superellipse.svg"
    _write_test_svg(str(filename))
    root = ET.parse(filename).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    paths = root.findall("svg:path", namespace)
    assert len(paths) == 9
    assert len(root.findall("svg:polygon", namespace)) == 9
    assert all(
        path.attrib["d"].startswith("M ") and path.attrib["d"].endswith("Z")
        for path in paths
    )


@pytest.mark.parametrize("k", [-3.0, -2.99, -2.9, -2.5])
@pytest.mark.parametrize("num_curves", range(4, 41, 4))
@pytest.mark.parametrize("vertices", [False, True])
def test_near_axis_tangents(k: float, num_curves: int, vertices: bool) -> None:
    """Very flat concave segments remain finite and connected at all resolutions."""
    curves = rhombus_superellipse(
        _SQUARE, k, num_curves=num_curves, use_vertice_endpoints=vertices
    )
    assert len(curves) == num_curves
    for index, curve in enumerate(curves):
        assert curve[-1] == pytest.approx(curves[(index + 1) % len(curves)][0])
        for point in curve:
            assert all(math.isfinite(value) for value in point)
            assert all(abs(value) <= 1.0 + 1e-9 for value in point)


def test_parallel_tangent_fallback_is_straight() -> None:
    """Midpoint fallback produces straight cubics at the formerly failing seams."""
    curves = rhombus_superellipse(_SQUARE, -3.0, num_curves=40)
    for quadrant in range(4):
        for index in (quadrant * 10, quadrant * 10 + 9):
            start, c1, c2, end = curves[index]
            for control, fraction in ((c1, 1.0 / 3.0), (c2, 2.0 / 3.0)):
                expected = tuple(a + fraction * (b - a) for a, b in zip(start, end))
                assert control == pytest.approx(expected, abs=1e-12)
