
import geom2d
from geom2d.point import P
from geom2d.arc import Arc

arc1 = Arc(
    P(2.9339552, 4.5527481),
    P(2.796207118678724, 5.066830803567288),
    1.0281651740545676,
    0.5235989064807819,
    P(1.9057900259454326, 4.5527481)
)
arc2 = Arc(
    P(2.796207118678724, 5.066830803567288),
    P(2.4198724, 5.443165499999999),
    1.028165930499886,
    0.5235985729456468,
    P(1.9057893708446176, 4.5527477217772585)
)



def test_g1():
    """Test G1 (tangential connection)."""
    assert geom2d.float_eq(arc1.end_tangent_angle(), arc2.start_tangent_angle())

