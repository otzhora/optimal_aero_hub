import pytest

from hub.geodesic import Point, distance


def test_distance():
    moscow = Point(55.751244, 37.618423)
    spb = Point(59.9342802, 30.3350986)

    assert 633.4538325946983 == pytest.approx(distance(moscow, spb))

    newport = Point(41.49008, -71.312796)
    cleveland = Point(41.499498, -81.695391)

    assert 864.2144943393627 == pytest.approx(distance(newport, cleveland))
