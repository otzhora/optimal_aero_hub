import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from hub.geodesic import Point, distance, distance_tensors


def test_distance():
    moscow = Point(55.751244, 37.618423)
    spb = Point(59.9342802, 30.3350986)

    assert 633.4538325946983 == pytest.approx(distance(moscow, spb))

    newport = Point(41.49008, -71.312796)
    cleveland = Point(41.499498, -81.695391)

    assert 864.2144943393627 == pytest.approx(distance(newport, cleveland))


def test_distance_tensors():
    moscow = torch.tensor([[55.751244, 37.618423]])
    spb = torch.tensor([[59.9342802, 30.3350986]])

    d = distance_tensors(moscow, spb).numpy()

    assert_allclose(np.array([633.4538325946983]), d, rtol=1e-03)

    newport = torch.tensor([[41.49008, -71.312796]])
    cleveland = torch.tensor([[41.499498, -81.695391]])

    d = distance_tensors(newport, cleveland).numpy()

    assert_allclose(np.array([864.2144943393627]), d, rtol=1e-03)
