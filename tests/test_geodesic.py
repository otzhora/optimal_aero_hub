import numpy as np
import pandas as pd
import torch
from numpy.testing import assert_allclose

from hub.geodesic import Points


def test_points():
    moscow = pd.DataFrame([{"lat": 55.751244, "lng": 37.618423, "passengers": 1}])
    spb = torch.tensor([[59.9342802, 30.3350986]])

    Points_msk = Points(moscow)

    d = Points_msk.distances_to_point(spb).numpy()

    assert_allclose(np.array([633.4538325946983]), d, rtol=1e-03)

    df = pd.DataFrame([{"lat": 41.49008, "lng": -71.312796, "passengers": 1},
                       {"lat": 41.499498, "lng": -81.695391, "passengers": 1},
                       {"lat": 55.751244, "lng": 37.618423, "passengers": 1}])
    Points_df = Points(df)

    d = Points_df.distances_to_point(spb).numpy()

    assert_allclose(np.array([6689.4517, 7156.5312, 633.4543]), d, rtol=1e-03)
