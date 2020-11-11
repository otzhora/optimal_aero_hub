import pandas as pd
import torch

EARTH_RADIUS = 6371.009


class Points:
    """
    Class to store geodesic coordinates of airports
    """
    def __init__(self, airports: pd.DataFrame):
        """
        Store coordinates and population of airports. Dataframe should have lat, lng and passengers in columns.
        :param airports: DataFrame with airports data
        """
        self.airports = airports
        self.coordinates = torch.tensor([*zip(airports.loc[:, "lat"], airports.loc[:, 'lng'])],
                                        dtype=torch.float64)
        self.passengers = torch.tensor(airports.loc[:, "passengers"], dtype=torch.float64)
        self.normalized_passengers = self.passengers / self.passengers.sum()

    def __str__(self):
        return self.airports.__str__()

    def __repr__(self):
        return self.airports.__repr__()

    def distances_to_point(self, other: torch.Tensor) -> torch.Tensor:
        """
        Given coordinates of some point returns distances to this point from every point
        :param other: coordinates of other point
        :return: tensor of distances
        """
        assert other.shape == (1, 2)

        points_1 = torch.deg2rad(other)
        points_2 = torch.deg2rad(self.coordinates)

        sin_lat1, cos_lat1 = torch.sin(points_1[:, 0]), torch.cos(points_1[:, 0])
        sin_lat2, cos_lat2 = torch.sin(points_2[:, 0]), torch.cos(points_2[:, 0])

        delta_lng = points_2[:, 1] - points_1[:, 1]
        cos_delta_lng, sin_delta_lng = torch.cos(delta_lng), torch.sin(delta_lng)

        d = torch.atan2(torch.sqrt(torch.pow((cos_lat2 * sin_delta_lng), 2) +
                                   torch.pow((cos_lat1 * sin_lat2 -
                                              sin_lat1 * cos_lat2 * cos_delta_lng), 2)),
                        sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)

        return EARTH_RADIUS * d
