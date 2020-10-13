from typing import Optional

import numpy as np
import torch

EARTH_RADIUS = 6371.009


# TODO: __repr__, __str__, etc
# TODO: init from string, tuple
class Point:
    """
    Class to store geodesic coordinates of points
    """
    longitude: Optional[float] = None
    latitude: Optional[float] = None

    def __init__(self, latitude: float, longitude: float):
        self.longitude = longitude
        self.latitude = latitude


def distance(a: Point, b: Point) -> float:
    """
    Given geodesic coordinates of two point return distance between them. I assume Earth to be sphere.

    Example:
    >>>> moscow = Point(55.751244, 37.618423)
    >>>> spb = Point(59.9342802, 30.3350986)
    >>>> distance(moscow, spb)
    633.4538325946983

    :param a: coordinates of point a
    :param b: coordinates of point b
    :return: distance between a and b in kilometers
    """
    lat1, lng1 = np.radians(a.latitude), np.radians(a.longitude)
    lat2, lng2 = np.radians(b.latitude), np.radians(b.longitude)

    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)

    delta_lng = lng2 - lng1
    cos_delta_lng, sin_delta_lng = np.cos(delta_lng), np.sin(delta_lng)

    d = np.arctan2(np.sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                           (cos_lat1 * sin_lat2 -
                            sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
                   sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)

    return EARTH_RADIUS * d


def distance_tensors(points_1: torch.tensor, points_2: torch.tensor):
    """
    Given geodesic coordinates of two tensors of point return the pairwise distance between them.
    I assume Earth to be a sphere.

    Example:
    >>>> moscow = torch.tensor([55.751244, 37.618423])
    >>>> spb = torch.tensor([59.9342802, 30.3350986])
    >>>> distance_tensors(moscow, spb)
    tensor([633.4543])

    :param points_1: coordinates of points
    :param points_2: coordinates of points
    :return: pairwise disctances between points_1 and points_2 in kilometers
    """

    points_1 = torch.deg2rad(points_1)
    points_2 = torch.deg2rad(points_2)

    sin_lat1, cos_lat1 = torch.sin(points_1[:, 0]), torch.cos(points_1[:, 0])
    sin_lat2, cos_lat2 = torch.sin(points_2[:, 0]), torch.cos(points_2[:, 0])

    delta_lng = points_2[:, 1] - points_1[:, 1]
    cos_delta_lng, sin_delta_lng = torch.cos(delta_lng), torch.sin(delta_lng)

    d = torch.atan2(torch.sqrt(torch.pow((cos_lat2 * sin_delta_lng), 2) +
                               torch.pow((cos_lat1 * sin_lat2 -
                                          sin_lat1 * cos_lat2 * cos_delta_lng), 2)),
                    sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)

    return EARTH_RADIUS * d
