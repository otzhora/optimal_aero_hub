import torch

from hub.geodesic import Points


def exp_cost(hub_coords: torch.Tensor, points: Points, C: float = 1.) -> torch.Tensor:
    """
    Calculate cost of placing hub in a hub_coords
    :param C: impact of distance to result
    :param hub_coords: possible coords of hub
    :param points: list of zones
    :return: cost
    """
    return (torch.exp(C * points.distances_to_point(hub_coords) / 1000) * points.normalized_passengers).sum()
