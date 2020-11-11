import torch

from hub.geodesic import Points


def exp_cost(hub_coords: torch.Tensor, points: Points) -> torch.Tensor:
    """
    Calculate cost of placing hub in a hub_coords
    :param hub_coords: possible coords of hub
    :param points: list of zones
    :return: cost
    """
    return (torch.exp(points.distances_to_point(hub_coords) / 1000) * points.normalized_passengers).sum()
