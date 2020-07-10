import numpy as np
import pytest
from distance_center import distance_center, minimum_distance_center


@pytest.fixture()
def example_points_center():
    points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    centers = np.array([[1, 0, 0], [0, 1, 0]])
    result = np.array([[3.6056, 8.3666, 13.4536], [3.3166, 8.2462, 13.3791]])
    return points, centers, result


def test_distance(example_points_center):
    points, centers, result = example_points_center
    clusters_id = np.array([1, 1, 1])
    assert np.all(result == distance_center(points, centers))
    assert np.all(clusters_id == minimum_distance_center(result))
