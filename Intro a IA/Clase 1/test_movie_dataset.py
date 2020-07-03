import numpy as np
from movie_dataset import MovieDatasetStructured


def test_movie():
    dataset = MovieDatasetStructured('ratings_formato.csv')
    assert dataset.get_first_five_rows()[1] == np.array([(1, 147, 4.5, 1425942435)], dtype=dataset.get_dtype())
