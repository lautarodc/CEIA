import numpy as np
import pickle


class MovieDatasetStructured(object):
    instance = None

    def __new__(cls, file_name):
        if MovieDatasetStructured.instance is None:
            MovieDatasetStructured.instance = super(MovieDatasetStructured, cls).__new__(cls)
            return MovieDatasetStructured.instance
        else:
            return MovieDatasetStructured.instance

    def __init__(self, file_name):
        self.mtype = np.dtype(
            [('userId', np.int64), ('movieId', np.int64), ('rating', np.float32), ('timestamp', np.int64)])
        try:
            self.structured_dataset = pickle.load(open("dataset.pickle", "rb"))
        except(OSError, IOError) as e:
            mdata = np.genfromtxt(file_name, dtype=self.mtype, delimiter=",", skip_header=1)
            pickle.dump(mdata, open("dataset.pickle", "wb"))
            self.structured_dataset = pickle.load(open("dataset.pickle", "rb"))

    def get_first_five_rows(self):
        return self.structured_dataset[0:5]

    def get_dtype(self):
        return self.mtype