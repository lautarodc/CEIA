import numpy as np


class Data(object):

    def __init__(self, file_name_position, file_name_velocity, file_name_acceleration):
        self.position = self._build_dataset(file_name_position)
        self.velocity = self._build_dataset(file_name_velocity)
        self.acceleration = self._build_dataset(file_name_acceleration)

    def _build_dataset(self, file_name):
        self.structured_type = np.dtype(
            [('id', np.int64), ('var_x', np.float32), ('var_y', np.float32), ('var_z', np.float32)])
        return np.genfromtxt(file_name, dtype=self.structured_type, delimiter=None)

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_acceleration(self):
        return self.acceleration

    def get_position_gaussian(self, media, std):
        data_size = self.position['var_x'].shape
        px_gaussian = self.position['var_x'] + np.random.normal(media, std, data_size)
        py_gaussian = self.position['var_y'] + np.random.normal(media, std, data_size)
        pz_gaussian = self.position['var_z'] + np.random.normal(media, std, data_size)
        r = np.eye(3) * np.array([std ** 2, std ** 2, std ** 2])
        q = np.eye(9) * 0.3
        y = np.array([px_gaussian, py_gaussian, pz_gaussian])
        c = np.concatenate((np.eye(3), np.zeros((3, 6))), axis=1)
        return r, q, c, y

    def get_position_uniform(self, std):
        data_size = self.position['var_x'].shape
        # Get the low and high values of uniform distribution based on standard deviation
        var = std**2
        interval = (np.sqrt(12*var))
        high = interval/2
        low = -high
        px_gaussian = self.position['var_x'] + np.random.uniform(low, high, data_size)
        py_gaussian = self.position['var_y'] + np.random.uniform(low, high, data_size)
        pz_gaussian = self.position['var_z'] + np.random.uniform(low, high, data_size)
        r = np.eye(3) * np.array([std ** 2, std ** 2, std ** 2])
        q = np.eye(9) * 0.3
        y = np.array([px_gaussian, py_gaussian, pz_gaussian])
        c = np.concatenate((np.eye(3), np.zeros((3, 6))), axis=1)
        return r, q, c, y

    def get_pos_vel_gaussian(self, media_p, std_p, media_v, std_v):
        data_size = self.position['var_x'].shape
        px_gaussian = self.position['var_x'] + np.random.normal(media_p, std_p, data_size)
        py_gaussian = self.position['var_y'] + np.random.normal(media_p, std_p, data_size)
        pz_gaussian = self.position['var_z'] + np.random.normal(media_p, std_p, data_size)
        vx_gaussian = self.velocity['var_x'] + np.random.normal(media_v, std_v, data_size)
        vy_gaussian = self.velocity['var_y'] + np.random.normal(media_v, std_v, data_size)
        vz_gaussian = self.velocity['var_z'] + np.random.normal(media_v, std_v, data_size)
        r = np.eye(6) * np.array([std_p ** 2, std_p ** 2, std_p ** 2, std_v ** 2, std_v ** 2, std_v ** 2])
        q = np.eye(9) * 0.3
        y = np.array([px_gaussian, py_gaussian, pz_gaussian, vx_gaussian, vy_gaussian, vz_gaussian])
        c = np.concatenate((np.eye(6), np.zeros((6, 3))), axis=1)
        return r, q, c, y
