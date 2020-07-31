import numpy as np
import matplotlib.pyplot as plt
from Dataset import Data
from KalmanFilter import KalmanFilter
from Metric import MSE

# Dataset Values
dataset = Data('posicion.dat', 'velocidad.dat', 'aceleracion.dat')
position = dataset.get_position()
velocity = dataset.get_velocity()
acceleration = dataset.get_acceleration()

# Initial Conditions
x0 = np.array([10.7533, 36.6777, -45.1769, 1.1009, -17.0, 35.7418, -5.7247, 3.4268, 5.2774])
p0 = np.diag(np.array([100, 100, 100, 1, 1, 1, 0.1, 0.1, 0.1]))

# Input Matrix
b = np.eye(9)

# Sample time
h = 1

# Process Matrix
eye = np.eye(3)
a_1 = np.hstack((eye, eye * h, eye * ((h ** 2) * 0.5)))
a_2 = np.hstack((np.zeros(eye.shape), eye, eye * h))
a_3 = np.hstack((np.zeros(eye.shape), np.zeros(eye.shape), eye))
a = np.vstack((a_1, a_2, a_3))

# Get measurements
r_pv, q_pv, c_pv, y_pv = dataset.get_pos_vel_gaussian(0, 10, 0, 0.2)
r_g, q_g, c_g, y_g = dataset.get_position_gaussian(0, 10)
r_u, q_u, c_u, y_u = dataset.get_position_uniform(10)

# Instantiate Kalman Filters
kalman_gaussian = KalmanFilter(r_g, q_g, y_g, x0, p0, a, b, c_g)
kalman_uniform = KalmanFilter(r_u, q_u, y_u, x0, p0, a, b, c_u)
kalman_velocity = KalmanFilter(r_pv, q_pv, y_pv, x0, p0, a, b, c_pv)

# Get predictions, covariance matrices and Kalman gains
x_g, p_g, kn_g = kalman_gaussian.get_prediction()
x_u, p_u, kn_u = kalman_uniform.get_prediction()
x_pv, p_pv, kn_pv = kalman_velocity.get_prediction()

# MSE calculation for different scenarios
mse = MSE()

mse_gaussian_position = np.zeros((9, 100))
mse_uniform = np.zeros((9, 100))
mse_gaussian_velocity = np.zeros((9, 100))

for i in range(100):
    r_pv, q_pv, c_pv, y_pv = dataset.get_pos_vel_gaussian(0, 10, 0, 0.2)
    r_g, q_g, c_g, y_g = dataset.get_position_gaussian(0, 10)
    r_u, q_u, c_u, y_u = dataset.get_position_uniform(10)
    kalman_gaussian = KalmanFilter(r_g, q_g, y_g, x0, p0, a, b, c_g)
    kalman_uniform = KalmanFilter(r_u, q_u, y_u, x0, p0, a, b, c_u)
    kalman_velocity = KalmanFilter(r_pv, q_pv, y_pv, x0, p0, a, b, c_pv)
    x_g, p_g, kn_g = kalman_gaussian.get_prediction()
    x_u, p_u, kn_u = kalman_uniform.get_prediction()
    x_pv, p_pv, kn_pv = kalman_velocity.get_prediction()
    mse_gaussian_position[:, i] = np.array([mse(position['var_x'], x_g[0, 1:]), mse(position['var_y'], x_g[1, 1:]),
                                            mse(position['var_z'], x_g[2, 1:]), mse(velocity['var_x'], x_g[3, 1:]),
                                            mse(velocity['var_y'], x_g[4, 1:]),
                                            mse(velocity['var_z'], x_g[5, 1:]), mse(acceleration['var_x'], x_g[6, 1:]),
                                            mse(acceleration['var_y'], x_g[7, 1:]),
                                            mse(acceleration['var_z'], x_g[8, 1:])])
    mse_uniform[:, i] = np.array([mse(position['var_x'], x_u[0, 1:]), mse(position['var_y'], x_u[1, 1:]),
                                  mse(position['var_z'], x_u[2, 1:]), mse(velocity['var_x'], x_u[3, 1:]),
                                  mse(velocity['var_y'], x_u[4, 1:]),
                                  mse(velocity['var_z'], x_u[5, 1:]), mse(acceleration['var_x'], x_u[6, 1:]),
                                  mse(acceleration['var_y'], x_u[7, 1:]),
                                  mse(acceleration['var_z'], x_u[8, 1:])])
    mse_gaussian_velocity[:, i] = np.array([mse(position['var_x'], x_pv[0, 1:]), mse(position['var_y'], x_pv[1, 1:]),
                                            mse(position['var_z'], x_pv[2, 1:]), mse(velocity['var_x'], x_pv[3, 1:]),
                                            mse(velocity['var_y'], x_pv[4, 1:]),
                                            mse(velocity['var_z'], x_pv[5, 1:]),
                                            mse(acceleration['var_x'], x_pv[6, 1:]),
                                            mse(acceleration['var_y'], x_pv[7, 1:]),
                                            mse(acceleration['var_z'], x_pv[8, 1:])])

mse_gaussian = np.mean(mse_gaussian_position, axis=1)
mse_uniform_mean = np.mean(mse_uniform, axis=1)
mse_velocity = np.mean(mse_gaussian_velocity, axis=1)
measure_gaussian = np.array([mse(position['var_x'], y_g[0, :]), mse(position['var_y'], y_g[1, :]),
                             mse(position['var_z'], y_g[2, :])])
measure_uniform = np.array([mse(position['var_x'], y_u[0, :]), mse(position['var_y'], y_u[1, :]),
                            mse(position['var_z'], y_u[2, :])])
measure_velocity = np.array([mse(position['var_x'], y_pv[0, :]), mse(position['var_y'], y_pv[1, :]),
                             mse(position['var_z'], y_pv[2, :])])
measure_velocity_vel = np.array([mse(velocity['var_x'], y_pv[3, :]), mse(velocity['var_y'], y_pv[4, :]),
                                 mse(velocity['var_z'], y_pv[5, :])])
mse_prediction = np.vstack(
    (mse_gaussian[0:3], measure_gaussian, mse_uniform_mean[0:3], measure_uniform, mse_velocity[0:3], measure_velocity))
mse_vel_prediction = np.vstack(
    (mse_gaussian[3:6], mse_uniform_mean[3:6], mse_velocity[3:6], measure_velocity_vel))

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.axis('off')
rows = ['Gaussian Pos Prediction', 'Gaussian Pos Measurement ', 'Uniform', 'Uniform Measurement',
        'Gaussian Pos-Vel Prediction',
        'Gaussian Pos-Vel Measurement']
cols = ["MSE Position X", "MSE Position Y", "MSE Position Z"]
table = ax.table(cellText=mse_prediction, colLabels=cols, rowLabels=rows, loc='upper center', cellLoc='center')
table.auto_set_font_size(True)

fig = plt.figure(4)
ax = fig.add_subplot(111)
ax.axis('off')
rows = ['Gaussian Velocity Prediction', 'Uniform Velocity Prediction',
        'Gaussian Pos-Vel Velocity Prediction',
        'Gaussian Pos-Vel Velocity Measurement']
cols = ["MSE Velocity X", "MSE Velocity Y", "MSE Velocity Z"]
table = ax.table(cellText=mse_vel_prediction, colLabels=cols, rowLabels=rows, loc='upper center', cellLoc='center')
table.auto_set_font_size(True)

# Graphics

# Time vector for plotting
time = np.arange(y_g.shape[1]) * h

# Comparision of Kalman Filter, Measurement and True Value
plt.figure(2)
plt.subplot(411)
plt.subplots_adjust(hspace=0.5)

plt.grid(True)
plt.title('Comparision of KF Position Estimate, Position Measurement and True Value')
plt.plot(time, position['var_x'], label="True Position X")
plt.plot(time, x_u[0, 1:], label="KF Prediction Position X")
plt.plot(time, y_u[0, :], label="Measurements Position X")
ax = plt.gca()
ax.set_xlim(340, 350)
ax.set_ylim(-5600, -5300)
plt.legend()

plt.subplot(412)
plt.grid(True)
plt.title('Comparision of KF Position Estimate, Position Measurement and True Value')
plt.plot(time, position['var_x'], label="True Position X")
plt.plot(time, x_u[0, 1:], label="KF Prediction Position X")
plt.plot(time, y_u[0, :], label="Measurements Position X")
ax = plt.gca()
ax.set_xlim(349, 350)
ax.set_ylim(-5550, -5570)
plt.legend()

plt.subplot(413)
plt.grid(True)
plt.title('Evolution of the covariance matrix trace of estimate')
plt.plot(time, np.trace(p_u[:, :, 1:]), label="Trace of covariance matrix")
plt.legend()

plt.subplot(414)
plt.grid(True)
plt.title('Evolution of Kalman Gain for Position X')
plt.plot(time, kn_u[0, 0, 1:], label="Kalman Gain for Position X")
plt.legend()

# Comparision of different of case 1, 2 and 3
plt.figure(3)
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.grid(True)
plt.title('Comparision of different filters predictions vs. true x position values ')
plt.plot(time, x_g[0, 1:], label="Gaussian Predictions Position X")
plt.plot(time, x_u[0, 1:], label="Uniform Predictions Position X")
plt.plot(time, x_pv[0, 1:], label="Position-Velocity Predictions Position X")
plt.plot(time, position['var_x'], label="True Position X")
plt.legend()

plt.subplot(312)
plt.grid(True)
plt.title('Comparision of different filters predictions vs. true y position values ')
plt.plot(time, x_g[1, 1:], label="Gaussian Predictions Pos Y")
plt.plot(time, x_u[1, 1:], label="Uniform Predictions Pos Y")
plt.plot(time, x_pv[1, 1:], label="Position-Velocity Predictions Pos Y")
plt.plot(time, position['var_y'], label="True Pos Y")
plt.legend()

plt.subplot(313)
plt.grid(True)
plt.title('Comparision of different filters predictions vs. true z position values ')
plt.plot(time, x_g[2, 1:], label="Gaussian Predictions Pos Z")
plt.plot(time, x_u[2, 1:], label="Uniform Predictions Pos Z")
plt.plot(time, x_pv[2, 1:], label="Position-Velocity Predictions Pos Z")
plt.plot(time, position['var_z'], label="True Pos Z")
plt.legend()

plt.show()
