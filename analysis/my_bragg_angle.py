# Imports

import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets.QScroller import velocity
from scipy.stats import linregress

# Given constants
n = 2.2
wavelength = 532e-9  # in meters
L = 0.02  # in meters
V = 4260  # in m/s


# Define the function for Q as a function of F
def Q(F):
    return (2 * np.pi * wavelength * L * F**2) / (n**2 * V**2)


# Generate a range of frequencies
F_values = np.linspace(1e3, 1e8, 1000)  # from 1 kHz to 1 mHz

# Calculate Q for each frequency
Q_values = Q(F_values)


def theta(F):
    return wavelength * F / V


F = np.array([75.4, 79.4, 84.4]) *1e6
print(theta(F)*1.2)


distances = np.array([9.75/2, 10.35/2, 11.25/2]) * 1e-3
distances_2 = distances + 1.25e-3


def theta_V(fs, distances):
    theta = distances / 1.26 #+- 1

    return wavelength * fs / (2 * theta)


#theta_V(distances_2)

theta_values = theta(F_values) *180/np.pi
# Plotting


f_werte = np.array([95.8, 89.7, 84.4, 79.5, 75.4, 71.6, 68.3, 65.3, 62.4]) * 1e6
f_unsicher = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.3, 0.3, 0.3, 0.4])
distances_3 = np.array([15.75, 14.25, 13.25])*5e-4
distances_3_2 = np.array([15.20, 14.28, 13.13, 12.61, 12.34, 11.5, 11, 10.75, 10]) * 5e-4


def velocity_error(_lambda, delta_lambda, frequency, delta_frequency, bragg_angle, delta_bragg_angle):
    dv_dl = frequency * delta_lambda / (2 * bragg_angle)
    dv_df = _lambda * delta_frequency / (2 * bragg_angle)
    dv_dt = _lambda * frequency * delta_bragg_angle / (2 * bragg_angle ** 2)
    errors = np.sqrt(dv_dl ** 2 + dv_df ** 2 + dv_dt ** 2)

    return errors


def get_bragg_angle_errors(dot_distances, dot_distance_errors):
    hypotenuse = 1.26

    # Do a real error calculation / assumption here...
    hypotenuse_error = 0.02

    dt_dg = (dot_distance_errors / hypotenuse) ** 2
    dt_da = (dot_distances * hypotenuse_error / (hypotenuse ** 2)) ** 2

    bragg_angle_errors = np.sqrt(dt_dg + dt_da)

    return bragg_angle_errors


# print('bragg angle error:', get_bragg_angle_errors(15 * 5))


v_errors = velocity_error(wavelength, f_werte, distances_3_2/1.26, 1e-10, f_unsicher, 1e-3)

v_values = theta_V(f_werte, distances_3_2)
weights = 1/(v_errors**2)
v_mean = np.sum(v_values*weights)/np.sum(weights)
v_mean_errors = np.sqrt(1/np.sum(1/v_errors))
slope, intercept, r_value, p_value, std_err = linregress(f_werte, v_values)
print(v_mean, v_mean_errors)
# Plot
plt.figure(figsize=(10, 6))
plt.scatter(f_werte, v_values, label='Datenpunkte')
plt.plot(f_werte, slope * f_werte + intercept, color='red', label=f'Lineare Regression\ny = {slope:.5f}x + {intercept:.2f}\n$R^2$ = {r_value**2:.4f}')
plt.xlabel('Frequenz (Hz)')
plt.ylabel('Schallgeschwindigkeit (m/s)')
plt.title('Lineare Regression der Schallgeschwindigkeit in Abhängigkeit von der Frequenz')
plt.legend()
plt.grid(True)