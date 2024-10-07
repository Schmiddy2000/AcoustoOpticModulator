# Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import odr

# Given constants
n = 2.2
wavelength = 532e-9  # in meters
L = 0.02  # in meters
V = 4260  # in m/s
rad_to_degree_factor = 180 / np.pi

print(0.006 * rad_to_degree_factor, 0.000325 * rad_to_degree_factor)


# Define the function for Q as a function of F
def Q(F):
    return (2 * np.pi * wavelength * L * F ** 2) / (n ** 2 * V ** 2)


# Generate a range of frequencies
F_values = np.linspace(6e7, 1e8, 1000)  # from 1 kHz to 1 mHz

# Calculate Q for each frequency
Q_values = Q(F_values)


def theta(F):
    return 0.5 * wavelength * F / V


def get_bragg_angle(distances):
    return 0.5 * np.arctan(distances / 1.2614)


def get_bragg_angle_errors(dot_distances, dot_distance_errors):
    hypotenuse = 1.2614

    # Do a real error calculation / assumption here...
    hypotenuse_error = np.sqrt(0.025 ** 2 + 0.41 ** 2)

    dt_dg = (dot_distance_errors / hypotenuse) ** 2
    dt_da = (dot_distances * hypotenuse_error / (hypotenuse ** 2)) ** 2

    bragg_angle_errors = np.sqrt(dt_dg + dt_da)

    return bragg_angle_errors


# F = np.array([75.4, 79.4, 84.4]) *1e6 testvalues
def delta_theta(x, delta_x):  ##with tan
    L = 1.2614
    delta_L = 0.4e-3
    return np.sqrt((L * delta_x / (x ** 2 + L ** 2)) ** 2 + (x * delta_L / (L ** 2 + x ** 2)) ** 2)


# distances = np.array([9.75/2, 10.35/2, 11.25/2]) * 1e-3
# distances_2 = distances + 1.25e-3
# distances_3 = np.array([15.75, 14.25, 13.25])*5e-4

def theta_V(fs, distances):
    theta = distances / 1.2614

    return wavelength * fs / (np.sin(theta))


#hypotenuse_error = np.sqrt(0.025**2 + 0.41**2)


#theta_values = theta(F_values) *180/np.pi

# Plotting


f_werte = np.array([95.8, 89.7, 84.4, 79.5, 75.4, 71.6, 68.3, 65.3, 62.4]) * 1e6
f_unsicher = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.3, 0.3, 0.3, 0.4]) * 1e6
f_error = f_werte * 1e-6
distances_3_2 = np.array([15.20, 14.28, 13.13, 12.61, 12.34, 11.5, 11, 10.75, 10]) * 1e-3
theta_b = get_bragg_angle(distances_3_2)
theta_b_unsicher = delta_theta(distances_3_2, 0.82e-3) * 0.5

print(list(zip(theta_b, theta_b_unsicher)))


## ODR-Modell definieren
def linear_func(p, x):
    return p[0] * x + p[1]


# ODR einrichten
linear_model = odr.Model(linear_func)
data = odr.RealData(f_werte, theta_b, sx=f_unsicher, sy=theta_b_unsicher)
odr_instance = odr.ODR(data, linear_model, beta0=[1e-16, 1e-8])

# ODR ausführen
output = odr_instance.run()
slope, intercept = output.beta
slope_err, intercept_err = output.sd_beta

# Berechnung der vorhergesagten y-Werte
theta_pred = slope * f_werte + intercept
theta_pred_minus = (slope + slope_err) * f_werte + (intercept + intercept_err)
theta_pred_plus = (slope - slope_err) * f_werte + (intercept - intercept_err)

# Plotten der Daten und der ODR-Fit-Linie
plt.figure(figsize=(12, 5))
plt.errorbar(f_werte, theta_b, yerr=theta_b_unsicher, xerr=f_unsicher, fmt='o', label='Data with uncertainties',
             capsize=5, ecolor='k')
plt.plot(f_werte, theta_pred, 'r-', label=f'ODR Fit\ny = {slope:.2e}x + {intercept:.2e}')
plt.fill_between(f_werte, theta_pred_minus, theta_pred_plus, ls='--', color='r', alpha=0.2)
plt.xlabel('Frequenz (Hz)')
plt.ylabel('Bragg Winkel (rad)')
plt.title('ODR Fit Frequency vs. Bragg Angle')
plt.legend()
plt.grid(True)
plt.savefig('Bragg_angle_vs_frequency.png', dpi=200)

# plt.show()


# velocity
def velocity_steigung(slope):
    n = 2.3
    return wavelength / (n * slope)


def velocity_steigung_err(slope, slope_err):
    n = 2.3
    dw = 0.5e-9
    dv_dw = 1 / (slope * n)
    dv_ds = wavelength / (slope ** 2 * n)
    return np.sqrt((dv_dw * dw) ** 2 + (dv_ds * slope_err) ** 2)


# print(delta_theta(distances_3_2, 0.8e-3), velocity_steigung_err(slope, slope_err))
print('v(slope), v_err:\n', velocity_steigung(slope), velocity_steigung_err(slope, slope_err))


def velocity_error(_lambda, delta_lambda, frequency, delta_frequency, bragg_angle
                   , delta_bragg_angle):
    dv_dl = frequency * delta_lambda / (2 * np.sin(bragg_angle))
    dv_df = _lambda * delta_frequency / (2 * np.sin(bragg_angle))
    dv_dt = _lambda * frequency * delta_bragg_angle / (2 * bragg_angle ** 2)
    errors = np.sqrt(dv_dl ** 2 + dv_df ** 2 + dv_dt ** 2)

    return errors


v_errors = velocity_error(wavelength, 0.05e-9, f_werte, f_unsicher, theta_b, theta_b_unsicher)
v_values = theta_V(f_werte, distances_3_2)
weights = 1 / (v_errors ** 2)
v_mean = np.sum(v_values * weights) / np.sum(weights)
v_mean_errors = np.sqrt(1 / np.sum(1 / v_errors ** 2))

print('v_values, v_errors:')
print(list(v_values), list(v_errors))
print(theta_b_unsicher)
print(v_mean, v_mean_errors)
