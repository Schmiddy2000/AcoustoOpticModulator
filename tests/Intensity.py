import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Daten
inten_frequencies = np.array([5, 4.5, 4, 3.5, 3, 2, 1])
inten_frequencies_2 = np.array([4.5, 4, 3.5, 3, 2, 1])
intensities_5khz = np.array([808, 875, 828, 826, 747, 589, 354])
intensities_1khz = np.array([1160, 1180, 1120, 1130, 1080, 905, 578])
intensities_100hz = np.array([1200, 1160, 1160, 1100, 900, 584])
intensities_20hz =np.array([1200, 1210, 1220, 1180, 1130, 913, 596])
chopper_1or = np.array([438, 462, 447, 450, 462, 438, 441, 436, 382, 360, 328,
                        295, 262, 221, 170, 119, 58.3, 24.1, 18, 15.6])
chopper_0or = np.array([584, 594, 590, 585, 597, 592, 626, 597, 593, 596, 594, 604, 599, 613, 626, 618])
chopper_frequencies = np.array([5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.9, 0.8, 0.7,
                                0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.002])
chopper_frequencies_0 = np.array([5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.9, 0.8, 0.7,
                                0.6, 0.5, 0.4, 0.1])

# Angepasste Fit-Funktion: Normale Sinusfunktion
def func(x, a, b):
    return a * np.sin(np.sqrt(b * x))**2

# Curve fitting für jede Datenreihe
popt_5khz, _ = curve_fit(func, inten_frequencies, intensities_5khz)
popt_1khz, _ = curve_fit(func, inten_frequencies, intensities_1khz)
popt_20hz, _ = curve_fit(func, inten_frequencies, intensities_20hz)
popt_100hz, _ = curve_fit(func, inten_frequencies_2, intensities_100hz)
popt_chopper, _ = curve_fit(func, chopper_frequencies, chopper_1or)
# Erstellen von Werten für die Fit-Kurven
x_fit = np.linspace(1, 5, 100)
x_fit_chopper= np.linspace(0, 5, 100)
y_fit_5khz = func(x_fit, *popt_5khz)
y_fit_1khz = func(x_fit, *popt_1khz)
y_fit_20hz = func(x_fit, *popt_20hz)
y_fit_100hz = func(x_fit, *popt_100hz)
y_fit_chopper = func(x_fit_chopper, *popt_chopper)
# Plots mit Markierungen
plt.plot(chopper_frequencies, chopper_1or, 'cx', label='chopper Data')
#plt.plot(chopper_frequencies_0, chopper_0or, 'g', label='chopper Data')
plt.plot(x_fit_chopper, y_fit_chopper, 'c-', label='Chopper Fit')
# Achsenbeschriftungen und Titel
plt.xlabel('Amplitude of Soundwave (V)', fontsize=12)
plt.ylabel('Intensity of 1st Order Beam (mV)', fontsize=12)
plt.title('Intensity vs Amplitude measured with chopper', fontsize=14)


plt.legend()
plt.grid(True)
plt.savefig('intensities_chopper.png', dpi=200)
plt.show()


plt.plot(inten_frequencies, intensities_5khz, 'bx', label='5 kHz Data')
plt.plot(x_fit, y_fit_5khz, 'b-', label='5 kHz Fit')
plt.plot(inten_frequencies, intensities_1khz, 'gx', label='1 kHz Data')
plt.plot(x_fit, y_fit_1khz, 'g-', label='1 kHz Fit')
plt.plot(inten_frequencies, intensities_20hz, 'rx', label='20 Hz Data')
plt.plot(x_fit, y_fit_20hz, 'r-', label='20 Hz Fit')
plt.plot(inten_frequencies_2, intensities_100hz, 'cx', label='100 Hz Data')
plt.plot(x_fit, y_fit_100hz, 'c-', label='100 Hz Fit')

# Achsenbeschriftungen und Titel
plt.xlabel('Amplitude of Soundwave (V)', fontsize=12)
plt.ylabel('Intensity of 1st Order Beam (mV)', fontsize=12)
plt.title('Intensity vs Amplitude for Different Frequencies', fontsize=14)


plt.legend()
plt.grid(True)
plt.savefig('intensities_modulated.png', dpi=200)
#plt.show()
#plt.plot(chopper_frequencies, chopper_1or, 'yo', label='chopper Data', marker = 'x')
#plt.plot(x_fit_chopper, y_fit_chopper, 'r-', label='Chopper Fit')
