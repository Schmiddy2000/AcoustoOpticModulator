import matplotlib.pyplot as plt
import numpy as np
from scipy.odr import ODR, Model, RealData

# Daten
v = np.array([4229.57655852, 4215.39203984, 4313.70083065, 4230.81207037, 4100.41272247,
              4178.16567748, 4166.75516014, 4076.37784988, 4187.4882147])
delta_v = np.array([456.34533573, 484.18841418, 538.99798669, 550.32256011, 544.95690479,
                    596.08115109, 621.47699374, 622.14946518, 687.25616132])
f_werte = np.array([95.8, 89.7, 84.4, 79.5, 75.4, 71.6, 68.3, 65.3, 62.4]) * 1e6
f_unsicher = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.3, 0.3, 0.3, 0.4]) * 1e6

# Definiere das Modell (Lineare Funktion für den ODR)
def linear_func(B, x):
    return B[0] * x + B[1]

# Daten für den ODR
data = RealData(f_werte, v, sx=f_unsicher, sy=delta_v)

# Modell für den ODR
model = Model(linear_func)

# ODR ausführen
odr = ODR(data, model, beta0=[1e-2, 1e3])  # Erste Schätzung für die Parameter
output = odr.run()

# Fit-Ergebnisse
beta = output.beta  # Fit-Parameter

# Fit-Kurve erstellen
f_fit = np.linspace(min(f_werte), max(f_werte), 500)
v_fit = linear_func(beta, f_fit)

# Plot mit Fehlerbalken
plt.errorbar(f_werte, v, xerr=f_unsicher, yerr=delta_v, fmt='o', label='Data with error bars', capsize=5)

# ODR-Fit plotten
plt.plot(f_fit, v_fit, 'r-', label=f'ODR Fit: v = {beta[0]:.4e} * f + {beta[1]:.4e}')

# Achsenbeschriftungen
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Velocity (m/s)', fontsize=12)
plt.title('Velocity vs Frequency with ODR Fit', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('velocities_vs_frequencies.png', dpi= 200)
plt.show()
