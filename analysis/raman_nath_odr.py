import numpy as np
from scipy.odr import ODR, Model, RealData
import matplotlib.pyplot as plt
from raman_nath_diffraction import data_dict, height_dict

# Deine Daten
x_diff = np.array([-31.730103806228374, -15.7439446366782, 15.709342560553633, 31.453287197231834, 47.61245674740485])
diff = 1 / 2 * (height_dict.get('left_pixels') + height_dict.get('right_pixels')) / height_dict.get('pixel_density')

# Fehler
x_diff_errors = np.zeros(5) + (6 / np.sqrt(2)) / height_dict.get('pixel_density')
diff_errors = np.zeros(5) + (6 / np.sqrt(2)) / height_dict.get('pixel_density')

# Lineare Funktion mit und ohne Offset
def lin_func(p, x):
    a, b = p
    return a * x + b

def lin_no_offset_func(p, x):
    a = p[0]
    return a * x

# Set up für das Modell mit ODR
model = Model(lin_func)
data = RealData(x_diff, diff, sx=x_diff_errors, sy=diff_errors)
odr = ODR(data, model, beta0=[1.0, 0.0])  # Startwerte für a und b
output = odr.run()

# Ohne Offset Modell
model_no_offset = Model(lin_no_offset_func)
odr_no_offset = ODR(data, model_no_offset, beta0=[1.0])  # Startwert für a
output_no_offset = odr_no_offset.run()

# Ergebnisse
popt = output.beta
perr = output.sd_beta
popt_no = output_no_offset.beta
perr_no = output_no_offset.sd_beta

# Plotten der Ergebnisse
x_lin = np.linspace(1.2 * min(x_diff), 1.2 * max(x_diff), 100)

plt.figure(figsize=(12, 4))
plt.title('Diffraction order height offset vs. distance', fontsize=16)
plt.xlabel('Distance from 0th order in [mm]', fontsize=13)
plt.ylabel('Height offset in [mm]', fontsize=13)

# Plot für lineare Anpassung mit Offset
plt.plot(x_lin, lin_func(popt, x_lin), c='r', lw=1,
         label=f'Best fit with a, b:\n({popt[0]:.2f}±{perr[0]:.2f})x + ({popt[1]:.2f}±{perr[1]:.2f})')
plt.errorbar(x_diff, diff, x_diff_errors, diff_errors, ecolor='k', capsize=5, capthick=0.85, elinewidth=0.85,
             label='Data points with uncertainties', fmt='o')
plt.fill_between(x_lin, lin_func([popt[0] + perr[0], popt[1] + perr[1]], x_lin),
                 lin_func([popt[0] - perr[0], popt[1] - perr[1]], x_lin), color='r', alpha=0.2,
                 label=r'1-$\sigma$ confidence band')

# Plot für lineare Anpassung ohne Offset
plt.plot(x_lin, lin_no_offset_func(popt_no, x_lin), c='g', lw=1,
         label=f'Best fit with a:\n({popt_no[0]:.2f}±{perr_no[0]:.2f})x')
plt.fill_between(x_lin, lin_no_offset_func([popt_no[0] + perr_no[0]], x_lin),
                 lin_no_offset_func([popt_no[0] - perr_no[0]], x_lin), color='g', alpha=0.2,
                 label=r'1-$\sigma$ confidence band')

# Datenpunkte plotten
# plt.scatter(x_diff, diff, marker='x')
plt.tight_layout()
plt.xlim(-35, 50)
plt.legend()
plt.savefig('diffraction_order_vs_height_offset.png', dpi=200)
plt.show()
