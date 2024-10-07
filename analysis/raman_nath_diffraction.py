# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Data convention:
# pixel_density describes the number of pixels that make up 1mm in an image.
# Here we assume a standard error of 1 pixel.

# pixel_errors describes the standard error in pixels to determine the left / right position

# left_pixels describe the distance from the left side of the center point to the left side
# of the other orders, starting with the left-most order. The entry in the middle describes
# the distance from the left side of the 0th order to the right.

# right_pixels describes the same as left pixels but this time from the right side to the
# middle to the right side of the other orders' points.

# pixel_density_50 = 13
# pixel_errors_50 = 5
# left_pixels_50 = np.array([232, 118, 48, 122, 240])
# right_pixels_50 = np.array([242, 118, 47, 106, 233])

data_dict = {
    50: {
        'pixel_density': 14.7,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 0, 1, 2]),
        'left_pixels': np.array([232, 118, 48, 122, 240]),
        'right_pixels': np.array([242, 118, 47, 106, 233])
    },
    55: {
        'pixel_density': 15.4,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 0, 1, 2]),
        'left_pixels': np.array([268, 136, 48, 134, 275]),
        'right_pixels': np.array([283, 137, 49, 138, 265])
    },
    60: {
        'pixel_density': 14.5,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 0, 1, 2, 3, 4, 5]),
        'left_pixels': np.array([279, 144, 44, 142, 276, 417, 554, 688]),
        'right_pixels': np.array([281, 137, 46, 131, 275, 402, 539, 677])
    },
    65: {
        'pixel_density': 14.4,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 0, 1, 2, 3]),
        'left_pixels': np.array([300, 150, 42, 153, 294, 444]),
        'right_pixels': np.array([302, 152, 42, 143, 299, 439])
    },
    70: {
        'pixel_density': 15.2,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 0, 1, 2, 3, 4]),
        'left_pixels': np.array([333, 161, 51, 180, 340, 513, 678]),
        'right_pixels': np.array([349, 176, 49, 166, 334, 496, 662])
    },
    75: {
        'pixel_density': 15,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 0, 1, 2]),
        'left_pixels': np.array([357, 176, 50, 197, 365]),
        'right_pixels': np.array([366, 180, 50, 173, 355])
    },
    80: {
        'pixel_density': 14.7,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 0, 1, 2, 3, 4]),
        'left_pixels': np.array([377, 192, 48, 187, 375, 563, 745]),
        'right_pixels': np.array([389, 184, 49, 188, 370, 552, 734])
    },
    85: {
        'pixel_density': 14.2,
        'pixel_errors': 5,
        'orders': np.array([-1, 0, 1, 2, 3, 4]),
        'left_pixels': np.array([195, 45, 195, 382, 572, 753]),
        'right_pixels': np.array([188, 48, 180, 373, 559, 743])
    },
    90: {
        'pixel_density': 14,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 0, 1, 2, 3, 4]),
        'left_pixels': np.array([408, 209, 44, 203, 402, 604, 794]),
        'right_pixels': np.array([420, 204, 45, 194, 394, 596, 776])
    },
    91: {
        'pixel_density': 14.5,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 0, 1, 2, 3]),
        'left_pixels': np.array([421, 198, 48, 212, 417, 624]),
        'right_pixels': np.array([436, 233, 48, 200, 411, 614])
    },
    92: {
        'pixel_density': 14.6,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 0, 1, 2, 3]),
        'left_pixels': np.array([429, 200, 49, 218, 426, 639]),
        'right_pixels': np.array([444, 235, 49, 203, 420, 631])
    },
    93: {
        'pixel_density': 14.65,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 0, 1, 2, 3]),
        'left_pixels': np.array([438, 214, 47, 223, 436, 660]),
        'right_pixels': np.array([449, 231, 48, 208, 437, 653])
    },
    94: {
        'pixel_density': 14.65,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 0, 1, 2, 3]),
        'left_pixels': np.array([438, 214, 48, 224, 437, 674]),
        'right_pixels': np.array([451, 231, 48, 208, 437, 653])
    },
    95: {
        'pixel_density': 14.4,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 0, 1, 2, 3]),
        'left_pixels': np.array([436, 210, 47, 221, 433, 661]),
        'right_pixels': np.array([443, 229, 46, 209, 432, 653])
    },
    99: {
        'pixel_density': 14.45,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 0, 1, 2, 3]),
        'left_pixels': np.array([456, 222, 47, 236, 453, 685]),
        'right_pixels': np.array([461, 233, 47, 218, 456, 691])
    }
}


# Height information
height_dict = {
        'pixel_density': 14.45,
        'pixel_errors': 5,
        'orders': np.array([-2, -1, 1, 2, 3]),
        '0th_order_height': 45,
        'left_pixels': np.array([-13, 0, 16, 38, 50]),
        'right_pixels': np.array([-35, -25, -1, -2, 13])
    }

# x_diff = 1 / 2 * (data_dict.get(99).get('left_pixels') +
#                   data_dict.get(99).get('right_pixels')) / data_dict.get(99).get('pixel_density')

x_diff = np.array([-31.730103806228374, -15.7439446366782, 15.709342560553633, 31.453287197231834, 47.61245674740485])
diff = 1 / 2 * (height_dict.get('left_pixels') + height_dict.get('right_pixels')) / height_dict.get('pixel_density')
print(np.abs(diff))

x_diff_errors = np.zeros(5) + (6 / np.sqrt(2))
diff_errors = np.zeros(5) + (6 / np.sqrt(2))


def lin_func(x, a, b):
    return a * x + b


def lin_no_offset(x, a):
    return a * x


popt, pcov = curve_fit(lin_func, x_diff, diff)
perr = np.sqrt(np.diag(pcov))

popt_no, pcov_no = curve_fit(lin_no_offset, x_diff, diff)

print(popt, perr)

x_lin = np.linspace(1.1 * min(x_diff), 1.1 * max(x_diff), 10)

plt.figure(figsize=(12, 4))

print(popt_no, pcov_no)

plt.plot(x_lin, lin_func(x_lin, *popt), c='r', lw=1,
         label=f'Best fit with a, b:\n({popt[0]:.2f}±{perr[0]:.2f})x + ({popt[1]:.2f}±{perr[1]:.2f})')
plt.fill_between(x_lin, lin_func(x_lin, popt[0] + perr[0], popt[1] + perr[1]),
                 lin_func(x_lin, popt[0] - perr[0], popt[1] - perr[1]), ls='--', color='r', alpha=0.2,
                 label=r'1-$\sigma$ confidence band')

plt.plot(x_lin, lin_no_offset(x_lin, popt_no), c='g', lw=1,
         label=f'Best fit with a:\n({popt[0]:.2f}±{perr[0]:.2f})x')
plt.fill_between(x_lin, lin_no_offset(x_lin, popt_no + pcov_no[0]),
                 lin_no_offset(x_lin, popt_no - pcov_no[0]), ls='--', color='g', alpha=0.2,
                 label=r'1-$\sigma$ confidence band')

plt.scatter(x_diff, diff)
plt.tight_layout()
plt.legend()
# plt.savefig()
# plt.show()


def get_velocities(frequency: int):
    data = data_dict.get(frequency)

    order_positions = 1e-3 / 2 * (data.get('left_pixels') + data.get('right_pixels')) / data.get('pixel_density')
    actual_positions = np.sqrt(order_positions ** 2 + (0.0457039 * order_positions) ** 2)
    order_positions = actual_positions
    order_angles = 0.5 * np.arctan(order_positions / 1.2614)
    velocities = 532e-9 * frequency * 1e6 * np.abs(data.get('orders')) / (2 * np.sin(order_angles))

    return velocities


# Dictionary zum Speichern der Geschwindigkeiten und Standardabweichungen für jede Order (außer 0)
velocities_per_order = {order: [] for order in range(-2, 6) if order != 0}

# Schleife über alle Frequenzen im data_dict
for frequency in data_dict.keys():
    velocities = get_velocities(frequency)
    data = data_dict[frequency]

    # Zuordnen der Geschwindigkeiten basierend auf der Order (0. Order ausschließen)
    for i, order in enumerate(data.get('orders')):
        if order != 0 and order != 5:
            velocities_per_order[order].append(velocities[i])

# Berechnen der Standardabweichung pro Order (0. Order wurde ausgeschlossen)
std_per_order = {order: np.std(vels) for order, vels in velocities_per_order.items() if len(vels) > 1}

# Berechnen des gewichteten Mittelwerts der Geschwindigkeiten
weighted_numerator = 0
weighted_denominator = 0

for order, vels in velocities_per_order.items():
    if len(vels) > 1:  # Wir benötigen mindestens zwei Punkte, um eine Unsicherheit zu haben
        std = std_per_order[order]
        mean_velocity = np.mean(vels)
        weighted_numerator += mean_velocity / std ** 2
        weighted_denominator += 1 / std ** 2

# Gewichteter Mittelwert
if weighted_denominator != 0:
    weighted_mean_velocity = weighted_numerator / weighted_denominator
    weighted_mean_std = np.sqrt(1 / weighted_denominator) if weighted_denominator != 0 else np.nan
    print(f"Standardabweichung des gewichteten Mittelwerts: {weighted_mean_std:.3f} m/s")

    print(f"Gewichteter Mittelwert der Geschwindigkeiten (ohne 0. Ordnung): {weighted_mean_velocity:.3f} m/s")
else:
    print("Keine ausreichenden Daten für den gewichteten Mittelwert")

# Ausgabe der Standardabweichungen
print("\nStandardabweichungen pro Order (ohne 0. Ordnung):")
for order, std in std_per_order.items():
    print(f"Order {order}: {std:.3f} m/s")


# Berechnung der Geschwindigkeiten und Standardabweichungen für jede Order (ohne 0. Ordnung)
orders = []
mean_velocities = []
std_velocities = []

for order, vels in velocities_per_order.items():
    if order != 0:
        orders.append(order)
        mean_velocities.append(np.mean(vels))
        std_velocities.append(np.std(vels))

# Plotten der Geschwindigkeiten mit Unsicherheiten (Fehlerbalken)
plt.figure(figsize=(12, 5))
plt.errorbar(orders, mean_velocities, yerr=std_velocities, fmt='o', capsize=5, label='Measured velocities')

# Horizontale Linie für den finalen gewichteten Mittelwert der Geschwindigkeiten
plt.axhline(y=weighted_mean_velocity, color='r', linestyle='--', label=f'Weighted mean velocity ({weighted_mean_velocity:.2f} m/s)')
plt.fill_between([-3, -2, -1, 0, 1, 2, 3, 4, 5], weighted_mean_velocity - weighted_mean_std, weighted_mean_velocity + weighted_mean_std, color='r', alpha=0.2, label=r'1-$\sigma$ confidence band')

# Horizontale Linie für den Literaturwert (4260 m/s)
plt.axhline(y=4260, color='g', linestyle='-', label='Literature value (4260 m/s)')

# Achsenbeschriftungen
plt.xlabel('Order')
plt.ylabel('Velocity (m/s)')
plt.title('Velocities by Order with Uncertainties')

# Legende hinzufügen
plt.legend()
plt.xlim(-2.5, 4.5)

# Plot anzeigen
plt.tight_layout()
plt.savefig('weighted_mean_raman_nath.png', dpi=200)
plt.show()

# all_velocities = []
#
# plt.figure(figsize=(12, 5))
# for key in data_dict.keys():
#     velocities = get_velocities(key)
#     plt.scatter(data_dict.get(key).get('orders'), velocities, label=f'{key} MHz')
#     avg_velocity = sum(velocities) / (len(velocities) - 1)
#     all_velocities.append(avg_velocity)
#     print(key, ':\n', velocities, avg_velocity)
#
# print(np.mean(all_velocities), np.std(all_velocities))
#
# plt.show()

