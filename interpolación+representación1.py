import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Función de interpolación 
def interpolate(data, data_error, ref):
    for i in range(len(ref) - 1):
        t1, p1 = ref[i]
        t2, p2 = ref[i + 1]
        if t1 <= data <= t2:
            P = p1 + (p2 - p1) * (data - t1) / (t2 - t1)
            P_error = abs((p2 - p1) / (t2 - t1)) * data_error
            return P, P_error

# Datos de referencia 
referencia_TP = [
    (0.000, 4.5851), (1.000, 4.9291), (2.000, 5.2958), (3.000, 5.6864),
    (4.000, 6.1024), (5.000, 6.5450), (6.000, 7.0159), (7.000, 7.5164),
    (8.000, 8.0482), (9.000, 8.6122), (10.000, 9.2115), (11.000, 9.8476),
    (12.000, 10.521), (13.000, 11.235), (14.000, 11.992), (15.000, 12.793),
    (16.000, 13.640), (17.000, 14.536), (18.000, 15.484), (19.000, 16.485),
    (20.000, 17.542), (21.000, 18.659), (22.000, 19.837), (23.000, 21.080),
    (24.000, 22.389), (25.000, 23.769), (26.000, 25.224), (27.000, 26.755),
    (28.000, 28.366), (29.000, 30.061), (30.000, 31.844), (31.000, 33.718),
    (32.000, 35.686), (33.000, 37.754), (34.000, 39.925), (35.000, 42.204),
    (36.000, 44.593), (37.000, 47.100), (38.000, 49.728), (39.000, 52.481),
    (40.000, 55.365), (41.000, 58.385), (42.000, 61.546), (43.000, 64.853),
    (44.000, 68.312), (45.000, 71.929), (46.000, 75.711), (47.000, 79.657),
    (48.000, 83.789), (49.000, 88.095), (50.000, 92.588), (51.000, 97.283),
    (52.000, 102.18), (53.000, 107.28), (54.000, 112.60), (55.000, 118.15),
    (56.000, 123.93), (57.000, 129.94), (58.000, 136.20), (59.000, 142.72),
    (60.000, 149.50), (61.000, 156.56), (62.000, 163.90), (63.000, 171.52),
    (64.000, 179.45), (65.000, 187.68), (66.000, 196.24), (67.000, 205.12),
    (68.000, 214.34), (69.000, 223.91), (70.000, 233.84), (71.000, 244.14),
    (72.000, 254.81), (73.000, 265.88), (74.000, 277.36), (75.000, 289.25),
    (76.000, 301.56), (77.000, 314.31), (78.000, 327.51), (79.000, 341.18),
    (80.000, 355.33), (81.000, 369.96), (82.000, 385.10), (83.000, 400.74),
    (84.000, 416.92), (85.000, 433.65), (86.000, 450.93), (87.000, 468.78),
    (88.000, 487.23), (89.000, 506.26), (90.000, 525.92), (91.000, 546.22),
    (92.000, 567.25), (93.000, 588.75), (94.000, 611.04), (95.000, 634.02),
    (96.000, 657.71), (97.000, 682.14), (98.000, 707.32), (99.000, 733.25)
]

# Datos experimentales
T_exp = np.array([18.8, 27.8, 69.4, 73.2, 79.2, 82.5, 85.6, 87.7, 89.4, 90.6, 91.5, 94.4])
P_exp = np.array([26, 28, 243, 283, 365, 425, 467, 507, 535, 555, 572, 637])
datos_error = 0.1

# Interpolación 
P_teo_interp = []
P_teo_error = []
for temp in T_exp:
    p, p_err = interpolate(temp, datos_error, referencia_TP)
    P_teo_interp.append(p)
    P_teo_error.append(p_err)
P_teo_interp = np.array(P_teo_interp)

# Cálculo de línea de tendecia
T_teo_full = np.array([x[0] for x in referencia_TP])
P_teo_full = np.array([x[1] for x in referencia_TP])


def exponential_model(T, a, b, c):
    return a * np.exp(b * T) + c


popt_exp, _ = curve_fit(exponential_model, T_exp, P_exp, p0=[1, 0.05, 0])
a_exp, b_exp, c_exp = popt_exp


popt_teo, _ = curve_fit(exponential_model, T_teo_full, P_teo_full, p0=[1, 0.05, 0])
a_teo, b_teo, c_teo = popt_teo


T_smooth = np.linspace(min(T_exp.min(), T_teo_full.min()), 
                       max(T_exp.max(), T_teo_full.max()), 300)
P_exp_fit = exponential_model(T_smooth, a_exp, b_exp, c_exp)
P_teo_fit = exponential_model(T_smooth, a_teo, b_teo, c_teo)

# Gráfica 
plt.figure(figsize=(12, 7))


plt.scatter(T_exp, P_exp, color='red', s=80, label='Presión experimental', zorder=5)

plt.plot(T_smooth, P_exp_fit, color='red', linestyle='-', linewidth=2, 
         label='Línea de tendencia', alpha=0.8)

plt.scatter(T_exp, P_teo_interp, color='blue', s=80, 
           label='Presión interpolada', zorder=5)

plt.plot(T_smooth, P_teo_fit, color='blue', linestyle='-', linewidth=2,
         label='Línea de tendencia', alpha=0.8)

plt.title('Comparación de presiones experimentales y teóricas', fontsize=14)
plt.xlabel('T (°C)', fontsize=12)
plt.ylabel('P (mmHg)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()

# Mostrar tabla de datos interpolados 
print("\nTabla de datos interpolados:")
print("T (°C)\t\tP teórica (mmHg)\t Error(P) (mmHg)")
print("------------------------------------------------")
for t, p, e in zip(T_exp, P_teo_interp, P_teo_error):
    print(f"{t:.1f}\t\t{p:.2f}\t\t\t{e:.4f}")

plt.show()