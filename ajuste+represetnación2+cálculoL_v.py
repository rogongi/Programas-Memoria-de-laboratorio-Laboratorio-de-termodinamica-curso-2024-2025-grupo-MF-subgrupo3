import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Datos experimentales
inverse_T = np.array([3425.22, 3322.81, 2919.30, 2887.30, 2838.10, 
                      2811.80, 2787.50, 2771.20, 2758.20, 2749.10, 
                      2742.40, 2720.70]) * 1e-6

ln_P = np.array([3.25, 3.32, 5.492, 5.644, 5.899, 
                 6.051, 6.146, 6.228, 6.282, 6.318, 
                 6.349, 6.456])

error_inverse_T = np.array([1.2, 1.1, 0.9, 0.8, 0.8, 0.8, 
                            0.8, 0.8, 0.8, 0.8, 0.8, 0.7]) * 1e-6
error_ln_P = np.array([0.04, 0.04, 0.004, 0.004, 0.003, 
                       0.002, 0.002, 0.002, 0.002, 0.002, 
                       0.002, 0.002])

T1 = 1/inverse_T[0]  # en K
P1 = np.exp(ln_P[0]) # en mmHg

# Ajuste lineal
def linear_func(x, a, b):
    return a * x + b

popt, pcov = curve_fit(linear_func, inverse_T, ln_P, sigma=error_ln_P, absolute_sigma=True)
a, b = popt
a_err, b_err = np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1])

# Cálculo de Lv
R = 8.314  # J/(mol·K)
M_water = 18.015  # g/mol

# De la pendiente
Lv_slope = -a * R 
Lv_slope_err = a_err * R
Lv_slope_cal_g = Lv_slope/4.184/M_water
Lv_slope_cal_g_err = Lv_slope_err/4.184/M_water

# Del intercepto
Lv_intercept = (b - ln_P[0]) * R * T1
Lv_intercept_err = np.sqrt((b_err * R * T1)**2 + (error_ln_P[0] * R * T1)**2)
Lv_intercept_cal_g = Lv_intercept/4.184/M_water
Lv_intercept_cal_g_err = Lv_intercept_err/4.184/M_water

# Valores ponderados
weights = np.array([1/Lv_slope_err**2, 1/Lv_intercept_err**2])
Lv = np.average([Lv_slope, Lv_intercept], weights=weights)
Lv_err = 1/np.sqrt(np.sum(weights))
Lv_cal_g = np.average([Lv_slope_cal_g, Lv_intercept_cal_g], weights=weights)
Lv_cal_g_err = 1/np.sqrt(np.sum(weights))

# Resultados
print(f"Resultados del ajuste lineal:")
print(f"Pendiente (a) = {a:.2e} ± {a_err:.2e} K")
print(f"Intercepto (b) = {b:.2f} ± {b_err:.2f}")
print(f"\nCalor latente de vaporización:")
print(f"De la pendiente: Lv = {Lv_slope/1000:.2f} ± {Lv_slope_err/1000:.2f} kJ/mol = {Lv_slope_cal_g:.2f} ± {Lv_slope_cal_g_err:.2f} cal/g")
print(f"Del intercepto: Lv = {Lv_intercept/1000:.2f} ± {Lv_intercept_err/1000:.2f} kJ/mol = {Lv_intercept_cal_g:.2f} ± {Lv_intercept_cal_g_err:.2f} cal/g")
print(f"\nValor medio ponderado: Lv = {Lv/1000:.2f} ± {Lv_err/1000:.2f} kJ/mol = {Lv_cal_g:.2f} ± {Lv_cal_g_err:.2f} cal/g")

# Gráfica 
plt.figure(figsize=(10, 6))

plt.plot(inverse_T*1e6, ln_P, 'o', color='blue', 
         markersize=8, markeredgecolor='black', 
         markeredgewidth=0.5, label='Datos experimentales')

plt.plot(inverse_T*1e6, linear_func(inverse_T, *popt), 
         'r-', linewidth=2, label='Ajuste lineal')

plt.xlabel('1/T₂ (×10$^{-6}$ K$^{-1}$)', fontsize=12)
plt.ylabel('ln(P₂/mmHg)', fontsize=12)
plt.title('ln(P₂/mmHg) frente a 1/T₂', fontsize=14)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

x_min, x_max = inverse_T.min()*1e6, inverse_T.max()*1e6
y_min, y_max = ln_P.min(), ln_P.max()
plt.xlim(x_min - 50, x_max + 50)
plt.ylim(y_min - 0.5, y_max + 0.5)

plt.tight_layout()
plt.show()