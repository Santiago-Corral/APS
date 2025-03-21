import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.close('all')

# Parámetros del circuito 
R = 1  # Resistencia en ohmios
L = 1  # Inductancia en Henrios
C = 1  # Capacitancia en Faradios

def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
    N = nn
    ts = 1/fs # tiempo de muestreo
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (N-1)*ts, N) #Grilla total de tiempo 
    
    xx = vmax*np.sin(2*np.pi*ff*tt+ph) + dc #Sen (W0 . t + fase) // W0 = 2pi . t 
    
    return tt, xx

#%% Funcion 1

# Definir la función de transferencia H(s) = (R sC) / (1 + R sC + s^2 LC)
num = [R, 0]  # Numerador: R * s
den = [L, R, 1/C]  # Denominador: L * s^2 + R * s + 1/C
w0 = 1/np.sqrt(L*C) 

w0 = 1/np.sqrt(L*C)
Q = (np.sqrt(L*C)*L)/R
BW=w0/Q
# Crear el sistema en función de s
system = sig.TransferFunction(num, den)

# Definir el rango de frecuencias para el análisis
w = np.logspace(-1, 5, 1000) # Frecuencia de 10^-1 a 10^5 rad/s #!! De cambiar los valores de RLC hay que prestar atencion al rango de frecuencias
# Obtener la respuesta en frecuencia
w, mag, phase = sig.bode(system, w)

# Graficar la respuesta en magnitud
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.semilogx(w, mag, label="Módulo |H(jω)| (dB)")
#w0
## Señalar el valor específico w0 en el eje x con una marca
##plt.scatter([w0], [np.interp(w0, w, mag)], color='r', zorder=5)  # Marca el punto en (x=w0, y=mag(w0))

# Añadir una pequeña línea vertical en w0 (acortar la longitud de la línea)
plt.vlines(w0, ymin=np.interp(w0, w, mag) - 5, ymax=np.interp(w0, w, mag) + 5, color='b', linestyle='-', label='w0')

#Ancho de banda
plt.vlines([w0 - BW/2, w0 + BW/2], ymin=np.min(mag)-10, ymax=np.max(mag)+10, color='g', linestyle='--', label="Ancho de Banda (BW)")

plt.axhline(-20, color='r', linestyle='--', label="Asíntota baja frecuencia (-20 dB/dec)")
plt.axhline(0, color='g', linestyle='--', label="Asíntota alta frecuencia (0 dB)")
plt.ylabel("Magnitud (dB)")
plt.legend()
plt.grid(True, which="both", linestyle="--")

# Graficar la respuesta en fase
plt.subplot(2, 1, 2)
plt.semilogx(w, phase, label="Fase ∠H(jω) (°)")
plt.axhline(90, color='r', linestyle='--', label="Asíntota alta frecuencia (90°)")
plt.axhline(0, color='g', linestyle='--', label="Asíntota baja frecuencia (0°)")
plt.ylabel("Fase (°)")
plt.xlabel("Frecuencia (rad/s)")
plt.legend()
plt.grid(True, which="both", linestyle="--")

plt.show()



#%% Funcion 2

# Definir la función de transferencia H(s) = (R sC) / (1 + R sC + s^2 LC)
num = [1, 0, 0]  # Numerador: s**2
den = [1, 1/(R*C), 1/(L*C)]  # Denominador: s^2 + s / R*C + 1/L*C

w0 = 1/np.sqrt(L*C)
Q = (np.sqrt(L*C))/R*C
BW=w0/Q

# Crear el sistema en función de s
system = sig.TransferFunction(num, den)

# Definir el rango de frecuencias para el análisis
w = np.logspace(-1, 5, 1000)  # Frecuencia de 10^-1 a 10^5 rad/s

# Obtener la respuesta en frecuencia
w, mag, phase = sig.bode(system, w)

# Graficar la respuesta en magnitud
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.semilogx(w, mag, label="Módulo |H(jω)| (dB)")
#w0
## Señalar el valor específico w0 en el eje x con una marca
##plt.scatter([w0], [np.interp(w0, w, mag)], color='r', zorder=5)  # Marca el punto en (x=w0, y=mag(w0))

# Añadir una pequeña línea vertical en w0 (acortar la longitud de la línea)
plt.vlines(w0, ymin=np.interp(w0, w, mag) - 5, ymax=np.interp(w0, w, mag) + 5, color='b', linestyle='-', label='w0')

#Ancho de banda
plt.vlines([w0 - BW/2, w0 + BW/2], ymin=np.min(mag)-10, ymax=np.max(mag)+10, color='g', linestyle='--', label="Ancho de Banda (BW)")

plt.axhline(-20, color='r', linestyle='--', label="Asíntota baja frecuencia (-20 dB/dec)")
plt.axhline(0, color='g', linestyle='--', label="Asíntota alta frecuencia (0 dB)")
plt.ylabel("Magnitud (dB)")
plt.legend()
plt.grid(True, which="both", linestyle="--")

# Graficar la respuesta en fase
plt.subplot(2, 1, 2)
plt.semilogx(w, phase, label="Fase ∠H(jω) (°)")
plt.axhline(90, color='r', linestyle='--', label="Asíntota alta frecuencia (90°)")
plt.axhline(0, color='g', linestyle='--', label="Asíntota baja frecuencia (0°)")
plt.ylabel("Fase (°)")
plt.xlabel("Frecuencia (rad/s)")
plt.legend()
plt.grid(True, which="both", linestyle="--")

plt.show()

# # %%Simulacion sinusoidal

# tt, xx = mi_funcion_sen(1, 0, 100, 0, 1000, 1000)

# # Simular la salida del sistema a la señal sinusoidal
# tt2, xx2, _ = sig.lsim(system, xx, tt) #lsim simula una salida y(t) para una entrada x(t), (funcion transferencia, señal de entrada, tiempo)

# # Graficar la entrada y la salida
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(tt, xx, label="Entrada: Sinusoidal")
# plt.title("Entrada y salida del sistema")
# plt.ylabel("Amplitud")
# plt.legend()
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.plot(tt2, xx2, label="Salida: Respuesta del sistema")
# plt.ylabel("Amplitud")
# plt.xlabel("Tiempo (s)")
# plt.legend()
# plt.grid(True)

# plt.show()
