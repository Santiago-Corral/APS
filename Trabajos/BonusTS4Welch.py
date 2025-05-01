#%% módulos y funciones a importar
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal.windows as windows
import pandas as pd

#%% Datos de la simulación
R = 200 #Realizaciones
N = 1000 # cantidad de muestras
fs =  1000 # frecuencia de muestreo (Hz)
SNR = 3 #dB, piso de ruido

#%%
ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral

tt = np.linspace (0, (N-1)*ts, N).reshape((N,1)) #vector de tiempo
ff = np.linspace(0, (N-1)*df, N) #Grilla de sampleo frecuencial

#generacion de la señal senoidal
tt = np.tile(tt, (1,R))

fr = np.random.uniform(-1/2,1/2, size = (1,R))
fr = np.reshape(fr, (1,R))

omega_0 = fs/4 #Mitad de banda digital
omega_1 = omega_0 +  fr * df #como luego lo usamos para el sesgo, multiplicamos por 2pi dentro del seno
a1 = np.sqrt(2)
xx = a1 * np.sin(2*np.pi*omega_1*tt) 
#El A1 lo usamos para normalizar

#%% Genero Ruido

pot_ruido = 10**(-SNR/10)
nn = np.random.normal(0,np.sqrt(pot_ruido),(N,R)) #señal de ruido analogico, le doy el tamaño de xx

#%%
ss = xx+nn #Señal mas ruido

#%% Implemento Welch 

(ff,a2_hann) = sp.signal.welch(ss,fs = fs,window="hann",nperseg = N/8, axis = 0) #Implementa el método de Welch, con sus promedios
(ff,a2_flattop) = sp.signal.welch(ss,fs = fs,window="flattop",nperseg = N/8, axis = 0) #Implementa el método de Welch, con sus promedios

## El tener mas promedios (aumentar N en welch, largo de segmentos) disminuimos aun mas la varianza
## Pero sacrificamos resolución espectral, ya que nos quedamos con menos muestras

#%% Estimador
a2_hann_max = np.max(a2_hann , axis = 0)
a2_flattop_max = np.max(a2_flattop , axis = 0)

est_omega_hann = ff[np.argmax(a2_hann, axis=0)]
est_omega_flattop = ff[np.argmax(a2_flattop, axis=0)]

plt.figure()
bins = 20

plt.hist(a2_hann_max.flatten(), bins=bins, label = "Estimador con metodo de Welch ventana Hann ")
plt.hist(a2_flattop_max.flatten(), bins=bins, label = "Estimador con metodo de Welch ventana Flattop")
plt.legend()

plt.xlabel("Amplitud Estimada")
plt.ylabel("Cantidad de ocurrencias")
plt.title(f"Histograma de estimadores de amplitudes con SNR = {SNR} dB" )

#%% Histograma de frecuencias
plt.figure()
bins = 30
plt.hist(est_omega_hann.flatten(), bins=bins, label = "Estimador con ventana Hann", alpha=0.5)
plt.hist(est_omega_flattop.flatten(), bins=bins, label = "Estimador con ventana Flattop", alpha=0.5)
plt.legend()
plt.grid("on")

plt.xlabel("Frecuencia estimada [Hz]")
plt.ylabel("Cantidad de ocurrencias")
plt.title(f"Histograma de estimadores de frecuencias con SNR = {SNR} dB" )

#%% Estimador omega
esperanza_o_hann = np.mean(est_omega_hann)
esperanza_o_flattop = np.mean(est_omega_flattop)

sesgo_o_hann = esperanza_o_hann - omega_0
sesgo_o_flattop = esperanza_o_flattop - omega_0

varianza_o_hann = np.var(est_omega_hann)
varianza_o_flattop = np.var(est_omega_flattop)

#%% Analizo varianza y sesgo a1

esperanza_welch_hann = np.mean(a2_hann_max)
esperanza_welch_flattop = np.mean(a2_flattop_max)

sesgo_a2_max_hann = esperanza_welch_hann - a1
sesgo_a2_max_flattop = esperanza_welch_flattop - a1

varianza_welch_hann = np.var(a2_hann_max)
varianza_welch_flattop = np.var(a2_flattop_max) 

#%% Crear la tabla con tus valores (reemplazá con tus propios datos si es necesario)
tabla_resultados = pd.DataFrame({
    "Ventana": ["Hann", "Flattop"],
    "Sesgo a1": [round(sesgo_a2_max_hann, 3), round(sesgo_a2_max_flattop, 3)],
    "Varianza a1": [varianza_welch_hann, varianza_welch_flattop],
    "Sesgo Ω1": [round(sesgo_o_hann, 3), round(sesgo_o_flattop, 3)],
    "Varianza Ω1": [varianza_o_hann, varianza_o_flattop]
})

# Redondear para mejorar la estética
tabla_resultados = tabla_resultados.round(6)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')  # Ocultar los ejes

# Dibujar la tabla
tabla = ax.table(cellText=tabla_resultados.values,
                 colLabels=tabla_resultados.columns,
                 cellLoc='center',
                 loc='center')

tabla.scale(1.2, 1.5)  # Escalar tabla para mejor legibilidad

# Título de la tabla
plt.title(f"Sesgo y Varianza de Estimadores con SNR {SNR} dB", fontsize=14, pad=20)

plt.tight_layout()
plt.show()