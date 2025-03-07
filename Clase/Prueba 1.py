#matplotlib qt, abre una ventana con los graficos


import numpy as np
import matplotlib.pyplot as plt

# Datos generales de la simulación
fs = 1000 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras
f0 = 2001 #tiempo (Hz)  

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral

# grilla de sampleo temporal
tt = np.linspace(0, (N-1)*ts, N) #Grilla total de tiempo 
xx = np.sin(2*np.pi*f0*tt)

plt.plot(tt,xx)