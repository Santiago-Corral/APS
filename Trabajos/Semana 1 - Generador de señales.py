#matplotlib qt, abre una ventana con los graficos

import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
    ts = 1/fs # tiempo de muestreo
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (N-1)*ts, N) #Grilla total de tiempo 
    
    xx = vmax*np.sin(2*np.pi*ff*tt+ph) + dc #Sen (W0 . t + fase) // W0 = 2pi . t 
    
    return tt, xx

def mi_funcion_square (vmax, dc, ff, ph, nn, fs):
    ts = 1/fs # tiempo de muestreo
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (N-1)*ts, N) #Grilla total de tiempo 
    
    xx = vmax * np.sign(np.sin(2 * np.pi * ff * tt + ph)) + dc #Existen funciones que generan squares, pero asi es mas facil
    
    return tt, xx

# Datos generales de la simulación
fs = 1000 # frecuencia de muestreo (Hz)
N = 1000  # cantidad de muestras
f0 = 2001 #tiempo (Hz)  

plt.figure()

tt, xx = mi_funcion_sen( vmax = 2, dc =0, ff = f0, ph=0, nn = N, fs = fs)
tt, xx2 = mi_funcion_square( vmax = 2, dc =0, ff = f0, ph=0, nn = N, fs = fs)

plt.plot(tt,xx)
plt.plot(tt,xx2)
plt.grid 

"""def generador_senal(tipo="senoidal", vmax=1, dc=0, ff=10, ph=0, nn=1000, fs=1000):
    ts = 1 / fs  # Tiempo de muestreo
    tt = np.linspace(0, (nn - 1) * ts, nn)  # Vector de tiempo

    # Generación de señales según el tipo
    if tipo == "senoidal":
        xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    elif tipo == "cuadrada":
        xx = vmax * np.sign(np.sin(2 * np.pi * ff * tt + ph)) + dc
    elif tipo == "triangular":
        xx = vmax * (2 * np.abs(2 * ((tt * ff) % 1) - 1) - 1) + dc
    elif tipo == "diente_de_sierra":
        xx = vmax * (2 * ((tt * ff) % 1) - 1) + dc
    else:
        raise ValueError("Tipo de señal no reconocido")

    return tt, xx
""" #Generador de varias señales distintas