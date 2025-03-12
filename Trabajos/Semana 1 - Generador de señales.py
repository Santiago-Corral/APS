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

#Distintas simulaciónes variando f0
f0 = 1 #tiempo (Hz)  
tt1, xx1 = mi_funcion_sen( vmax = 1, dc = 0, ff = f0, ph=0, nn = N, fs = fs)
f0 = 10 #tiempo (Hz)  
tt6, xx6 = mi_funcion_sen( vmax = 1, dc = 0, ff = f0, ph=0, nn = N, fs = fs)
f0 = 500 #tiempo (Hz)  
tt2, xx2 = mi_funcion_sen( vmax = 1, dc = 0, ff = f0, ph=0, nn = N, fs = fs)
f0 = 999 #tiempo (Hz)  
tt3, xx3 = mi_funcion_sen( vmax = 1, dc = 2, ff = f0, ph=0, nn = N, fs = fs)
f0 = 1001 #tiempo (Hz)  
tt4, xx4 = mi_funcion_sen( vmax = 1, dc = 4, ff = f0, ph=0, nn = N, fs = fs)
f0 = 2001 #tiempo (Hz)  
tt5, xx5 = mi_funcion_sen( vmax = 1, dc = 4, ff = f0, ph=0, nn = N, fs = fs)

#Ploteo las simulaciones
plt.figure()
plt.plot(tt1,xx1,label="ff = 1Hz, dc = 0")
plt.plot(tt2,xx2,label="ff = 500Hz, dc = 0")
plt.plot(tt3,xx3,label="ff = 999Hz, dc = 2")
plt.plot(tt4,xx4,label="ff = 1001Hz, dc = 4")
plt.plot(tt5,xx5,label="ff = 2001Hz, dc = 4")
plt.plot(tt6,xx6,label="ff = 10Hz, dc = 0")
plt.title("Señal: ")
plt.xlabel("tiempo [segundos]")
plt.ylabel("Amplitud")
plt.legend()
plt.show()
plt.grid 

#Distintas simulaciónes variando f0
f0 = 1 #tiempo (Hz)  
tt1, xx1 = mi_funcion_square( vmax = 1, dc = 0, ff = f0, ph=0, nn = N, fs = fs)
f0 = 999 #tiempo (Hz)  
tt3, xx3 = mi_funcion_square( vmax = 1, dc = 5, ff = f0, ph=0, nn = N, fs = fs)
f0 = 1001 #tiempo (Hz)  
tt4, xx4 = mi_funcion_square( vmax = 1, dc = -5, ff = f0, ph=0, nn = N, fs = fs)

#Ploteo las simulaciones
plt.figure()
plt.plot(tt1,xx1,label="ff = 1Hz, dc = 0") #Genera un primer valor de 0!!
plt.plot(tt3,xx3,label="ff = 999Hz, dc = 5")
plt.plot(tt4,xx4,label="ff = 1001Hz, dc = -5")
plt.title("Señal: ")
plt.xlabel("tiempo [segundos]")
plt.ylabel("Amplitud")
plt.legend()
plt.show()

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