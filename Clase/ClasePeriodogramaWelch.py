#%% El Welch esta bien aplicado, hay que ver el blackman tukey


#%% módulos y funciones a importar
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal.windows as windows
import pandas as pd

def blackman_tukey(x, fs=1.0, window='hamming', nperseg=None, axis=-1):
    """
    Estimador de densidad espectral de potencia tipo Blackman-Tukey,
    compatible con parámetros similares a scipy.signal.welch.
    
    Parámetros:
        x: array 1D o 2D (cada columna puede ser una realización)
        fs: frecuencia de muestreo
        window: tipo de ventana ('hamming', 'hann', 'bartlett', 'blackman', o None)
        nperseg: cantidad de retardos usados (M)
        axis: eje de tiempo (por defecto -1)
    
    Retorna:
        f: frecuencias en Hz
        Pxx: espectro estimado
    """
    x = np.moveaxis(np.asarray(x), axis, 0)
    N, R = x.shape if x.ndim == 2 else (x.shape[0], 1)
    x = x if R > 1 else x.reshape(N, 1)

    if nperseg is None:
        nperseg = N // 4
    M = int(nperseg)

    # Selección de ventana
    if window == 'hamming':
        w = windows.hamming(M)
    elif window == 'hann':
        w = windows.hann(M)
    elif window == 'bartlett':
        w = windows.bartlett(M)
    elif window == 'blackman':
        w = windows.blackman(M)
    else:
        w = np.ones(M)

    Pxx = np.zeros((1024, R))

    for i in range(R):
        xi = x[:, i]
        Rxx = np.correlate(xi, xi, mode='full') / len(xi)
        mid = len(Rxx) // 2
        r = Rxx[mid:mid + M]
        r_win = np.concatenate([r[:0:-1], r]) * np.concatenate([w[:0:-1], w])
        Px = np.abs(np.fft.fft(r_win, n=1024))
        Pxx[:, i] = Px

    f = np.linspace(0, fs, 1024)
    return f, Pxx

#%% Datos de la simulación

R = 200 #Realizaciones
N = 1000 # cantidad de muestras
fs =  1000 # frecuencia de muestreo (Hz)
SNR = 10 #dB, piso de ruido

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

pot_ruido = 10**(-SNR/10) #Como se define esto?

# ff = np.arange(0, 1, 1/1000).reshape((1000,1))
# ff = np.repeat(tt, 200, axis = 1)
nn = np.random.normal(0,np.sqrt(pot_ruido),(N,R)) #señal de ruido analogico, le doy el tamaño de xx

#%%
ss = xx+nn #Señal mas ruido

#%% Implemento Welch 

(ff,a2) = sp.signal.welch(ss,fs = fs,window="flattop",nperseg = N/128, axis = 0) #Implementa el metodo de Welch, con sus promedios

## El tener mas promedios (aumentar N en welch, largo de segmentos) disminuimos aun mas la varianza
## Pero sacrificamos resolución espectral, ya que nos quedamos con menos muestras?

bfrec = ff <= fs/2
plt.plot(ff[bfrec], 10* np.log10(2*np.abs(a2[bfrec])**2)) 
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
plt.title("Representación espectral")
axes_hdl = plt.gca()

#%% Estimador
a2_max = np.max(a2 , axis = 0)

plt.figure()
bins = 20
plt.hist(a2_max.flatten(), bins=bins, label = "Estimador con metodo de Welch")

plt.xlabel("Amplitud Estimada")
plt.ylabel("Cantidad de ocurrencias")
plt.title(f"Histograma de estimadores de amplitudes con SNR = {SNR} dB" )

#%% Analizo varianza y sesgo

esperanza_welch = np.mean(a2_max)
sesgo_a2_max = esperanza_welch - a1
varianza_welch = np.var(a2_max) 

#%% Blackman Tukey
f_bt, a_bt = blackman_tukey(ss, fs=fs, window='hamming', nperseg=N//4, axis=0)

bfrec = f_bt <= fs/2
plt.plot(f_bt[bfrec], 10* np.log10(2*np.abs(a_bt[bfrec])**2)) 
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
plt.title("Representación espectral, Blackman Tukey")
axes_hdl = plt.gca()

#%% Estimador
a_bt = np.max(a_bt , axis = 0)

plt.figure()
bins = 20
plt.hist(a_bt.flatten(), bins=bins, label = "Estimador con metodo de Welch")

plt.xlabel("Amplitud Estimada")
plt.ylabel("Cantidad de ocurrencias")
plt.title(f"Histograma de estimadores de amplitudes con SNR = {SNR} dB, Blackman Tukey" )

#%% Analizo varianza y sesgo

esperanza_tukey = np.mean(a_bt)
sesgo_a_bt = esperanza_welch - a1
varianza_tukey = np.var(a_bt) 