#%% módulos y funciones a importar
import numpy as np
import matplotlib.pyplot as plt

#%% Datos de la simulación

R = 10 #Realizaciones
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

sigma_0 = fs/4 #Mitad de banda digital
sigma_1 = sigma_0 +  fr * 2 * (np.pi/N)

xx = np.sin(sigma_1*tt) 

#%% Genero Ruido

pot_ruido = 10**(-SNR/10) #Como se define esto?

# ff = np.arange(0, 1, 1/1000).reshape((1000,1))
# ff = np.repeat(tt, 200, axis = 1)
nn = np.random.normal(0,np.sqrt(pot_ruido),(N,R)) #señal de ruido analogico, le doy el tamaño de xx

#%%
ss = xx+nn #Señal mas ruido y plots

plt.plot(tt, ss) 