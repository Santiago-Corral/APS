#%% módulos y funciones a importar
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as windows

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

sigma_0 = fs/4 #Mitad de banda digital
sigma_1 = (sigma_0 +  fr) * 2 * (np.pi)

A1 = np.sqrt(2)
xx = A1 * np.sin(sigma_1*tt) 
#El A1 lo usamos para normalizar

#%% Genero Ruido

pot_ruido = 10**(-SNR/10) #Como se define esto?

# ff = np.arange(0, 1, 1/1000).reshape((1000,1))
# ff = np.repeat(tt, 200, axis = 1)
nn = np.random.normal(0,np.sqrt(pot_ruido),(N,R)) #señal de ruido analogico, le doy el tamaño de xx

#%%
ss = xx+nn #Señal mas ruido

#%%
blackmanharris = windows.blackmanharris(N).reshape(N,1) # Reshape para multiplicar por la matriz ss
flattop = windows.flattop(N).reshape(N,1)
lanczos = windows.lanczos(N).reshape(N,1)

#Multiplico por la ventana
ssblackman = ss * blackmanharris
ssflattop = ss * flattop
sslanczos = ss * lanczos

#%% FFT, respuesta en frecuencia
ft_sinc = 1/N*np.fft.fft(ss, axis = 0) #es como multiplicar por uno el sinc
ft_blackman = 1/N*np.fft.fft(ssblackman, axis = 0)
ft_flattop = 1/N*np.fft.fft(ssflattop, axis = 0)
ft_lanczos = 1/N*np.fft.fft(sslanczos, axis = 0)

bfrec = ff <= fs/2

#plt.plot(ff[bfrec], 10* np.log10(2*np.abs(ft_blackman[bfrec])**2)) 
#plt.plot(ff[bfrec], 10* np.log10(2*np.abs(ft_flattop[bfrec])**2)) 
#plt.plot(ff[bfrec], 10* np.log10(2*np.abs(ft_lanczos[bfrec])**2))
plt.plot(ff[bfrec], 10* np.log10(2*np.abs(ft_sinc[bfrec])**2)) 
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()

## Estimadores de a para cada ventana
# Tomo una feta en N/4 (250) donde tengo el pulso para ver el histograma
est_a_sinc = np.abs(ft_sinc [N//4,:])
est_a_blackman = np.abs(ft_blackman [N//4,:])
est_a_flattop = np.abs(ft_flattop [N//4,:])
est_a_lanczos = np.abs(ft_lanczos [N//4,:])

#Histograma
plt.figure()
bins = 10
plt.hist(est_a_sinc.flatten(), bins=bins, label = "Sinc")
plt.hist(est_a_blackman.flatten(), bins=bins, label = "Blackman")
plt.hist(est_a_flattop.flatten(), bins=bins, label = "flattop")
plt.hist(est_a_lanczos.flatten(), bins=bins, label = "lanczos")
plt.legend()

plt.xlabel("Estimador")
plt.ylabel("Cantidad de ocurrencias")

#Cada ventana concentra distinto la energía 


