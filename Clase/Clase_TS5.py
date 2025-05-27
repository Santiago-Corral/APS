#Utilizar el estimador de frecuencia favorito con tu señal favorita :)

import scipy.signal.windows as windows

import scipy as sp
import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

def vertical_flaten(a):
    return a.reshape(a.shape[0],1)

#%%
##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz, ECG

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat') #El archivo tiene que estar en la misma carpeta del proyecto

ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
N = len(ecg_one_lead)

hb_1 = vertical_flaten(mat_struct['heartbeat_pattern1'])
hb_2 = vertical_flaten(mat_struct['heartbeat_pattern2'])

plt.figure()
plt.plot(ecg_one_lead[5000:12000])

# plt.figure()
# plt.plot(hb_1)

# plt.figure()
# plt.plot(hb_2)

#Estimo con welch

ecg_one_lead = ecg_one_lead/np.max(ecg_one_lead) #normalizo, divido por el maximo de la señal

df = fs_ecg/N
(fwelch,pxx) = sp.signal.welch(ecg_one_lead, fs_ecg ,nfft = N, window = 'flattop', nperseg = N/32, axis = 0)

plt.figure()
ff = np.linspace(0, (N-1)*df, N)
bfrec = ff <= fs_ecg/2
plt.plot(fwelch, 10*np.log10(2*np.abs(pxx)**2))
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
plt.title("Representación espectral")
plt.show

#%% Verifico el teo. de parseval
fftecg = np.fft.fft(ecg_one_lead)
fftECG = np.abs(fftecg)**2

energia_espectral = np.sum((df/2)*(fftECG[:-1] + fftECG[1:])) #Singifica desde el primer element al penultimo (-1) + desde el segundo elemento al ultimo 

energia_espectral2 = np.mean(fftECG) #Mas sencillamente con el valor medio (ver notas)

energia_temporal = np.sum(ecg_one_lead**2)

#%%

#Energia total
PAtotal = np.sum(pxx[0:12000]) #Porque esto nos da la potencia total???? Area total?

#Energia acumulada 
PAcumulada = np.cumsum(pxx[0:12000]) / PAtotal #podemos ver que aprox en la muestra 300 se acumula toda la energía (satura)

#%%
####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

# Cargar el archivo CSV como un array de NumPy
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe

plt.figure()
plt.plot(ppg)

#Estimo con welch

ppg = ppg/np.max(ppg) #normalizo, divido por el maximo de la señal

df = fs_ppg/N
(fwelch,wppg) = sp.signal.welch(ppg, fs_ppg ,nfft = N, window = 'flattop', nperseg = N/32, axis = 0)

plt.figure()
ff = np.linspace(0, (N-1)*df, N)
bfrec = ff <= fs_ecg/2
plt.plot(fwelch, 10*np.log10(2*np.abs(wppg)**2))
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
plt.title("Representación espectral")
plt.show

#%%

#Energia total
PAtotal = np.sum(wppg) #Porque esto nos da la potencia total???? Area total?

#Energia acumulada 
PAcumulada = np.cumsum(wppg) / PAtotal #podemos ver que aprox en la muestra 300 se acumula toda la energía (satura)

#%%
####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
# fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')

plt.figure()
plt.plot(wav_data)

# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
 #import sounddevice as sd
 #sd.play(wav_data, fs_audio) #No esta funcionando hay q instalar el pip
 


