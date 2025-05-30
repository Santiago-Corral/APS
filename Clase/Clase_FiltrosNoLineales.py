import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from pytc2.sistemas_lineales import plot_plantilla

aprox_name = 'butter'

fs = 1000 #Hz
nyq_frec = fs/2
fpass = np.array ( [1.0, 35.0 ] )
ripple = 1.0 #dB
fstop = np.array( [.1 , 50.] )
attenuation = 40 #dB

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

#%% Señal ECG 

mat_struct = sio.loadmat('./ECG_TP4.mat') #El archivo tiene que estar en la misma carpeta del proyecto

ecg_one_lead = mat_struct['ecg_lead'].flatten() #700 000 y 745 000

#%% Filtro Mediana
ecg_one_lead = ecg_one_lead[700000:745000]

#Filtro mediana 200 muestras (200ms)
ecgfiltered_1 = sig.medfilt(ecg_one_lead, kernel_size=201)

#Filtro mediana 200 muestras (200ms)
ecgfiltered_2 = sig.medfilt(ecgfiltered_1, kernel_size=601)

#Plots
plt.figure()
plt.plot(ecg_one_lead, label = "ECG")
plt.plot(ecgfiltered_2, label = "ECG - Filtrado MEDIANA")
plt.title ("ECG Filtrado")
plt.grid();
plt.legend();

#%% Filtro splines cuicos

ecg_one_lead

qrs = mat_struct['qrs_detections']

plt.figure()
plt.plot(ecg_one_lead)
plt.plot(qrs, ecg_one_lead[qrs],'rx', label='QRS detectados')
plt.title('Señal ECG')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()

qrs2 = qrs - 90 #Resto 90 ms

