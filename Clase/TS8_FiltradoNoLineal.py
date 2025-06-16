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

#Filtro mediana 600 muestras (600ms)
ecgfiltered_2 = sig.medfilt(ecgfiltered_1, kernel_size=601)

#Plots
plt.figure()
plt.plot(ecg_one_lead, label = "ECG")
plt.plot(ecgfiltered_2, label = "ECG - Filtrado MEDIANA")
plt.title ("ECG Con linea de base")
plt.grid();
plt.legend();

ECGfiltrado = ecg_one_lead - ecgfiltered_2

#Plots
plt.figure()
plt.plot(ECGfiltrado, label = "ECG Filtrado")
plt.title ("ECG Filtrado")
plt.grid();
plt.legend();

#%% Filtro splines cuicos
from scipy.interpolate import CubicSpline

ecg_one_lead = mat_struct['ecg_lead'].flatten()

qrs = mat_struct['qrs_detections']

qrs2 = qrs - 90 #Resto 90 ms

qrs2 = qrs2.flatten()

splines = CubicSpline(qrs2,ecg_one_lead[qrs2])

plt.figure()
plt.plot(ecg_one_lead)
plt.plot(qrs, ecg_one_lead[qrs],'rx', label='QRS detectados')
plt.plot(qrs2, ecg_one_lead[qrs2],'yx', label='Segmento PQ')
plt.plot(splines(np.arange(len(ecg_one_lead))), label='Interpolación')
plt.title('Señal ECG')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.xlim(700000 , 745000)
plt.legend()
plt.grid()
plt.show()

ECGfiltrado = ecg_one_lead - splines(np.arange(len(ecg_one_lead)))

plt.figure()
plt.plot(ECGfiltrado)
plt.title('Señal ECG')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.xlim(700000 , 745000)
plt.legend()
plt.grid()
plt.show()

#%% Filtro splines cuicos
from scipy.interpolate import CubicSpline

ecg_one_lead = mat_struct['ecg_lead'].flatten()

qrs = mat_struct['qrs_detections']

qrs2 = qrs - 90 #Resto 90 ms

qrs2 = qrs2.flatten()

splines = CubicSpline(qrs2,ecg_one_lead[qrs2])

plt.figure()
plt.plot(ecg_one_lead)
plt.plot(qrs, ecg_one_lead[qrs],'rx', label='QRS detectados')
plt.plot(qrs2, ecg_one_lead[qrs2],'yx', label='Segmento PQ')
plt.plot(splines(np.arange(len(ecg_one_lead))), label='Interpolación')
plt.title('Señal ECG')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.xlim(700000 , 745000)
plt.legend()
plt.grid()
plt.show()

ECGfiltrado = ecg_one_lead - splines(np.arange(len(ecg_one_lead)))

plt.figure()
plt.plot(ECGfiltrado)
plt.title('Señal ECG')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.xlim(700000 , 745000)
plt.legend()
plt.grid()
plt.show()

#%% Detección de picos

qrs = mat_struct['qrs_pattern1'] / np.std(mat_struct['qrs_pattern1'])

# Correlación como filtro adaptado
correlatedECG = np.correlate(ecg_one_lead, qrs.flatten(), mode='same')
correlatedECG = correlatedECG / np.std(correlatedECG)

peaks_true = mat_struct['qrs_detections'].flatten()

# Visualización de la correlación
plt.figure()
plt.plot(ecg_one_lead, label="ECG")
plt.plot(peaks_true, ecg_one_lead[peaks_true], 'rx', label = 'picos reales')
plt.plot(correlatedECG, label="ECG - Correlado")
plt.title("ECG Correlado (Filtro Adaptado)")
plt.grid()
plt.legend()

# Detección de picos
peaks, _ = sig.find_peaks(correlatedECG, prominence=1.8)

# Visualización de los picos detectados
plt.figure()
plt.plot(peaks, ecg_one_lead[peaks], 'rx', label="Picos")
plt.plot(peaks_true, ecg_one_lead[peaks_true], 'yx', label="Picos Reales")
plt.plot(ecg_one_lead, label="ECG - Correlado")
plt.title("Detección de picos con filtro adaptado")
plt.grid()
plt.legend()

# Comparación con detecciones reales
qrs_ref = mat_struct['qrs_detections'].flatten()
tolerancia = 100
TP = 0

comparaciones = min(len(peaks), len(qrs_ref)) #Cantidad de comparaciones
indices = np.arange(comparaciones) #Genero un vector para recorrer los indices

for i in indices:
    if abs(peaks[i] - qrs_ref[i]) <= tolerancia:
        TP += 1

FN = len(qrs_ref) - TP
FP = len(peaks) - TP

Se = TP / (TP + FN)
PPV = TP / (TP + FP)

print(f"Sensibilidad (Se): {Se:.3f}")
print(f"Valor predictivo positivo (PPV): {PPV:.3f}")
print(f"Sensibilidad (Se): {TP:.3f}")

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio

# Cargar señal y patrón
mat_struct = sio.loadmat('ECG_TP4.mat')
ecg = mat_struct['ecg_lead'].flatten()
pattern = mat_struct['qrs_pattern1'].flatten()
qrs_ref = mat_struct['qrs_detections'].flatten()

# Normalizar
ecg = ecg / np.std(ecg)
pattern = pattern / np.std(pattern)

# Filtro adaptado = correlación con el patrón
matched_output = np.correlate(ecg, pattern, mode='same')
matched_output = matched_output / np.std(matched_output)

# Detección de picos en la señal filtrada
peaks, _ = sig.find_peaks(matched_output, distance=150, prominence=1.0)

# Visualización
plt.figure(figsize=(12, 4))
plt.plot(ecg, label='ECG')
plt.plot(matched_output, label='Filtro adaptado')
plt.plot(peaks, matched_output[peaks], 'rx', label='Detecciones')
plt.legend()
plt.title('Detección con filtro adaptado')
plt.grid()
plt.show()
