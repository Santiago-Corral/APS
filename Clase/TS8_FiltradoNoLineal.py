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

#%% Detección de picos
fs = 1000

qrs = mat_struct['qrs_pattern1'] / np.std(mat_struct['qrs_pattern1'])

# Correlación como filtro adaptado
correlatedECG = np.correlate(ecg_one_lead, qrs.flatten(), mode='same')
correlatedECG = correlatedECG / np.std(correlatedECG)

peaks_true = mat_struct['qrs_detections'].flatten()

# Detección de picos
peaks, _ = sig.find_peaks(correlatedECG, prominence = 1.8)

# Visualización de los picos detectados
plt.figure()
plt.plot(peaks, ecg_one_lead[peaks], 'rx', label="Picos")
plt.plot(peaks_true, ecg_one_lead[peaks_true], 'yx', label="Picos Reales")
plt.plot(ecg_one_lead, label="ECG - Correlado")
plt.title("Detección de picos con filtro adaptado")
plt.xlim(700000 , 745000)
plt.grid()
plt.legend()

# Visualización de los picos detectados
plt.figure()
plt.plot(peaks, ecg_one_lead[peaks], 'rx', label="Picos")
plt.plot(peaks_true, ecg_one_lead[peaks_true], 'yx', label="Picos Reales")
plt.plot(ecg_one_lead, label="ECG - Correlado")
plt.title("Detección de picos con filtro adaptado")
plt.xlim(700000 , 705000)
plt.grid()
plt.legend()

# Visualización de los picos detectados
plt.figure()
plt.plot(peaks, ecg_one_lead[peaks], 'rx', label="Picos")
plt.plot(peaks_true, ecg_one_lead[peaks_true], 'yx', label="Picos Reales")
plt.plot(ecg_one_lead, label="ECG - Correlado")
plt.title("Detección de picos con filtro adaptado")
plt.xlim( 32500 , 40000)
plt.grid()
plt.legend()


# Comparación con detecciones reales dentro de un rango específico
inicio = 700000
fin = 745000
tolerancia = 150
TP = 0

# Filtrar picos detectados y reales dentro del rango
peaks_region = peaks[(peaks >= inicio) & (peaks <= fin)]
qrs_ref_region = peaks_true[(peaks_true >= inicio) & (peaks_true <= fin)]
comparaciones = min(len(peaks_region), len(qrs_ref_region))
indices = np.arange(comparaciones)

for i in indices:
    if abs(peaks_region[i] - qrs_ref_region[i]) <= tolerancia:
        TP += 1

FN = len(qrs_ref_region) - TP
FP = len(peaks_region) - TP

Se = TP / (TP + FN) if (TP + FN) > 0 else 0
PPV = TP / (TP + FP) if (TP + FP) > 0 else 0

print(f"Sensibilidad (Se): {Se:.3f}")
print(f"Valor predictivo positivo (PPV): {PPV:.3f}")
print(f"Cantidad de detecciónes verdaderas: {TP}")
print(f"Cantidad de picos reales en la región: {len(qrs_ref_region)}")
print(f"Cantidad de picos detectados en la región: {len(peaks_region)}")

#%%
qrs = mat_struct['qrs_pattern1'] / np.std(mat_struct['qrs_pattern1'])

# Segmento de ECG
ecg_segment = ecg_one_lead[300000:312000]

# Invierto el patrón del filtro adaptado
h = qrs_pattern1[::-1]

# Correlación en modo 'same' para que tenga el mismo largo que ecg_segment
correlation = correlate(ecg_segment, h, mode='same')

# Detección de picos en la correlación
peaks, _ = find_peaks(correlation, height=np.max(correlation)*0.3, distance=int(fs*0.6))

# Picos verdaderos dentro del segmento (convertidos a relativo al segmento)
peaks_true = qrs_detections[(qrs_detections >= 300000) & (qrs_detections < 312000)] - 300000


# Visualizacion 
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

axs[0].plot(ecg_segment, label='ECG')
axs[0].plot(peaks_true, ecg_segment[peaks_true], 'go', label='Latidos reales')
axs[0].plot(peaks, ecg_segment[peaks], 'rx', label='Latidos detectados')
axs[0].set_title('Señal ECG (latidos reales)')
axs[0].set_xlabel('Muestras')
axs[0].set_ylabel('Amplitud')
axs[0].grid(True)
axs[0].legend()

# Correlación con detección de picos
axs[1].plot(correlation, label='Correlación (matched filter)')
axs[1].plot(peaks_true, correlation[peaks_true], 'go', label='Latidos reales')
axs[1].plot(peaks, correlation[peaks], 'ro', label='Latidos detectados')
axs[1].set_title('Filtro Adaptado - Correlación y Detecciones')
axs[1].set_xlabel('Muestras')
axs[1].set_ylabel('Amplitud')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

