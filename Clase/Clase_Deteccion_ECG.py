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

#%% Correlación del ECG (pre filtrado, con ruido)
mat_struct = sio.loadmat('./ECG_TP4.mat') #El archivo tiene que estar en la misma carpeta del proyecto

ecg_one_lead = mat_struct['ecg_lead'].flatten()

#Normalizo
ecg_one_lead = ecg_one_lead/np.std(ecg_one_lead)
qrs = mat_struct['qrs_pattern1']/np.std(mat_struct['qrs_pattern1'])

correlatedECG = np.correlate(ecg_one_lead,qrs.flatten())

correlatedECG = correlatedECG/np.std(correlatedECG)

plt.figure()
plt.plot(ecg_one_lead, label = "ECG")
plt.plot(correlatedECG, label = "ECG - Correlado")
plt.title ("ECG Correlado")
plt.grid();
plt.legend();

peaks = sig.find_peaks(correlatedECG,  distance=150)
np.diff(peaks)

plt.figure()
plt.plot(peaks, ecg_one_lead[peaks], "Peaks")
plt.plot(correlatedECG, label = "ECG - Correlado")
plt.title ("ECG Correlado")
plt.grid();
plt.legend();

#%% Filtro

mat_struct = sio.loadmat('./ECG_TP4.mat') #El archivo tiene que estar en la misma carpeta del proyecto

ecg_one_lead = mat_struct['ecg_lead'].flatten()

#ECGfiltrado = sig.sosfilt(mi_sos, ecg_one_lead) #Tenemos problemas de demora y distorsión de fase
ECGfiltrado = sig.sosfiltfilt(mi_sos, ecg_one_lead) #Anulamos los problemas de fase

plt.figure()
plt.plot(ecg_one_lead, label = "ECG")
plt.plot(ECGfiltrado, label = "ECG - Filtrado FILT")
plt.title ("ECG Filtrado")
plt.grid();
plt.legend();

#%% Regiones de interes
cant_muestras = np.size(ECGfiltrado)
ECGfiltrado_win = ECGfiltrado
demora = 0 # Vi ese offset a ojo, corregimos la demora del filt

regs_interes = ( 
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )

for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    
    plt.figure(figsize=(16, 8), facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
    plt.plot(zoom_region, ECGfiltrado_win[zoom_region + demora], label='Win')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()
    
    regs_interes = ( 
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    
    plt.figure(figsize=(16, 8), facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
    plt.plot(zoom_region, ECGfiltrado_win[zoom_region + demora], label='Win')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()
