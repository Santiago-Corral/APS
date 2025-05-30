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

#%% Analisis
# Diseñamos el filtro, HACER RTA DE FASE Y DE DEMORA

mi_sos = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=aprox_name, output='sos', fs=fs)

#Analizamos el filtro
npoints = 1000

w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi

w, hh = sig.sosfreqz(mi_sos, worN=w_rad)

w, hh = sig.sosfreqz(mi_sos, worN=npoints)

fasehh = np.angle(hh) #Rta de fase de hh
np.unwrap(fasehh)
#Derivar rta de fase con y(n) = x(n) - x(n-1), en esta pendiente esta el retardo del filtro

plt.figure()

plt.plot(w/np.pi, 20*np.log10(np.abs(hh)+1e-15), label= 'mi_sos')

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

ax = plt.gca()
ax.set_xlim([0, 1])
ax.set_ylim([-60, 1])

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()
plt.show()

plt.figure()

plt.plot(w/np.pi, np.unwrap(fasehh))

plt.title('Fase_SOS')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

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

#%% Diseño un nuevo filtro

# Definir frecuencias normalizadas (0 a 1)
ff = np.array([0, fstop[0], fpass[0], fpass[1], fstop[1], nyq_frec])
hh = np.array([0, 0, 1, 1, 0, 0])

# Diseño del filtro
Filtro_Ventana = sig.firwin2(numtaps = 2501, freq = ff, gain = hh, fs = 1000)
#Necesito tener orden impar! Estudiar esto de los filtros fir

w, hh = sig.freqz(Filtro_Ventana, worN=npoints) #Interpolo los puntos obtenidos

plt.figure()

plt.plot(w/np.pi, 20*np.log10(np.abs(hh)+1e-15), label= 'FirWin')

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

# ax = plt.gca()
# ax.set_xlim([0, 1])
# ax.set_ylim([-60, 1])

# plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
# plt.legend()
# plt.show()
