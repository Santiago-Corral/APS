import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from pytc2.sistemas_lineales import plot_plantilla

aprox_name = 'butter'

M = 100 #Diezmado, BW = fs/2 /10 = 50Hz, si diezmo por mas es mas facil ver las componentes de baja frecuencia
fs = 1000 #Hz
nyq_frec = fs/2
fpass = (nyq_frec/M)-1 #con 
ripple = 1.0 #dB
fstop = nyq_frec/M
attenuation = 20 #dB

#%% Diseñamos el filtro

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

plot_plantilla(filter_type = 'lowpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()
plt.show()

plt.figure()

plt.plot(w/np.pi, np.unwrap(fasehh))

#%% Filtrado y Diezmado de la señal (Ancho de banda reducido)

mat_struct = sio.loadmat('./ECG_TP4.mat') #El archivo tiene que estar en la misma carpeta del proyecto

ecg_one_lead = mat_struct['ecg_lead'].flatten()

# Normalización
ecg_one_lead = ecg_one_lead / np.std(ecg_one_lead)

# Vector de tiempo original
t_original = np.arange(len(ecg_one_lead)) / fs

ECGfiltrado = sig.sosfiltfilt(mi_sos, ecg_one_lead) #Anulamos los problemas de fase

# Diezmado
D_ECG = ECGfiltrado[::M]
t_diezmado = t_original[::M]  # Tiempo correspondiente al diezmado

# Gráfico
plt.figure(figsize=(10, 4))
plt.plot(t_original, ecg_one_lead, label="ECG original", alpha=0.7)
plt.plot(t_diezmado, D_ECG, label="ECG diezmado")
plt.title("ECG y su versión diezmada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud normalizada")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


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