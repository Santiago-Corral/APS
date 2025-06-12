import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from pytc2.sistemas_lineales import plot_plantilla

#Muestreos del ECG
fs = 1000 #Hz
nyq_frec = fs/2
ripple = 1.0 #dB
attenuation = 40 #dB
fpass = np.array ( [1.0, 35.0 ] )
fstop = np.array( [.1 , 50.] )

#%% Inicio la señal del ECG
mat_struct = sio.loadmat('./ECG_TP4.mat') #El archivo tiene que estar en la misma carpeta del proyecto

ecg_one_lead = mat_struct['ecg_lead'].flatten()

#%% Estimación del ancho de banda

#Estimo respuesta en frecuencia con welch
(fwelch_ecgn, pxx_ecgn) = sp.signal.welch(ecg_one_lead_n, fs_ecg ,nfft = N1, window = 'hamming', nperseg = N1//6, axis = 0)
(fwelch_ecg, pxx_ecg) = sp.signal.welch(ecg_one_lead, fs_ecg ,nfft = N2, window = 'hamming', nperseg = N2//6, axis = 0)

#Potencia total de la señal 
potencia_total_ecgn = np.sum(pxx_ecgn) #Devuelve un valor
#Potencia acumulada
potencia_acumulada_ecgn = np.cumsum(pxx_ecgn)

umbral_98_n = 0.98 * potencia_total_ecgn #Tomo como umbral el 98% de energía total de la señal
idc_98_n = np.argmax(potencia_acumulada_ecgn >= umbral_98_n)

#Bw
BW_98_n = fwelch_ecgn[idc_98_n] #Frecuencia a la que se llega al indice

#%% diseño de plantilla
####### Filtro con pytc2 #######
fpass = np.array ( [1.0, 35.0 ] )
fstop = np.array( [.1 , 50.] )

npoints = 1000

fpass = np.array ( [1.0, 35.0 ] )
fstop = np.array( [.1 , 50.] )

plt.figure(figsize=(10, 6))

plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
ax = plt.gca()

plt.title('Plantilla de Diseño de Filtro Pasabanda para ECG')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.legend()

plt.show()

#%% Filtros IIR, Maxima Planicidad

aprox_name = 'butter'

# Diseñamos el filtro, HACER RTA DE FASE Y DE DEMORA

mi_sos = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=aprox_name, output='sos', fs=fs)

#Analizamos el filtro
npoints = 1000

w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi

w, hh = sig.sosfreqz(mi_sos, worN=w_rad)
w, hh = sig.sosfreqz(mi_sos, worN=npoints)

plt.figure()

plt.plot(w / np.pi * fs / 2, 20 * np.log10(np.abs(hh) + 1e-15), label= 'mi_sos')

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()
plt.show()

#Filtramos
ECGfiltrado_maxp = sig.sosfiltfilt(mi_sos, ecg_one_lead) #Anulamos los problemas de fase

#%% Filtros IIR, Cheby 1
fpass = np.array ( [1.0, 35.0 ] )
fstop = np.array( [.1 , 50.] )

aprox_name = 'cheby1'

# Diseñamos el filtro, HACER RTA DE FASE Y DE DEMORA

mi_sos = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=aprox_name, output='sos', fs=fs)

#Analizamos el filtro
npoints = 1000

w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi

w, hh = sig.sosfreqz(mi_sos, worN=npoints)

plt.figure()

plt.plot(w / np.pi * fs / 2, 20 * np.log10(np.abs(hh) + 1e-15), label= 'mi_sos')

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()
plt.show()

#Filtramos
ECGfiltrado_cheby1 = sig.sosfiltfilt(mi_sos, ecg_one_lead) #Anulamos los problemas de fase

#%% Filtros IIR, Cheby 2
fpass = np.array ( [1.0, 35.0 ] )
fstop = np.array( [.1 , 50.] )

aprox_name = 'cheby2'

# Diseñamos el filtro, HACER RTA DE FASE Y DE DEMORA

mi_sos = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=aprox_name, output='sos', fs=fs)

#Analizamos el filtro
npoints = 1000

w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi

w, hh = sig.sosfreqz(mi_sos, worN=w_rad)

plt.figure()

plt.plot(w / np.pi * fs / 2, 20 * np.log10(np.abs(hh) + 1e-15), label= 'mi_sos')

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()
plt.show()
sig.sosfiltfilt(mi_sos, ecg_one_lead) #Anulamos los problemas de fase

#Filtramos
ECGfiltrado_cheby2 = sig.sosfiltfilt(mi_sos, ecg_one_lead) #Anulamos los problemas de fase
#%% Filtros IIR, Elipse
fpass = np.array ( [1.0, 35.0 ] )
fstop = np.array( [.1 , 50.] )

aprox_name = 'ellip'

# Diseñamos el filtro, HACER RTA DE FASE Y DE DEMORA

mi_sos = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=aprox_name, output='sos', fs=fs)

#Analizamos el filtro
npoints = 1000

w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi

w, hh = sig.sosfreqz(mi_sos, worN=w_rad)

plt.figure()

plt.plot(w / np.pi * fs / 2, 20 * np.log10(np.abs(hh) + 1e-15), label= 'mi_sos')

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()
plt.show()

#Filtramos
ECGfiltrado_Eli = sig.sosfiltfilt(mi_sos, ecg_one_lead) #Anulamos los problemas de fase

#%% Filtros FIR, Ventana

# Definir frecuencias normalizadas (0 a 1)
ff = np.array([0, fstop[0], fpass[0], fpass[1], fstop[1], nyq_frec])
hh = np.array([0, 0, 1, 1, 0, 0])

# Diseño del filtro
Filtro_Ventana = sig.firwin2(numtaps = 2501, freq = ff, gain = hh, fs = 1000)
#Necesito tener orden impar! Estudiar esto de los filtros fir

w, hh = sig.freqz(Filtro_Ventana, worN=npoints) #Interpolo los puntos obtenidos

plt.figure()

plt.plot(w/np.pi * fs / 2, 20*np.log10(np.abs(hh)+1e-15), label= 'FirWin', alpha=0)

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()
plt.show()

#Filtramos
ECGfiltrado_ventana = np.convolve(ecg_one_lead, Filtro_Ventana, mode='same')

#%% Filtros FIR, Cuadrados Minimos

npoints = 1000

# Definir frecuencias normalizadas (0 a 1)
ff = np.array([0, fstop[0], fpass[0], fpass[1], fpass[1]+1, nyq_frec]) #fpass[1]+1 De esta manera trabajamos con una banda de paso simetrica
hh = np.array([0, 0, 1, 1, 0, 0]) 

# Diseño del filtro
Filtro_Ventana = sig.firls(numtaps = 1501, bands = ff, desired = hh, fs = 1000)
#Necesito tener orden impar! Estudiar esto de los filtros fir

w, hh = sig.freqz(Filtro_Ventana, worN=npoints) #Interpolo los puntos obtenidos

plt.figure()

plt.plot(w/np.pi * fs / 2, 20*np.log10(np.abs(hh)+1e-15), label= 'FirWin')

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()
plt.show()

#Filtramos
ECGfiltrado_Cuadrados = np.convolve(ecg_one_lead, Filtro_Ventana, mode='same')

#%% Filtros FIR, Remez
#Para este caso partiremos el filtro en un pasa bajos y un pasa altos

# Definir frecuencias normalizadas (0 a 1)
ffH = np.array([0, fstop[0], fpass[0], nyq_frec]) 
hhH = np.array([0, 1]) 

ffL = np.array([0, fpass[1], fstop[1], nyq_frec]) 
hhL = np.array([1, 0]) 

# Diseño del filtro
Remez_High = sig.remez(numtaps = 1001, bands = ffH, desired = hhH, fs = 1000) 
Remez_Low= sig.remez(numtaps = 501, bands = ffL, desired = hhL, fs = 1000)
#Necesito un orden mucho mayor para aplicar el pasa altos

w1, hh1 = sig.freqz(Remez_Low, worN=npoints) #Interpolo los puntos obtenidos
w2, hh2 = sig.freqz(Remez_High, worN=npoints) #Interpolo los puntos obtenidos

plt.figure()

plt.plot(w/np.pi * fs / 2, 20*np.log10(np.abs(hh1)+1e-15), label= 'RemezHigh')
plt.plot(w/np.pi * fs / 2, 20*np.log10(np.abs(hh2)+1e-15), label= 'RemezLow')

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()
plt.show()

#Filtramos
ECGfiltrado_Remez_Low = np.convolve(ecg_one_lead, Remez_Low, mode='same')
ECGfiltrado_Remez = np.convolve(ECGfiltrado_Remez_Low, Remez_High, mode='same')

#%% Regiones de interes
cant_muestras = np.size(ECGfiltrado_Remez)
ECGfiltrado_win = ECGfiltrado_Remez
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
