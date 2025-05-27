import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
#from pytc2.sistemas_lineales import plot_plantilla

aprox_name = 'butter'

fs = 1000 #Hz
nyq_frec = fs/2
fpass = np.array ( [1.0, 35.0 ] )
ripple = 1.0 #dB
fstop = np.array( [.1 , 50.] )
attenuation = 40 #dB

#%% Analisis
# Diseñamos el 

mi_sos = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=aprox_name, output='sos', fs=fs)

#Analizamos el filtro
npoints = 1000

w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi

w, hh = sig.sosfreqz(mi_sos, worN=w_rad)

w, hh = sig.sosfreqz(mi_sos, worN=npoints)
plt.plot(w/np.pi, 20*np.log10(np.abs(hh)+1e-15), label= 'mi_sos')

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

ax = plt.gca()
ax.set_xlim([0, 1])
ax.set_ylim([-60, 1])

#plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()
plt.show()


#%% Filtro

mat_struct = sio.loadmat('./ECG_TP4.mat') #El archivo tiene que estar en la misma carpeta del proyecto

ecg_one_lead = mat_struct['ecg_lead'].flatten()

ECGfiltrado = sig.sosfiltfilt(mi_sos, ecg_one_lead)


plt.figure()
plt.plot(ecg_one_lead[5000:12000])
plt.plot(ECGfiltrado[5000:12000])
