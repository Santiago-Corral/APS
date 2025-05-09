import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def mi_funcion_sen (vmax, dc, ff, ph, N, fs):
    #fs frecuencia de muestreo (Hz)
    #N cantidad de muestras
    
    ts = 1/fs # tiempo de muestreo o periodo
    tt=np.linspace (0, (N-1)*ts, N) #vector de tiempo
    
    #generacion de la señal senoidal
    xx= dc + vmax*np.sin(2*np.pi*ff*tt + ph)
    #la señal debe generarse con la formula: x(t)=DC+Vmax*sen(2pift+fase)
    
    return tt, xx

#%%Datos para el sampleo
fs =  1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras
# con 1000 para cada una normalizamos la resolucion espectral

# Datos del ADC
B = 16 # bits (los elegimos entre todos)
Vf = 2 # rango simétrico de +/- Vf Volts 
q = 2*Vf/(2**B)# paso de cuantización de q Volts

# datos del ruido (potencia de la señal normalizada, es decir 1 W)
pot_ruido_cuant = (q**2)/12 # Watts 
kn = 10. # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn 

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral

#%%

tt, xx = mi_funcion_sen(2, 0, 1, 0, N, fs)

##normalizar para que la potencia sea 1
##con desvio estandar:
xn = xx/np.std(xx) #varianza unitaria con potencia = 1

nn=np.random.normal(0,np.sqrt(pot_ruido_analog),N) #señal de ruido analogico

#%% Generacion de ruido

analog_sig = xn # señal analógica sin ruido

sr = xn+nn # señal analógica de entrada al ADC (con ruido analógico)

srq = np.round(sr/q)*q# señal cuantizada, (señal divida la cantidad total de bits)

nq =  srq-sr# señal de ruido de cuantización

#%% Graficos y analisis
##################
# Señal temporal
##################

plt.figure(1)


#Grafico de la señal sin ruido
plt.plot(tt, analog_sig, lw=1,linestyle='-', color ="red", label='$ s $ = señal analogica' )
# Gráfico de señal cuantizada con ruido
plt.plot(tt, srq, lw=2, linestyle='-', color='blue', marker='o', markersize=3, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label=' $ srq $ = señal cuantizada')
# Gráfico de señal analógica con ruido
plt.plot(tt, sr, lw=1, color='black', marker='x', linestyle='dotted', markersize=2, markerfacecolor='red', markeredgecolor='black', fillstyle='none', label='$ sr $ =  señal analogica con ruido')


plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


###########
# Espectro
###########
plt.figure(2)
ft_SR = 1/N*np.fft.fft( sr)
ft_Srq = 1/N*np.fft.fft( srq)
ft_As = 1/N*np.fft.fft( analog_sig)
ft_Nq = 1/N*np.fft.fft( nq)
ft_Nn = 1/N*np.fft.fft( nn)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.figure(10)

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)' ) #El puntero se usa para quedarnos con la mitad del vector (por paridad no necesitamos mas)
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

#############
# Histograma
#############

plt.figure(3)
bins = 10
plt.hist(nq.flatten()/(q), bins=bins)
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

plt.xlabel('Pasos de cuantización (q) [V]')

#%% Analisis mas preciso

plt.figure(4)


#Grafico de la señal sin ruido
plt.plot(tt, analog_sig, lw=1,linestyle='-', color ="red", label='$ s $ = señal analogica' )
# Gráfico de señal cuantizada con ruido
plt.plot(tt, srq, lw=2, linestyle='-', color='blue', marker='o', markersize=3, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label=' $ srq $ = señal cuantizada')
# Gráfico de señal analógica con ruido
plt.plot(tt, sr, lw=1, color='black', marker='x', linestyle='dotted', markersize=2, markerfacecolor='red', markeredgecolor='black', fillstyle='none', label='$ sr $ =  señal analogica con ruido')


plt.xlim(0.140,0.360)
plt.ylim(1,1.8)
plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()
