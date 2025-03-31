#%% módulos y funciones a importar

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import lanczos

def mi_funcion_sen (vmax, dc, ff, ph, N, fs):
    #fs frecuencia de muestreo (Hz)
    #N cantidad de muestras
    
    ts = 1/fs # tiempo de muestreo o periodo
    tt=np.linspace (0, (N-1)*ts, N) #vector de tiempo
    
    #generacion de la señal senoidal
    xx= dc + vmax*np.sin(2*np.pi*ff*tt + ph)
    #la señal debe generarse con la formula: x(t)=DC+Vmax*sen(2pift+fase)
    
    return tt, xx

N = 1000
fs =  1000 # frecuencia de muestreo (Hz)
df = fs/N # resolución espectral


window = lanczos(N)

##normalizar para que la potencia sea 1
##uno es viendo la varianza pero no 
tt, xx = mi_funcion_sen(1.4, 0, 250, 0, 1000, 1000)
##print (np.var(xx)) #Imprime la varianza de la funcion
##con desvio estandar:
xw=xx*window/np.std(xx*window)
xn=xx/np.std(xx)

plt.figure(1)
plt.plot(tt, xw) 
plt.title("Señal limpia normalizada:")
plt.xlabel("tiempo [segundos]")
plt.ylabel("Amplitud")
plt.legend()
plt.show()

analog_sig = xn # señal analógica sin ruido
analog_sig2 = xw

#%% Visualización de resultados

# cierro ventanas anteriores
# plt.close('all')

###########
# Espectro
###########

plt.figure(2)
ft_As = 1/N*np.fft.fft( analog_sig)
ft_As2 = 1/N*np.fft.fft( analog_sig2)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)', marker = 'x' ) #El puntero se usa para quedarnos con la mitad del vector (por paridad no necesitamos mas)
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As2[bfrec])**2), color='red', ls='dotted', label='$ s $ (sig.)' , marker = 'o')
plt.title('Comparativa señal muestreada con cuadrado y con lanczos' )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()
