#%% módulos y funciones a importar
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as windows
import pandas as pd

#%% Datos de la simulación

R = 200 #Realizaciones
N = 1000 # cantidad de muestras
fs =  1000 # frecuencia de muestreo (Hz)
SNR = 10 #dB, piso de ruido

#%%

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral

tt = np.linspace (0, (N-1)*ts, N).reshape((N,1)) #vector de tiempo
ff = np.linspace(0, (N-1)*df, N) #Grilla de sampleo frecuencial

#generacion de la señal senoidal
tt = np.tile(tt, (1,R))

fr = np.random.uniform(-1/2,1/2, size = (1,R))
fr = np.reshape(fr, (1,R))

omega_0 = fs/4 #Mitad de banda digital
omega_1 = omega_0 +  fr * df #como luego lo usamos para el sesgo, multiplicamos por 2pi dentro del seno


a1 = np.sqrt(2)
xx = a1 * np.sin(2*np.pi*omega_1*tt) 
#El A1 lo usamos para normalizar

#%% Genero Ruido

pot_ruido = 10**(-SNR/10) #Como se define esto?

# ff = np.arange(0, 1, 1/1000).reshape((1000,1))
# ff = np.repeat(tt, 200, axis = 1)
nn = np.random.normal(0,np.sqrt(pot_ruido),(N,R)) #señal de ruido analogico, le doy el tamaño de xx

#%%
ss = xx+nn #Señal mas ruido

#%%
blackmanharris = windows.blackmanharris(N).reshape(N,1) # Reshape para multiplicar por la matriz ss
flattop = windows.flattop(N).reshape(N,1)
lanczos = windows.lanczos(N).reshape(N,1)

#Multiplico por la ventana
ssblackman = ss * blackmanharris
ssflattop = ss * flattop
sslanczos = ss * lanczos

#%% FFT, respuesta en frecuencia para cada ventana
ft_sinc = 1/N*np.fft.fft(ss, axis = 0) #es como multiplicar por uno el sinc
ft_blackman = 1/N*np.fft.fft(ssblackman, axis = 0)
ft_flattop = 1/N*np.fft.fft(ssflattop, axis = 0)
ft_lanczos = 1/N*np.fft.fft(sslanczos, axis = 0)

bfrec = ff <= fs/2

#plt.plot(ff[bfrec], 10* np.log10(2*np.abs(ft_blackman[bfrec])**2)) 
#plt.plot(ff[bfrec], 10* np.log10(2*np.abs(ft_flattop[bfrec])**2)) 
#plt.plot(ff[bfrec], 10* np.log10(2*np.abs(ft_lanczos[bfrec])**2))
plt.plot(ff[bfrec], 10* np.log10(2*np.abs(ft_sinc[bfrec])**2)) 
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
plt.title("Representación espectral")
axes_hdl = plt.gca() 

## Estimadores de a para cada ventana
# Tomo una feta en N/4 (250) donde tengo el pulso para ver el histograma
est_a_sinc = np.abs(ft_sinc [N//4,:])
est_a_blackman = np.abs(ft_blackman [N//4,:])
est_a_flattop = np.abs(ft_flattop [N//4,:])
est_a_lanczos = np.abs(ft_lanczos [N//4,:])

#%% Histograma de amplitudes
plt.figure()
bins = 10
plt.hist(est_a_sinc.flatten(), bins=bins, label = "Estimador con ventana Sinc")
plt.hist(est_a_blackman.flatten(), bins=bins, label = "Estimador con ventana Blackman")
plt.hist(est_a_flattop.flatten(), bins=bins, label = "Estimador con ventana flattop")
plt.hist(est_a_lanczos.flatten(), bins=bins, label = "Estimador con ventana lanczos")
plt.legend()
plt.grid("on")

plt.xlabel("Amplitud Estimada")
plt.ylabel("Cantidad de ocurrencias")
plt.title(f"Histograma de estimadores de amplitudes con SNR = {SNR} dB" )

#Cada ventana concentra distinto la energía 

#%% Estimadores de omega para cada ventana

ftabs_sinc = np.abs(ft_sinc[:N//2, :])
ftabs_blackman = np.abs(ft_blackman[:N//2, :])
ftabs_flattop = np.abs(ft_flattop[:N//2, :])
ftabs_lanczos = np.abs(ft_lanczos[:N//2, :])

est_omega_sinc = np.argmax(ftabs_sinc, axis = 0) * df
est_omega_blackman = np.argmax(ftabs_blackman, axis = 0) * df
est_omega_flattop = np.argmax(ftabs_flattop, axis = 0) * df
est_omega_lanczos = np.argmax(ftabs_lanczos, axis = 0) * df

#%% Histograma de frecuencias
plt.figure()
bins = 30
plt.hist(est_omega_sinc.flatten(), bins=bins, label = "Estimador con ventana Sinc", alpha=0.5)
plt.hist(est_omega_blackman.flatten(), bins=bins, label = "Estimador con ventana Blackman", alpha=0.5)
plt.hist(est_omega_flattop.flatten(), bins=bins, label = "Estimador con ventana flattop", alpha=0.5)
plt.hist(est_omega_lanczos.flatten(), bins=bins, label = "Estimador con ventana lanczos", alpha=0.5)
plt.legend()
plt.grid("on")

plt.xlabel("Frecuencia estimada [Hz]")
plt.ylabel("Cantidad de ocurrencias")
plt.title(f"Histograma de estimadores de frecuencias con SNR = {SNR} dB" )

#%% Sesgo y varianza de los estimadores de aplitud

esperanza_a_sinc = np.mean(est_a_sinc)
esperanza_a_blackman = np.mean(est_a_blackman)
esperanza_a_flattop = np.mean(est_a_flattop)
esperanza_a_lanczos = np.mean(est_a_lanczos)

sesgo_a_sinc = esperanza_a_sinc - a1 #siendo a1 el valor real, no el estimado
sesgo_a_blackman = esperanza_a_blackman - a1
sesgo_a_flattop = esperanza_a_flattop - a1
sesgo_a_lanczos = esperanza_a_lanczos - a1

varianza_a_sinc = np.var(est_a_sinc)
varianza_a_blackman = np.var(est_a_blackman)
varianza_a_flattop = np.var(est_a_flattop)
varianza_a_lanczos = np.var(est_a_lanczos)

## Sesgo y varianza de los estimadores de frecuencia espectral

esperanza_o_sinc = np.mean(est_omega_sinc)
esperanza_o_blackman = np.mean(est_omega_blackman)
esperanza_o_flattop = np.mean(est_omega_flattop)
esperanza_o_lanczos = np.mean(est_omega_lanczos)

sesgo_o_sinc = esperanza_o_sinc - omega_0 #siendo omega_1 el valor real, no el estimado
sesgo_o_blackman = esperanza_o_blackman - omega_0
sesgo_o_flattop = esperanza_o_flattop - omega_0
sesgo_o_lanczos = esperanza_o_lanczos - omega_0

varianza_o_sinc = np.var(est_omega_sinc)
varianza_o_blackman = np.var(est_omega_blackman)
varianza_o_flattop = np.var(est_omega_flattop)
varianza_o_lanczos = np.var(est_omega_lanczos)
#%% Tabla de resultados
 #Importe una libreria llamada pandas la cual resulta muy util para este tipo de tablas

# Crear la tabla con tus valores (reemplazá con tus propios datos si es necesario)
tabla_resultados = pd.DataFrame({
    "Ventana": ["Sinc", "Blackman", "Flattop", "Lanczos"],
    "Sesgo A1": [sesgo_a_sinc, sesgo_a_blackman, sesgo_a_flattop, sesgo_a_lanczos],
    "Varianza A1": [varianza_a_sinc, varianza_a_blackman, varianza_a_flattop, varianza_a_lanczos],
    "Sesgo Ω1": [sesgo_o_sinc, sesgo_o_blackman, sesgo_o_flattop, sesgo_o_lanczos],
    "Varianza Ω1": [varianza_o_sinc, varianza_o_blackman, varianza_o_flattop, varianza_o_lanczos]
})

# Redondear para mejorar la estética
tabla_resultados = tabla_resultados.round(6)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')  # Ocultar los ejes

# Dibujar la tabla
tabla = ax.table(cellText=tabla_resultados.values,
                 colLabels=tabla_resultados.columns,
                 cellLoc='center',
                 loc='center')

tabla.scale(1.2, 1.5)  # Escalar tabla para mejor legibilidad

# Título de la tabla
plt.title(f"Sesgo y Varianza de Estimadores con SNR {SNR} dB", fontsize=14, pad=20)

plt.tight_layout()
plt.show()

#%% Bonus - Zero Padding

N2 = 10*N #Tomo un N mas grande (calcula 10 veces mas puntos que los datos reales)
#En la fft vemos ceros extras despues de la señal
df_zp = fs/N2

#Rta en frecuencia para cada ventana
ft_sinc_zp = 1/N*np.fft.fft(ss, n = N2, axis = 0) #es como multiplicar por uno el sinc
ft_blackman_zp = 1/N*np.fft.fft(ssblackman, n = N2, axis = 0)
ft_flattop_zp = 1/N*np.fft.fft(ssflattop, n = N2, axis = 0)
ft_lanczos_zp = 1/N*np.fft.fft(sslanczos, n = N2, axis = 0)

ftabs_sinc_zp = np.abs(ft_sinc_zp[:N2//2, :])
ftabs_blackman_zp = np.abs(ft_blackman_zp[:N2//2, :])
ftabs_flattop_zp = np.abs(ft_flattop_zp[:N2//2, :])
ftabs_lanczos_zp = np.abs(ft_lanczos_zp[:N2//2, :])

est_omega_sinc_zp = np.argmax(ftabs_sinc_zp, axis = 0) * df_zp
est_omega_blackman_zp = np.argmax(ftabs_blackman_zp, axis = 0) * df_zp
est_omega_flattop_zp = np.argmax(ftabs_flattop_zp, axis = 0) * df_zp
est_omega_lanczos_zp = np.argmax(ftabs_lanczos_zp, axis = 0) * df_zp

#%% Histograma de frecuencias
plt.figure()
bins = 30
plt.hist(est_omega_sinc_zp.flatten(), bins=bins, label = "Estimador con ventana Sinc", alpha=0.5)
plt.hist(est_omega_blackman_zp.flatten(), bins=bins, label = "Estimador con ventana Blackman", alpha=0.5)
plt.hist(est_omega_flattop_zp.flatten(), bins=bins, label = "Estimador con ventana flattop", alpha=0.5)
plt.hist(est_omega_lanczos_zp.flatten(), bins=bins, label = "Estimador con ventana lanczos", alpha=0.5)
plt.legend()
plt.grid("on")

plt.xlabel("Frecuencia estimada [Hz]")
plt.ylabel("Cantidad de ocurrencias")
plt.title(f"Histograma de estimadores de frecuencias con SNR = {SNR} dB // ZERO PADDING" )
## Sesgo y varianza de los estimadores de frecuencia espectral

## Sesgo y varianza de los estimadores de frecuencia espectral

esperanza_o_sinc = np.mean(est_omega_sinc_zp)
esperanza_o_blackman = np.mean(est_omega_blackman_zp)
esperanza_o_flattop = np.mean(est_omega_flattop_zp)
esperanza_o_lanczos = np.mean(est_omega_lanczos_zp)

sesgo_o_sinc = esperanza_o_sinc - omega_0 #siendo omega_1 el valor real, no el estimado
sesgo_o_blackman = esperanza_o_blackman - omega_0
sesgo_o_flattop = esperanza_o_flattop - omega_0
sesgo_o_lanczos = esperanza_o_lanczos - omega_0

varianza_o_sinc = np.var(est_omega_sinc_zp)
varianza_o_blackman = np.var(est_omega_blackman_zp)
varianza_o_flattop = np.var(est_omega_flattop_zp)
varianza_o_lanczos = np.var(est_omega_lanczos_zp)
#%% Tabla de resultados

 #Importe una libreria llamada pandas la cual resulta muy util para este tipo de tablas

# Crear la tabla con tus valores (reemplazá con tus propios datos si es necesario)
tabla_resultados = pd.DataFrame({
    "Ventana": ["Sinc", "Blackman", "Flattop", "Lanczos"],
    "Sesgo Ω1": [sesgo_o_sinc, sesgo_o_blackman, sesgo_o_flattop, sesgo_o_lanczos],
    "Varianza Ω1": [varianza_o_sinc, varianza_o_blackman, varianza_o_flattop, varianza_o_lanczos]
})

# Redondear para mejorar la estética
tabla_resultados = tabla_resultados.round(6)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')  # Ocultar los ejes

# Dibujar la tabla
tabla = ax.table(cellText=tabla_resultados.values,
                 colLabels=tabla_resultados.columns,
                 cellLoc='center',
                 loc='center')

tabla.scale(1.2, 1.5)  # Escalar tabla para mejor legibilidad

# Título de la tabla
plt.title(f"Sesgo y Varianza de Estimadores con SNR {SNR} dB // ZERO PADDING", fontsize=14, pad=20)

plt.tight_layout()
plt.show()
