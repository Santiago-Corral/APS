import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt

#%%
def Analisis_Frecuencial(H, Numeradores, Denominadores):
    omega = np.linspace(0, np.pi, 1000) #Frecuencias normalizadas

    H_Modulo = np.abs(H)
    H_Fase = np.angle(H)

    # Graficar módulo
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(omega, H_Modulo)
    plt.title('Módulo de H(e^{jω})')
    plt.xlabel('ω [rad/muestra]')
    plt.ylabel('|H(e^{jω})|')
    plt.grid(True)

    # Graficar fase
    plt.subplot(1, 2, 2)
    plt.plot(omega, H_Fase)
    plt.title('Fase de H(e^{jω})')
    plt.xlabel('ω [rad/muestra]')
    plt.ylabel('Fase [rad]')
    plt.grid(True)

    plt.tight_layout()
    plt.show() 

    #POLOS y CEROS
    ceros = np.roots(Numeradores)
    polos = np.roots(Denominadores)

    # Círculo unitario
    theta = np.linspace(0, 2*np.pi, 300)
    circ_x = np.cos(theta)
    circ_y = np.sin(theta)

    # Gráfico
    fig, ax = plt.subplots()
    ax.plot(circ_x, circ_y, 'k--', label='Círculo unitario')
    ax.plot(np.real(ceros), np.imag(ceros), 'o', label='Ceros', markersize=10, markerfacecolor='none')
    ax.plot(np.real(polos), np.imag(polos), 'x', label='Polos', markersize=10)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_title('Diagrama de Polos y Ceros')
    ax.grid(True)
    ax.axis('equal')
    ax.legend()
    plt.show()

#%% Evalulo las transferencias

omega = np.linspace(0, np.pi, 1000) #Frecuencias normalizadas

H_a = 1 + np.exp(-1j*omega) + np.exp(-2j*omega) + np.exp(-3j*omega)
Numeradores_a = [1, 1, 1, 1]
Denominadores_a = [1, 0, 0, 0]

H_b = 1 + np.exp(-1j*omega) + np.exp(-2j*omega) + np.exp(-3j*omega) + np.exp(-4j*omega)
Numeradores_b = [1, 1, 1, 1, 1]
Denominadores_b = [1, 0, 0, 0, 0]

H_c = 1 - np.exp(-1j*omega)
Numeradores_c = [1, -1]
Denominadores_c = [1, 0]

H_d = 1 - np.exp(-2j*omega)
Numeradores_d = [1, 0 , -1]
Denominadores_d = [1, 0, 0]

Analisis_Frecuencial(H_a, Numeradores_a, Denominadores_a)
Analisis_Frecuencial(H_b, Numeradores_b, Denominadores_b)
Analisis_Frecuencial(H_c, Numeradores_c, Denominadores_c)
Analisis_Frecuencial(H_d, Numeradores_d, Denominadores_d)