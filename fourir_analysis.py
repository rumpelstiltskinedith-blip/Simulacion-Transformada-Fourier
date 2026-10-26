import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import unit_impulse

# ==========================================
# Configuración global de la simulación
# ==========================================
Fs = 1000          # Frecuencia de muestreo (Hz)
T = 1.0            # Duración de la señal (segundos)
N = int(Fs * T)    # Número total de muestras
t = np.linspace(0, T, N, endpoint=False) # Vector de tiempo
freqs = np.fft.fftfreq(N, 1/Fs)          # Vector de frecuencias

# Función auxiliar para graficar señales y su FFT (Magnitud y Fase)
def plot_signal_and_fft(t, x, freqs, X, title_name):
    # Aplicamos fftshift para centrar las frecuencias en 0 (componentes negativas y positivas)
    freqs_shifted = np.fft.fftshift(freqs)
    X_shifted = np.fft.fftshift(X)
    
    magnitude = np.abs(X_shifted) / N  # Magnitud normalizada
    # Filtramos fases donde la magnitud es insignificante (ruido numérico)
    phase = np.angle(X_shifted)
    phase[magnitude < 1e-4] = 0 

    plt.figure(figsize=(15, 4))
    
    # Dominio del tiempo
    plt.subplot(1, 3, 1)
    plt.plot(t, x, 'b-', linewidth=1.5)
    plt.title(f'{title_name} - Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    
    # Magnitud en Frecuencia
    plt.subplot(1, 3, 2)
    plt.stem(freqs_shifted, magnitude, 'r', markerfmt='ro', basefmt=' ')
    plt.xlim(-20, 20)  # Zoom en frecuencias bajas de interés
    plt.title('Espectro de Magnitud')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.grid(True)
    
    # Fase en Frecuencia
    plt.subplot(1, 3, 3)
    plt.stem(freqs_shifted, phase, 'g', markerfmt='go', basefmt=' ')
    plt.xlim(-20, 20)
    plt.title('Espectro de Fase')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Fase (rad)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# PARTE 1: Señales Elementales
# ==========================================

print("--- Generando Señales Elementales y sus Transformadas ---")

# 1. Función Senoidal (Frecuencia de 5 Hz)
f0 = 5
x_sine = np.sin(2 * np.pi * f0 * t)
X_sine = np.fft.fft(x_sine)
plot_signal_and_fft(t, x_sine, freqs, X_sine, "Señal Senoidal (5 Hz)")

# 2. Pulso Rectangular (Ancho de 0.1 segundos en el centro)
x_rect = np.where((t >= 0.45) & (t <= 0.55), 1.0, 0.0)
X_rect = np.fft.fft(x_rect)
plot_signal_and_fft(t, x_rect, freqs, X_rect, "Pulso Rectangular")

# 3. Función Escalón Unitario (Inicia en t = 0.3 segundos)
x_step = np.where(t >= 0.3, 1.0, 0.0)
X_step = np.fft.fft(x_step)
plot_signal_and_fft(t, x_step, freqs, X_step, "Función Escalón")


# ==========================================
# PARTE 2: Verificación de Propiedades
# ==========================================
print("\n--- Verificando Propiedades de la Transformada de Fourier ---")

# --- 1. Linealidad: F(a*x1 + b*x2) = a*F(x1) + b*F(x2) ---
a, b = 2.0, 3.5
x1 = np.sin(2 * np.pi * 3 * t)  # 3 Hz
x2 = np.cos(2 * np.pi * 7 * t)  # 7 Hz

# Combinación lineal en el tiempo y luego FFT
x_combo = a * x1 + b * x2
X_combo_directa = np.fft.fft(x_combo)

# Combinación lineal directamente en frecuencias
X_combo_separada = a * np.fft.fft(x1) + b * np.fft.fft(x2)

error_linealidad = np.max(np.abs(X_combo_directa - X_combo_separada))
print(f"Error de linealidad (debe ser cercano a 0): {error_linealidad:.2e}")


# --- 2. Desplazamiento en el Tiempo: f(t - t0) -> F(w) * e^(-j * w * t0) ---
# Desplazamos un pulso rectangular en el tiempo por t0 = 0.2 segundos
t0 = 0.2
# Señal original (pulso en t=0) y señal desplazada
x_orig = np.where((t >= 0.1) & (t <= 0.2), 1.0, 0.0)
x_shifted = np.where((t >= 0.1 + t0) & (t <= 0.2 + t0), 1.0, 0.0)

X_orig = np.fft.fft(x_orig)
X_shifted_real = np.fft.fft(x_shifted)

# Aplicación teórica del desfase exponencial
# w = 2 * pi * f
fases_teoricas = np.exp(-1j * 2 * np.pi * freqs * t0)
X_shifted_teorica = X_orig * fases_teoricas

error_desplazamiento = np.max(np.abs(X_shifted_real - X_shifted_teorica))
print(f"Error de desplazamiento en tiempo (debe ser cercano a 0): {error_desplazamiento:.2e}")


# --- 3. Escalamiento en Frecuencia / Tiempo: f(a*t) -> (1/|a|) * F(w/a) ---
# Generamos dos senoidales: una base y otra con el doble de frecuencia (tiempo comprimido)
x_lenta = np.sin(2 * np.pi * 4 * t)  # 4 Hz
x_rapida = np.sin(2 * np.pi * 8 * t) # 8 Hz (Escalamiento en tiempo a=2)

X_lenta = np.fft.fft(x_lenta)
X_rapida = np.fft.fft(x_rapida)

# Graficamos ambas para comprobar visualmente la propiedad de escalamiento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t, x_lenta, 'b', label='Original (4 Hz)')
plt.plot(t, x_rapida, 'r--', label='Comprimida t (8 Hz)')
plt.title('Escalamiento en el Tiempo')
plt.xlim(0, 0.5)
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(np.fft.fftshift(freqs), np.abs(np.fft.fftshift(X_lenta))/N, 'b', basefmt=' ', label='Espectro 4 Hz')
plt.stem(np.fft.fftshift(freqs), np.abs(np.fft.fftshift(X_rapida))/N, 'r--', basefmt=' ', label='Espectro 8 Hz')
plt.title('Efecto en el Espectro de Frecuencia')
plt.xlim(-15, 15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
