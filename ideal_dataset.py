from scipy.signal import convolve, correlate, hann
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, log, exp

# --------------[FFT on mix of 2 frequency Sine Wave]--------------------
# Number of sample points
N = 500
# sampling frequency in Hz
fs = 800
# sample spacing in seconds
T = 1.0 / fs
x = np.linspace(0.0, N*T, N)
# 80 Hz and 50Hz sine wave
# Wave Formula = sin(wt) = sin(2*pi*f*t)
y1 = np.sin(10 * 2.0*pi*x) + 0.5*np.sin(80.0 * 2.0*pi*x)
y2 = np.sin(10*2*pi*x)
y3 = np.sin(10*2*pi*x + pi)

plt.plot(x, y2, color='r', label='sin(10*2*pi*x)')
plt.plot(x, y3, color='b', label='sin(10*2*pi*x + pi)')
plt.show()


def sweep(f_start, f_end, interval, n_steps):
    b = log(f_end/f_start) / interval
    a = 2 * pi * f_start / b
    for i in range(n_steps):
        delta = i / float(n_steps)
        t = interval * delta
        g_t = a * exp(b * t)
        print(t, 3 * sin(g_t))

# sweep(1, 10, 5, 1000)
