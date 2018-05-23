from scipy.signal import convolve, correlate, hann
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, log, exp
# self defined library
from dsp_tools import spectrogram_scipy


# --------------[Mix of 2 frequency Sine Wave]--------------------
def n_sine():
    # Number of sample points
    N = 500
    # sampling frequency in Hz
    fs = 800
    # sample spacing in seconds
    T = 1.0 / fs
    t = np.linspace(0.0, N*T, N)
    # 80 Hz and 50Hz sine wave
    # Wave Formula = sin(wt) = sin(2*pi*f*t)
    y1 = np.sin(10 * 2.0*pi*t) + 0.5*np.sin(80.0 * 2.0*pi*t)
    return t, y1


# --------------[Sine wave of increasing freq]--------------------
# Source: https://stackoverflow.com/questions/19771328/
# sine-wave-that-exponentialy-changes-between-frequencies-f1-and-f2-at-given-time
def sweep_linear(f_start, f_end, interval, n_steps, amplitude=1):
    '''
    :param f_start: starting freq
    :param f_end: ending freq of the sine wave
    :param interval: the total time of the signal
    :param n_steps: total sample points in the interval
    :param amplitude: amplitude of the sine wave
    :return: lists of sampled points of the wave and the time steps
    Sampling rate will simply be n_steps / interval
    '''
    x_t, y = [], []
    for i in range(n_steps):
        delta = i / float(n_steps)
        t = interval * delta
        phase = 2 * pi * t * (f_start + (f_end - f_start) * delta / 2)
        # collect the time steps
        x_t.append(t)
        # collecting wave output
        y.append(amplitude*sin(phase))
        # print(t, phase * 180 / pi, amplitude * sin(phase))
    fs = n_steps / interval
    return x_t, y, fs


# https://stackoverflow.com/questions/19771328/sine-wave-that-exponentialy-changes-between-frequencies-f1-and-f2-at-given-time
def sweep_exponential(f_start, f_end, interval, n_steps):
    b = log(f_end/f_start) / interval
    a = 2 * pi * f_start / b
    for i in range(n_steps):
        delta = i / float(n_steps)
        t = interval * delta
        g_t = a * exp(b * t)
        print(t, 3 * sin(g_t))

# sweep(1, 10, 5, 1000)


x, y, fs = sweep_linear(f_start=100, f_end=10000, interval=5, n_steps=int(100e3), amplitude=6)
print(fs)
spectrogram_scipy(y,
                  fs=fs,
                  nperseg=500,
                  noverlap=100,
                  verbose=True,
                  visualize=True,
                  vis_max_freq_range=10000)

# # plt.plot(x, y2, color='r', label='sin(10*2*pi*x)')
# plt.plot(x, y, color='b', label='linear change in f')
# plt.show()
