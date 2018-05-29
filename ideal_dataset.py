from scipy.signal import iirdesign, filtfilt, butter
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, log, exp
# self defined library
from dsp_tools import spectrogram_scipy


# --------------[Sine wave of increasing freq]--------------------
# https://stackoverflow.com/questions/19771328/
# sine-wave-that-exponentialy-changes-between-frequencies-f1-and-f2-at-given-time
def sweep_exponential(f_start, f_end, interval, n_steps):
    b = log(f_end/f_start) / interval
    a = 2 * pi * f_start / b
    for i in range(n_steps):
        delta = i / float(n_steps)
        t = interval * delta
        g_t = a * exp(b * t)
        print(t, 3 * sin(g_t))

# sine wave sweeping f with noise---------------------
# fs = 10e3
# N = 10e4
# amp = 2 * np.sqrt(2)
# noise_power = 0.01 * fs / 2  # nyquist f x 0.001
# time = np.arange(N) / fs
# freq = np.linspace(1, 2e3, int(N))  # freq sweep range of sine wave
# x = amp * np.sin(2*np.pi*freq*time + 1)
# x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
# noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)


def white_noise(fs, duration, power):
    total_point = int(fs * duration)
    noise = np.random.normal(scale=np.sqrt(power), size=total_point)

    return noise


def sine_wave_continuous(fs, duration, amplitude, fo, phase=0):
    total_point = int(fs * duration)
    time_axis = np.linspace(0, duration, total_point)
    y = amplitude * np.cos(2 * np.pi * fo * time_axis + phase)

    return y, time_axis


# equal to 5 seconds
def sine_pulse():
    fs = 10e3
    f_nyquist = fs / 2
    total_sample = 70e3
    time_axis = np.arange(total_sample) / fs
    zero = np.array([0]*30000)
    sine = 3 * np.sin(2*np.pi*1e3*time_axis[int(30e3):int(40e3)])

    pulse_sine = np.concatenate((zero, sine, zero), axis=0)

    spectrogram_scipy(pulse_sine,
                      fs=fs,
                      nperseg=500,
                      noverlap=100,
                      verbose=True,
                      visualize=True,
                      vis_max_freq_range=fs/2)


