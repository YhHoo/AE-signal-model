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


# assume this noise has freq from 20-10kHz
# Here we let ensure the white noise is sampled at 20kHz to be able to capture 10kHz
def white_noise(mean, std, interval, max_cap_freq):
    '''
    :param mean: of the white noise
    :param std: std deviation of the noise
    :param interval: length of the noise in time (s)
    :param max_cap_freq: the maximum freq to be capture
    :return:
    '''
    fs = max_cap_freq * 2  # (nyquist f) to able to capture 10kHz
    sample_size = int(fs * interval)
    white_noise = np.random.normal(mean, std, size=sample_size)
    return white_noise, fs


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


