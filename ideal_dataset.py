from scipy.signal import iirdesign, filtfilt, butter, stft
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, log, exp
from keras.preprocessing.sequence import pad_sequences
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


def white_noise(time_axis, power):
    noise = np.random.normal(scale=np.sqrt(power), size=time_axis.size)

    return noise


def sine_wave_continuous(time_axis, amplitude, fo, phase=0):
    y = amplitude * np.sin(2 * np.pi * fo * time_axis + phase)

    return y


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


# Noise shift Signal--------------------------------------------

# time axis setting
fs = 1000
duration = 100  # tune this for duration
total_point = int(fs * duration)
time_axis = np.linspace(0, duration, total_point)


time_shift = [0, 100, 200, 300]  # 0.1, 0.2 .. seconds
np.random.seed(45)
noise = white_noise(time_axis=time_axis, power=1)

signal = []
for shift in time_shift:
    signal.append(np.concatenate((np.zeros(shift), noise), axis=0))

signal = pad_sequences(signal, maxlen=total_point + 500, dtype='float32', padding='post')

# new time axis setting
duration2 = 100.5  # tune this for duration
total_point2 = int(fs * duration2)
time_axis2 = np.linspace(0, duration2, total_point2)

# plot all raw signals
i = 1
for s in signal:
    plt.subplot(6, 1, i)
    plt.plot(time_axis2, s)
    i += 1
plt.show()

# sliced to take only 1-3 seconds
signal_sliced = signal[:, 1000:3000]

phase_map = []
for s in signal_sliced:
    t, f, Sxx = spectrogram_scipy(signal_sliced[0],
                                  fs=fs,
                                  nperseg=100,
                                  noverlap=85,
                                  mode='angle',
                                  visualize=False,
                                  verbose=False,
                                  vis_max_freq_range=fs/2)
    phase_map.append(Sxx)

phase_map = np.array(phase_map)
print('Original Data Dim: ', phase_map.shape)

print(phase_map[0, :, 1].shape)
test = np.array([phase_map[0, :, 0], phase_map[1, :, 0]])
print(test.shape)

class_1, class_2, class_3 = [], [], []
# for all time step
for i in range(phase_map.shape[2]):
    concat_phase = [phase_map[0, :, i], phase_map[1, :, i]]
    class_1.append(concat_phase)
class_1 = np.array(class_1)
print(class_1.shape)








