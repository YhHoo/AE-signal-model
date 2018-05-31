from scipy.signal import iirdesign, filtfilt, butter, stft
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, log, exp
from keras.preprocessing.sequence import pad_sequences
# self defined library
from dsp_tools import spectrogram_scipy
from utils import break_into_train_test


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
def noise_time_shift_dataset(time_axis):
    # # time axis setting
    # fs = 1000
    # duration = 100  # tune this for duration
    # total_point = int(fs * duration)
    # time_axis = np.linspace(0, duration, total_point)

    time_shift = [0, 100, 200, 300]  # 0.1, 0.2 .. seconds
    np.random.seed(45)
    noise = white_noise(time_axis=time_axis, power=1)

    signal = []
    for shift in time_shift:
        signal.append(np.concatenate((np.zeros(shift), noise), axis=0))

    signal = pad_sequences(signal, maxlen=(time_shift[-1] + 500), dtype='float32', padding='post')

    # plot all raw signals
    i = 1
    for s in signal:
        plt.subplot(6, 1, i)
        plt.plot(s)
        i += 1
    plt.show()

    # sliced to take only 1-3 seconds
    signal_sliced = signal[:, 1000:3000]

    phase_map = []
    for s in signal_sliced:
        t, f, Sxx = spectrogram_scipy(s,
                                      fs=fs,
                                      nperseg=100,
                                      noverlap=85,
                                      mode='angle',
                                      visualize=False,
                                      verbose=False,
                                      vis_max_freq_range=fs/2)
        phase_map.append(Sxx)

    phase_map = np.array(phase_map)
    print('Original Data Dim (Sensor, Freq, Time): ', phase_map.shape)

    dataset, label = [], []
    class_no = 0

    # for all sensor pair (all classes)
    for i in range(phase_map.shape[0] - 1):
        # for all time steps (all samples)
        for j in range(phase_map.shape[2]):
            concat_phase = [phase_map[0, :, j], phase_map[i+1, :, j]]
            dataset.append(concat_phase)
            label.append(class_no)
        class_no += 1

    # convert to Ndarray
    dataset = np.array(dataset)
    label = np.array(label)

    print('Data set Dim: ', dataset.shape)
    print('Label Dim: ', label.shape)
    print(label)


# time axis setting
fs = 1000
duration = 100  # tune this for duration
total_point = int(fs * duration)
time_axis = np.linspace(0, duration, total_point)
noise_time_shift_dataset(time_axis)

# train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
#                                                          label=label,
#                                                          num_classes=3,
#                                                          train_split=0.6,
#                                                          verbose=True)








