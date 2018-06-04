from scipy.signal import iirdesign, filtfilt, butter, stft
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, log, exp
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
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


def noise_time_shift_dataset(time_axis, fs, random_seed=None, num_series=2,
                             visualize_time_series=False, verbose=False):
    '''
    :param time_axis: White nosie will consists of (time_axis.size) points
    :param fs: sampling freq of the system
    :param num_series: num of diff random series,it controls the sample size
    :param random_seed: If stated, the seed is fixed to that, orelse it is random everytime it is called
    :param visualize_time_series: Plot the time series for checking delay in time
    :param verbose: print the dimension of arranged FFT phase map
    :return: a 3d data set where shape[0] is
    AIM--------------
    Create a time series white noise and applied delay to them
    '''
    # Scaler declaration
    scaler = MinMaxScaler(feature_range=(0, 1))

    # whether u wat the random series to be same all the time
    if random_seed is None:
        pass
    else:
        np.random.seed(random_seed)

    random_set = []
    # generating different random series to increase sample size
    for i in range(num_series):
        # add as many shift as u want, fr small to big
        time_shift = [0, 100, 200, 300]  # 0.1, 0.2 .. seconds,
        noise = white_noise(time_axis=time_axis, power=1)

        signal = []
        for shift in time_shift:
            signal.append(np.concatenate((np.zeros(shift), noise), axis=0))
        # so that all time shifted series are of same length
        signal = pad_sequences(signal, maxlen=(signal[-1].size + 500), dtype='float32', padding='post')

        # visualize the time series signal after shift
        if visualize_time_series:
            # plot all raw signals
            i = 1
            for s in signal:
                plt.subplot(6, 1, i)
                plt.plot(s)
                i += 1
            plt.show()
            plt.close()

        # sliced to take only 1-9 seconds
        signal_sliced = signal[:, 1000:9000]

        # convert all time series to F-T representation, form the phase map----------
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
            # scaling of all phases from -2pi to 2pi in 2d matrix of Sxx to 0-1
            Sxx = scaler.fit_transform(Sxx.ravel().reshape((-1, 1))).reshape((Sxx.shape[0], Sxx.shape[1]))
            print(Sxx)
            phase_map.append(Sxx)
        # convert to ndarray
        phase_map = np.array(phase_map)
        # put all 3d phase map array into list
        random_set.append(phase_map)

    # concatenate all 3d ndarray in random_set list in axis 2
    phase_map = np.concatenate(random_set, axis=2)

    # data slicing and labelling--------------------------------------------------
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

    if verbose:
        print('Original Data Dim (Sensor, Freq, Time): ', phase_map.shape)
        print('Paired Data set Dim (Sample size, Sensor, Freq): ', dataset.shape)
        print('Label Dim: ', label.shape)

    return dataset, label







