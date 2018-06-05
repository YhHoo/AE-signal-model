from scipy.signal import filtfilt, butter, spectrogram
from scipy.fftpack import fft
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
# self defined library
from ideal_dataset import noise_time_shift_dataset, white_noise


# time axis setting
fs = 1000
duration = 20  # tune this for duration
total_point = int(fs * duration)
time_axis = np.linspace(0, duration, total_point)

# Inside the noise_time_shift_dataset() -------------------------------------------

time_shift = [0, 100, 200, 300]  # 0.1, 0.2 .. seconds,
noise = white_noise(time_axis=time_axis, power=1)

signal = []
for shift in time_shift:
    signal.append(np.concatenate((np.zeros(shift), noise), axis=0))
# so that all time shifted series are of same length
signal = pad_sequences(signal, maxlen=(signal[-1].size + 500), dtype='float32', padding='post')

# visualize the time series signal after shift
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
i = 1
for s in signal_sliced:
    f, t, Sxx = spectrogram(s,
                            fs=fs,
                            scaling='spectrum',
                            nperseg=100,
                            noverlap=81,
                            mode='angle')
    plt.figure(i)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    # display only 0Hz to 300kHz
    plt.ylim((0, fs/2))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title('Spectrogram Phase Information {}'.format(i))
    plt.grid()
    plt.xlabel('Time [Sec]')
    plt.colorbar()
    i += 1

plt.show()