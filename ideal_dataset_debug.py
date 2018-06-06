from scipy.signal import filtfilt, butter, spectrogram
from scipy.fftpack import fft
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# self defined library
from ideal_dataset import noise_time_shift_dataset, white_noise, sine_wave_continuous
from utils import three_dim_visualizer

# time axis setting
fs = 1000
duration = 20  # tune this for duration
total_point = int(fs * duration)
time_axis = np.linspace(0, duration, total_point)

# Inside the noise_time_shift_dataset() -------------------------------------------

time_shift = [0, 100, 200, 300]  # 0.1, 0.2 .. seconds,
# noise
noise = white_noise(time_axis=time_axis, power=1)
# sine
sine = sine_wave_continuous(time_axis=time_axis, amplitude=1, fo=5)


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
                            noverlap=0,
                            mode='angle')
    f_res = fs / (2 * (f.size - 1))
    t_res = (s.shape[0] / fs) / t.size
    print('\n----------SPECTROGRAM OUTPUT---------')
    print('Time Segment....{}\nFirst 5: {}\nLast 5: {}\n'.format(t.size, t[:5], t[-5:]))
    print('Frequency Segment....{}\nFirst 5: {}\nLast 5: {}\n'.format(f.size, f[:5], f[-5:]))
    print('Spectrogram Dim: {}\nF-Resolution: {}Hz/Band\nT-Resolution: {}'.format(Sxx.shape, f_res, t_res))

    phase_map.append(Sxx)
    # plt.figure(i)
    #
    # plt.subplot(211)
    # plt.pcolormesh(t, f, Sxx)
    # plt.ylabel('Frequency [Hz]')
    # # display only 0Hz to 300kHz
    # plt.ylim((0, fs/2))
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.title('Spectrogram Phase Information {}'.format(i))
    # plt.grid()
    # plt.xlabel('Time [Sec]')
    # plt.colorbar()
    #
    # plt.subplot(212)
    # f_selected = np.arange(0, 51, 10)
    # for i in range(f_selected.size):
    #     plt.plot(t, Sxx[i], label=f[f_selected[i]])
    i += 1


phase_map = np.array(phase_map)

# cross cor for map 1 and map 2 with smallest time shift
# for all frequency band
lx = []
for i in range(phase_map.shape[1]):
    x_cor = np.correlate(phase_map[0, i], phase_map[3, i], 'full')
    lx.append(x_cor)
lx = np.array(lx)
plt.pcolormesh(np.arange(1, 160, 1), f, lx)
plt.colorbar()
plt.show()

# three_dim_visualizer(x_axis=np.arange(1, 160, 1), y_axis=f, zxx=lx, label=['time shit', 'frequency', 'correlation score'])


