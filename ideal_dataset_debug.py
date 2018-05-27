from scipy.signal import filtfilt, butter
import numpy as np
import matplotlib.pyplot as plt
# self defined library
from dsp_tools import spectrogram_scipy, butter_bandpass_filtfilt, fft_scipy
from ideal_dataset import white_noise, sine_wave_continuous

# simulated leak noise -------------------------------------
# # assume it has active freq only in certain range
# leak_noise = white_noise(fs=10e3, duration=1, power=1)
# fft_scipy(leak_noise, fs=10e3, visualize=True)
#
# # bandpass filter
# filtered = butter_bandpass_filtfilt(sampled_data=leak_noise, fs=fs, f_hicut=3e3, f_locut=2e3)
# spectrogram_scipy(filtered,
#                   fs=fs,
#                   nperseg=500,
#                   noverlap=100,
#                   mode='angle',
#                   verbose=True,
#                   visualize=True,
#                   vis_max_freq_range=fs/2)

fs = 10e3
sine_wave, sine_wave_time_axis = sine_wave_continuous(fs=fs, duration=1, amplitude=1, fo=2500, phase=0)
spectrogram_scipy(sine_wave,
                  fs=fs,
                  nperseg=500,
                  noverlap=100,
                  mode='magnitude',
                  verbose=True,
                  visualize=True,
                  vis_max_freq_range=fs/2)






# # noisy environment -----------------------------------------
# env_noise = white_noise(fs=10e3, duration=1, power=0.3)
# # spectrogram_scipy(env_noise,
# #                   fs=10e3,
# #                   nperseg=500,
# #                   noverlap=100,
# #                   verbose=True,
# #                   visualize=True,
# #                   vis_max_freq_range=fs/2)
#
# # assert filtered.shape[0] == env_noise[0], 'Filtered and Env_noise must have equal length'
# # mix the 2 signals
# mix_signal = []
# for i in range(filtered.shape[0]):
#     mix_signal.append(filtered[i] + env_noise[i])
#
# spectrogram_scipy(mix_signal,
#                   fs=10e3,
#                   nperseg=500,
#                   noverlap=100,
#                   verbose=True,
#                   visualize=True,
#                   vis_max_freq_range=fs/2)
