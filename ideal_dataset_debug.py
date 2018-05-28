from scipy.signal import filtfilt, butter
from scipy.fftpack import fft
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


fs = 1e3
sine_wave, sine_wave_time_axis = sine_wave_continuous(fs=fs, duration=1, amplitude=1, fo=50, phase=1)
# plt.plot(sine_wave_time_axis, sine_wave)
# plt.grid()
# plt.show()
# time_axis, _, Sxx = spectrogram_scipy(sine_wave,
#                                       fs=fs,
#                                       nperseg=80,
#                                       noverlap=30,
#                                       mode='complex',
#                                       verbose=True,
#                                       visualize=False,
#                                       vis_max_freq_range=100)
#
# color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
# for i in range(3, 10, 1):
#     angle = []
#     for complex in Sxx[i]:
#         angle.append(np.angle(complex))
#     plt.plot(time_axis, angle, color=color[i-3])
# plt.plot(sine_wave_time_axis, sine_wave, color='r')
# plt.grid()
# plt.show()


sine_fft = fft(sine_wave)
N = sine_wave.size
sine_fft_mag = (2.0/N) * np.abs(sine_fft[0: N//2])
sine_fft_phase = np.angle(sine_fft[0: N//2])
f_axis = np.linspace(0.0, fs/2, N//2)

plt.plot(f_axis, sine_fft_mag)
plt.plot(f_axis, sine_fft_phase, color='r')
plt.plot([0.5*np.pi] * f_axis.size, color='k')
plt.plot([np.pi] * f_axis.size, color='k')
plt.show()
