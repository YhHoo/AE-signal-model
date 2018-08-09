# This is juz for testing out small function from dsp_tools before integration on bigger one

import numpy as np
from scipy.signal import chirp, cwt, ricker
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import pywt
# self lib
from src.utils.dsp_tools import spectrogram_scipy
from src.utils.helpers import direct_to_dir, read_all_tdms_from_folder

data_dir = direct_to_dir(where='yh_laptop_test_data') + 'plb/'
n_channel_data_near = read_all_tdms_from_folder(data_dir)
n_channel_data_near = np.swapaxes(n_channel_data_near, 1, 2)
print('Swap Axis: ', n_channel_data_near.shape)
signal = n_channel_data_near[0, 7, 90000:130000]
width = np.linspace(1, 30, 50)
samp_period = 1/1000000
# CWT--------------------------------
# using scipy
# cwtmatr = cwt(signal, ricker, width)
# using pywt
cwt_out, freq = pywt.cwt(signal, scales=width, wavelet='gaus2')
print(freq)
print(freq.shape)
fig = plt.figure()
fig.subplots_adjust(hspace=0.3)
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
ax1 = fig.add_axes([0.1, 0.7, 0.8, 0.2])
ax2 = fig.add_axes([0.1, 0.2, 0.8, 0.4], sharex=ax1)
colorbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.01])
ax1.set_title('Time Series ')
ax2.set_title('CWT using Gaus1')
ax1.plot(signal)
i = ax2.imshow(cwt_out, extent=[0, 40000, 1, 10], cmap='seismic', aspect='auto')  # extent=[x_start, x_end, y_start, y_end]
plt.colorbar(i, cax=colorbar_ax, orientation='horizontal')

_, _, _, fig2 = spectrogram_scipy(sampled_data=signal, fs=1e6, nperseg=100,
                                  noverlap=0, nfft=500, return_plot=True, mode='magnitude')

plt.show()





# fs = 8000
# T = 10
# t = np.linspace(0, T, T*fs, endpoint=False)
# w = chirp(t, f0=1000, f1=50, t1=10, method='linear')
# print(w.shape)
# cA, cD = pywt.dwt(w, 'db2')
#
# _, _, _, fig1 = spectrogram_scipy(sampled_data=w, fs=fs, nperseg=100, nfft=500, plot_title='Ori',
#                                   noverlap=50, return_plot=True, verbose=True)
# _, _, _, fig2 = spectrogram_scipy(sampled_data=cA, fs=fs, nperseg=100, nfft=500, plot_title='cA',
#                                   noverlap=50, return_plot=True, verbose=True)
# _, _, _, fig3 = spectrogram_scipy(sampled_data=cD, fs=fs, nperseg=100, nfft=500, plot_title='cD',
#                                   noverlap=50, return_plot=True, verbose=True)


# # --------------[FFT on mix of 2 frequency Sine Wave]--------------------
# # Number of sample points
# N = 600
# # sample spacing
# T = 1.0 / 800.0
# x = np.linspace(0.0, N*T, N)
# # 80 Hz and 50Hz sine wave
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# # yf = fft(y)
# # xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# # plt.plot(xf, (2/N) * np.abs(yf[0:N//2]))
# # plt.show()
# print(y.size)
#
#
# # SPECTROGRAM
# f, t, Sxx = spectrogram(y, fs=800)
# plt.pcolormesh(t, f, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# plt.xlabel('Time [Sec]')
# plt.savefig('test')
# plt.show()


