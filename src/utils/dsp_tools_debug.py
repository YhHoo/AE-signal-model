# This is juz for testing out small function from dsp_tools before integration on bigger one

import numpy as np
from scipy.signal import chirp
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import pywt
# self lib
from src.utils.dsp_tools import spectrogram_scipy


fs = 8000
T = 10
t = np.linspace(0, T, T*fs, endpoint=False)
w = chirp(t, f0=1000, f1=50, t1=10, method='linear')
print(w.shape)
cA, cD = pywt.dwt(w, 'db2')

_, _, _, fig1 = spectrogram_scipy(sampled_data=w, fs=fs, nperseg=100, nfft=500, plot_title='Ori',
                                  noverlap=50, return_plot=True, verbose=True)
_, _, _, fig2 = spectrogram_scipy(sampled_data=cA, fs=fs, nperseg=100, nfft=500, plot_title='cA',
                                  noverlap=50, return_plot=True, verbose=True)
_, _, _, fig3 = spectrogram_scipy(sampled_data=cD, fs=fs, nperseg=100, nfft=500, plot_title='cD',
                                  noverlap=50, return_plot=True, verbose=True)


plt.show()







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


