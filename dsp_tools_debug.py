# This is juz for testing out small function from dsp_tools before integration on bigger one

import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


# --------------[FFT on mix of 2 frequency Sine Wave]--------------------
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
# 80 Hz and 50Hz sine wave
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# yf = fft(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# plt.plot(xf, (2/N) * np.abs(yf[0:N//2]))
# plt.show()
print(y.size)




# SPECTROGRAM
f, t, Sxx = spectrogram(y, fs=800)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('Time [Sec]')
plt.show()


