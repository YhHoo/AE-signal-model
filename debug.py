# This is juz for testing out small function bfore integration on bigger one

import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt


# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
# xf =

plt.plot(y)
plt.show()