from scipy.signal import filtfilt, butter
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
# self defined library
from dsp_tools import spectrogram_scipy, butter_bandpass_filtfilt, fft_scipy
from ideal_dataset import white_noise, sine_wave_continuous


# time axis setting
fs = 5000
duration = 4  # tune this for duration
total_point = int(fs * duration)
time_axis = np.linspace(0, duration, total_point)

# sine wave
sine = sine_wave_continuous(time_axis=time_axis, amplitude=10, fo=50, phase=0)
sine2 = sine_wave_continuous(time_axis=time_axis, amplitude=10, fo=50, phase=1)
# noise
noise = white_noise(time_axis=time_axis, power=10)


t, f, Sxx1 = spectrogram_scipy(sine,
                               fs=fs,
                               nperseg=1000,
                               noverlap=500,
                               mode='angle',
                               visualize=False,
                               verbose=True)

plt.plot(t, Sxx1[10], label='Phase 0')
t, f, Sxx2 = spectrogram_scipy(sine2,
                               fs=fs,
                               nperseg=1000,
                               noverlap=500,
                               mode='angle',
                               visualize=False,
                               verbose=True)

plt.plot(t, Sxx2[10], label='Phase 1')
# plt.plot(t, Sxx[30], label=f[30])
# plt.plot(t, Sxx[100], label=f[100])
# plt.plot(t, Sxx[300], label=f[300])
# plt.plot(t, Sxx[450], label=f[450])

plt.legend()
plt.show()

diff = []
for i in range(Sxx1.shape[1]):
    diff.append(Sxx2[10, i] - Sxx1[10, i])

print(diff)
plt.plot(diff)
plt.show()

