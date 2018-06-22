import numpy as np
import matplotlib.pyplot as plt
# self lib
from src.utils.dsp_tools import spectrogram_scipy
from src.controlled_dataset.ideal_dataset import sine_wave_continuous

# time axis setting
fs = 100
duration = 5  # tune this for duration
total_point = int(fs * duration)
time_axis = np.linspace(0, duration, total_point)

signal = sine_wave_continuous(time_axis=time_axis, amplitude=1, phase=0, fo=3)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(time_axis, signal, marker='x')
ax1.grid()
plt.show()

_, _, mat = spectrogram_scipy(sampled_data=signal,
                              fs=fs,
                              nperseg=50,
                              noverlap=0,
                              mode='angle',
                              verbose=True,
                              return_plot=True)
print(mat)

_, _, mat = spectrogram_scipy(sampled_data=signal,
                              fs=fs,
                              nperseg=50,
                              noverlap=0,
                              mode='magnitude',
                              verbose=True,
                              return_plot=True)

print(mat)