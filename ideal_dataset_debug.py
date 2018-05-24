from scipy.signal import filtfilt, butter
import numpy as np
import matplotlib.pyplot as plt
# self defined library
from dsp_tools import spectrogram_scipy

# sine wave sweeping f with noise---------------------
# fs = 10e3
# N = 10e4
# amp = 2 * np.sqrt(2)
# noise_power = 0.01 * fs / 2  # nyquist f x 0.001
# time = np.arange(N) / fs
# freq = np.linspace(1, 2e3, int(N))  # freq sweep range of sine wave
# x = amp * np.sin(2*np.pi*freq*time + 1)
# x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
# noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)


def white_noise(fs, duration, power):
    total_point = int(fs * duration)
    noise = np.random.normal(scale=np.sqrt(power), size=total_point)

    return noise


def butter_bandpass_filtfilt(sampled_data, fs, f_hicut, f_locut, order=5):
    f_nyquist = fs / 2
    low = f_locut / f_nyquist
    high = f_hicut / f_nyquist
    b, a = butter(order, [low, high], btype='band')  # ignore warning
    # using zero phase filter (so no phase shift after filter)
    filtered_signal = filtfilt(b, a, sampled_data)
    return filtered_signal


# simulated leak noise -------------------------------------
# assume it has active freq only in certain range
fs = 10e3
leak_noise = white_noise(fs=10e3, duration=1, power=1)

# bandpass filter
filtered = butter_bandpass_filtfilt(sampled_data=leak_noise, fs=fs, f_hicut=3e3, f_locut=2e3)
# spectrogram_scipy(filtered,
#                   fs=fs,
#                   nperseg=500,
#                   noverlap=100,
#                   verbose=True,
#                   visualize=True,
#                   vis_max_freq_range=fs/2)


# noisy environment -----------------------------------------
env_noise = white_noise(fs=10e3, duration=1, power=0.3)
# spectrogram_scipy(env_noise,
#                   fs=10e3,
#                   nperseg=500,
#                   noverlap=100,
#                   verbose=True,
#                   visualize=True,
#                   vis_max_freq_range=fs/2)


# shifting in leak noise
shift = 


# assert filtered.shape[0] == env_noise[0], 'Filtered and Env_noise must have equal length'
# mix the 2 signals
mix_signal = []
for i in range(filtered.shape[0]):
    mix_signal.append(filtered[i] + env_noise[i])


spectrogram_scipy(mix_signal,
                  fs=10e3,
                  nperseg=500,
                  noverlap=100,
                  verbose=True,
                  visualize=True,
                  vis_max_freq_range=fs/2)
