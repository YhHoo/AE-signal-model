# ------------------------------------------------------
# Process the Raw AE signals for training-ready
# The Raw signal is sampled at 5MHz, So time btw points = 2e-7 s
# ------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.fftpack import fft
from scipy.signal import spectrogram, decimate

# ----------------------[RAW DATA IMPORT]-------------------------
# data files location
path_noleak_2bar = 'E://Experiment 1//pos_0m_2m//No_Leak//2_bar//Set_1//Sensor_1//'
path_leak_2bar = 'E://Experiment 1//pos_0m_2m//Leak//2_bar//Set_1//'
# the sensors
data_1 = 'STREAM 06.03.201820180306-143237-780_1_1048500_2096999'

data_noleak_raw_1 = read_csv(path_noleak_2bar + data_1 + '.csv',
                             skiprows=12,
                             names=['Data_Point', 'Vibration_In_Volt'])

print('----------RAW DATA SET---------')

# ----------------------[DOWN-SAMPLING]-------------------------
# DOWNSAMPLING (sampling freq from 5MHz to 1MHz) q=scaling factor
data_noleak_1_downsample_zerop = decimate(data_noleak_raw_1['Vibration_In_Volt'], q=5, zero_phase=True)
data_noleak_1_downsample = decimate(data_noleak_raw_1['Vibration_In_Volt'], q=5, zero_phase=False)


# VISUALIZE
# plt.figure(1)
# # no leak plot
# plt.subplot(211)
# plt.plot(data_noleak_1_downsample_zerop)
# plt.title('ZERO PHASE')
# # leak plot
# plt.subplot(212)
# plt.plot(data_noleak_1_downsample)
# plt.title('NO ZERO PHASE')
# plt.show()

# ----------------------[SIGNAL TRANSFORMATION]-------------------------


# FAST FOURIER TRANSFORM (FFT)
def fft_scipy(sampled_data=None, fs=1, visualize=True):
    '''
    :param sampled_data: A one dimensional data (Size = N), can be list or series
    :param fs: Sampling frequency
    :param visualize: Plot or not (Boolean)
    :return: amplitude and the frequency spectrum (Size = N // 2)
    '''
    # Sample points and sampling frequency
    N = sampled_data.size
    fs = fs
    # fft
    print('Scipy.FFT on {} points...'.format(N), end='')
    # take only half of the FFT output because it is a reflection
    # take abs because the FFT output is complex
    # divide by N to reduce the amplitude to correct one
    # times 2 to restore the discarded reflection amplitude
    y_fft = fft(sampled_data)
    y_fft = (2.0/N) * np.abs(y_fft[0: N//2])
    # x-axis - only half of N
    f_axis = np.linspace(0.0, fs/2, N//2)

    if visualize:
        plt.plot(f_axis, y_fft)
        # use sci. notation at the x-axis value
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # plot only 0Hz to 300kHz
        plt.xlim((0, 300e3))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.title('Fast Fourier Transform')
        plt.show()
    print('[Done]')
    return y_fft, f_axis


fft1, faxis1 = fft_scipy(data_noleak_1_downsample_zerop, fs=1e6, visualize=False)
fft2, faxis2 = fft_scipy(data_noleak_1_downsample, fs=1e6, visualize=False)


# Bfore Downsample
plt.subplot(211)
plt.plot(faxis1, fft1)
plt.xlim((0, 300e3))
plt.title('ZERO PHASE')
# After Downsample
plt.subplot(212)
plt.plot(faxis2, fft2)
plt.xlim((0, 300e3))
plt.title('NO ZERO PHASE')
plt.show()


# SPECTROGRAM
def spectrogram_scipy(sampled_data=None, fs=1, visualize=True):
    '''
    :param sampled_data: A one dimensional data (Size = N), can be list or series
    :param fs: Sampling frequency
    :param visualize: Plot Spectrogram or not (Boolean)
    :return: time axis, frequency band and the Amplitude in 2D matrix
    '''
    f, t, Sxx = spectrogram(sampled_data,
                            fs=fs,
                            scaling='spectrum',
                            nperseg=100000,  # Now 5kHz is sliced into 100 pcs i.e. 500Hz/pcs
                            noverlap=1000)
    print('----------SPECTROGRAM OUTPUT---------')
    print('Time Segment....{}\n'.format(t.size), t)
    print('Frequency Segment....{}\n'.format(f.size), f)
    print('Power Density....{}\n'.format(Sxx.shape), Sxx)
    if visualize:
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.xlabel('Time [Sec]')
        plt.show()

    return t, f, Sxx




