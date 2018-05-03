# ------------------------------------------------------
# Process the Raw AE signals for training-ready
# The Raw signal is sampled at 5MHz, So time btw points = 2e-7 s
# ------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.fftpack import fft
from scipy.signal import spectrogram

# ----------------------[RAW DATA IMPORT]-------------------------
# data files location
path_noleak_2bar = 'E://Experiment 1//pos_0m_2m//No_Leak//2_bar//Set_1//'
path_leak_2bar = 'E://Experiment 1//pos_0m_2m//Leak//2_bar//Set_1//'
# the sensors
data_1 = 'STREAM 06.03.201820180306-143237-780_1_1048500_2096999'  # no leak
data_2 = 'STREAM 06.03.201820180306-143732-581_1_1048500_2096999'  # leak

data_noleak_raw = read_csv('dataset//' + data_1 + '.csv',
                           skiprows=12,
                           names=['Data_Point', 'Vibration_In_Volt'])
data_leak_raw = read_csv('dataset//' + data_2 + '.csv',
                         skiprows=12,
                         names=['Data_Point', 'Vibration_In_Volt'])
print('----------RAW DATA SET---------')
print(data_noleak_raw.head())
print(data_noleak_raw.shape)
print(data_leak_raw.head())
print(data_leak_raw.shape)

# VISUALIZE
# plt.figure(1)
# # no leak plot
# plt.subplot(211)
# plt.plot(data_noleak_raw['Vibration_In_Volt'])
# plt.title('No Leak (Raw Signal)')
# # leak plot
# plt.subplot(212)
# plt.plot(data_leak_raw['Vibration_In_Volt'])
# plt.title('Leak (Raw Signal)')
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
    print('FFT with {} points...'.format(N))
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
        # plot only 0Hz to 500kHz
        plt.xlim((0, 500e3))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.title('Fast Fourier Transform')
        plt.show()

    return y_fft, f_axis


fft_scipy(data_noleak_raw['Vibration_In_Volt'], fs=5e6)


# SPECTROGRAM
f, t, Sxx = spectrogram(data_noleak_raw['Vibration_In_Volt'],
                        fs=5e6,
                        scaling='density',
                        nperseg=100000,  # Now 5kHz is sliced into 100 pcs i.e. 500Hz/pcs
                        noverlap=1000)
print('Time Segment....{}\n'.format(t.size), t)
print('Frequency Segment....{}\n'.format(f.size), f)
print('Power Density....{}\n'.format(Sxx.shape), Sxx)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.xlabel('Time [Sec]')
plt.show()


