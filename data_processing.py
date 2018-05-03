# ------------------------------------------------------
# Process the Raw AE signals for training-ready
# The Raw signal is sampled at 5MHz, So time btw points = 2e-7 s
# ------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.fftpack import fft

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
def fft_scipy(sampled_data=None):
    '''
    :param sampled_data: A one dimensional data, can be list or series
    :return: amplitude and the frequency spectrum
    '''
    # Sample points and sampling frequency
    N = sampled_data.size
    fs = 5e6
    # fft
    print('FFT with {} points...'.format(N))
    y_fft = fft(sampled_data)
    f_axis = np.linspace(0.0, fs/2, N//2)
    # take only half of the FFT output because it is a reflection
    # take abs because the FFT output is complex
    # divide by N to reduce the amplitude to correct one
    # times 2 to restore the discarded reflection amplitude
    plt.plot(f_axis, (2.0/N) * np.abs(y_fft[0: N//2]))
    # use sci. notation at the x-axis value
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()


fft_scipy(data_noleak_raw['Vibration_In_Volt'])
