'''
THIS SCRIPT IS FOR EXPERIMENTING AND FINDING THE BEST CONFIG OF DENOISING ON RAW AE SIGNAL BFORE SENT TO CWT+XCOR.
'''
import matplotlib.pyplot as plt
import pywt
import numpy as np
# self lib
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.helpers import direct_to_dir, read_all_tdms_from_folder, plot_multiple_level_decomposition
from src.utils.dsp_tools import dwt_smoothing

# CONFIG --------------------------------------------------------------------------------------------------------------
# CWT
m_wavelet = 'gaus1'
scale = np.linspace(2, 30, 100)
fs = 1e6
# for splitting the 5M points signal
no_of_segment = 50

# DATA READING ---------------------------------------------------------------------------------------------------------
on_pc = True

if on_pc:
    data = AcousticEmissionDataSet_13_7_2018(drive='F')
    n_channel_leak = data.test_data(sensor_dist='near', pressure=1, leak=True)
else:
    data_dir = direct_to_dir(where='yh_laptop_test_data') + '1bar_leak/'
    n_channel_leak = read_all_tdms_from_folder(data_dir)
    n_channel_leak = np.swapaxes(n_channel_leak, 1, 2)
    n_channel_leak = n_channel_leak[0]

# processing


# break into a list of segmented points
n_channel_leak = np.split(n_channel_leak, axis=1, indices_or_sections=no_of_segment)
print('Total Segment: ', len(n_channel_leak))
print('Each Segment Dim: ', n_channel_leak[0].shape)

# DWT DENOISING --------------------------------------------------------------------------------------------------------
input_signal = n_channel_leak[0][1, :]

w = pywt.Wavelet('db4')
dec_level = 5
print('MAX DEC LEVEL: ', pywt.dwt_max_level(data_len=len(input_signal), filter_len=w.dec_len))

# smoothing using dwt
input_signal_denoised = dwt_smoothing(x=input_signal, wavelet='db4', level=3)
print(len(input_signal_denoised))

plt.plot(input_signal, c='b', label='Ori')
plt.plot(input_signal_denoised, alpha=0.5, c='r', label='Denoised')
plt.legend()
plt.grid(linestyle='dotted')

# # decomposition
# coeff = pywt.wavedec(input_signal, wavelet='db4', mode="per", level=dec_level)
#
# # visualize
# fig = plot_multiple_level_decomposition(ori_signal=input_signal,
#                                         dec_signal=coeff,
#                                         dec_level=dec_level,
#                                         main_title='Wavelet Decomposition',
#                                         fs=fs)
plt.show()
