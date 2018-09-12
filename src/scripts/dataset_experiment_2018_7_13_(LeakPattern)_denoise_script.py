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
# wavelet
cwt_wavelet = 'gaus1'
dwt_wavelet = 'haar'
dwt_smooth_level = 2
# dwt_dec_level = 5
scale = np.linspace(2, 30, 100)
fs = 1e6

# segmentation
no_of_segment = 10  # 10 is showing a consistent pattern

# DATA READING ---------------------------------------------------------------------------------------------------------
on_pc = False

if on_pc:
    data = AcousticEmissionDataSet_13_7_2018(drive='F')
    n_channel_leak = data.test_data(sensor_dist='near', pressure=1, leak=True)
else:
    data_dir = direct_to_dir(where='yh_laptop_test_data') + '1bar_leak/'
    n_channel_leak = read_all_tdms_from_folder(data_dir)
    n_channel_leak = np.swapaxes(n_channel_leak, 1, 2)
    n_channel_leak = n_channel_leak[0]

# break into a list of segmented points
n_channel_leak = np.split(n_channel_leak, axis=1, indices_or_sections=no_of_segment)
print('Total Segment: ', len(n_channel_leak))
print('Each Segment Dim: ', n_channel_leak[0].shape)

# signal selection
input_signal_1 = n_channel_leak[3][1, :]
input_signal_2 = n_channel_leak[3][7, :]

# DWT DENOISING --------------------------------------------------------------------------------------------------------

print('MAX DEC LEVEL: ', pywt.dwt_max_level(data_len=len(input_signal_1), filter_len=pywt.Wavelet(dwt_wavelet).dec_len))

# smoothing using dwt
input_signal_denoised_1 = dwt_smoothing(x=input_signal_1, wavelet=dwt_wavelet, level=dwt_smooth_level)
input_signal_denoised_2 = dwt_smoothing(x=input_signal_2, wavelet=dwt_wavelet, level=dwt_smooth_level)


fig = plt.figure()
fig.suptitle('{} + smooth using level: {}'.format(dwt_wavelet, dwt_smooth_level))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

ax1.set_title('Sensor 1')
ax1.plot(input_signal_1, c='b', label='Ori')
ax1.plot(input_signal_denoised_1, alpha=0.5, c='r', label='Denoised')
ax1.legend()
ax1.grid(linestyle='dotted')
ax1.set_ylim(bottom=-0.4, top=0.4)


ax2.set_title('Sensor 2')
ax2.plot(input_signal_2, c='b', label='Ori')
ax2.plot(input_signal_denoised_2, alpha=0.5, c='r', label='Denoised')
ax2.legend()
ax2.grid(linestyle='dotted')
ax2.set_ylim(bottom=-0.4, top=0.4)



# # decomposition
# coeff = pywt.wavedec(input_signal_1, wavelet='db4', mode="per", level=5)
#
# # visualize
# fig = plot_multiple_level_decomposition(ori_signal=input_signal_1,
#                                         dec_signal=coeff,
#                                         dec_level=5,
#                                         main_title='Wavelet Decomposition',
#                                         fs=fs)
# plt.show()
