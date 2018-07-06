'''
This script is to use for leak detection on continuous time series AE signal
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, ricker, correlate
# self lib
from src.experiment_dataset.dataset_experiment_30_5_2018 import AcousticEmissionDataSet_30_5_2018
from src.utils.dsp_tools import one_dim_xcor_2d_input, butter_bandpass_filtfilt, spectrogram_scipy
from src.utils.helpers import three_dim_visualizer
from src.utils.plb_analysis_tools import dual_sensor_xcor_with_stft_qiuckview

data = AcousticEmissionDataSet_30_5_2018(drive='F')
n_channel_data_leak = data.leak_noleak_4_sensor(leak=True)
n_channel_data_noleak = data.leak_noleak_4_sensor(leak=False)
set_no = [0, 1, 2]
savepath = 'C:/Users/YH/PycharmProjects/AE-signal-model/result/'

# ----------------------[Visualize in Time and Saving]----------------------------
visualize_in_time = False
if visualize_in_time:
    for i in set_no:
        fig_time_series = plt.figure(figsize=(10, 12))
        fig_time_series.suptitle('[Set {}] Sensors data in Time Series, Leak @ 0m'.format(i))
        # create subplots
        ax_time_sensor_1 = fig_time_series.add_subplot(4, 2, 1)
        ax_time_sensor_2 = fig_time_series.add_subplot(4, 2, 3)
        ax_time_sensor_3 = fig_time_series.add_subplot(4, 2, 5)
        ax_time_sensor_4 = fig_time_series.add_subplot(4, 2, 7)
        ax_time_sensor_5 = fig_time_series.add_subplot(4, 2, 2)
        ax_time_sensor_6 = fig_time_series.add_subplot(4, 2, 4)
        ax_time_sensor_7 = fig_time_series.add_subplot(4, 2, 6)
        ax_time_sensor_8 = fig_time_series.add_subplot(4, 2, 8)
        # naming subplots
        ax_time_sensor_1.set_title('Sensor[-2m] No Leak')
        ax_time_sensor_2.set_title('Sensor[-1m] No Leak')
        ax_time_sensor_3.set_title('Sensor[22m] No Leak')
        ax_time_sensor_4.set_title('Sensor[23m] No Leak')
        ax_time_sensor_5.set_title('Sensor[-2m] Leak')
        ax_time_sensor_6.set_title('Sensor[-1m] Leak')
        ax_time_sensor_7.set_title('Sensor[22m] Leak')
        ax_time_sensor_8.set_title('Sensor[23m] Leak')
        # plot
        ax_time_sensor_1.plot(n_channel_data_noleak[i, :, 0])
        ax_time_sensor_2.plot(n_channel_data_noleak[i, :, 1])
        ax_time_sensor_3.plot(n_channel_data_noleak[i, :, 2])
        ax_time_sensor_4.plot(n_channel_data_noleak[i, :, 3])
        ax_time_sensor_5.plot(n_channel_data_leak[i, :, 0])
        ax_time_sensor_6.plot(n_channel_data_leak[i, :, 1])
        ax_time_sensor_7.plot(n_channel_data_leak[i, :, 2])
        ax_time_sensor_8.plot(n_channel_data_leak[i, :, 3])
        # setting
        plt.subplots_adjust(hspace=0.6)
        # path = '{}Set{}.png'.format(savepath, i)
        # plt.savefig(path)
        # plt.close()
        # print('saved !')
        plt.show()


# ----------------------[STFT + XCOR]----------------------------
# signal segmentation
start = int(100e3)
set_no = 2
input_signal_1 = n_channel_data_leak[set_no, start:start+100000, 1]
input_signal_2 = n_channel_data_leak[set_no, start:start+100000, 2]
# quick peek at the signal
visualize = False
if visualize:
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(input_signal_1)
    ax2.plot(input_signal_2)

    plt.show()
# bandpass
filtered_signal_1 = butter_bandpass_filtfilt(sampled_data=input_signal_1, fs=1e6, f_hicut=1e5, f_locut=20e3)
filtered_signal_2 = butter_bandpass_filtfilt(sampled_data=input_signal_2, fs=1e6, f_hicut=1e5, f_locut=20e3)
# print(filtered_signal_1.shape)
# print(filtered_signal_2.shape)
# xcor_map_in_time = correlate(filtered_signal_1,
#                              filtered_signal_2,
#                              'full',
#                              method='fft')
#
# widths_2 = np.arange(1, 20, 0.5)
# cwtmatr_1 = cwt(xcor_map_in_time, ricker, widths_2)
# fig_cwt_1 = three_dim_visualizer(x_axis=np.arange(1, cwtmatr_1.shape[1] + 1, 1),
#                                  y_axis=widths_2,
#                                  zxx=cwtmatr_1,
#                                  label=['time steps', 'Wavelet Width', 'CWT Coefficient'],
#                                  output='2d',
#                                  title='CWT Coef of Xcor Map')
# plt.show()


# STFT + XCOR
fig_time, fig_stft_1, fig_stft_2, fig_xcor = dual_sensor_xcor_with_stft_qiuckview(data_1=filtered_signal_1,
                                                                                  data_2=filtered_signal_2,
                                                                                  stft_mode='magnitude',
                                                                                  stft_nperseg=100,
                                                                                  plot_label=['0m', '-1m', '22m'])
plt.show()

# path_temp = '{}Sensor[-1m]_leak[{}m]_set{}'.format(savepath, 0, set_no)
# fig_stft_1.savefig(path_temp)
# path_temp = '{}Sensor[22m]_leak[{}m]_set{}'.format(savepath, 0, set_no)
# fig_stft_2.savefig(path_temp)
# path_temp = '{}XcorMap_leak[{}m]_set{}'.format(savepath, 0, set_no)
# fig_xcor.savefig(path_temp)
# print('saved')
# plt.close('all')
