'''
This script is to use for leak detection on continuous time series AE signal
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, ricker, correlate
# self lib
from src.experiment_dataset.dataset_experiment_2018_5_30 import AcousticEmissionDataSet_30_5_2018
from src.utils.dsp_tools import one_dim_xcor_2d_input, butter_bandpass_filtfilt, spectrogram_scipy
from src.utils.helpers import heatmap_visualizer
from src.utils.plb_analysis_tools import dual_sensor_xcor_with_stft_qiuckview

data = AcousticEmissionDataSet_30_5_2018(drive='F')
n_channel_data_leak = data.leak_noleak_4_sensor(leak=True)
# n_channel_data_noleak = data.leak_noleak_4_sensor(leak=False)
savepath = 'C:/Users/YH/Desktop/hooyuheng.masterWork/MASTER_PAPERWORK/My Practical Work------------' \
           '/Exp30_5_2018/LeakNoLeak Data/STFT+Xcor/10mmLeak_1bar_seg=100e3_nperseg=100/'


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
start = [0.2e6, 0.5e6, 1e6, 1.4e6, 2e6, 2.5e6, 2.8e6, 3.2e6, 3.8e6, 4.3e6, 4.7e6]
start = [int(i) for i in start]
segment_len = int(100e3)
set_no = 0

for s in start:
    # sensor[-1m]
    input_signal_1 = n_channel_data_leak[set_no, s:(s+segment_len), 1]
    # sensor[22m]
    input_signal_2 = n_channel_data_leak[set_no, s:(s+segment_len), 2]
    # bandpass
    filtered_signal_1 = butter_bandpass_filtfilt(sampled_data=input_signal_1, fs=1e6, f_hicut=1e5, f_locut=20e3)
    filtered_signal_2 = butter_bandpass_filtfilt(sampled_data=input_signal_2, fs=1e6, f_hicut=1e5, f_locut=20e3)
    # STFT + XCOR
    fig_time, fig_stft_1, fig_stft_2, fig_xcor = dual_sensor_xcor_with_stft_qiuckview(data_1=filtered_signal_1,
                                                                                      data_2=filtered_signal_2,
                                                                                      stft_mode='magnitude',
                                                                                      stft_nperseg=100,
                                                                                      plot_label=['0m', '-1m', '22m'])

    path_temp = '{}TimeSeries_leak[{}m]_set{}_[{}_{}]'.format(savepath, 0, set_no, s, (s+segment_len))
    fig_time.savefig(path_temp)
    path_temp = '{}Sensor[-1m]_leak[{}m]_set{}_[{}_{}]'.format(savepath, 0, set_no, s, (s+segment_len))
    fig_stft_1.savefig(path_temp)
    path_temp = '{}Sensor[22m]_leak[{}m]_set{}_[{}_{}]'.format(savepath, 0, set_no, s, (s+segment_len))
    fig_stft_2.savefig(path_temp)
    path_temp = '{}XcorMap_leak[{}m]_set{}_[{}_{}]'.format(savepath, 0, set_no, s, (s+segment_len))
    fig_xcor.savefig(path_temp)
    # clear memory
    plt.close('all')

    print('saved')

