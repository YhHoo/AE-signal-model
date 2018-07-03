'''
This script is to use the
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, ricker
# self lib
from src.experiment_dataset.dataset_experiment_30_5_2018 import AcousticEmissionDataSet_30_5_2018
from src.utils.dsp_tools import one_dim_xcor_freq_band, butter_bandpass_filtfilt
from src.utils.helpers import three_dim_visualizer
from src.utils.plb_analysis_tools import dual_sensor_xcor_with_stft_qiuckview

data = AcousticEmissionDataSet_30_5_2018(drive='E')
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
