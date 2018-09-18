'''
This script is finding the segments with leak-caused correlated peaks, or another word or saying in DSP, the event
detection by Weitang.
'''
import numpy as np
import peakutils
from multiprocessing import Pool
import gc
from random import shuffle
from scipy.signal import gausspulse
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import signal
from scipy.signal import correlate as correlate_scipy
from numpy import correlate as correlate_numpy
import pandas as pd
import pywt
import time
from os import listdir
from scipy.signal import correlate
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, LabelBinarizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

# self lib
from src.controlled_dataset.ideal_dataset import white_noise
from src.utils.dsp_tools import spectrogram_scipy, one_dim_xcor_2d_input, dwt_smoothing, one_dim_xcor_1d_input, \
                                detect_ae_event_by_sandwich_sensor, detect_ae_event_by_v_sensor
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.helpers import plot_heatmap_series_in_one_column, read_single_tdms, direct_to_dir, ProgressBarForLoop, \
                              break_balanced_class_into_train_test, ModelLogger, reshape_3d_to_4d_tocategorical, \
                              scatter_plot_3d_vispy, scatter_plot, plot_multiple_timeseries, plot_cwt_with_time_series, \
                              plot_multiple_timeseries_with_roi, lollipop_plot
from src.model_bank.dataset_2018_7_13_leak_localize_model import fc_leak_1bar_max_vec_v1


# CONFIG ---------------------------------------------------------------------------------------------------------------
fs = 1e6
# dwt denoising setting
dwt_wavelet = 'db2'
dwt_smooth_level = 3

# cwt
cwt_wavelet = 'gaus1'
scale = np.linspace(2, 30, 100)

# segmentation
no_of_segment = 1

# roi
roi_width = (int(1.5e3), int(11e3))

# DATA READING AND PRE-PROCESSING --------------------------------------------------------------------------------------
# tdms file reading
folder_path = 'E:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/1 bar/Leak/'
all_file_path = [(folder_path + f) for f in listdir(folder_path) if f.endswith('.tdms')]

# file of interest
foi = all_file_path[0]
n_channel_data_near_leak = read_single_tdms(foi)
n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)

print('Read Data Dim: ', n_channel_data_near_leak.shape)

# SIGNAL PROCESSING ----------------------------------------------------------------------------------------------------
# denoising
denoise = False
if denoise:
    print('DWT Denoising with wavelet: {}, level: {}'.format(dwt_wavelet, dwt_smooth_level))
    temp = []
    # for all channel of sensor
    for channel in n_channel_data_near_leak:
        denoised_signal = dwt_smoothing(x=channel, wavelet=dwt_wavelet, level=dwt_smooth_level)
        temp.append(denoised_signal)
    n_channel_data_near_leak = np.array(temp)

# segment of interest
# soi = 0
# n_channel_data_near_leak = np.split(n_channel_data_near_leak, indices_or_sections=no_of_segment, axis=1)
# signal_1 = n_channel_data_near_leak[soi]


# PEAK DETECTION AND ROI -----------------------------------------------------------------------------------------------
# peak finding for sensor [-2m] and [2m] only
peak_list = []
time_start = time.time()
for channel in n_channel_data_near_leak:
    print('Detecting')
    peak_list.append(peakutils.indexes(channel, thres=0.55, min_dist=5000))
print('Time Taken for peakutils.indexes(): {:.4f}s'.format(time.time()-time_start))

# detect leak caused peaks
# leak_caused_peak = detect_ae_event_by_sandwich_sensor(x1=peak_list[0],
#                                                       x2=peak_list[1],
#                                                       threshold1=1000,
#                                                       threshold2=100000)

leak_caused_peak = detect_ae_event_by_v_sensor(x1=peak_list[0],
                                               x2=peak_list[1],
                                               x3=peak_list[2],
                                               x4=peak_list[3],
                                               threshold_list=[500, 1250, 2500],  # calc by dist*fs/800
                                               threshold_x=10000)
print(leak_caused_peak)
# if the list is empty
if not leak_caused_peak:
    print('No Leak Caused Peak Detected !')
    leak_caused_peak = None


# just to duplicate the list for plot_multiple_timeseries_with_roi() usage
temp = []
for _ in range(8):
    temp.append(leak_caused_peak)

# VISUALIZING ----------------------------------------------------------------------------------------------------------

subplot_titles = np.arange(0, 8, 1)

# fig_timeseries = plot_multiple_timeseries(input=n_channel_data_near_leak,
#                                           subplot_titles=subplot_titles,
#                                           main_title=foi)

fig_timeseries = plot_multiple_timeseries_with_roi(input=n_channel_data_near_leak,
                                                   subplot_titles=subplot_titles,
                                                   main_title=foi,
                                                   peak_center_list=temp,
                                                   roi_width=roi_width)

fig_lollipop = lollipop_plot(x_list=peak_list[:4],
                             y_list=[n_channel_data_near_leak[0][peak_list[0]],
                                     n_channel_data_near_leak[1][peak_list[1]],
                                     n_channel_data_near_leak[2][peak_list[2]],
                                     n_channel_data_near_leak[3][peak_list[3]]],
                             hit_point=leak_caused_peak,
                             label=['Sensor[-3m]', 'Sensor[-2m]', 'Sensor[2m]', 'Sensor[4m]'])


plt.show()

# CWT + XCOR -----------------------------------------------------------------------------------------------------------
# xcor pairing commands - [near] = 0m, 1m,..., 10m
# sensor_pair_near = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]
#
# sample_no = 0
# # for all roi segments
# for p in leak_caused_peak:
#     signal_roi = n_channel_data_near_leak[:, (p-roi_width[0]):(p+roi_width[1])]
#     dist_diff = 0
#     # for all sensor combination
#     for sensor_pair in sensor_pair_near:
#         # method 1
#         xcor_cwt = True
#         if xcor_cwt:
#             xcor_1d = correlate(in1=signal_roi[sensor_pair[0]],
#                                 in2=signal_roi[sensor_pair[1]],
#                                 mode='full',
#                                 method='fft')
#             xcor, _ = pywt.cwt(xcor_1d, scales=scale, wavelet=cwt_wavelet)
#
#         # method 2
#         cwt_xcor = False
#         if cwt_xcor:
#             # CWT
#             pos1_leak_cwt, _ = pywt.cwt(signal_roi[sensor_pair[0]], scales=scale, wavelet=cwt_wavelet)
#             pos2_leak_cwt, _ = pywt.cwt(signal_roi[sensor_pair[1]], scales=scale, wavelet=cwt_wavelet)
#
#             # Xcor
#             xcor, _ = one_dim_xcor_2d_input(input_mat=np.array([pos1_leak_cwt, pos2_leak_cwt]), pair_list=[(0, 1)])
#             xcor = xcor[0]
#
#         fig_title = 'Xcor of CWT of Sensor[{}] and Sensor[{}] -- Dist_Diff[{}m] -- Roi[{}]'.format(sensor_pair[0],
#                                                                                                    sensor_pair[1],
#                                                                                                    dist_diff,
#                                                                                                    sample_no)
#
#         fig_cwt = plot_cwt_with_time_series(time_series=[signal_roi[sensor_pair[0]], signal_roi[sensor_pair[1]]],
#                                             no_of_time_series=2,
#                                             cwt_mat=xcor,
#                                             cwt_scale=scale,
#                                             title=fig_title,
#                                             maxpoint_searching_bound=(roi_width[1]-roi_width[0]-1))
#
#         # only for showing the max point vector
#         show_xcor = False
#         if show_xcor:
#             plt.show()
#
#         # saving the plot ----------------------------------------------------------------------------------------------
#         saving = False
#         if saving:
#             filename = direct_to_dir(where='google_drive') + \
#                        'xcor_cwt_DistDiff[{}m]_roi[{}]'.format(dist_diff, sample_no)
#
#             fig_cwt.savefig(filename)
#             plt.close('all')
#             print('Saving --> Dist_diff: {}m, Roi: {}'.format(dist_diff, sample_no))
#
#         dist_diff += 1
#
#     sample_no += 1














