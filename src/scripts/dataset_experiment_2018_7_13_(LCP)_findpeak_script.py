'''
This script is finding the segments with leak-caused correlated peaks, or another word or saying in DSP, the event
detection by Weitang.
'''
import numpy as np
import peakutils
import matplotlib.pyplot as plt
import time
from os import listdir
import gc
from scipy import signal

# self lib
from src.controlled_dataset.ideal_dataset import white_noise
from src.utils.dsp_tools import spectrogram_scipy, one_dim_xcor_2d_input, dwt_smoothing, one_dim_xcor_1d_input, \
                                detect_ae_event_by_sandwich_sensor, detect_ae_event_by_v_sensor
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.helpers import *
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
no_of_segment = 2

# roi
roi_width = (int(1e3), int(16e3))

# channel naming
label = ['-4.5m', '-2m', '2m', '5m', '8m', '10m', '17m']

# DATA READING AND PRE-PROCESSING --------------------------------------------------------------------------------------
# tdms file reading
folder_path = 'E:/Experiment_3_10_2018/-4.5, -2, 2, 5, 8, 10, 17 (leak 1bar)/'
all_file_path = [(folder_path + f) for f in listdir(folder_path) if f.endswith('.tdms')]


n_channel_data_near_leak = read_single_tdms(all_file_path[10])
n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)

# discard channel 7
n_channel_data_near_leak = n_channel_data_near_leak[:-1]

print('Read Data Dim: ', n_channel_data_near_leak.shape)

# split 5M into half
n_channel_split = np.split(n_channel_data_near_leak, axis=1, indices_or_sections=no_of_segment)

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


# PEAK DETECTION AND ROI -----------------------------------------------------------------------------------------------

time_start = time.time()

# detect peak by segments, to avoid affects by super peaks
peak_ch0, peak_ch1, peak_ch2, peak_ch3 = [], [], [], []
for seg, count in zip(n_channel_split, [0, 1]):
    peak_ch0.append([(x + (count*2500000)) for x in peakutils.indexes(seg[0], thres=0.5, min_dist=1500)])
    peak_ch1.append([(x + (count*2500000)) for x in peakutils.indexes(seg[1], thres=0.58, min_dist=5000)])
    peak_ch2.append([(x + (count*2500000)) for x in peakutils.indexes(seg[2], thres=0.58, min_dist=5000)])
    peak_ch3.append([(x + (count*2500000)) for x in peakutils.indexes(seg[3], thres=0.5, min_dist=1500)])

# convert list of list into single list
peak_ch0 = [i for sublist in peak_ch0 for i in sublist]
peak_ch1 = [i for sublist in peak_ch1 for i in sublist]
peak_ch2 = [i for sublist in peak_ch2 for i in sublist]
peak_ch3 = [i for sublist in peak_ch3 for i in sublist]
peak_list = [peak_ch0, peak_ch1, peak_ch2, peak_ch3]

# USING peakutils
# peak_list = []
# time_start = time.time()
# # channel 0
# print('Peak Detecting')
# peak_list.append(peakutils.indexes(n_channel_data_near_leak[0], thres=0.5, min_dist=1500))
# # channel 1
# print('Peak Detecting')
# peak_list.append(peakutils.indexes(n_channel_data_near_leak[1], thres=0.6, min_dist=5000))
# # channel 2
# print('Peak Detecting')
# peak_list.append(peakutils.indexes(n_channel_data_near_leak[2], thres=0.6, min_dist=5000))
# # channel 3
# print('Peak Detecting')
# peak_list.append(peakutils.indexes(n_channel_data_near_leak[3], thres=0.5, min_dist=1500))

# for channel in n_channel_data_near_leak:
#     print('Peak Detecting')
#     peak_list.append(peakutils.indexes(channel, thres=0.5, min_dist=5000))
print('Time Taken for peakutils.indexes(): {:.4f}s'.format(time.time()-time_start))

# USING Scipy
# peak_list = []
# time_start = time.time()
# for channel in n_channel_data_near_leak:
#     print('Peak Detecting')
#     peak, _ = signal.find_peaks(x=channel, distance=5000, prominence=(0.1, None))
#     peak_list.append(peak)
# print('Time Taken for peakutils.indexes(): {:.4f}s'.format(time.time()-time_start))

leak_caused_peak = detect_ae_event_by_v_sensor(x1=peak_list[0],
                                               x2=peak_list[1],
                                               x3=peak_list[2],
                                               x4=peak_list[3],
                                               n_ch_signal=n_channel_data_near_leak[:4],
                                               threshold_list=[550, 3300, 3500],  # calc by dist*fs/800
                                               threshold_x=10000)
print(leak_caused_peak)

# if the list is empty
if not leak_caused_peak:
    print('No Leak Caused Peak Detected !')
    leak_caused_peak = None

# gc.collect()

# just to duplicate the list for plot_multiple_timeseries_with_roi() usage
# temp = []
# for _ in range(8):
#     temp.append(leak_caused_peak)

# VISUALIZING ----------------------------------------------------------------------------------------------------------

# fig_timeseries = plot_multiple_timeseries(input=n_channel_data_near_leak,
#                                           subplot_titles=subplot_titles,
#                                           main_title=foi)

# fig_timeseries = plot_multiple_timeseries_with_roi(input=n_channel_data_near_leak[:4],
#                                                    subplot_titles=label[:4],
#                                                    main_title=foi,
#                                                    all_ch_peak=peak_list[:4],
#                                                    lcp_list=leak_caused_peak,
#                                                    roi_width=roi_width)

# fig_lollipop = lollipop_plot(x_list=peak_list[:4],
#                              y_list=[n_channel_data_near_leak[0][peak_list[0]],
#                                      n_channel_data_near_leak[1][peak_list[1]],
#                                      n_channel_data_near_leak[2][peak_list[2]],
#                                      n_channel_data_near_leak[3][peak_list[3]]],
#                              hit_point=leak_caused_peak,
#                              label=['Sensor[-4.5m]', 'Sensor[-2m]', 'Sensor[2m]', 'Sensor[5m]'])
for lcp in leak_caused_peak:
    roi = n_channel_data_near_leak[:, (lcp - roi_width[0]):(lcp + roi_width[1])]
    flag = picklist_multiple_timeseries(input=roi,
                                        subplot_titles=['-4.5m [0]', '-2m [1]', '2m [2]', '5m [3]',
                                                        '8m [4]', '10m [5]', '17m [6]'],
                                        main_title='Manual Filtering of Non-LCP (Click [X] to discard)')

    print('Ch_0 = ', flag['ch0'])
    print('Ch_1 = ', flag['ch1'])
    print('Ch_2 = ', flag['ch2'])
    print('Ch_3 = ', flag['ch3'])
    print('Ch_4 = ', flag['ch4'])
    print('Ch_5 = ', flag['ch5'])
    print('Ch_6 = ', flag['ch6'])
    print('Ch_all = ', flag['all'])




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














