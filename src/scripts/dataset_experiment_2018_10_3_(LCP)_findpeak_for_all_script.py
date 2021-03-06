'''
THIS SCRIPT FINDS ALL POSSIBLE LCP INDEXES USING THE PEAKS DETECTION AND detect_ae_event_by_v_sensor(). THEN FOLLOWED BY
MANUAL FILTERING BY USERS,TO REMOVE FALSELY DETECTED LCP. USER CAN DISCARD ONLY FAULTY CHANNELS. PROCEED TO
(LCP)_Data_preparation_script.py TO EXTRAC THE REAL AE DATA.
'''

import numpy as np
import peakutils
import matplotlib.pyplot as plt
import pandas as pd
import time
from os import listdir

# self lib
from src.utils.dsp_tools import dwt_smoothing, detect_ae_event_by_v_sensor
from src.utils.helpers import *

# ** = tunable param
# CONFIG ---------------------------------------------------------------------------------------------------------------
fs = 1e6
sensor_position = ['-4.5m', '-2m', '2m', '5m', '8m', '10m']

# dwt denoising setting
# denoise = False
# dwt_wavelet = 'db2'
# dwt_smooth_level = 3

# peak detect (by peakutils.indexes())
# thre_peak = 0.55  # (in % of the range)
# min_dist_btw_peak = 5000

# ae event detect (by detect_ae_event_by_v_sensor)
thre_event = [550, 3300, 3500]
threshold_x = 10000

# cwt
# cwt_wavelet = 'gaus1'
# scale = np.linspace(2, 30, 100)

# roi
roi_width = (int(1e3), int(16e3))

# segmentation
no_of_segment = 2

# saving filename

filename_to_save = 'lcp_index_1bar_near_segmentation4_p8.csv'

# DATA READING AND PRE-PROCESSING --------------------------------------------------------------------------------------
# tdms file reading
folder_path = 'F:/Experiment_3_10_2018/-4.5, -2, 2, 5, 8, 10, 17 (leak 1bar)/'
all_file_path = [(folder_path + f) for f in listdir(folder_path) if f.endswith('.tdms')]
for f in all_file_path[140:160]:
    print(f)

lcp_list, lcp_ch_list, lcp_filename_list = [], [], []

# for all tdms file
for foi in all_file_path[140:160]:
    # take the last filename
    filename = foi.split(sep='/')[-1]
    filename = filename.split(sep='.')[0]

    # start reading fr drive
    n_channel_data_near_leak = read_single_tdms(foi)
    n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)

    # discard channel 7 and 6
    n_channel_data_near_leak = n_channel_data_near_leak[:-2]

    print('Swap Dim: ', n_channel_data_near_leak.shape)

    # split into 2 to avoid peakdetect affect by super peak
    n_channel_split = np.split(n_channel_data_near_leak, axis=1, indices_or_sections=no_of_segment)

    # SIGNAL PROCESSING ------------------------------------------------------------------------------------------------
    # denoising
    # if denoise:
    #     print('DWT Denoising with wavelet: {}, level: {}'.format(dwt_wavelet, dwt_smooth_level))
    #     temp = []
    #     for channel in n_channel_data_near_leak:
    #         denoised_signal = dwt_smoothing(x=channel, wavelet=dwt_wavelet, level=dwt_smooth_level)
    #         temp.append(denoised_signal)
    #     n_channel_data_near_leak = np.array(temp)

    # PEAK DETECTION AND ROI -------------------------------------------------------------------------------------------
    # peak finding for sensor -4.5m, -2m, 2m, 5m
    time_start = time.time()
    print('Peak localizing ...')

    peak_ch0, peak_ch1, peak_ch2, peak_ch3 = [], [], [], []
    for seg, count in zip(n_channel_split, [0, 1]):
        peak_ch0.append([(x + (count * 2500000)) for x in peakutils.indexes(seg[0], thres=0.5, min_dist=1500)])
        peak_ch1.append([(x + (count * 2500000)) for x in peakutils.indexes(seg[1], thres=0.62, min_dist=5000)])
        peak_ch2.append([(x + (count * 2500000)) for x in peakutils.indexes(seg[2], thres=0.62, min_dist=5000)])
        peak_ch3.append([(x + (count * 2500000)) for x in peakutils.indexes(seg[3], thres=0.5, min_dist=1500)])

    # convert list of list into single list
    peak_ch0 = [i for sublist in peak_ch0 for i in sublist]
    peak_ch1 = [i for sublist in peak_ch1 for i in sublist]
    peak_ch2 = [i for sublist in peak_ch2 for i in sublist]
    peak_ch3 = [i for sublist in peak_ch3 for i in sublist]

    lcp_per_file = detect_ae_event_by_v_sensor(x1=peak_ch0,
                                               x2=peak_ch1,
                                               x3=peak_ch2,
                                               x4=peak_ch3,
                                               n_ch_signal=n_channel_data_near_leak[:4],
                                               threshold_list=thre_event,  # ** calc by dist*fs/800
                                               threshold_x=threshold_x)  # **

    # if the list is empty, skip to next tdms file
    if not lcp_per_file:
        print('No Leak Caused Peak Detected !')
        lcp_per_file = None
        # skip to the nex tdms file
        continue

    print('Leak Caused Peak: ', lcp_per_file)

    # MANUAL FILTERING OF NON-LCP DATA ---------------------------------------------------------------------------------
    lcp_index_temp, lcp_ch_temp = [], []
    for lcp in lcp_per_file:
        print('LCP {}/{} ---------'.format((lcp_per_file.index(lcp) + 1), len(lcp_per_file)))
        roi = n_channel_data_near_leak[:, (lcp - roi_width[0]):(lcp + roi_width[1])]
        flag = picklist_multiple_timeseries_3(input=roi,
                                              subplot_titles=['-4.5m [0]', '-2m [1]', '2m [2]', '5m [3]',
                                                              '8m [4]', '10m [5]'],
                                              main_title='Manual Filtering of Non-LCP (Click [X] to discard)')

        print('Ch_0 = ', flag['ch0'])
        print('Ch_1 = ', flag['ch1'])
        print('Ch_2 = ', flag['ch2'])
        print('Ch_3 = ', flag['ch3'])
        print('Ch_4 = ', flag['ch4'])
        print('Ch_5 = ', flag['ch5'])
        print('Ch_all = ', flag['all'])

        # if user wan to discard all channels -> false LCP
        if flag['all'] is 1:
            print('LCP discarded')
            continue
        else:
            lcp_index_temp.append(lcp)

        # store all ch in flag into a list o f8
        channel_multiple_hot = []
        for i in range(6):
            channel_multiple_hot.append(flag['ch{}'.format(i)])

        lcp_ch_temp.append(channel_multiple_hot)

    # if all lcp are discarded by users, proceed to nex tdms
    if not lcp_index_temp:
        print('all LCP discarded !')
        continue
    else:
        # store only LCP that survives
        # list [filename]
        lcp_filename_list.append(filename)
        # list of list [filename, LCP]
        lcp_list.append(lcp_index_temp)
        # list of list of list [filename, LCP, ch no.]
        lcp_ch_list.append(lcp_ch_temp)

    # VISUALIZING ------------------------------------------------------------------------------------------------------
    # save lollipop plot
    # fig_lollipop = lollipop_plot(x_list=peak_list[:4],
    #                              y_list=[n_channel_data_near_leak[0][peak_list[0]],
    #                                      n_channel_data_near_leak[1][peak_list[1]],
    #                                      n_channel_data_near_leak[2][peak_list[2]],
    #                                      n_channel_data_near_leak[3][peak_list[3]]],
    #                              hit_point=leak_caused_peak,
    #                              label=['Sensor[-3m]', 'Sensor[-2m]', 'Sensor[2m]', 'Sensor[4m]'])
    # fig_lollipop_filename = direct_to_dir(where='result') + '{}_lollipop'.format(filename)
    # fig_lollipop.savefig(fig_lollipop_filename)
    # plt.close('all')
    # print('Lollipop fig saved --> ', fig_lollipop_filename)

    # roi_no = 0
    # # for all roi by lcp
    # for lcp in lcp_per_file:
    #
    #     roi = n_channel_data_near_leak[:, lcp-roi_width[0]:lcp+roi_width[1]]
    #
    #     # save roi time series plot
    #     fig_timeseries = plot_multiple_timeseries(input=roi[:5],
    #                                               subplot_titles=sensor_position[:5],
    #                                               main_title=foi)
    #     fig_timeseries_filename = direct_to_dir(where='result') + '{}_roi[{}]'.format(filename, roi_no)
    #     fig_timeseries.savefig(fig_timeseries_filename)
    #     plt.close('all')
    #     print('Time Series fig saved --> ', fig_timeseries_filename)

        # CWT + XCOR ---------------------------------------------------------------------------------------------------
        # # xcor pairing commands - [near] = 0m, 1m,..., 10m
        # sensor_pair_near = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]
        #
        # dist_diff = 0
        # # for all sensor combination
        # for sensor_pair in sensor_pair_near:
        #     # CWT
        #     pos1_leak_cwt, _ = pywt.cwt(roi[sensor_pair[0]], scales=scale, wavelet=cwt_wavelet)
        #     pos2_leak_cwt, _ = pywt.cwt(roi[sensor_pair[1]], scales=scale, wavelet=cwt_wavelet)
        #
        #     # Xcor
        #     xcor, _ = one_dim_xcor_2d_input(input_mat=np.array([pos1_leak_cwt, pos2_leak_cwt]), pair_list=[(0, 1)])
        #     xcor = xcor[0]
        #     fig_title = 'Xcor_CWT_[{}]x[{}]_Dist_Diff[{}m]_[{}]_Roi[{}]'.format(sensor_pair[0],
        #                                                                         sensor_pair[1],
        #                                                                         dist_diff,
        #                                                                         filename,
        #                                                                         roi_no)
        #     fig_cwt_xcor = plot_cwt_with_time_series(time_series=[roi[sensor_pair[0]], roi[sensor_pair[1]]],
        #                                              no_of_time_series=2,
        #                                              cwt_mat=xcor,
        #                                              cwt_scale=scale,
        #                                              title=fig_title,
        #                                              maxpoint_searching_bound=(roi_width[1]-roi_width[0]-1))
        #     fig_cwt_filename = direct_to_dir(where='result') + 'xcor_cwt_[{}m]_{}_roi[{}]'.format(dist_diff,
        #                                                                                           filename,
        #                                                                                           roi_no)
        #     fig_cwt_xcor.savefig(fig_cwt_filename)
        #
        #     plt.close('all')
        #     print('CWR_XCOR fig saved --> ', fig_cwt_filename)
        #
        #     dist_diff += 1

        # roi_no += 1

# STORE TO CSV ---------------------------------------------------------------------------------------------------------
# converting lcp_list, lcp_filename_list, lcp_ch_list into 1d list of same length, to be placed into df

lcp_col, filename_col, channel_array = [], [], []
# lcp_list is a list of list
for lcp_per_file, f, ch_per_file in zip(lcp_list, lcp_filename_list, lcp_ch_list):
    # labelling all lcp with f
    for p, ch in zip(lcp_per_file, ch_per_file):
        lcp_col.append(p)
        filename_col.append(f)
        channel_array.append(ch)

lcp_df = pd.DataFrame()
lcp_df['lcp'] = lcp_col
lcp_df['filename'] = filename_col

ch_df = pd.DataFrame(data=channel_array, columns=['ch{}'.format(i) for i in range(6)])

final_df = pd.concat([lcp_df, ch_df], axis=1)

df_filename = direct_to_dir(where='result') + filename_to_save
final_df.to_csv(df_filename)

print('data saved --> ', df_filename)
