import numpy as np
import peakutils
import matplotlib.pyplot as plt
import pandas as pd
import time
from os import listdir

# self lib
from src.utils.dsp_tools import dwt_smoothing, detect_ae_event_by_v_sensor
from src.utils.helpers import read_single_tdms, direct_to_dir, plot_multiple_timeseries, lollipop_plot

# ** = tunable param
# CONFIG ---------------------------------------------------------------------------------------------------------------
fs = 1e6
sensor_position = ['-3m', '-2m', '2m', '4m', '6m', '8m', '10m', '12m']

# dwt denoising setting
denoise = False
dwt_wavelet = 'db2'
dwt_smooth_level = 3

# peak detect (by peakutils.indexes())
thre_peak = 0.55  # (in % of the range)
min_dist_btw_peak = 5000

# ae event detect (by detect_ae_event_by_v_sensor)
thre_event = [400, 1250, 2500]
threshold_x = 10000

# cwt
cwt_wavelet = 'gaus1'
scale = np.linspace(2, 30, 100)

# roi
roi_width = (int(1e3), int(5e3))

# DATA READING AND PRE-PROCESSING --------------------------------------------------------------------------------------
# tdms file reading
folder_path = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/2 bar/Leak/'
all_file_path = [(folder_path + f) for f in listdir(folder_path) if f.endswith('.tdms')]
for f in all_file_path:
    print(f)

lcp_list, lcp_filename_list = [], []

# for all tdms file
for foi in all_file_path:
    # take the last filename
    filename = foi.split(sep='/')[-1]
    filename = filename.split(sep='.')[0]

    # start reading fr drive
    n_channel_data_near_leak = read_single_tdms(foi)
    n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)
    print('Read Data Dim: ', n_channel_data_near_leak.shape)

    # SIGNAL PROCESSING ------------------------------------------------------------------------------------------------
    # denoising
    if denoise:
        print('DWT Denoising with wavelet: {}, level: {}'.format(dwt_wavelet, dwt_smooth_level))
        temp = []
        for channel in n_channel_data_near_leak:
            denoised_signal = dwt_smoothing(x=channel, wavelet=dwt_wavelet, level=dwt_smooth_level)
            temp.append(denoised_signal)
        n_channel_data_near_leak = np.array(temp)

    # PEAK DETECTION AND ROI -------------------------------------------------------------------------------------------
    # peak finding for sensor [-3m] to [4m] only
    peak_list = []
    time_start = time.time()
    print('Peak localizing ...')
    for channel in n_channel_data_near_leak[:4]:
        peak_list.append(peakutils.indexes(channel, thres=thre_peak, min_dist=min_dist_btw_peak))  # **
    print('Time Taken for peakutils.indexes(): {:.4f}s'.format(time.time() - time_start))

    leak_caused_peak = detect_ae_event_by_v_sensor(x1=peak_list[0],
                                                   x2=peak_list[1],
                                                   x3=peak_list[2],
                                                   x4=peak_list[3],
                                                   threshold_list=thre_event,  # ** calc by dist*fs/800
                                                   threshold_x=threshold_x)  # **
    print('Leak Caused Peak: ', leak_caused_peak)
    # if the list is empty
    if not leak_caused_peak:
        print('No Leak Caused Peak Detected !')
        leak_caused_peak = None
        # skip to the nex tdms file
        continue
    else:
        # to be stored into csv
        lcp_list.append(leak_caused_peak)
        lcp_filename_list.append(filename)

    # VISUALIZING ------------------------------------------------------------------------------------------------------
    # save lollipop plot
    fig_lollipop = lollipop_plot(x_list=peak_list[:4],
                                 y_list=[n_channel_data_near_leak[0][peak_list[0]],
                                         n_channel_data_near_leak[1][peak_list[1]],
                                         n_channel_data_near_leak[2][peak_list[2]],
                                         n_channel_data_near_leak[3][peak_list[3]]],
                                 hit_point=leak_caused_peak,
                                 label=['Sensor[-3m]', 'Sensor[-2m]', 'Sensor[2m]', 'Sensor[4m]'])
    fig_lollipop_filename = direct_to_dir(where='result') + '{}_lollipop'.format(filename)
    fig_lollipop.savefig(fig_lollipop_filename)
    plt.close('all')
    print('Lollipop fig saved --> ', fig_lollipop_filename)

    roi_no = 0
    # for all roi by lcp
    for lcp in leak_caused_peak:

        roi = n_channel_data_near_leak[:, lcp-roi_width[0]:lcp+roi_width[1]]

        # save roi time series plot
        fig_timeseries = plot_multiple_timeseries(input=roi[:5],
                                                  subplot_titles=sensor_position[:5],
                                                  main_title=foi)
        fig_timeseries_filename = direct_to_dir(where='result') + '{}_roi[{}]'.format(filename, roi_no)
        fig_timeseries.savefig(fig_timeseries_filename)
        plt.close('all')
        print('Time Series fig saved --> ', fig_timeseries_filename)

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

        roi_no += 1

# STORE TO CSV ---------------------------------------------------------------------------------------------------------
lcp_col, filename_col = [], []
for lcp, f in zip(lcp_list, lcp_filename_list):
    for p in lcp:
        lcp_col.append(p)
        filename_col.append(f)

lcp_df = pd.DataFrame()
lcp_df['lcp'] = lcp_col
lcp_df['filename'] = filename_col

df_filename = direct_to_dir(where='result') + 'lcp_1bar_near.csv'
lcp_df.to_csv(df_filename)

print('data saved --> ', df_filename)
