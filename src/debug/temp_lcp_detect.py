
import numpy as np
import peakutils
import matplotlib.pyplot as plt
import time
from os import listdir
from src.utils.helpers import *
from src.utils.dsp_tools import *

# CONFIG ---------------------------------------------------------------------------------------------------------------
fs = 1e6
sensor_position = ['-4.5m', '-2m', '2m', '5m', '8m', '10m', '17m']

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
roi_width = (int(1e3), int(16e3))

# saving filename
filename_to_save = 'lcp_index_1bar_near_segmentation3_p0.csv'

# DATA READING AND PRE-PROCESSING --------------------------------------------------------------------------------------
# tdms file reading
folder_path = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/1 bar/Leak/'
all_file_path = [(folder_path + f) for f in listdir(folder_path) if f.endswith('.tdms')]
for f in all_file_path:
    print(f)

lcp_list, lcp_ch_list, lcp_filename_list = [], [], []


lcp_count = 0
# for all tdms file
for foi in all_file_path[30:]:
    # take the last filename
    filename = foi.split(sep='/')[-1]
    filename = filename.split(sep='.')[0]

    # start reading fr drive
    n_channel_data_near_leak = read_single_tdms(foi)
    n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)
    print('Swap Dim: ', n_channel_data_near_leak.shape)

    # PEAK DETECTION AND ROI -------------------------------------------------------------------------------------------
    # peak finding for sensor -3m, -2m, 2m, 4m
    peak_list = []
    time_start = time.time()
    print('Peak localizing ...')
    for channel in n_channel_data_near_leak[:4]:
        peak_list.append(peakutils.indexes(channel, thres=thre_peak, min_dist=min_dist_btw_peak))  # **
    print('Time Taken for peakutils.indexes(): {:.4f}s'.format(time.time() - time_start))

    lcp_per_file = detect_ae_event_by_v_sensor(x1=peak_list[0],
                                               x2=peak_list[1],
                                               x3=peak_list[2],
                                               x4=peak_list[3],
                                               threshold_list=thre_event,  # ** calc by dist*fs/800
                                               threshold_x=threshold_x)  # **

    # if the list is empty, skip to next tdms file
    if not lcp_per_file:
        print('No Leak Caused Peak Detected !')
        lcp_per_file = None
        # skip to the nex tdms file
        continue

    lcp_count += len(lcp_per_file)
    print('Leak Caused Peak: ', lcp_per_file)
    print('LCP COUNT: ', lcp_count)



    # MANUAL FILTERING OF NON-LCP DATA ---------------------------------------------------------------------------------
    # lcp_index_temp, lcp_ch_temp = [], []
    # for lcp in lcp_per_file:
    #     print('LCP {}/{} ---------'.format((lcp_per_file.index(lcp) + 1), len(lcp_per_file)))
    #     roi = n_channel_data_near_leak[:, lcp - roi_width[0]:lcp + roi_width[1]]
    #     flag = picklist_multiple_timeseries(input=roi,
    #                                         subplot_titles=['-3m [0]', '-2m [1]', '2m [2]', '4m [3]',
    #                                                         '6m [4]', '8m [5]', '10m [6]', '12m [7]'],
    #                                         main_title='Manual Filtering of Non-LCP (Click [X] to discard)')
    #
    #     print('Ch_0 = ', flag['ch0'])
    #     print('Ch_1 = ', flag['ch1'])
    #     print('Ch_2 = ', flag['ch2'])
    #     print('Ch_3 = ', flag['ch3'])
    #     print('Ch_4 = ', flag['ch4'])
    #     print('Ch_5 = ', flag['ch5'])
    #     print('Ch_6 = ', flag['ch6'])
    #     print('Ch_7 = ', flag['ch7'])
    #     print('Ch_all = ', flag['all'])
    #
    #     # if user wan to discard all channels -> false LCP
    #     if flag['all'] is 1:
    #         print('LCP discarded')
    #         continue
    #     else:
    #         lcp_index_temp.append(lcp)
    #
    #     # store all ch in flag into a list o f8
    #     channel_multiple_hot = []
    #     for i in range(8):
    #         channel_multiple_hot.append(flag['ch{}'.format(i)])
    #
    #     lcp_ch_temp.append(channel_multiple_hot)
    #
    # # if all lcp are discarded by users, proceed to nex tdms
    # if not lcp_index_temp:
    #     print('all LCP discarded !')
    #     continue
    # else:
    #     # store only LCP that survives
    #     # list [filename]
    #     lcp_filename_list.append(filename)
    #     # list of list [filename, LCP]
    #     lcp_list.append(lcp_index_temp)
    #     # list of list of list [filename, LCP, ch no.]
    #     lcp_ch_list.append(lcp_ch_temp)

    # VISUALIZING ------------------------------------------------------------------------------------------------------
    # save lollipop plot
    fig_lollipop = lollipop_plot(x_list=peak_list[:4],
                                 y_list=[n_channel_data_near_leak[0][peak_list[0]],
                                         n_channel_data_near_leak[1][peak_list[1]],
                                         n_channel_data_near_leak[2][peak_list[2]],
                                         n_channel_data_near_leak[3][peak_list[3]]],
                                 hit_point=lcp_per_file,
                                 label=['Sensor[-3m]', 'Sensor[-2m]', 'Sensor[2m]', 'Sensor[4m]'])
    # fig_lollipop_filename = direct_to_dir(where='result') + '{}_lollipop'.format(filename)
    # fig_lollipop.savefig(fig_lollipop_filename)
    # plt.close('all')
    # print('Lollipop fig saved --> ', fig_lollipop_filename)

    temp = []
    for _ in range(8):
        temp.append(lcp_per_file)

    fig_timeseries = plot_multiple_timeseries_with_roi(input=n_channel_data_near_leak,
                                                       subplot_titles=['-4.5', '-2', '2', '4', '8', ],
                                                       main_title='TESTING',
                                                       all_ch_peak=temp,
                                                       roi_width=roi_width)


