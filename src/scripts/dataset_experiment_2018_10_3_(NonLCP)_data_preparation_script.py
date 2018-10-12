'''
EXTRACTING NON LCP PEAK FROM noleak data
WARNING: CHANNEL 6 (20m) DATA IS BROKEN DUE TO BAD ATTACHMENT OF SENSOR, CHANNEL 6 IS DELETEC AND REPLACED
WITH CHANNEL 7.
'''
import time
import peakutils
import gc
import numpy as np
import csv
from src.utils.helpers import *

# CONFIG ---------------------------------------------------------------------------------------------------------------

non_lcp_dataset_save_filename = direct_to_dir(where='result') + 'non_lcp_1bar.csv'
folder_path = 'F:/Experiment_2_10_2018/-4.5,-2,2,5,8,17,20,23/no_leak/'
all_file_path = [(folder_path + f) for f in listdir(folder_path) if f.endswith('.tdms')]
roi_width = (int(1e3), int(5e3))

total_non_lcp = 0

# FOR ALL TDMS ---------------------------------------------------------------------------------------------------------
# write header to csv
header = np.arange(0, roi_width[0] + roi_width[1], 1).tolist() + ['channel']
with open(non_lcp_dataset_save_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)


for foi in all_file_path:
    n_channel_data_near_leak = read_single_tdms(foi)
    n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)

    # extract tdms name
    filename = foi.split(sep='/')[-1]
    filename = filename.split(sep='.')[0]

    # delete broken channel 6
    n_channel_data_near_leak = np.delete(n_channel_data_near_leak, axis=0, obj=6)
    soi = n_channel_data_near_leak
    print(soi.shape)

    time_start = time.time()
    print('Peak localizing ...')

    ch_no = 0
    for ch in n_channel_data_near_leak:
        peak_per_ch = peakutils.indexes(ch, thres=0.65, min_dist=6000)

        # save all lcp in one ch into csv
        with open(non_lcp_dataset_save_filename, 'a', newline='') as f:
            writer = csv.writer(f)

            # all ch for 1 lcp
            for peak_index in peak_per_ch:
                list_to_csv = ch[(peak_index - roi_width[0]):(peak_index + roi_width[1])].tolist() + [ch_no]
                writer.writerow(list_to_csv)

        ch_no += 1

    print('NonLCP written --> ', non_lcp_dataset_save_filename)

    # peak_ch0 = peakutils.indexes(soi[0], thres=0.65, min_dist=6000)
    # peak_ch1 = peakutils.indexes(soi[1], thres=0.65, min_dist=6000)
    # peak_ch2 = peakutils.indexes(soi[2], thres=0.65, min_dist=6000)
    # peak_ch3 = peakutils.indexes(soi[3], thres=0.65, min_dist=6000)
    # peak_ch4 = peakutils.indexes(soi[4], thres=0.65, min_dist=6000)
    # peak_ch5 = peakutils.indexes(soi[5], thres=0.65, min_dist=6000)
    # peak_ch6 = peakutils.indexes(soi[6], thres=0.65, min_dist=6000)


    # peak_all = [peak_ch0, peak_ch1, peak_ch2, peak_ch3, peak_ch4, peak_ch5, peak_ch6]

    # non_lcp_count = len([lcp for lcp_list in peak_all for lcp in lcp_list])
    # print('Total nonLCP count: ', non_lcp_count)
    # total_non_lcp += non_lcp_count
    #
    # fig = plot_multiple_timeseries_with_roi(input=n_channel_data_near_leak,
    #                                         subplot_titles=['-4.5m [0]', '-2m [1]', '2m [2]', '5m [3]', '8m [4]',
    #                                                         '17m [5]', '23m [7]'],
    #                                         main_title='NoLeak_{}'.format(filename),
    #                                         all_ch_peak=peak_all)
    # fig_save_filename = '{}NonLCP_noleak_{}.png'.format(direct_to_dir(where='result'), filename)
    # plt.savefig(fig_save_filename)
    #
    # plt.close('all')
    #
    # print('plot saved --> ', fig_save_filename)

    gc.collect()

