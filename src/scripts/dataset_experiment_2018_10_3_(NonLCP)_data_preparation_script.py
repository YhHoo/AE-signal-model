'''
EXTRACTING NON LCP PEAK FROM noleak data
'''
import time
import peakutils
import numpy as np
from src.utils.helpers import *


folder_path = 'F:/Experiment_2_10_2018/-4.5,-2,2,5,8,17,20,23/no_leak/'
all_file_path = [(folder_path + f) for f in listdir(folder_path) if f.endswith('.tdms')]

for f in all_file_path:
    print(f)

n_channel_data_near_leak = read_single_tdms(all_file_path[0])
n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)
soi = n_channel_data_near_leak

time_start = time.time()
print('Peak localizing ...')

peak_ch0, peak_ch1, peak_ch2, peak_ch3, peak_ch4, peak_ch5, peak_ch6, peak_ch7 = [], [], [], [], [], [], [], []
peak_ch0.append(peakutils.indexes(soi, thres=0.5, min_dist=1500))
peak_ch1.append(peakutils.indexes(soi, thres=0.5, min_dist=1500))
peak_ch2.append(peakutils.indexes(soi, thres=0.5, min_dist=1500))
peak_ch3.append(peakutils.indexes(soi, thres=0.5, min_dist=1500))
peak_ch4.append(peakutils.indexes(soi, thres=0.5, min_dist=1500))
peak_ch5.append(peakutils.indexes(soi, thres=0.5, min_dist=1500))
peak_ch6.append(peakutils.indexes(soi, thres=0.5, min_dist=1500))
peak_ch7.append(peakutils.indexes(soi, thres=0.5, min_dist=1500))

peak_all = [peak_ch0, peak_ch1, peak_ch2, peak_ch3, peak_ch4, peak_ch5, peak_ch6, peak_ch7]


fig = plot_multiple_timeseries_with_roi(input=n_channel_data_near_leak,
                                        subplot_titles=['-4.5m [0]', '-2m [1]', '2m [2]', '5m [3]', '8m [4]', '17m [5]',
                                                        '20m [6]', '23m [7]'],
                                        main_title='Non Leak',
                                        all_ch_peak=peak_all)
#
# fig = plot_multiple_timeseries(input=n_channel_data_near_leak,
#                                subplot_titles=['-4.5m [0]', '-2m [1]', '2m [2]', '5m [3]', '8m [4]', '17m [5]',
#                                                '20m [6]', '23m [7]'],
#                                main_title='Non Leak')

plt.show()