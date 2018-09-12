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
from src.utils.dsp_tools import spectrogram_scipy, one_dim_xcor_2d_input, dwt_smoothing, one_dim_xcor_1d_input
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.helpers import plot_heatmap_series_in_one_column, read_single_tdms, direct_to_dir, ProgressBarForLoop, \
                              break_balanced_class_into_train_test, ModelLogger, reshape_3d_to_4d_tocategorical, \
                              scatter_plot_3d_vispy, scatter_plot, plot_multiple_timeseries, plot_cwt_with_time_series, \
                              plot_multiple_timeseries_with_peak
from src.model_bank.dataset_2018_7_13_leak_localize_model import fc_leak_1bar_max_vec_v1


# CONFIG ---------------------------------------------------------------------------------------------------------------
fs = 1e6
# dwt denoising setting
dwt_wavelet = 'db2'
dwt_smooth_level = 4

# cwt
cwt_wavelet = 'gaus1'
scale = np.linspace(2, 30, 100)

# segmentation
no_of_segment = 1


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
    temp = []
    # for all channel of sensor
    for channel in n_channel_data_near_leak:
        denoised_signal = dwt_smoothing(x=channel, wavelet=dwt_wavelet, level=dwt_smooth_level)
        temp.append(denoised_signal)
    n_channel_data_near_leak = np.array(temp)

# segment of interest
soi = 0
n_channel_data_near_leak = np.split(n_channel_data_near_leak, indices_or_sections=no_of_segment, axis=1)
signal_1 = n_channel_data_near_leak[soi]

# peak finding
peak_list = []
time_start = time.time()
for channel in signal_1[1:3]:
    peak_list.append(peakutils.indexes(channel, thres=0.7, min_dist=10000))
print('Time Taken for peakutils.indexes(): {:.4f}'.format(time.time()-time_start))

# visualize
# main_title = '{}--Segment_{}'.format(foi, soi)
# subplot_titles = np.arange(0, 8, 1)
# fig_timeseries = plot_multiple_timeseries_with_peak(input=signal_1,
#                                                     subplot_titles=subplot_titles,
#                                                     main_title=main_title,
#                                                     peak_list=peak_list)

l = [1, 6, 9]
m = [12, 4, 8]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
markerline, stemlines, baseline = ax.stem(peak_list[0], signal_1[1][peak_list[0]], '-', label='Sensor[-2m]')
markerline2, stemlines2, baseline2 = ax.stem(peak_list[1], signal_1[2][peak_list[1]], '-', label='Sensor[2m]')

plt.setp(markerline, markerfacecolor='b')
plt.setp(stemlines, color='b', linewidth=1, linestyle='dotted')
plt.setp(baseline, visible=False)

plt.setp(markerline2, markerfacecolor='r', markeredgecolor='r')
plt.setp(stemlines2, color='r', linewidth=1, linestyle='dotted')
plt.setp(baseline2, visible=False)

ax.grid(linestyle='dotted')
ax.legend()
plt.show()






















