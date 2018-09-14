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
                              scatter_plot_3d_vispy, scatter_plot, plot_multiple_timeseries, plot_cwt_with_time_series,\
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

for foi in all_file_path:
    n_channel_data_near_leak = read_single_tdms(foi)
    n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)

    print('Read Data Dim: ', n_channel_data_near_leak.shape)

    # SIGNAL PROCESSING ------------------------------------------------------------------------------------------------
    # denoising
    denoise = False
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
    for channel in n_channel_data_near_leak:
        peak_list.append(peakutils.indexes(channel, thres=0.55, min_dist=5000))
    print('Time Taken for peakutils.indexes(): {:.4f}s'.format(time.time() - time_start))

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

    # VISUALIZING ------------------------------------------------------------------------------------------------------

    subplot_titles = ['-3m', '-2m', '2m', '4m', '6m', '8m', '10m', '12m']

    fig_timeseries = plot_multiple_timeseries(input=n_channel_data_near_leak[:, ],
                                              subplot_titles=subplot_titles,
                                              main_title=foi)

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
                                 test_point=leak_caused_peak,
                                 label=['Sensor[-3m]', 'Sensor[-2m]', 'Sensor[2m]', 'Sensor[4m]'])

    plt.show()
