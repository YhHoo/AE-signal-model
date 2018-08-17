import itertools
import numpy as np
import pywt
import gc
from multiprocessing import Pool
# from random import shuffle
# from scipy.signal import gausspulse
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import AxesGrid
# from scipy import signal
# from scipy.signal import correlate as correlate_scipy
# from numpy import correlate as correlate_numpy
# import pandas as pd
# from os import listdir
# from keras.utils import to_categorical
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
# # self lib
# from src.controlled_dataset.ideal_dataset import white_noise
# from src.utils.dsp_tools import spectrogram_scipy, one_dim_xcor_2d_input
# from src.experiment_dataset.dataset_experiment_2018_5_30 import AcousticEmissionDataSet_30_5_2018
# from src.utils.helpers import plot_heatmap_series_in_one_column, read_single_tdms, direct_to_dir, ProgressBarForLoop, \
#                               break_into_train_test, ModelLogger, reshape_3d_to_4d_tocategorical
# from src.model_bank.dataset_2018_7_13_leak_model import fc_leak_1bar_max_vec_v1

l = []
x_in = np.arange(0, 10, 1)
a = 12


def addon(a, x):
    return a*x*x


if __name__ == '__main__':
    pool = Pool()
    result = pool.apply(addon, args=(a, x_in))
    print(result)





# dist_diff = 0
# # for all sensor combination
# sensor_pair_near = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]
#
# for sensor_pair in sensor_pair_near:
#     segment_no = 0
#     pb = ProgressBarForLoop(title='CWT+Xcor using {}'.format(sensor_pair), end=len(n_channel_leak))
#     # for all segmented signals
#     for segment in n_channel_leak:
#         pos1_leak_cwt, _ = pywt.cwt(segment[sensor_pair[0]], scales=scale, wavelet=m_wavelet,
#                                     sampling_period=1 / fs)
#         pos2_leak_cwt, _ = pywt.cwt(segment[sensor_pair[1]], scales=scale, wavelet=m_wavelet,
#                                     sampling_period=1 / fs)
#
#         # xcor for every pair of cwt
#         xcor, _ = one_dim_xcor_2d_input(input_mat=np.array([pos1_leak_cwt, pos2_leak_cwt]),
#                                         pair_list=[(0, 1)])
#         xcor = xcor[0]
#
#         # midpoint in xcor
#         mid = xcor.shape[1] // 2 + 1
#
#         max_xcor_vector = []
#         # for every row of xcor, find max point index
#
#         for row in xcor:
#             max_along_x = np.argmax(row)
#             max_xcor_vector.append(max_along_x - mid)
#         # store all feature vector for same class
#         all_class['class_[{}]'.format(dist_diff)].append(max_xcor_vector)
#
#         pb.update(now=segment_no)
#         segment_no += 1
#     dist_diff += 1
#     pb.destroy()
# # just to display the dict full dim
# l = []
# for _, value in all_class.items():
#     l.append(value)
# l = np.array(l)
# print(l.shape)
#
# # free up memory for unwanted variable
# pos1_leak_cwt, pos2_leak_cwt, n_channel_data_near_leak, l = None, None, None, None
# gc.collect()
#

