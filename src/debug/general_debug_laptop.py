import itertools
import time
import types
import numpy as np
import pywt
import gc
from multiprocessing import Pool
from random import shuffle
from scipy.signal import gausspulse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import signal
from scipy.signal import correlate as correlate_scipy
from numpy import correlate as correlate_numpy
import pandas as pd
from os import listdir
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
# self lib
from src.controlled_dataset.ideal_dataset import white_noise
from src.utils.dsp_tools import spectrogram_scipy, one_dim_xcor_2d_input, detect_ae_event_by_v_sensor
from src.experiment_dataset.dataset_experiment_2018_5_30 import AcousticEmissionDataSet_30_5_2018
from src.utils.helpers import plot_heatmap_series_in_one_column, read_single_tdms, direct_to_dir, ProgressBarForLoop, \
                              break_balanced_class_into_train_test, ModelLogger, reshape_3d_to_4d_tocategorical, \
                              scatter_plot, scatter_plot_3d_vispy, lollipop_plot
from src.model_bank.dataset_2018_7_13_leak_localize_model import fc_leak_1bar_max_vec_v1


# x = [[1, 25, 67], [2, 24, 70], [3, 20, 58]]
# y = [[1, 1.04, 1.2], [1, 2, 1], [0.7, 1, 1]]
# label = ['one', 'two', 'three']
#
# fig = lollipop_plot(x_list=x, y_list=y, label=label, test_point=[2, 50, 80])
# plt.show()

# x = np.arange(0, 10, 1)
# x2 = np.arange(0, 10, 2)
# y = [5, 10]
# for i in y:
#     print('Start First Loop')
#     for j in x:
#         print('Start Second Loop')
#         print(j)
#         if j > i:
#             print('Oppssss')
#             break
#     for k in x2:
#         print('Start Third Loop')
#         print(k)
#         if k > i:
#             print('Oppssss')
#             break

l = [1]
if not l:
    print('empty list')
# ax = Axes3D(fig)

# ax.scatter(x, y, z, cmap=cm.rainbow, c=[0, 0.5, 0.9])

# plt.show()


# label_to_take = [1, 3]
# data_selected = data_df.loc[data_df['label'].isin(label_to_take)]
# print(data_selected)

# ----------------------------------------------------------------------------------
# max_vec_list = np.linspace(0, 10, 44).reshape((11, 2, 2))
# print(max_vec_list.shape)
#
# all_class = {}
# for i in range(0, 11, 1):
#     all_class['class_[{}]'.format(i)] = []
#     all_class['class_[{}]'.format(i)].append(max_vec_list[i])
#
# # just to display the dict full dim
# temp = []
# for _, value in all_class.items():
#     temp.append(value[0])
# temp = np.array(temp)
# # print(temp)
# print('all_class dim: ', temp.shape)
#
# dataset = []
# label = []
# for i in range(0, 11, 1):
#     for sample in all_class['class_[{}]'.format(i)][0]:
#         print(sample)
#         dataset.append(sample)
#         label.append(i)
#
# # convert to array
# dataset = np.array(dataset)
# label = np.array(label)
# print('Dataset Dim: ', dataset.shape)
# print('Label Dim: ', label.shape)
#
# # save to csv
# label = label.reshape((-1, 1))
# all_in_one = np.concatenate([dataset, label], axis=1)
# print(all_in_one.shape)
# ----------------------------------------------------------------------------------

