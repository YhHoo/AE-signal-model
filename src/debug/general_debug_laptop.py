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
from matplotlib.widgets import Button
from os import listdir
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn import svm, datasets
from scipy.interpolate import interp1d
import keras.backend as K
from keras.utils import plot_model
from itertools import islice

# self lib
from src.controlled_dataset.ideal_dataset import white_noise
from src.utils.dsp_tools import spectrogram_scipy, one_dim_xcor_2d_input, detect_ae_event_by_v_sensor, dwt_smoothing
from src.experiment_dataset.dataset_experiment_2018_5_30 import AcousticEmissionDataSet_30_5_2018
from src.utils.helpers import *
from src.model_bank.dataset_2018_7_13_lcp_recognition_model import lcp_recognition_binary_model_2
from collections import deque
from itertools import islice





# it = iter(l)
# m = tuple(islice(it, 2, None))
#
# for i in it:
#     print(m[1:])
#     print(i)
#     result = m[1:] + (i, )
#
#     print(result)

# l = [1, 2, 3]
#
# x = []
# for _ in range(3):
#     x.append(l)
#
# print(x)



# result = tuple(islice(it, 2))
#
# print(result[1:])


# x = slide_window(seq=l, n=3)
# for i in x:
#     print('here')
#     print(i)


# model = lcp_recognition_binary_model_2()
#
# filename = direct_to_dir(where='result') + 'model.png'
# plot_model(model, to_file=filename, show_shapes=True)

# PLB DSP TECHNIQUES TESTING -------------------------------------------------------------------------------------------
# wave_speed_filename = direct_to_dir(where='desktop') + 'F11_vdisp.csv'
#
# df = pd.read_csv(wave_speed_filename)
#
# print(df.head())
#
# print(df.values.shape)
#
# x = df['frequency'].values
# y = df['wavespeed'].values
#
# f = interp1d(x=x, y=y, kind='cubic')

# x_new = np.linspace(20000, 50000, 1100)
# plt.plot(x_new, f(x_new), '--', x, y, '-')
# plt.legend(['cubic', 'ori'], loc='best')

# plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# tdms_filename = 'C:/Users/YH/Desktop/hooyuheng.masterWork/LCP DATASET OCT 3 1BAR/sample_data/11436_test_0001.tdms'
# lcp_index_filename = 'C:/Users/YH/Desktop/hooyuheng.masterWork/LCP DATASET OCT 3 1BAR/' \
#                      'lcp_index_1bar_near_segmentation4_p0.csv'
# roi_width = (int(1e3), int(5e3))
# offset_all = [1700, 0, 0, 1600, 3400, 6000]
#
# n_channel_data_near_leak = read_single_tdms(tdms_filename)
# n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)
# n_channel_data_near_leak = n_channel_data_near_leak[:-2]
#
# print(n_channel_data_near_leak.shape)
#
# # take the last filename
# filename = tdms_filename.split(sep='/')[-1]
# filename = filename.split(sep='.')[0]
#
# # look up the LCP indexes from LCP_indexes.csv
# print('LCP lookup --> ', lcp_index_filename)
# df = pd.read_csv(lcp_index_filename, index_col=0)
# df_selected = df[df['filename'] == filename]
# print(df_selected)
# lcp_index = df_selected.values[-1, 0]
#
# ch_no = 1
# soi = n_channel_data_near_leak[ch_no, (lcp_index - roi_width[0] + offset_all[ch_no]):
#                                       (lcp_index + roi_width[1] + offset_all[ch_no])]
# plot_title = 'ch1_-1'
# fig_time = plt.figure()
# ax_time = fig_time.add_subplot(1, 1, 1)
# ax_time.plot(soi)
# ax_time.set_title(plot_title)
#
# fig = spectrogram_scipy(sampled_data=soi, fs=1e6, nperseg=1000, noverlap=900, nfft=2000, return_plot=True,
#                         vis_max_freq_range=100e3, verbose=True, plot_title=plot_title)
# plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# all_tdms_file = [(tdms_dir + f) for f in listdir(tdms_dir) if f.endswith('.tdms')]
#
#
# n_channel_data_near_leak = read_single_tdms(all_tdms_file[0])
# n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)
#
# coi = n_channel_data_near_leak[1, :]
#
# # coi_smooth = dwt_smoothing(x=coi, level=4)
#
# peak_list, properties = find_peaks(x=coi, prominence=(0.7, None), wlen=10000)
#
# print(len(peak_list))
#
# print(peak_list)
#
#
# fig = plt.figure(figsize=(7, 3))
# ax1 = fig.add_subplot(1, 1, 1)
# # signal
# ax1.plot(coi)
#
# # peak marker
# ax1.plot(peak_list, coi[peak_list], marker='o', ls='', ms=3, mfc='red')
#
# plt.show()


# ----------------------------------------------------------------------------------------------------------------------
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

