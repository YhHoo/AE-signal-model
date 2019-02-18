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
import os
from scipy.signal import decimate
# self lib
from src.controlled_dataset.ideal_dataset import white_noise
from src.utils.dsp_tools import fft_scipy, spectrogram_scipy, one_dim_xcor_2d_input, detect_ae_event_by_v_sensor, dwt_smoothing
from src.experiment_dataset.dataset_experiment_2018_5_30 import AcousticEmissionDataSet_30_5_2018
from src.utils.helpers import *
from src.model_bank.dataset_2018_7_13_lcp_recognition_model import lcp_recognition_binary_model_2
from collections import deque
from itertools import islice
from scipy.signal import decimate

tdms_file = 'E:/Experiment_3_1_2019/-4,-2,2,4,6,8,10/1.5 bar/Leak/Train & Val data/2019.01.03_101026_001.tdms'

n_channel_data = read_single_tdms(tdms_file)
n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-1, :]  # drop last channel, due to no sensor

temp, temp2 = [], []
DOWNSAMPLE_FACTOR_1 = 50
DOWNSAMPLE_FACTOR_2 = 10

for channel in n_channel_data:
    temp.append(decimate(x=channel, q=DOWNSAMPLE_FACTOR_1))

for channel in temp:
    temp2.append(decimate(x=channel, q=DOWNSAMPLE_FACTOR_2))

temp2 = np.array(temp2)

print(temp2.shape)
print(temp2)

# l = [1, 4, 7, 1]
# file_dir = direct_to_dir(where='result') + 'test.csv'
#
# if not os.path.exists(file_dir):
#     df = pd.DataFrame(data=l, columns=['unseen_leak'])
#     df.to_csv(file_dir)
# else:
#     df = pd.read_csv(file_dir, index_col=0)
#     print(df)
#     df['seen_leak'] = l
#     df.to_csv(file_dir)

# unseen_data_labels = [
#                       'sensor@[-3m]',
#                       'sensor@[-2m]',
#                       'sensor@[0m]',
#                       'sensor@[5m]',
#                       'sensor@[7m]',
#                       'sensor@[16m]',
#                       'sensor@[17m]'
#                      ]
#
# seen_data_labels = [
#                     'sensor@[-4m]',
#                     'sensor@[-2m]',
#                     'sensor@[2m]',
#                     'sensor@[4m]',
#                     'sensor@[6m]',
#                     'sensor@[8m]',
#                     'sensor@[10m]'
#                    ]
#
# file_dir = direct_to_dir(where='result') + 'test.csv'
# df = pd.read_csv(file_dir)
#
# # calc mean unseen score
# unseen_mean_acc = np.mean([df['unseen_leak'].values, df['unseen_noleak']], axis=0)
# seen_mean_acc = np.mean([df['seen_leak'].values, df['seen_noleak']], axis=0)


# print(df['unseen_leak'].values)
# print(df['unseen_leak'].values.shape)






# file_dir = direct_to_dir(where='result') + 'LNL_25x1_result.txt'
#
# with open(file_dir) as file:
#     file_contents = file.read()
#     print(file_contents)


# FILENAME_TO_SAVE = 'result.txt'
# result_1 = 'acc2: 0.875'
# result_2 = 'acc3: 0.174'
# with open(FILENAME_TO_SAVE, 'a') as f:
#     f.write('\n' + result_1)
#
# with open(FILENAME_TO_SAVE, 'a') as f:
#     f.write('\n' + result_2)


# np.random.seed(42)
# print(np.random.permutation(10))
# print(np.random.permutation(37))



# tdms_test = direct_to_dir(where='result') + 'dataset_leak_random_1.5bar_[0]_ds.csv'
#
#
# test_df = pd.read_csv(tdms_test)
# print('dataset_leak_random_1.5bar_[0]_ds.csv DIM: ', test_df.values.shape)
#
# tdms_test = direct_to_dir(where='result') + 'dataset_noleak_random_1.5bar_[0]_ds.csv'
# test_df = pd.read_csv(tdms_test)
# print('dataset_noleak_random_1.5bar_[0]_ds.csv DIM: ', test_df.values.shape)
#
# tdms_test = direct_to_dir(where='result') + 'dataset_leak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds.csv'
# test_df = pd.read_csv(tdms_test)
# print('dataset_leak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds.csv DIM: ', test_df.values.shape)
#
# tdms_test = direct_to_dir(where='result') + 'dataset_noleak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds.csv'
# test_df = pd.read_csv(tdms_test)
# print('dataset_noleak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds.csv DIM: ', test_df.values.shape)


# print('Main Df DIM: ', test_df.values.shape)
# for i in range(7):
#     x = test_df.loc[test_df['channel'] == i].values[:, :-1]
#     print('Index[{}]: '.format(i), x.shape)

# fig1 = plot_multiple_timeseries(input=n_channel_data,
#                                 subplot_titles=['-4', '-2', '2', '4', '6', '8', '10'],
#                                 main_title='B4 Downsample')

# n_channel_data_downsampled = []
# time_start = time.time()
# for channel in n_channel_data:
#     n_channel_data_downsampled.append(decimate(x=channel, q=5))
# n_channel_data_downsampled = np.array(n_channel_data_downsampled)

# print('Time taken to downsample: {}s'.format(time.time()-time_start))
# print('Dime: ', n_channel_data_downsampled.shape)
# fig2 = plot_multiple_timeseries(input=n_channel_data_downsampled,
#                                 subplot_titles=['-4', '-2', '2', '4', '6', '8', '10'],
#                                 main_title='AFTER Downsample')
#
# plt.show()

# n_channel_data_downsampled_fft = []
# for channel in n_channel_data_downsampled:
#     f_mag_unseen, _, f_axis = fft_scipy(sampled_data=channel, fs=int(200e3), visualize=False)
#     n_channel_data_downsampled_fft.append(f_mag_unseen)
# label = ['-4', '-2', '2', '4', '6', '8', '10']
#
# for channel, l, index in zip(n_channel_data_downsampled, label, range(7)):
#     f_mag_unseen, _, f_axis = fft_scipy(sampled_data=channel, fs=int(200e3), visualize=False)
#     fig_fft = plt.figure(figsize=(14, 8))
#     ax_fft = fig_fft.add_subplot(1, 1, 1)
#     ax_fft.grid('on')
#     ax_fft.plot(f_axis[10:], f_mag_unseen[10:], alpha=0.5)
#     ax_fft.set_ylim(bottom=0, top=0.001)
#     ax_fft.set_title('FFT_{}'.format(l))
#     save_filename = direct_to_dir(where='result') + 'FFT_{}.png'.format(index)
#     fig_fft.savefig(save_filename)
#
#     plt.close('all')
#
# plt.show()


# n_channel_data_downsampled_fft = np.array(n_channel_data_downsampled_fft)
# fig = heatmap_visualizer(x_axis=f_axis,
#                          y_axis=[-4, -2, 2, 4, 6, 8, 10],
#                          zxx=n_channel_data_downsampled_fft,
#                          label=['position', 'frequency', 'amplitude'], output='3d')
#
# plt.show()



# tdms_dir_leak = ''
# tdms_dir_leak_2 =''
# all_tdms = [(tdms_dir_leak + f) for f in listdir(tdms_dir_leak) if f.endswith('.tdms')]
#
# n_channel_data = read_single_tdms(filename=all_tdms[30])
# n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-1, :]
#
# fig1 = plot_multiple_timeseries(input=n_channel_data,
#                                 subplot_titles=['-4', '-2', '2', '4', '6', '8', '10'],
#                                 main_title='Leak 21 Dec',
#                                 ylim=3)
#
# # ----
# all_tdms = [(tdms_dir_leak_2 + f) for f in listdir(tdms_dir_leak_2) if f.endswith('.tdms')]
#
# n_channel_data = read_single_tdms(filename=all_tdms[30])
# n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-1, :]
#
# fig2 = plot_multiple_timeseries(input=n_channel_data,
#                                 subplot_titles=['-3', '-2', '2', '4', '6', '8', '10', '12'],
#                                 main_title='Leak 13 July',
#                                 ylim=3)
#
# plt.show()

# unseen_data_filename = 'E:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/2 bar/No_Leak/test_0017.tdms'
# train_data_filename = 'E:/Experiment_2_10_2018/-4.5,-2,2,5,8,17,20,23/no_leak/test1_0017.tdms'
# train_data_filename_2 = 'E:/Experiment_2_10_2018/-4.5,-2,2,5,8,17,20,23/no_leak/test1_0040.tdms'
# unseen_data = read_single_tdms(unseen_data_filename)
# unseen_data = np.swapaxes(unseen_data, 0, 1)
# train_data = read_single_tdms(train_data_filename)
# train_data = np.swapaxes(train_data, 0, 1)
#
# # normalize
# scaler = MinMaxScaler(feature_range=(-1, 1))
# signal_1 = scaler.fit_transform(unseen_data[1].reshape(-1, 1)).ravel()
# signal_2 = scaler.fit_transform(train_data[1].reshape(-1, 1)).ravel()
#
# f_mag_unseen, _, f_axis = fft_scipy(sampled_data=signal_1, fs=int(1e6), visualize=False)
# f_mag_train, _, _ = fft_scipy(sampled_data=signal_2, fs=int(1e6), visualize=False)
#
# plt.plot(f_axis, f_mag_unseen, color='b', alpha=0.5, label='signal 1')
# plt.plot(f_axis, f_mag_train, color='r', alpha=0.5, label='signal 2')
# plt.grid('on')
# plt.legend()
# plt.show()


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

