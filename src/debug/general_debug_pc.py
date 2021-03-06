import numpy as np
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
import peakutils
import os
from scipy.signal import decimate
from os import listdir
from scipy.signal import correlate
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, LabelBinarizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
import csv
import argparse
from sklearn.metrics import confusion_matrix
# self lib
# from src.controlled_dataset.ideal_dataset import white_noise
# from src.utils.dsp_tools import spectrogram_scipy, one_dim_xcor_2d_input, dwt_smoothing, one_dim_xcor_1d_input
from src.experiment_dataset.dataset_experiment_2018_10_3 import AcousticEmissionDataSet_3_10_2018
import matplotlib.patches as mpatches
from src.utils.helpers import *
# from src.model_bank.dataset_2018_7_13_leak_localize_model import fc_leak_1bar_max_vec_v1

print(10//3)



# tdms_file = 'G:/Experiment_3_1_2019/-4,-2,2,4,6,8,10/1.5 bar/Leak/Train & Val data/2019.01.03_101106_009.tdms'
#
# n_channel_data = read_single_tdms(tdms_file)
# n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-1, :]  # drop last channel, due to no sensor
# print(n_channel_data.shape)
#
# fig1 = plot_multiple_timeseries(input=n_channel_data,
#                                 subplot_titles=['-4', '-2', '2', '4', '6', '8', '10'],
#                                 main_title='Fs=1MHz')
# temp, temp2, temp3 = [], [], []
#
# for channel in n_channel_data:
#     temp3.append(decimate(x=channel, q=20, ftype='iir'))
#
# fig3 = plot_multiple_timeseries(input=temp3,
#                                 subplot_titles=['-4', '-2', '2', '4', '6', '8', '10'],
#                                 main_title='single')

# # first downsample
# for channel in n_channel_data:
#     temp.append(decimate(x=channel, q=50))
# # second downsample
# for channel in temp:
#     temp2.append(decimate(x=channel, q=10))
#
# n_channel_data = np.array(temp2)
# print('Dim After Downsample: ', n_channel_data.shape)
#
#
# fig2 = plot_multiple_timeseries(input=n_channel_data,
#                                 subplot_titles=['-4', '-2', '2', '4', '6', '8', '10'],
#                                 main_title='double')

plt.show()


# filename = 'G:/Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_leak_random_1.5bar_[0]_ds5.csv'
# data = pd.read_csv(filename)
# print(data.values.shape)
# print(data.head())




# plt.plot(n_channel_data.values[1045, :-1])
# plt.show()

# directory = direct_to_dir(where='result') + 'new_one/'
# if not os.path.exists(directory):
#     os.makedirs(directory)
#
# df_pred = pd.DataFrame(data=np.arange(9).reshape((3, 3)),
#                        columns=[1, 2, 3])
# directory = direct_to_dir(where='result') + 'new_one/'
#
# for i in range(4, 5, 1):
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#         filename = directory + 'x_{}.csv'.format(i)
#         df_pred.to_csv(filename)
#     else:
#         filename = directory + 'x_{}.csv'.format(i)
#         df_pred.to_csv(filename)



# change the filename to the one we wish to norm
# dataset_noleak_rand_filename = direct_to_dir(where='result') + 'dataset_leak_random_1bar_2.csv'
# print('Reading --> ', dataset_noleak_rand_filename)
# df_data = pd.read_csv(dataset_noleak_rand_filename)
# print(df_data.values.shape)
#
# label_to_dist = {0: 'nonLCP',
#                  1: '2m',
#                  2: '4.5m',
#                  3: '5m',
#                  4: '8m',
#                  5: '10m'}
#
# lcp_dataset_filename = 'F:/Experiment_3_10_2018/LCP x NonLCP DATASET/' \
#                        'dataset_lcp_1bar_seg4_norm.csv'
# non_lcp_dataset_filename = 'F:/Experiment_3_10_2018/LCP x NonLCP DATASET/' \
#                            'dataset_non_lcp_1bar_seg1_norm.csv'
#
# random_leak_dataset_filename = 'F:/Experiment_3_10_2018/LCP x NonLCP DATASET/' \
#                                'dataset_leak_random_1bar_norm.csv'
# random_noleak_dataset_filename = 'F:/Experiment_3_10_2018/LCP x NonLCP DATASET/' \
#                                'dataset_noleak_random_2bar_norm.csv'
#
# # reading lcp data fr csv
# time_start = time.time()
# print('Reading --> ', lcp_dataset_filename)
# df_leak_rand = pd.read_csv(lcp_dataset_filename)
#
# print('File Read Time: {:.4f}s'.format(time.time() - time_start))
# print('Random Leak Dataset Dim: ', df_leak_rand.values.shape)
#
# print(df_leak_rand.head(100))
#
# print('Reading --> ', random_noleak_dataset_filename)
# df_noleak_rand = pd.read_csv(random_noleak_dataset_filename)
#
# print('File Read Time: {:.4f}s'.format(time.time() - time_start))
# print('Random no leak Dataset Dim: ', df_noleak_rand.values.shape)

# [WARNING] PLOTTING TAKES FOREVER TO PLOT -----------------------------------------------------------------------------
# color_seq = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
# color_label = ['-4.5m', '-2m', '2m', '5m', '8m', '10m']
# color_dict = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm', 5: 'y'}
# r_leg = mpatches.Patch(color='r', label=color_label[0])
# g_leg = mpatches.Patch(color='g', label=color_label[1])
# b_leg = mpatches.Patch(color='b', label=color_label[2])
# c_leg = mpatches.Patch(color='c', label=color_label[3])
# m_leg = mpatches.Patch(color='m', label=color_label[4])
# y_leg = mpatches.Patch(color='y', label=color_label[5])
#
# figure_result = plt.figure(figsize=(5, 8))
# figure_result.suptitle('Model prediction by 6k Sliding Window, Stride: {}'.format(window_stride),
#                        fontweight="bold",
#                        size=8)
# figure_result.subplots_adjust(hspace=0.7, top=0.9, bottom=0.03)
# ax1 = figure_result.add_subplot(6, 1, 1)
# ax1.set_title(color_label[0])
# ax1.plot(n_channel_data[0])
#
# # label the result
# for pred_index, pred in zip(window_index, prediction_all_ch[0]):
#     ax1.axvline(pred_index, color=color_dict[pred])
#
# for plot_no, plot_label, raw_signal, prediction_per_ch in zip(np.arange(2, 7, 1),
#                                                               color_label[1:],
#                                                               n_channel_data[1:],
#                                                               prediction_all_ch[1:]):
#     ax2 = figure_result.add_subplot(6, 1, plot_no, sharex=ax1)
#     ax2.set_title(plot_label)
#     ax2.plot(raw_signal)
#
#     # label the result
#     for pred_index, pred in zip(window_index, prediction_per_ch):
#         ax1.axvline(pred_index, color=color_dict[pred])
#
#
# plt.legend(handles=[r_leg, g_leg, b_leg, c_leg, m_leg, y_leg])
# plt.show()
#
#
# r_leg = mpatches.Patch(color='r', label='The red data')
# g_leg = mpatches.Patch(color='g', label='The green data')
# b_leg = mpatches.Patch(color='b', label='The blue data')
# c_leg = mpatches.Patch(color='c', label='The cyan data')
# m_leg = mpatches.Patch(color='m', label='The magenta data')
# y_leg = mpatches.Patch(color='y', label='The da data')
# k_leg = mpatches.Patch(color='k', label='The di data')
#
# l = np.linspace(0, 12, 100)
# print(l)
# x = [1, 5.4, 7, 11, 30, 40, 50]
# plt.plot(l)
# color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
# for c, i in zip(color, x):
#     plt.axvline(i, color=c)
# plt.legend(handles=[r_leg, g_leg, b_leg, c_leg, m_leg, y_leg, k_leg])
# plt.show()
#
# color_dict = {0: 'r', 1: 'g', 2:'b', 3:'c', 4:'m', 5:'y'}
#
# print(color_dict[5])
# ogger.save_recall_precision_f1(y_pred=prediction, y_true=actual_argmax, all_class_label=[0, 1, 2, 3, 4, 5])
# dataset_dir = 'F:/Experiment_3_10_2018/LCP x NonLCP DATASET/'
# dataset_lcp_filename = dataset_dir + 'dataset_lcp_1bar_seg4.csv'
# dataset_non_lcp_filename = dataset_dir + 'dataset_non_lcp_1bar_seg1_norm.csv'
# dataset_normalized_save_filename = direct_to_dir(where='result') + 'norm.csv'
#
# print('Reading --> ', dataset_non_lcp_filename)
# df_data = pd.read_csv(dataset_non_lcp_filename)
# column_label = df_data.columns.values
#
# print(df_data)
# print(df_data.values)
# print(df_data.values.shape)
#
# # drop rows that contains Nan
# df_data.dropna(inplace=True)
#
# df2 = pd.DataFrame(data=df_data.values, columns=column_label)
# df2.to_csv(dataset_normalized_save_filename, index=False)
#
# data_arr = df_data.values[:, :-1]
# label_arr = df_data.values[:, -1].reshape(-1, 1)
#
# temp = []
# scaler = MinMaxScaler(feature_range=(-1, 1))
#
# for row in data_arr:
#     temp.append(scaler.fit_transform(row.reshape(-1, 1)).ravel())
#
# temp = np.array(temp)
#
# print('INPUT DATA DIM:', df_data.values.shape)
#
# combine_arr = np.concatenate((temp, label_arr), axis=1)
#
# print('AFTER COMBINED DIM: ', combine_arr.shape)
#
# df_data_norm = pd.DataFrame(data=combine_arr, columns=column_label)
# df_data_norm.to_csv(dataset_normalized_save_filename, index=False)

# ----------------------------------------------------------------------------------------------------------------------

# fig = plot_multiple_timeseries(input=n_channel_data_near_leak[:, :2500000],
#                                subplot_titles=['-4.5m', '-2m', '2m', '5m', '8m', '17m', '20m', '23m'],
#                                main_title='test')
#
# plt.show()
# filename = 'F:\Experiment_13_7_2018\Experiment 1\-3,-2,2,4,6,8,10,12\LCP DATASET\dataset_lcp_2bar_near_seg3.csv'
#
# time_start = time.time()
# df = pd.read_csv(filename)
# print('Time Lapsed: {:.4f}s'.format(time.time() - time_start))
# print(df.head())
#
# random_pick = [0, 10, 15, 17, 19, 30]
#
# for r in random_pick:
#     temp = []
#     for i in range(8):
#         x = df[df['channel'] == i].values[r]
#         temp.append(x)
#
#     fig = plot_multiple_timeseries(input=temp,
#                                    subplot_titles=['-3m [0]', '-2m [1]', '2m [2]', '4m [3]',
#                                                    '6m [4]', '8m [5]', '10m [6]', '12m [7]'],
#                                    main_title='Random pick of the lcp data')
#
#     plt.savefig('lcp_sample[{}]'.format(r))

# lcp_df = pd.DataFrame()
# lcp_df['lcp'] = [1, 2, 3]
# lcp_df['filename'] = [4, 5, 6]
# print(lcp_df)
#
# l = [[1, 0, 1],
#      [0, 1, 1],
#      [1, 1, 1]]
# df = pd.DataFrame(data=l, columns=['ch0', 'ch1', 'ch2'])
# agg = pd.concat([lcp_df, df], axis=1)
# print(agg)

# header = ['0', '1', 'label']
#
# with open('test.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(header)
#
# for row in l:
#     with open('test.csv', 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(row)
#
# for row in l:
#     print([str(i) for i in row])
#     with open('test.csv', 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerows([str(i) for i in row])





# lcp_filename = direct_to_dir(where='result') + 'lcp_dataset_1.csv'
# lcp_dataset_df = pd.read_csv(lcp_filename, index_col=0)
# print(lcp_dataset_df.values.shape)
# print(lcp_dataset_df.values)

# folder_path = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/1 bar/Leak/'
# all_file_path = [(folder_path + f) for f in listdir(folder_path) if f.endswith('.tdms')]
#
# n_channel_data_near_leak = read_single_tdms(all_file_path[0])
# n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)
# print('Read Data Dim: ', n_channel_data_near_leak.shape)
#
# soi = n_channel_data_near_leak[1, 34350-1000:34350+5000]
# env = peakutils.envelope(y=soi, deg=100, tol=0.001)
# print(env.shape)
# print(soi.shape)
# plt.plot(soi, label='ori')
# plt.plot(env, label='env')
#
# plt.grid(linestyle='dotted')
# plt.show()

# dwt_wavelet = 'db2'
# dwt_smooth_level = 4
# cwt_wavelet = 'gaus1'
# scale = np.linspace(2, 30, 100)
# fs = 1e6
#
#
# folder_path = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/1 bar/No_Leak/'
# all_file_path = [(folder_path + f) for f in listdir(folder_path) if f.endswith('.tdms')]
#
# # read oni one tdms
# n_channel_data_near_leak = read_single_tdms(all_file_path[0])
# n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)
#
# # visualize in time
# title = np.arange(0, 8, 1)
# fig_0 = plot_multiple_timeseries(input=n_channel_data_near_leak,
#                                  subplot_titles=title,
#                                  main_title='5 sec AE data of 1bar leak')
#
# plt.show()
# # denoising
# temp = []
# # for all channel of sensor
# for signal in n_channel_data_near_leak:
#     denoised_signal = dwt_smoothing(x=signal, wavelet=dwt_wavelet, level=dwt_smooth_level)
#     temp.append(denoised_signal)
# n_channel_data_near_leak = np.array(temp)
# # segment of interest
# n_channel_data_near_leak = n_channel_data_near_leak[:, 3278910:3296410]
#
# # method 1
# pos1_leak_cwt, _ = pywt.cwt(n_channel_data_near_leak[1], scales=scale, wavelet=cwt_wavelet, sampling_period=1 / fs)
# pos2_leak_cwt, _ = pywt.cwt(n_channel_data_near_leak[7], scales=scale, wavelet=cwt_wavelet, sampling_period=1 / fs)
#
# xcor_1, _ = one_dim_xcor_2d_input(input_mat=np.array([pos1_leak_cwt, pos2_leak_cwt]), pair_list=[(0, 1)])
# xcor_1 = xcor_1[0]
# print('XCOR DIM 1: ', xcor_1.shape)
#
# fig_1 = plot_cwt_with_time_series(time_series=[n_channel_data_near_leak[1], n_channel_data_near_leak[7]],
#                                   no_of_time_series=2,
#                                   cwt_mat=xcor_1,
#                                   cwt_scale=scale,
#                                   maxpoint_searching_bound=(3296410-3278910))
#
# # method 2
# x_cor_1d = correlate(n_channel_data_near_leak[1], n_channel_data_near_leak[7], 'full', method='fft')
#
#
# xcor_len = x_cor_1d.shape[0]
# xcor_axis = np.arange(1, xcor_len + 1, 1) - xcor_len // 2 - 1
# # plt.axvline(x=)
#
# # plotting
#
# pos1_leak_cwt, _ = pywt.cwt(x_cor_1d, scales=scale, wavelet=cwt_wavelet, sampling_period=1 / fs)
# # pos2_leak_cwt, _ = pywt.cwt(n_channel_data_near_leak[2], scales=scale, wavelet=cwt_wavelet, sampling_period=1 / fs)
#
# fig_2 = plot_cwt_with_time_series(time_series=[x_cor_1d, x_cor_1d],
#                                   no_of_time_series=2,
#                                   cwt_mat=pos1_leak_cwt,
#                                   cwt_scale=scale,
#                                   maxpoint_searching_bound=(3296410 - 3278910))
#
#
# plt.show()


# plt.show()
#
#
# sensor_pair_near = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]
#
# dist_diff = 0
# # for all sensor combination
# for sensor_pair in sensor_pair_near:
#     signal_1 = n_channel_data_near_leak[sensor_pair[0]]
#     signal_2 = n_channel_data_near_leak[sensor_pair[1]]
#
#     # cwt
#     pos1_leak_cwt, _ = pywt.cwt(signal_1, scales=scale, wavelet=cwt_wavelet, sampling_period=1 / fs)
#     pos2_leak_cwt, _ = pywt.cwt(signal_2, scales=scale, wavelet=cwt_wavelet, sampling_period=1 / fs)
#
#     # xcor for every pair of cwt
#     xcor, _ = one_dim_xcor_2d_input(input_mat=np.array([pos1_leak_cwt, pos2_leak_cwt]), pair_list=[(0, 1)])
#     xcor = xcor[0]
#
#     # visualizing
#     fig_title = 'Xcor of CWT of Sensor[{}] and Sensor[{}] -- Dist_Diff[{}m] --[3278910:3296410]'.format(sensor_pair[0],
#                                                                                                       sensor_pair[1],
#                                                                                                       dist_diff)
#
#     fig = plot_cwt_with_time_series(time_series=[signal_1, signal_2],
#                                     no_of_time_series=2,
#                                     cwt_mat=xcor,
#                                     cwt_scale=scale,
#                                     title=fig_title,
#                                     maxpoint_searching_bound=(3296410-3278910))
#
#     # plt.show()
#
#     saving = True
#     if saving:
#         filename = direct_to_dir(where='result') + \
#                    'xcor_cwt_DistDiff[{}m]--[3278910_3296410]'.format(dist_diff)
#
#         fig.savefig(filename)
#         plt.close('all')
#         print('Saving --> Dist_diff: {}m'.format(dist_diff))
#
#     dist_diff += 1





# # for all tdms
# for tdms_file in all_file_path:
#     # read raw from drive
#     n_channel_data_near_leak = read_single_tdms(tdms_file)
#     n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)
#
#     temp = []
#     for signal in n_channel_data_near_leak:
#         denoised_signal = dwt_smoothing(x=signal, wavelet=dwt_wavelet, level=dwt_smooth_level)
#         temp.append(denoised_signal)
#     n_channel_data_near_leak = np.array(temp)
#     print(n_channel_data_near_leak.shape)
#
#     title = np.arange(0, 8, 1)
#     fig = plot_multiple_timeseries(input=n_channel_data_near_leak,
#                                    subplot_titles=title,
#                                    main_title='5 sec AE data of 1bar leak')
#     plt.show()


# SCRIPT FOR VISUALIZING TSNE DATA SAVED IN CSV-------------------------------------------------------------------------
# tsne_filename = direct_to_dir(where='result') + 'cwt_xcor_maxpoints_vector_dataset_bounded_xcor_3_(TSNE_5k_epoch_100_per).csv'
# tsne_df = pd.read_csv(tsne_filename, index_col=0)
# dataset = tsne_df.values[:, :2]
# label = tsne_df.values[:, -1]
# fig = scatter_plot(dataset=dataset,
#                    label=label,
#                    num_classes=11,
#                    feature_to_plot=(0, 1),
#                    annotate_all_point=True,
#                    title='cwt_xcor_maxpoints_vector_dataset_bounded_xcor_3_(TSNE_5k_epoch_100_per)')
#
# plt.show()


# l = [0, 0, 1, 0, 7, 11, 5, 2, 0, 0]
# m = [0, 7, 11, 5, 2, 0, 0, 1, 2, 2]
# z = correlate_scipy(in1=l, in2=m, mode='full', method='fft')
# print(z)

# testing grid search CV of sklearn ------------------------------------------------------------------------------------

# Use scikit-learn to grid search the batch size and epochs
# import numpy
# from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier

# def create_model():
#     # create model
#     model = Sequential()
#     model.add(Dense(12, input_dim=8, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
#
# # fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)
# # load dataset
# dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:, 0:8]
# Y = dataset[:, 8]
# # create model
# model = KerasClassifier(build_fn=create_model, verbose=0)
# # define the grid search parameters
# batch_size = [10, 20, 40, 60, 80, 100]
# epochs = [10, 50, 100]
# param_grid = dict(batch_size=batch_size, epochs=epochs)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
# grid_result = grid.fit(X, Y)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# ------------------------------------[TESTING MAX VEC GENERATOR]-------------------------------------------------------
# # wavelet
# m_wavelet = 'gaus1'
# scale = np.linspace(2, 30, 50)
# fs = 1e6
# # creating dict to store each class data
# all_class = {}
# for i in range(0, 11, 1):
#     all_class['class_[{}]'.format(i)] = []
#
#
# tdms_dir = direct_to_dir(where='yh_laptop_test_data') + '/1bar_leak/test_0001.tdms'
# # read raw from drive
# n_channel_data_near_leak = read_single_tdms(tdms_dir)
# n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)
#
# # split on time axis into no_of_segment
# n_channel_leak = np.split(n_channel_data_near_leak, axis=1, indices_or_sections=50)
#
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
# dataset = []
# label = []
# for i in range(0, 11, 1):
#     max_vec_list_of_each_class = all_class['class_[{}]'.format(i)]
#     dataset.append(max_vec_list_of_each_class)
#     label.append([i]*len(max_vec_list_of_each_class))
# dataset = np.concatenate(dataset, axis=0)
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
# column_label = ['Scale_{}'.format(i) for i in scale] + ['label']
# df = pd.DataFrame(all_in_one, columns=column_label)
# filename = direct_to_dir(where='result') + 'test.csv'
# df.to_csv(filename)

# ----------------------------------------------------------------------------------------------------------------------

# l = []
# for key, value in all_class.items():
#     l.append(value)
#
# l = np.array(l)
# print(l.shape)


# input = np.arange(300).reshape((10, 30))
#
# x = np.unravel_index(np.argmax(input, axis=None), input.shape)
# print(x)
# print(x[0])
# fig = plt.figure()
# cwt_ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
# colorbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.01])
# # title
# cwt_ax.set_title('Xcor of CWT')
# # plot
# cwt_ax.grid(linestyle='dotted')
# cwt_ax.axvline(x=input.shape[1] // 2 + 1, linestyle='dotted')
# cwt_ax.scatter(x[1], x[0], s=70, c='black', marker='x')
# cwt_ax.set_yticks([10, 50, 60, 70, 80, 82, 85, 89, 93, 99])
# i = cwt_ax.imshow(input, cmap='jet', aspect='auto', extent=[0, 30, 10, 100])
# plt.colorbar(i, cax=colorbar_ax, orientation='horizontal')
#
#
# plt.show()



# input = np.random.rand(3000).reshape((10, 300))
# act_1 = np.random.rand(2619).reshape((9, 291))
# act_2 = np.random.rand(2619).reshape((9, 291))
# act_3 = np.random.rand(2619).reshape((9, 291))
# act_4 = np.random.rand(2619).reshape((9, 291))
# act_5 = np.random.rand(2619).reshape((9, 291))
# act_6 = np.random.rand(2619).reshape((9, 291))
# act_7 = np.random.rand(2619).reshape((9, 291))
# act_8 = np.random.rand(2619).reshape((9, 291))
#
# act_11 = np.linspace(0, 1, 2619).reshape((9, 291))
# act_21 = np.linspace(0, 1, 2619).reshape((9, 291))
# act_31 = np.linspace(0, 1, 2619).reshape((9, 291))
# act_41 = np.linspace(0, 1, 2619).reshape((9, 291))
# act_51 = np.linspace(0, 1, 2619).reshape((9, 291))
# act_61 = np.linspace(0, 1, 2619).reshape((9, 291))
# act_71 = np.linspace(0, 1, 2619).reshape((9, 291))
# act_81 = np.linspace(0, 1, 2619).reshape((9, 291))
#
# val_test = [act_1, act_2, act_3, act_4, act_5, act_6, act_7, act_8]
# val_test_2 = [act_11, act_21, act_31, act_41, act_51, act_61, act_71, act_81]
# fig = plot_multiple_horizontal_heatmap(val_test, 'BIG TITLE', 'BIG TITLE')


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')


# col_labels = ['TargetLabel_class_1', 'TargetLabel_class_2', 'TargetLabel_class_3']
# index_labels = ['Predicted_class_1', 'Predicted_class_2', 'Predicted_class_3']
# conf_mat = pd.DataFrame(data=data.T, index=index_labels, columns=col_labels)
# # conf_mat['Total Prediction of Each Class'] = pd.DataFrame.sum(conf_mat, axis=1)
# diag = np.diag(conf_mat.values)
# total_pred_of_each_class = pd.DataFrame.sum(conf_mat, axis=1).values
# total_samples_of_each_class = pd.DataFrame.sum(conf_mat, axis=0).values
#
# recall_each_class = diag / total_samples_of_each_class
# precision_each_class = diag / total_pred_of_each_class
# print(conf_mat)
# print(diag)
# print(total_pred_of_each_class)
# print(total_samples_of_each_class)
# print('class recall: ', recall_each_class)
# print('class precision: ', precision_each_class)

# precision_c1 = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='micro')
# print(precision_c1)
# print(precision_c2)
# print(precision_c3)

# x = [1, 2, 3]
# dct = {}
# for i in x:
#     dct['lst_%s' % i] = []
#
# dct['lst_1'].append(np.arange(0, 10).reshape((2, 5)))
# dct['lst_1'].append(np.arange(5, 15).reshape((2, 5)))
# print(dct['lst_1'][0])


# x = [[i, i+1, i+2] for i in range(10)]
# print(x)
# shuffle(x)
# print(x)
#
#
# all_class = {}
# for i in range(-20, 21, 1):
#     all_class['class_[{}]'.format(i)] = []
#
# for i in range(5):
#     all_class['class_[0]'].append(np.linspace(0, i, 5))
#
# print(all_class['class_[0]'])
# l = np.array(all_class['class_[0]'])
# print(l)
# print(l.shape)

# shuffle(all_class['class_[0]'])
# # l = np.array(all_class['class_[0]'])
# # print(l)
#
# print(all_class['class_[0]'])
#
# all_class['class_[0]'] = all_class['class_[0]'][:2]
#
# print(all_class['class_[0]'])


# x = np.arange(0, 30, 1).reshape(2, 3, 5)
# x_shift = np.swapaxes(x, 1, 2)
# print(x)
# print(x_shift)

# fig = plt.figure(figsize=(5, 6))
# fig.suptitle('XCOR MAP of Leak Pos 1 & 2')
# ax1 = fig.add_axes([0.1, 0.51, 0.6, 0.39])  # [left, bottom, width, height]
# ax2 = fig.add_axes([0.1, 0.1, 0.6, 0.39])
# colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
# i = ax1.pcolormesh(np.arange(0, 600), np.arange(0, 41), values1)
# j = ax2.pcolormesh(np.arange(0, 600), np.arange(0, 41), values2)
# fig.colorbar(i, cax=colorbar_ax)
# fig.colorbar(j, cax=colorbar_ax)
# ax1.grid()
# ax2.grid()
# ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


# -----------------------------------------------------------------------------
# fig = plt.figure(figsize=(5, 8))
# ax1 = fig.add_subplot(4, 1, 1)
# ax2 = fig.add_subplot(4, 1, 2)
# ax3 = fig.add_subplot(4, 1, 3)
# ax4 = fig.add_subplot(4, 1, 4)
# ax1.set_title('Signal 1')
# ax2.set_title('Signal 2')
# ax3.set_title('Xcor Signal [numpy]')
# ax4.set_title('Xcor Signal [scipy + fft]')
# ax1.plot(l)
# ax2.plot(m)
# ax3.plot(z)
# ax4.plot(z2)
# plt.subplots_adjust(hspace=0.6)
# plt.show()

# t = np.linspace(0, 10, 11)
# f = np.linspace(10, 100, 11)
# mat = np.arange(0, 100, 1).reshape((10, 10))
# print(t.shape)
# print(f.shape)
# print(mat.shape)
# print(mat)
# x_axis = np.arange(1, 11, 1)
# y_axis = np.arange(1, 11, 1)
#
# fig = plt.figure()
# ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
# colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
# i = ax.pcolormesh(x_axis, y_axis, mat)
# ax.grid()
# fig.colorbar(i, cax=colorbar_ax)
# ax.grid()
# ax.set_xlabel('Time [Sec]')
# ax.set_ylabel('Frequency [Hz]')
# ax.set_ylim(bottom=0, top=6, auto=True)
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax.set_title(plot_title)

# plt.show()

#
#
# assert m == item for i in l

# def colormapplot():
#     fig = plt.figure()
#     ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
#     colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
#     i = ax.pcolormesh(t, f, mat)
#     ax.grid()
#     fig.colorbar(i, cax=colorbar_ax)
#
#     return fig
#
#
# for i in range(3):
#     _ = colormapplot()
#     plt.close()
#
# fig1 = colormapplot()
# plt.show()


# three_dim_visualizer()
# data_3d = np.array([[[1],
#                      [3]],
#                     [[2],
#                      [4]],
#                     [[3],
#                      [-5]]])
# print(data_3d.min())
# print(data_3d.shape)
# print(data_3d.shape[1])
# print(data_3d[0].shape[0])

# s = np.arange(0, 100, 1).reshape((5, 20))
# x = np.arange(0, 20, 1)
# y = np.arange(0, 5, 1)
# print(s)
# print(x)
# print(y)
# mlb.barchart(y, x, s)
# mlb.imshow()

# sig = np.repeat([0., 1., 0., 1], 100)
# win = signal.hann(50)
# print(sum(win))
# mat1 = np.array([3, 9, 2, 1, 0, 0, 0, 0])
# mat2 = np.array([0, 3, 9, 2, 1, 0, 0, 0])
# plt.plot(sig)
# plt.plot(win, marker='x')
# plt.show()


# ori_signal = np.concatenate((mat1, mat3), axis=0)
# lag_signal = np.concatenate((mat2, mat4), axis=0)
# print(ori_signal.shape)


