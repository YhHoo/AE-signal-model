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
from src.utils.dsp_tools import fft_scipy, spectrogram_scipy, one_dim_xcor_2d_input, detect_ae_event_by_v_sensor, dwt_smoothing
from src.experiment_dataset.dataset_experiment_2018_5_30 import AcousticEmissionDataSet_30_5_2018
from src.utils.helpers import *
from src.model_bank.dataset_2018_7_13_lcp_recognition_model import lcp_recognition_binary_model_2
from collections import deque
from itertools import islice


pred = [0, 0, 0, 1, 1, 1, 2, 1, 1]
actual = [0, 0, 0, 1, 1, 1, 2, 2, 2]

# conf_mat = confusion_matrix(y_true=actual, y_pred=pred)
conf_mat = np.array([[3, 0, 0],
                     [0, 3, 0],
                     [0, 2, 1],
                     [0, 1, 1]])


# plot confusion matrix
def plot_confusion_matrix(cm, col_label, row_label,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    col_label and row_label starts from top left of the matrix
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks_col = np.arange(len(col_label))
    tick_marks_row = np.arange(len(row_label))
    plt.xticks(tick_marks_col, col_label, rotation=45)
    plt.yticks(tick_marks_row, row_label)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return fig


fig = plot_confusion_matrix(cm=conf_mat, col_label=['a', 'b', 'c'], row_label=['w', 'x', 'y', 'z'])

plt.show()

# array = [[33,2,0,0,0,0,0,0,0,1,3],
#         [3,31,0,0,0,0,0,0,0,0,0],
#         [0,4,41,0,0,0,0,0,0,0,1],
#         [0,1,0,30,0,6,0,0,0,0,1],
#         [0,0,0,0,38,10,0,0,0,0,0],
#         [0,0,0,3,1,39,0,0,0,0,4],
#         [0,2,2,0,4,1,31,0,0,0,2],
#         [0,1,0,0,0,0,0,36,0,2,0],
#         [0,0,0,0,0,0,1,5,37,5,1],
#         [3,0,0,0,0,0,0,0,0,39,0],
#         [0,0,0,0,0,0,0,0,0,0,38]]
# df_cm = pd.DataFrame(array, index = [i for i in "ABCDEFGHIJK"],
#                      columns = [i for i in "ABCDEFGHIJK"])
#
# conf_mat, recall_each_class, precision_each_class, f1_score = compute_recall_precision_multiclass(y_true=,
#                                                                                                   y_pred=,
#                                                                                                   all_class_label=,
#                                                                                                   verbose=True)


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

