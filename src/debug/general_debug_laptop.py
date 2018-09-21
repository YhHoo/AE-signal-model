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
from sklearn.metrics import accuracy_score
# self lib
from src.controlled_dataset.ideal_dataset import white_noise
from src.utils.dsp_tools import spectrogram_scipy, one_dim_xcor_2d_input, detect_ae_event_by_v_sensor
from src.experiment_dataset.dataset_experiment_2018_5_30 import AcousticEmissionDataSet_30_5_2018
from src.utils.helpers import *
from src.model_bank.dataset_2018_7_13_leak_localize_model import fc_leak_1bar_max_vec_v1

# file reading
dataset_filename = 'E:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/1 bar/Leak/processed/' + \
                   'lcp_recog_1bar_near_segmentation2_dataset.csv'
lcp_model = load_model(model_name='LCP_Recog_1')
lcp_model.compile(loss='binary_crossentropy', optimizer='rmsprop')


print('Reading data --> ', dataset_filename)
time_start = time.time()
data_df = pd.read_csv(dataset_filename)
print('File Read Time: {:.4f}s'.format(time.time() - time_start))
print('Full Dim: ', data_df.values.shape)

lcp_data = data_df.loc[data_df['label'] == 1].values[:, :-1]
non_lcp_data = data_df.loc[data_df['label'] == 0].values[:, :-1]

fig = plot_multiple_timeseries(input=[lcp_data[10], non_lcp_data[10]],
                               subplot_titles=['LCP', 'Non LCP'],
                               main_title='LCP and Non LCP input')

lcp_data_test = lcp_data[10].reshape((6000, 1))
non_lcp_data_test = non_lcp_data[10].reshape((6000, 1))

activation = get_activations(lcp_model, model_inputs=[lcp_data_test, non_lcp_data_test], print_shape_only=True)
print(len(activation))

# first cnn layer
activation_test = np.swapaxes(activation[5], 1, 2)

fig2 = plot_multiple_timeseries(input=activation_test[0],
                                subplot_titles=['k1', 'k2', 'k3', 'k4', 'k5'],
                                main_title='cnn1d_1 activation [LCP]')

fig3 = plot_multiple_timeseries(input=activation_test[1],
                                subplot_titles=['k1', 'k2', 'k3', 'k4', 'k5'],
                                main_title='cnn1d_1 activation [NON LCP]')

plt.show()


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

